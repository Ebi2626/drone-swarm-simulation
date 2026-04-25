import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.algorithms.avoidance.AStarAvoidance import AStarAvoidance
from src.algorithms.avoidance.BaseAvoidance import EvasionContext
from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import KinematicState, ThreatAlert

@pytest.fixture
def mock_context():
    """Przygotowuje zunifikowany EvasionContext z fałszywym bazowym splinem."""
    drone_state = KinematicState(
        position=np.array([0.0, 0.0, 5.0]), 
        velocity=np.array([5.0, 0.0, 0.0]), 
        radius=0.4
    )
    obs_state = KinematicState(
        position=np.array([5.0, 0.0, 5.0]), 
        velocity=np.array([0.0, 0.0, 0.0]), 
        radius=0.5
    )
    threat = ThreatAlert(
        obstacle_state=obs_state, 
        distance=5.0, 
        time_to_collision=1.0, 
        relative_velocity=np.array([5.0, 0.0, 0.0])
    )
    
    # Mockujemy BSplineTrajectory, by uniknąć trudności z inicjalizacją krzywych w testach
    mock_base_spline = MagicMock()
    mock_base_spline.profile.cruise_speed = 5.0
    mock_base_spline.profile.max_accel = 2.0
    mock_base_spline.arc_length = 100.0
    
    return EvasionContext(
        drone_id=1,
        current_time=10.0,
        drone_state=drone_state,
        threat=threat,
        base_spline=mock_base_spline,
        rejoin_point=np.array([10.0, 0.0, 5.0]),
        rejoin_base_arc=10.0,
        world_bounds=(np.array([0.0, -10.0, 0.0]), np.array([20.0, 10.0, 10.0])),
        search_space_min=np.array([0.0, -5.0, 0.0]),
        search_space_max=np.array([10.0, 5.0, 10.0])
    )

@pytest.fixture
def planner():
    """Zwraca instancję planisty z WYŁĄCZONĄ wizualizacją (nie śmiecimy na dysku)."""
    return AStarAvoidance(visualize=False, grid_resolution=0.5)


def test_pick_preferred_axis_free_space(planner):
    """Weryfikuje heurystykę wyboru optymalnego kierunku uniku."""
    current_pos = np.array([0.0, 0.0, 5.0])
    obs_pos = np.array([5.0, 0.0, 5.0])
    forward_xy = np.array([1.0, 0.0, 0.0])
    lateral_xy = np.array([0.0, 1.0, 0.0])
    world_min = np.array([0.0, -10.0, 0.0])
    world_max = np.array([20.0, 10.0, 10.0])
    
    # Sufit jest na 10m (5m wolnego), podłoga na 0m (5m wolnego)
    # Zgodnie z listą preferencji ["up", "down", "right", "left"], powinien wybrać "up"
    axis_name, pdir = planner._pick_preferred_axis(
        current_pos, obs_pos, forward_xy, lateral_xy,
        floor_z=1.0, ceiling_z=9.0, world_min=world_min, world_max=world_max
    )
    
    assert axis_name == "up"
    np.testing.assert_array_equal(pdir, np.array([0.0, 0.0, 1.0]))


def test_pick_preferred_axis_blocked_up(planner):
    """Sprawdza wybór, gdy preferowany kierunek jest zablokowany (np. lot blisko sufitu)."""
    current_pos = np.array([0.0, 0.0, 8.5])  # Dron pod sufitem
    obs_pos = np.array([5.0, 0.0, 8.5])
    forward_xy = np.array([1.0, 0.0, 0.0])
    lateral_xy = np.array([0.0, 1.0, 0.0])
    
    # Zablokowany sufit (ceiling_z=9.0, zostaje 0.5m, a wymaga > 2.5m)
    axis_name, pdir = planner._pick_preferred_axis(
        current_pos, obs_pos, forward_xy, lateral_xy,
        floor_z=1.0, ceiling_z=9.0, 
        world_min=np.array([0.0, -10.0, 0.0]), 
        world_max=np.array([20.0, 10.0, 10.0])
    )
    
    assert axis_name == "down"
    np.testing.assert_array_equal(pdir, np.array([0.0, 0.0, -1.0]))


def test_douglas_peucker(planner):
    """Sprawdza upraszczanie ścieżki (Ramer-Douglas-Peucker)."""
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.05, 0.0],  # Ten punkt leży niemal w linii prostej i powinien wylecieć
        [2.0, 0.0, 0.0],
        [2.0, 5.0, 0.0]    # Ten punkt to ostre zagięcie, musi zostać
    ])
    
    simplified = planner._douglas_peucker(points, epsilon=0.1)
    
    assert len(simplified) == 3
    np.testing.assert_array_equal(simplified[0], points[0])
    np.testing.assert_array_equal(simplified[1], points[2])
    np.testing.assert_array_equal(simplified[2], points[3])


@patch("src.algorithms.avoidance.AStarAvoidance.splev")
@patch("src.algorithms.avoidance.AStarAvoidance.BSplineTrajectory")
@patch("src.algorithms.avoidance.AStarAvoidance.UAV3DGridSearch")
def test_compute_evasion_plan_success(mock_grid_search_cls, mock_bspline, mock_splev, planner, mock_context):
    """
    Testuje poprawny przepływ (Happy Path).
    A* znajduje ścieżkę, zwracana jest poprawna instancja EvasionPlan.
    """
    # Zabezpieczenie mock_splev. Gdy wywoływane wewnątrz AStarAvoidance do oceny
    # stycznej bazowej z _base_tangent_at_arc (der=1) albo pozycji.
    mock_splev.return_value = [1.0, 0.0, 0.0]  # sztuczny zwrot wartości zamiast błędu z unpack
    
    # Przygotowanie mocka dla A*
    mock_searcher = MagicMock()
    mock_grid_search_cls.return_value = mock_searcher
    
    # Symulacja znalezionej łamanej przez A*
    mock_searcher.astar.return_value = [
        (0.0, 0.0, 5.0),
        (2.5, 0.0, 6.0),
        (5.0, 0.0, 7.0),
        (7.5, 0.0, 6.0),
        (10.0, 0.0, 5.0)
    ]
    
    # Przygotowanie mocka dla wynikowego Splinu
    mock_evasion_spline_instance = MagicMock()
    mock_bspline.return_value = mock_evasion_spline_instance
    
    # Wywołanie testowanej metody
    plan = planner.compute_evasion_plan(mock_context)
    
    assert plan is not None
    assert plan.evasion_spline == mock_evasion_spline_instance
    assert plan.preferred_axis in ["up", "down", "left", "right"]
    np.testing.assert_array_equal(plan.rejoin_point, mock_context.rejoin_point)


@patch("src.algorithms.avoidance.AStarAvoidance.splev")
@patch("src.algorithms.avoidance.AStarAvoidance.BSplineTrajectory")
@patch("src.algorithms.avoidance.AStarAvoidance.UAV3DGridSearch")
def test_compute_evasion_plan_fallback(mock_grid_search_cls, mock_bspline, mock_splev, planner, mock_context):
    """
    Testuje zabezpieczenie (Fallback Path).
    Gdy A* wyrzuci błąd lub nie znajdzie drogi, planista musi uratować sytuację
    krzywą awaryjną za pomocą _fallback_path.
    """
    # Ominięcie operacji na Scipy by uniknąć unpack-errorów na tck.
    mock_splev.return_value = [1.0, 0.0, 0.0] 
    
    # Przygotowanie mocka dla A* symulującego porażkę (Exception)
    mock_searcher = MagicMock()
    mock_grid_search_cls.return_value = mock_searcher
    mock_searcher.astar.side_effect = Exception("Siatka nieciągła lub brak celu!")
    
    # Przygotowanie mocka dla wynikowego Splinu
    mock_evasion_spline_instance = MagicMock()
    mock_bspline.return_value = mock_evasion_spline_instance
    
    # Mimo że A* wywalił Exception, compute_evasion_plan ma go przechwycić
    # i zwrócić poprawny plan z wyliczonym awaryjnym waypointem!
    plan = planner.compute_evasion_plan(mock_context)
    
    assert plan is not None
    assert plan.evasion_spline == mock_evasion_spline_instance