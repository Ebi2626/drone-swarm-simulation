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
    lateral_xy = np.array([0.0, 1.0, 0.0])
    world_min = np.array([0.0, -10.0, 0.0])
    world_max = np.array([20.0, 10.0, 10.0])

    # Sufit jest na 10m (5m wolnego), podłoga na 0m (5m wolnego).
    # Bez wektora prędkości przeszkody składnik anty-VO jest zerowy,
    # więc tie-break po `prefer_axis_order` ["up","down","right","left"] → "up".
    axis_name, pdir = planner._pick_preferred_axis(
        current_pos, lateral_xy,
        floor_z=1.0, ceiling_z=9.0, world_min=world_min, world_max=world_max,
        obs_vel=None,
    )

    assert axis_name == "up"
    np.testing.assert_array_equal(pdir, np.array([0.0, 0.0, 1.0]))


def test_pick_preferred_axis_blocked_up(planner):
    """Sprawdza wybór, gdy preferowany kierunek jest zablokowany (np. lot blisko sufitu)."""
    current_pos = np.array([0.0, 0.0, 8.5])  # Dron pod sufitem
    lateral_xy = np.array([0.0, 1.0, 0.0])

    # Zablokowany sufit (ceiling_z=9.0, zostaje 0.5m, a wymaga > 2.5m)
    axis_name, pdir = planner._pick_preferred_axis(
        current_pos, lateral_xy,
        floor_z=1.0, ceiling_z=9.0,
        world_min=np.array([0.0, -10.0, 0.0]),
        world_max=np.array([20.0, 10.0, 10.0]),
        obs_vel=None,
    )

    assert axis_name == "down"
    np.testing.assert_array_equal(pdir, np.array([0.0, 0.0, -1.0]))


def test_pick_preferred_axis_sticky_hint_wins_when_viable(planner):
    """
    Sticky axis: przy replanie w trybie EVASION przekazujemy `axis_hint` z poprzedniego
    planu. Jeśli oś ma nadal wystarczającą przestrzeń, planner ją utrzymuje,
    eliminując flip-flopping up↔right↔left w korytarzu z wieloma przeszkodami.
    """
    current_pos = np.array([0.0, 0.0, 5.0])
    lateral_xy = np.array([0.0, 1.0, 0.0])
    world_min = np.array([0.0, -10.0, 0.0])
    world_max = np.array([20.0, 10.0, 10.0])

    # Bez hintu i przy v_obs wznoszącej się (up w stożku VO) ranking wskazuje "down".
    # Z hintem "right" (viable, 10m przestrzeni) oczekujemy "right" mimo to.
    axis_name, pdir = planner._pick_preferred_axis(
        current_pos, lateral_xy,
        floor_z=1.0, ceiling_z=9.0, world_min=world_min, world_max=world_max,
        obs_vel=np.array([0.0, 0.0, 3.0]),
        axis_hint="right",
    )

    assert axis_name == "right"
    np.testing.assert_array_equal(pdir, lateral_xy)


def test_pick_preferred_axis_sticky_hint_ignored_when_blocked(planner):
    """
    Sticky axis nie nadpisuje geometrii: jeśli oś z hintu nie ma już
    wystarczającej przestrzeni (np. dron dojechał do ściany), planner wybiera
    od nowa.
    """
    # Dron przy dolnej podłodze — "down" zablokowane (1.5m < 2.5m min_required).
    current_pos = np.array([0.0, 0.0, 2.5])
    lateral_xy = np.array([0.0, 1.0, 0.0])
    world_min = np.array([0.0, -10.0, 0.0])
    world_max = np.array([20.0, 10.0, 10.0])

    axis_name, _ = planner._pick_preferred_axis(
        current_pos, lateral_xy,
        floor_z=1.0, ceiling_z=9.0, world_min=world_min, world_max=world_max,
        obs_vel=None,
        axis_hint="down",
    )

    assert axis_name != "down"


def test_pick_preferred_axis_noise_level_obs_vz_does_not_override_prefer_axis_order():
    """
    Regresja Fazy 6 (eksperyment 2026-04-22, 21:07 forest_nsga-3): szum w
    składowej z wektora prędkości przeszkody (threat_vz ≈ 0.28 m/s, anti ≈ 0.047)
    nie może wygrywać nad `prefer_axis_order`. W buggy wersji `s * gain * anti`
    z `s_down = 6.5m` generowało score ≈ 0.33 > tie_break "right" (0.1),
    co kierowało drona w ziemię mimo konfiguracji priorytetu bocznego.
    """
    planner = AStarAvoidance(
        visualize=False, grid_resolution=0.5,
        prefer_axis_order=["right", "left", "up", "down"],
        axis_anti_obsvel_gain=1.0,
    )
    # Dron na wysokości 6.63 m (forest scenario), przeszkoda zbliżająca się head-on
    # z niemal pomijalnym wznoszeniem (vz=0.28). Prefer-axis order jest boczny.
    current_pos = np.array([18.77, 277.95, 6.63])
    forward_xy = np.array([0.0, 1.0, 0.0])
    lateral_xy = np.array([-forward_xy[1], forward_xy[0], 0.0])
    world_min = np.array([0.0, 0.0, 0.1])
    world_max = np.array([60.0, 600.0, 11.0])

    axis_name, _ = planner._pick_preferred_axis(
        current_pos, lateral_xy,
        floor_z=0.1, ceiling_z=11.0, world_min=world_min, world_max=world_max,
        obs_vel=np.array([0.07, -5.99, 0.28]),
    )
    assert axis_name in ("right", "left"), (
        f"Przy threat_vz = 0.28 m/s (szum) oczekujemy osi bocznej zgodnej z "
        f"prefer_axis_order, dostaliśmy: {axis_name}"
    )


@patch("src.algorithms.avoidance.AStarAvoidance.splev")
@patch("src.algorithms.avoidance.AStarAvoidance.BSplineTrajectory")
@patch("src.algorithms.avoidance.AStarAvoidance.UAV3DGridSearch")
def test_lateral_evasion_forces_horizontal_z_profile(
    mock_grid_search_cls, mock_bspline, mock_splev, planner, mock_context
):
    """
    Regresja Fazy 8.1 (eksperyment 2026-04-22, 21:24 forest_nsga-3): przy
    axis=right/left waypoints przekazywane do BSplineTrajectory muszą mieć Z
    liniowo interpolowane między current_z a rejoin_z. W buggy wersji A* potrafił
    wydać trasę z silnymi Z-oscylacjami (grid 3D bez bias Z), które w
    `constant_speed=True` BSpline tworzyły tangens z -Z składową na u=0,
    powodując spadek drona 2 z z=6.5m do z=0.12m w 1 sekundzie.
    """
    mock_splev.return_value = [1.0, 0.0, 0.0]
    mock_searcher = MagicMock()
    mock_grid_search_cls.return_value = mock_searcher

    # A* zwraca trasę z oscylacjami Z (cel uniku lateralny, ale grid 3D
    # mógł zaproponować "nurkowanie" żeby obejść przeszkodę).
    mock_searcher.astar.return_value = [
        (0.0, 0.0, 5.0),
        (2.5, 0.0, 2.0),   # A* nurkuje (bo grid 3D pozwala)
        (5.0, 0.0, 1.0),   # A* schodzi jeszcze niżej
        (7.5, 0.0, 3.0),
        (10.0, 0.0, 5.0),
    ]

    mock_evasion_spline_instance = MagicMock()
    mock_bspline.return_value = mock_evasion_spline_instance

    plan = planner.compute_evasion_plan(mock_context)

    assert plan is not None

    # Sprawdzamy, że wszystkie waypointy przekazane do BSplineTrajectory
    # mają Z w przedziale [z_start, z_end] (tu: 5.0) — monotoniczne, nie
    # oscylujące. Akceptujemy ε numeryczne na krańcach przez lead_in/lead_out.
    if plan.preferred_axis in ("right", "left"):
        # Sprawdzamy argument 'waypoints' ostatniego wywołania BSplineTrajectory
        last_call_waypoints = mock_bspline.call_args.kwargs["waypoints"]
        zs = last_call_waypoints[:, 2]
        z_start = float(zs[0])
        z_end = float(zs[-1])
        z_lo, z_hi = sorted([z_start, z_end])
        assert np.all(zs >= z_lo - 1e-6) and np.all(zs <= z_hi + 1e-6), (
            f"Przy axis={plan.preferred_axis} Z-profil waypointów powinien być "
            f"monotoniczny między {z_start} a {z_end}, a jest: {zs.tolist()}"
        )


def test_pick_preferred_axis_obs_vel_biases_against_obstacle(planner):
    """
    Faza 2.3: gdy przeszkoda wznosi się (v_obs = +z), oś 'up' jest w stożku
    kolizji. Ranking powinien preferować 'down' mimo listy prefer_axis_order
    zaczynającej się od 'up'.
    """
    current_pos = np.array([0.0, 0.0, 5.0])
    lateral_xy = np.array([0.0, 1.0, 0.0])
    world_min = np.array([0.0, -10.0, 0.0])
    world_max = np.array([20.0, 10.0, 10.0])

    axis_name, pdir = planner._pick_preferred_axis(
        current_pos, lateral_xy,
        floor_z=1.0, ceiling_z=9.0, world_min=world_min, world_max=world_max,
        obs_vel=np.array([0.0, 0.0, 3.0]),
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