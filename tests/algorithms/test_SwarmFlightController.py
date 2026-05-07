import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import importlib

TARGET_MODULE = "src.algorithms.SwarmFlightController"
AlgorithmModule = importlib.import_module(TARGET_MODULE)
SwarmFlightController = AlgorithmModule.SwarmFlightController

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_parent():
    """Atrapa obiektu zarządzającego (np. środowiska/NSGA), z którego algorytm czerpie dane."""
    parent = MagicMock()
    parent.drones_trajectories = [
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [4.0, 1.0, 0.0]])
    ]
    parent.current_states = [
        np.array([4.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([3.5, 1.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ]
    return parent

@pytest.fixture(autouse=True)
def mock_dependencies():
    """
    Automatycznie mockuje zależności sterowania i rysowania dla testów.
    """
    with patch(f"{TARGET_MODULE}.DSLPIDControl") as MockPID, \
         patch(f"{TARGET_MODULE}.plt") as mock_plt:
        
        instance = MockPID.return_value
        instance.computeControlFromState.return_value = (np.array([1, 2, 3, 4]), None, None)
        yield MockPID, mock_plt

# ==========================================
# TESTY INICJALIZACJI
# ==========================================

def test_initialization(mock_parent):
    """Sprawdzenie parametryzacji algorytmu na starcie."""
    params = {
        "ctrl_freq": 100,
        "hover_duration": 5.0,
        "finish_radius": 0.2,
        "cruise_speed": 10.0
    }
    
    algo = SwarmFlightController(mock_parent, num_drones=2, is_obstacle=False, params=params)

    assert algo._ctrl_timestep == 1.0 / 100
    assert algo.hover_duration == 5.0
    assert algo.finish_radius == 0.2
    assert algo.cruise_speed == 10.0
    assert len(algo.controllers) == 2

# ==========================================
# TESTY GEOMETRII
# ==========================================

def test_insert_midpoint_with_offset(mock_parent):
    """Weryfikuje poprawność matematyczną wstawiania wektora offsetu w środek najbliższego segmentu."""
    algo = SwarmFlightController(mock_parent, num_drones=1, is_obstacle=False)

    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0]
    ])

    target_pos = np.array([1.5, 0.0, 0.0])
    offset = np.array([0.0, 1.0, 0.0]) 
    
    new_wp = algo._insert_midpoint_with_offset(waypoints, target_pos, offset)
    
    assert len(new_wp) == 4
    expected_midpoint = np.array([1.0, 1.0, 0.0])
    np.testing.assert_allclose(new_wp[1], expected_midpoint)
    
# ==========================================
# TESTY ORKIESTRACJI TRAJEKTORII (KOLIZJE / NAPRAWA)
# ==========================================

@patch(f"{TARGET_MODULE}.check_collisions_njit")
def test_verify_trajectories_detects_collision(mock_check_collisions, mock_parent):
    """Weryfikacja integracji algorytmu z funkcją kolizji Numba JIT (check_collisions_njit)."""
    algo = SwarmFlightController(mock_parent, num_drones=2, is_obstacle=False)
    
    spline1, spline2 = MagicMock(), MagicMock()
    spline1.total_duration = 5.0
    spline2.total_duration = 5.0
    spline1.get_state_at_time.return_value = (np.array([10.0, 0.0, 0.0]), np.zeros(3))
    spline2.get_state_at_time.return_value = (np.array([10.5, 0.0, 0.0]), np.zeros(3))
    
    # Symulacja: kolizja drona 0 i 1 w ticku nr 25
    mock_check_collisions.return_value = (0, 1, 25)
    
    collision = algo._verify_trajectories([spline1, spline2])
    
    assert collision is not None
    assert collision[0] == 0
    assert collision[1] == 1
    np.testing.assert_array_equal(collision[2], [10.0, 0.0, 0.0])
    np.testing.assert_array_equal(collision[3], [10.5, 0.0, 0.0])

@patch.object(SwarmFlightController, '_verify_trajectories')
@patch.object(SwarmFlightController, '_visualize_trajectories')
@patch(f"{TARGET_MODULE}.NumbaTrajectoryProfile")
def test_prepare_trajectories_repair_loop(MockProfile, mock_visualize, mock_verify, mock_parent):
    """Sprawdzenie pętli retry (napraw i spróbuj ponownie)."""
    algo = SwarmFlightController(mock_parent, num_drones=2, is_obstacle=False)

    fake_collision = (0, 1, np.array([1,1,1]), np.array([1,1,1]))
    mock_verify.side_effect = [fake_collision, None]
    
    with patch.object(algo, '_repair_waypoints', wraps=algo._repair_waypoints) as spy_repair:
        splines = algo._prepare_trajectories()
        
        assert mock_verify.call_count == 2
        assert spy_repair.call_count == 1
        assert len(splines) == 2
        mock_visualize.assert_called_once()

# ==========================================
# TESTY AKCJI I STATUSU
# ==========================================

@patch.object(SwarmFlightController, '_prepare_trajectories')
def test_compute_actions_hover_and_flight(mock_prepare, mock_parent):
    """Hover: przed czasem startu pytamy o t=0.0. Lot: po hover pytamy o poprawiony czas lotu."""
    algo = SwarmFlightController(mock_parent, num_drones=1, is_obstacle=False)
    algo.hover_duration = 3.0
    
    mock_spline = MagicMock()
    mock_spline.get_state_at_time.return_value = (np.array([5, 5, 5]), np.array([1, 0, 0]))
    mock_prepare.return_value = [mock_spline]
    
    states = [np.zeros(13)]
    
    # Hover (t=1.0 < 3.0) -> flight_time = -2.0. Limit podłogi wymusza t=0.0
    algo.compute_actions(states, current_time=1.0)
    mock_spline.get_state_at_time.assert_called_with(0.0)
    
    # Lot (t=4.0 > 3.0) -> flight_time = 1.0
    algo.compute_actions(states, current_time=4.0)
    mock_spline.get_state_at_time.assert_called_with(1.0)

@patch.object(SwarmFlightController, '_prepare_trajectories')
def test_all_finished(mock_prepare, mock_parent):
    """Ostateczne zatrzymanie wymaga osiągnięcia przez wszystkie drony strefy lądowania (ostatniego węzła)."""
    algo = SwarmFlightController(mock_parent, num_drones=2, is_obstacle=False)
    algo.finish_radius = 1.0
    
    mock_spline1 = MagicMock()
    mock_spline1.waypoints = np.array([[0,0,0], [4.0, 0.0, 0.0]])
    mock_spline1.get_state_at_time.return_value = (np.zeros(3), np.zeros(3))
    
    mock_spline2 = MagicMock()
    mock_spline2.waypoints = np.array([[0,0,0], [4.0, 1.0, 0.0]])
    mock_spline2.get_state_at_time.return_value = (np.zeros(3), np.zeros(3))
    
    mock_prepare.return_value = [mock_spline1, mock_spline2]
    
    algo.compute_actions(mock_parent.current_states, current_time=0)
    
    # Dron 0 jest u celu (X=4.0), Dron 1 ma dystans X=0.5 -> 0.5 < 1.0
    assert algo.all_finished is True
    
    # Odsuwamy drona od celu tak, by dystans = 3.0 > radius(1.0)
    mock_parent.current_states[1][0] = 1.0
    assert algo.all_finished is False

# ==========================================
# TESTY LOGIKI DYNAMICZNYCH PRZESZKÓD
# ==========================================

def test_is_obstacle_flag_stored(mock_parent):
    """Izolacja flag obiektów aktywnych a pasywnych."""
    algo_main = SwarmFlightController(mock_parent, num_drones=2, is_obstacle=False)
    algo_obs = SwarmFlightController(mock_parent, num_drones=2, is_obstacle=True)
    assert algo_main.is_obstacle is False
    assert algo_obs.is_obstacle is True

def test_init_lidars_skipped_for_obstacles(mock_parent):
    """Obiekty z is_obstacle=True omijają ładowanie czujnika LiDAR."""
    algo = SwarmFlightController(mock_parent, num_drones=2, is_obstacle=True)
    algo.init_lidars(physics_client_id=99)
    assert algo._lidars is None

@patch(f"{TARGET_MODULE}.LidarSensor")
def test_init_lidars_created_for_main_drones(MockLidar, mock_parent):
    """Zwykłe roje ładują model lasera 3D."""
    algo = SwarmFlightController(mock_parent, num_drones=2, is_obstacle=False)
    algo.init_lidars(physics_client_id=99)
    assert MockLidar.call_count == 2
    assert algo._lidars is not None

@patch.object(SwarmFlightController, '_visualize_trajectories')
@patch(f"{TARGET_MODULE}.NumbaTrajectoryProfile")
def test_obstacle_trajectories_are_flipped_versions_of_main(MockProfile, mock_visualize, mock_parent):
    """Odwrotne podążanie ścieżką w ramach symulacji ataku czołowego."""
    algo = SwarmFlightController(mock_parent, num_drones=2, is_obstacle=True)
    algo._prepare_trajectories()

    expected_0 = np.flipud(mock_parent.drones_trajectories[0])
    expected_1 = np.flipud(mock_parent.drones_trajectories[1])

    actual_0 = MockProfile.call_args_list[0].args[0]
    actual_1 = MockProfile.call_args_list[1].args[0]

    np.testing.assert_array_equal(actual_0, expected_0)
    np.testing.assert_array_equal(actual_1, expected_1)

@patch.object(SwarmFlightController, '_verify_trajectories')
@patch.object(SwarmFlightController, '_visualize_trajectories')
@patch(f"{TARGET_MODULE}.NumbaTrajectoryProfile")
def test_obstacles_skip_collision_verification(MockProfile, mock_visualize, mock_verify, mock_parent):
    """Zagrożenia omijają detekcję zderzeń międzydronowych w przygotowaniu lotu."""
    algo = SwarmFlightController(mock_parent, num_drones=2, is_obstacle=True)
    algo._prepare_trajectories()
    mock_verify.assert_not_called()

# ==========================================
# REGRESJA: NumbaTrajectoryProfile API (bug 2026-04-29)
# ==========================================
# Po refaktorze TrajectoryFollowingAlgorithm → SwarmFlightController + NumbaTrajectoryProfile
# zanikł zagnieżdżony atrybut `.profile`; helper'y arc/time muszą sięgać po pola
# trapezoidalnego profilu bezpośrednio (ta/tc/td/sa/sc/v_peak/max_accel/total_*).

def _real_profile():
    from src.algorithms.abstraction.trajectory.strategies.shared.NumbaTrajectoryProfile import (
        NumbaTrajectoryProfile,
    )
    waypoints = np.array(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64
    )
    return NumbaTrajectoryProfile(waypoints=waypoints, cruise_speed=2.0, max_accel=1.0)


def test_base_arc_progress_uses_numba_profile_attrs(mock_parent):
    """Regresja: `_base_arc_progress` nie może wołać `.profile.get_state` (atrybut nie istnieje)."""
    algo = SwarmFlightController(mock_parent, num_drones=1, is_obstacle=False)
    profile = _real_profile()
    algo._base_trajectories = [profile]
    algo._tracking_start_times = np.array([0.0])

    assert algo._base_arc_progress(0, current_time=0.0) == pytest.approx(0.0)
    assert algo._base_arc_progress(0, current_time=10_000.0) == pytest.approx(
        profile.total_distance
    )

    # W fazie cruise (po t_a, przed t_a+t_c) prędkość = v_peak, dystans rośnie liniowo.
    t_mid_cruise = profile.ta + 0.5 * profile.tc
    expected_mid = profile.sa + profile.v_peak * 0.5 * profile.tc
    assert algo._base_arc_progress(0, current_time=t_mid_cruise) == pytest.approx(expected_mid)


def test_invert_profile_arc_to_time_roundtrip():
    """Regresja: `_invert_profile_arc_to_time` musi czytać `sa/sc/ta/tc/...` z NumbaTrajectoryProfile."""
    profile = _real_profile()

    assert SwarmFlightController._invert_profile_arc_to_time(profile, 0.0) == pytest.approx(0.0)
    assert SwarmFlightController._invert_profile_arc_to_time(
        profile, profile.total_distance
    ) == pytest.approx(profile.total_duration)

    for arc in (0.5 * profile.sa, profile.sa + 0.5 * profile.sc, profile.sa + profile.sc + 1e-3):
        t = SwarmFlightController._invert_profile_arc_to_time(profile, arc)
        # _arc_at_time(t) musi wrócić do tego samego arc (z numeryczną tolerancją).
        recovered = SwarmFlightController._arc_at_time(profile, t)
        assert recovered == pytest.approx(arc, abs=1e-6)


def test_prepare_trajectories_raises_when_source_missing(mock_parent):
    """Brak trajektorii optymalizacji wywołuje krytyczny błąd przygotowania lotu."""
    mock_parent.drones_trajectories = None

    algo_obs = SwarmFlightController(mock_parent, num_drones=2, is_obstacle=True)
    with pytest.raises(ValueError, match="Brak trajektorii"):
        algo_obs._prepare_trajectories()

    algo_main = SwarmFlightController(mock_parent, num_drones=2, is_obstacle=False)
    with pytest.raises(ValueError, match="Brak wyników optymalizacji"):
        algo_main._prepare_trajectories()