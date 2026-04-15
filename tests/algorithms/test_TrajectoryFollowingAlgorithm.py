import pytest
import numpy as np
from unittest.mock import patch, MagicMock


import importlib
TARGET_MODULE = "src.algorithms.TrajectoryFollowingAlgorithm"
AlgorithmModule = importlib.import_module(TARGET_MODULE)
TrajectoryFollowingAlgorithm = AlgorithmModule.TrajectoryFollowingAlgorithm

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_parent():
    """Atrapa obiektu zarządzającego (np. środowiska/NSGA), z którego algorytm czerpie dane."""
    parent = MagicMock()
    # Przykładowe trasy NSGA dla 2 dronów
    parent.trajectories = [
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [4.0, 1.0, 0.0]])
    ]
    # Przykładowe stany dronów (pybullet zwraca wektor z ~20 elementami, nas interesują pierwsze 3 - pozycja)
    parent.current_states = [
        np.array([4.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([3.5, 1.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ]
    return parent

@pytest.fixture(autouse=True)
def mock_dependencies():
    """
    Automatycznie mockuje ciężkie zależności (matplotlib i DSLPIDControl) 
    dla WSZYSTKICH testów w tym pliku.
    """
    with patch(f"{TARGET_MODULE}.DSLPIDControl") as MockPID, \
         patch(f"{TARGET_MODULE}.plt") as mock_plt:
        
        # Konfigurujemy, co ma zwracać atrapa kontrolera PID (action, _, _)
        instance = MockPID.return_value
        instance.computeControlFromState.return_value = (np.array([1, 2, 3, 4]), None, None)
        
        yield MockPID, mock_plt

# ==========================================
# TESTY INICJALIZACJI
# ==========================================

def test_initialization(mock_parent):
    """Intencja: Sprawdzenie, czy parametry są poprawnie wczytywane z kwargs."""
    params = {
        "ctrl_freq": 100,
        "hover_duration": 5.0,
        "finish_radius": 0.2,
        "cruise_speed": 10.0
    }
    
    algo = TrajectoryFollowingAlgorithm(mock_parent, num_drones=2, params=params)
    
    assert algo._ctrl_timestep == 1.0 / 100
    assert algo.hover_duration == 5.0
    assert algo.finish_radius == 0.2
    assert algo.cruise_speed == 10.0
    assert len(algo.controllers) == 2

# ==========================================
# TESTY GEOMETRII (Naprawa trasy)
# ==========================================

def test_insert_midpoint_near(mock_parent):
    """
    Intencja: Najważniejszy test matematyczny. Czy algorytm poprawnie wstawia 
    punkt pośrodku właściwego odcinka łamanej?
    """
    algo = TrajectoryFollowingAlgorithm(mock_parent, num_drones=1)
    
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0]
    ])
    
    # 1. Zderzenie wystąpiło w X=1.5 (bliżej środkowego waypointu, ale na PIERWSZYM odcinku)
    target_pos = np.array([1.5, 0.0, 0.0])
    new_wp = algo._insert_midpoint_near(waypoints, target_pos)
    
    assert len(new_wp) == 4
    # Środek między wp[0] a wp[1] to [1.0, 0.0, 0.0]
    np.testing.assert_array_equal(new_wp[1], [1.0, 0.0, 0.0])
    
    # 2. Zderzenie w X=3.5 (na DRUGIM odcinku)
    target_pos2 = np.array([3.5, 0.0, 0.0])
    new_wp2 = algo._insert_midpoint_near(waypoints, target_pos2)
    
    assert len(new_wp2) == 4
    # Środek między wp[1] a wp[2] to [3.0, 0.0, 0.0]
    np.testing.assert_array_equal(new_wp2[2], [3.0, 0.0, 0.0])

# ==========================================
# TESTY ORKIESTRACJI TRAJEKTORII
# ==========================================

@patch(f"{TARGET_MODULE}.BSplineTrajectory")
def test_verify_trajectories_detects_collision(MockSpline, mock_parent):
    """Intencja: Upewnienie się, że sprawdzanie w pętli czasowej wykryje przecięcie stref."""
    algo = TrajectoryFollowingAlgorithm(mock_parent, num_drones=2)
    algo.collision_radius = 1.0
    
    spline1, spline2 = MagicMock(), MagicMock()
    spline1.total_duration = 5.0
    spline2.total_duration = 5.0
    
    # W t=0 drony są daleko od siebie (-10 vs 10). 
    # Dopiero w ruchu (t>0) zbliżają się na kolizyjną odległość (10.0 vs 10.5 -> dystans 0.5 < radius 1.0)
    spline1.get_state_at_time.side_effect = lambda t: (np.array([10.0, 0.0, 0.0]), np.zeros(3)) if t > 0 else (np.array([-10.0, 0.0, 0.0]), np.zeros(3))
    spline2.get_state_at_time.side_effect = lambda t: (np.array([10.5, 0.0, 0.0]), np.zeros(3)) if t > 0 else (np.array([10.0, 0.0, 0.0]), np.zeros(3))
    
    collision = algo._verify_trajectories([spline1, spline2])
    
    assert len(collision) == 4
    assert collision[0] == 0
    assert collision[1] == 1
    # Upewniamy się, że kolizja to ta z ruchu (X=10.0)
    np.testing.assert_array_equal(collision[2], [10.0, 0.0, 0.0])

@patch.object(TrajectoryFollowingAlgorithm, '_verify_trajectories')
@patch.object(TrajectoryFollowingAlgorithm, '_visualize_trajectories')
@patch(f"{TARGET_MODULE}.BSplineTrajectory")
def test_prepare_trajectories_repair_loop(MockSpline, mock_visualize, mock_verify, mock_parent):
    """
    Intencja: Sprawdzenie pętli retry. Jeśli w pierwszej próbie jest kolizja,
    algorytm musi naprawić trasę i spróbować ponownie.
    """
    algo = TrajectoryFollowingAlgorithm(mock_parent, num_drones=2)
    
    # Symulacja: Za pierwszym wywołaniem (_verify_trajectories) zwraca kolizję, za drugim () pusto (bezpiecznie)
    fake_collision = (0, 1, np.array([1,1,1]), np.array([1,1,1]))
    mock_verify.side_effect = [fake_collision, ()]
    
    # Szpiegujemy oryginalną metodę naprawczą, by upewnić się, że zostanie użyta
    with patch.object(algo, '_repair_waypoints', wraps=algo._repair_waypoints) as spy_repair:
        splines = algo._prepare_trajectories()
        
        # Oczekujemy, że sprawdzanie wywołano 2 razy, a naprawę 1 raz
        assert mock_verify.call_count == 2
        assert spy_repair.call_count == 1
        assert len(splines) == 2
        mock_visualize.assert_called_once()

# ==========================================
# TESTY AKCJI I STATUSU
# ==========================================

@patch.object(TrajectoryFollowingAlgorithm, '_prepare_trajectories')
def test_compute_actions_hover_and_flight(mock_prepare, mock_parent):
    """
    Intencja: Weryfikacja fazy lotu. Przed upływem 'hover_duration' drony stoją w miejscu, 
    a dopiero po jego upływie zaczynają pobierać punkty z trajektorii.
    """
    algo = TrajectoryFollowingAlgorithm(mock_parent, num_drones=1)
    algo.hover_duration = 3.0
    
    # Przygotowujemy atrapę wygenerowanej trajektorii
    mock_spline = MagicMock()
    # Zwraca (pozycja, prędkość)
    mock_spline.get_state_at_time.return_value = (np.array([5, 5, 5]), np.array([1, 0, 0]))
    mock_prepare.return_value = [mock_spline]
    
    states = [np.zeros(13)]
    
    # 1. Czas t = 1.0s (Trwa Hovering). Algorytm powinien pytać o t=0.0 krzywej.
    algo.compute_actions(states, current_time=1.0)
    mock_spline.get_state_at_time.assert_called_with(0.0)
    
    # 2. Czas t = 4.0s (1 sekunda po zakończeniu Hovera).
    algo.compute_actions(states, current_time=4.0)
    # Powinno zapytać o czas 4.0 - 3.0 = 1.0s na samej krzywej
    mock_spline.get_state_at_time.assert_called_with(1.0)

@patch.object(TrajectoryFollowingAlgorithm, '_prepare_trajectories')
def test_all_finished(mock_prepare, mock_parent):
    """Intencja: Metoda all_finished sprawdza bliskość do OSTATNIEGO waypointu."""
    algo = TrajectoryFollowingAlgorithm(mock_parent, num_drones=2)
    algo.finish_radius = 1.0
    
    mock_spline1 = MagicMock()
    mock_spline1.waypoints = np.array([[0,0,0], [4.0, 0.0, 0.0]]) # Cel: 4.0
    # NAPRAWA: Zabezpieczenie przed rozpakowywaniem krotki w compute_actions
    mock_spline1.get_state_at_time.return_value = (np.zeros(3), np.zeros(3))
    
    mock_spline2 = MagicMock()
    mock_spline2.waypoints = np.array([[0,0,0], [4.0, 1.0, 0.0]]) # Cel: 4.0, 1.0
    # NAPRAWA: Zabezpieczenie przed rozpakowywaniem krotki w compute_actions
    mock_spline2.get_state_at_time.return_value = (np.zeros(3), np.zeros(3))
    
    mock_prepare.return_value = [mock_spline1, mock_spline2]
    
    # Wymuszamy wygenerowanie / pobranie z cache
    algo.compute_actions(mock_parent.current_states, current_time=0)
    
    # Z fixture: Dron0 jest na (4.0, 0, 0) -> u celu! Dron1 na (3.5, 1.0, 0) -> dist = 0.5 < radius 1.0 -> u celu!
    assert algo.all_finished is True
    
    # Odsunięcie drugiego drona z dala od celu (X = 1.0)
    mock_parent.current_states[1][0] = 1.0
    assert algo.all_finished is False