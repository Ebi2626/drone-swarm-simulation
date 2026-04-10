import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import importlib
TARGET_MODULE = "src.algorithms.abstraction.trajectory.strategies.nsga3_utils.swarm_evolution_strategy"
NSGA3Orchestrator = importlib.import_module(TARGET_MODULE)

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_world_data():
    world = MagicMock()
    # Zakładamy świat 100x100x100
    world.bounds = np.array([[0.0, 100.0], [0.0, 100.0], [0.0, 100.0]])
    return world

@pytest.fixture
def start_target_pos():
    start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) # 2 drony
    target = np.array([[10.0, 10.0, 10.0], [11.0, 10.0, 10.0]])
    return start, target

# ==========================================
# TESTY: RESAMPLE POLYLINE BATCH
# ==========================================

def test_resample_polyline_batch():
    """
    Intencja: Sprawdzenie, czy rzadka łamana jest poprawnie zagęszczana przez interpolację.
    """
    # 1 populacja, 1 dron, 3 waypointy wejściowe (Start, Srodek, Cel)
    # Trasa tworzy odwrócone V: (0,0,0) -> (5,0,10) -> (10,0,0)
    sparse = np.array([
        [
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 10.0],
                [10.0, 0.0, 0.0]
            ]
        ]
    ])
    
    # Zagęszczamy do 5 punktów.
    # Spodziewane czasy t: 0.0, 0.25, 0.5, 0.75, 1.0
    # Oczekiwane punkty na osi Z: 0.0, 5.0, 10.0, 5.0, 0.0
    dense = NSGA3Orchestrator.resample_polyline_batch(sparse, num_samples=5)
    
    assert dense.shape == (1, 1, 5, 3)
    
    # Sprawdzamy oś Z drona 0, populacji 0
    np.testing.assert_array_almost_equal(
        dense[0, 0, :, 2], 
        [0.0, 5.0, 10.0, 5.0, 0.0]
    )

# ==========================================
# TESTY: HEURISTIC SAMPLING
# ==========================================

def test_heuristic_sampling_generates_inner_points_only(start_target_pos):
    """
    Intencja: Sprawdzenie, czy sampler poprawnie kładzie wewnętrzne punkty 
    na prostej łączącej start z metą, omijając same skrajności.
    """
    start, target = start_target_pos
    n_inner = 3
    n_drones = 2
    n_samples = 4 # Wielkość populacji
    
    sampler = NSGA3Orchestrator.HeuristicSampling(start, target, n_inner, n_drones)
    
    # Tworzymy atrapę problemu z szerokimi granicami, by clipping nie zepsuł testu
    mock_problem = MagicMock()
    mock_problem.xl = np.array([-100.0])
    mock_problem.xu = np.array([100.0])
    
    # Wyłączamy szum (noise) na czas testu, by sprawdzić czystą interpolację
    with patch(f"{TARGET_MODULE}.np.random.normal", return_value=0.0):
        result = sampler._do(mock_problem, n_samples)
    
    # Oczekiwany kształt: (N_Samples, N_Drones * N_Inner * 3)
    assert result.shape == (4, 2 * 3 * 3)
    
    # Przebudowujemy z powrotem do (Pop, Drones, Inner, 3) by ułatwić asercję pierwszego osobnika
    reshaped = result.reshape(4, 2, 3, 3)
    
    # Dron 0 leci z (0,0,0) do (10,10,10). 3 wewnętrzne punkty to t=0.25, 0.5, 0.75
    # Czyli oczekujemy: (2.5,2.5,2.5), (5,5,5), (7.5,7.5,7.5)
    expected_d0 = np.array([
        [2.5, 2.5, 2.5],
        [5.0, 5.0, 5.0],
        [7.5, 7.5, 7.5]
    ])
    
    np.testing.assert_array_almost_equal(reshaped[0, 0], expected_d0)

# ==========================================
# TESTY: GŁÓWNA STRATEGIA (ORKIESTRATOR)
# ==========================================

@patch(f"{TARGET_MODULE}.minimize")
@patch(f"{TARGET_MODULE}.VectorizedEvaluator")
def test_nsga3_swarm_strategy_success(mock_eval, mock_minimize, mock_world_data, start_target_pos):
    """
    Intencja: Sprawdzenie przepływu dla udanej optymalizacji (algorytm znalazł Pareto front).
    Upewniamy się, że dokleja Start/Cel do Inner Waypoints i zwraca zagęszczoną tablicę.
    """
    start, target = start_target_pos
    n_drones = 2
    n_inner = 5 # Parametr dla mocka
    
    # Symulujemy udany wynik NSGA-III
    mock_res = MagicMock()
    
    # Zwrócone X (Zmienne decyzyjne - punkty wewnętrzne)
    # Kształt flat_X to (Pop_Size, N_Drones * N_Inner * 3). Zakładamy populację z 1 osobnikiem.
    fake_inner = np.ones((1, n_drones, n_inner, 3)) * 5.0 # Wszystkie punkty wewnętrzne mają współrzędne 5.0
    mock_res.X = fake_inner.reshape(1, -1)
    
    mock_res.F = np.array([[10.0, 0.0, 0.0]]) # 1 osobnik, 3 cele
    mock_res.G = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]) # Spełnia wszystkie ograniczenia
    mock_res.history = []
    
    mock_minimize.return_value = mock_res
    
    # Odpalamy strategię
    params = {"n_inner_waypoints": n_inner, "decision_mode": "knee_point"}
    final_trajectory = NSGA3Orchestrator.nsga3_swarm_strategy(
        start_positions=start,
        target_positions=target,
        obstacles_data=[],
        world_data=mock_world_data,
        number_of_waypoints=10, # Docelowe zagęszczenie
        drone_swarm_size=n_drones,
        algorithm_params=params
    )
    
    # Weryfikacja wyniku
    assert final_trajectory.shape == (2, 10, 3) # (Drony, Zageszczone_Waypointy, XYZ)
    
    # Punkt [0] musi być początkowy, a [-1] musi być celem, dla każdego drona
    np.testing.assert_array_equal(final_trajectory[0, 0], start[0])
    np.testing.assert_array_equal(final_trajectory[0, -1], target[0])
    np.testing.assert_array_equal(final_trajectory[1, 0], start[1])
    np.testing.assert_array_equal(final_trajectory[1, -1], target[1])

@patch(f"{TARGET_MODULE}.minimize")
@patch(f"{TARGET_MODULE}.VectorizedEvaluator")
def test_nsga3_swarm_strategy_fallback(mock_eval, mock_minimize, mock_world_data, start_target_pos):
    """
    Edge case: Algorytm ewolucyjny kompletnie zgubił rozwiązania (res.X is None lub len=0).
    Orkiestrator musi awaryjnie stworzyć czystą prostą.
    """
    start, target = start_target_pos
    
    # Symulujemy porażkę
    mock_res = MagicMock()
    mock_res.X = None
    mock_minimize.return_value = mock_res
    
    final_trajectory = NSGA3Orchestrator.nsga3_swarm_strategy(
        start_positions=start,
        target_positions=target,
        obstacles_data=[],
        world_data=mock_world_data,
        number_of_waypoints=5,
        drone_swarm_size=2
    )
    
    assert final_trajectory.shape == (2, 5, 3)
    
    # Dron 0: lot 0 -> 10, środek to dokładnie 5.0
    np.testing.assert_array_equal(final_trajectory[0, 2], [5.0, 5.0, 5.0])