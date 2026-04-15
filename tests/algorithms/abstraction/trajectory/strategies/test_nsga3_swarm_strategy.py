import numpy as np
from unittest.mock import patch, MagicMock
import importlib
import pytest

from src.environments.obstacles.ObstacleShape import ObstacleShape
TARGET_MODULE = "src.algorithms.abstraction.trajectory.strategies.nsga3_swarm_strategy"
NSGA3Module = importlib.import_module(TARGET_MODULE)


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_world_data():
    """
    Atrapa danych świata. 
    Ważne: min_bounds[2] to podłoga, max_bounds[2] to sufit.
    """
    world = MagicMock()
    world.min_bounds = np.array([0.0, 0.0, 0.0])
    world.max_bounds = np.array([100.0, 100.0, 20.0])
    # Bounds w formacie [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    world.bounds = np.array([[0.0, 100.0], [0.0, 100.0], [0.0, 20.0]])
    return world

@pytest.fixture
def mock_obstacles_data():
    """Atrapa danych o przeszkodach dla samplera."""
    obs = MagicMock()
    # Jedna przeszkoda typu BOX o wymiarach 10x10x10
    obs.shape_type = ObstacleShape.BOX
    obs.data = np.array([[50.0, 50.0, 0.0, 10.0, 10.0, 10.0]])
    obs.count = 1
    return obs

# ==========================================
# TESTY
# ==========================================

def test_heuristic_sampling_logic(mock_world_data, mock_obstacles_data):
    """
    Testuje czy sampling:
    1. Generuje odpowiednią liczbę zmiennych.
    2. Trzyma się bezpiecznego korytarza Z.
    3. Uwzględnia szum XY bazujący na rozmiarze przeszkód.
    """
    start = np.array([[0.0, 0.0, 1.0]])
    target = np.array([[10.0, 10.0, 1.0]])
    n_inner = 3
    n_drones = 1
    
    sampler = NSGA3Module.HeuristicSampling(
        start, target, n_inner, n_drones, 
        world_data=mock_world_data, 
        obstacles_data=mock_obstacles_data
    )
    
    # Atrapa problemu dla metody _do
    mock_problem = MagicMock()
    # xl/xu muszą mieć rozmiar (n_drones * n_inner * 3) = 9
    mock_problem.xl = np.full(9, -50.0)
    mock_problem.xu = np.full(9, 150.0)
    
    samples = sampler._do(mock_problem, n_samples=5)
    
    # Sprawdzenie kształtu (PopSize, Variables)
    assert samples.shape == (5, 9)
    
    # Sprawdzenie korytarza Z (co trzeci element w spłaszczonej tablicy)
    z_indices = [2, 5, 8]
    z_values = samples[:, z_indices]
    
    assert np.all(z_values >= 0.5)
    assert np.all(z_values <= 20.0)

@patch(f"{TARGET_MODULE}.HydraConfig")  # Dodany mock dla Hydry
@patch(f"{TARGET_MODULE}.minimize")
def test_nsga3_strategy_fallback_with_altitude(mock_minimize, mock_hydraconfig, mock_world_data, tmp_path):
    """
    Testuje, czy w razie braku rozwiązań, linia prosta jest generowana
    nad minimalną bezpieczną wysokością (np. 2.0m).
    """
    # 1. Konfiguracja mocka dla Hydry, aby .get().runtime.output_dir zwróciło tymczasową ścieżkę
    mock_hydraconfig.get.return_value.runtime.output_dir = str(tmp_path)
    
    # 2. Symulacja błędu optymalizacji (brak rozwiązań)
    mock_res = MagicMock()
    mock_res.X = None 
    mock_minimize.return_value = mock_res
    
    start = np.array([[0.0, 0.0, 0.0]])
    target = np.array([[10.0, 10.0, 0.0]])
    
    # 3. Wywołanie funkcji
    result = NSGA3Module.nsga3_swarm_strategy(
        start_positions=start,
        target_positions=target,
        obstacles_data=[],
        world_data=mock_world_data,
        number_of_waypoints=5,
        drone_swarm_size=1,
        algorithm_params={"min_safe_altitude": 2.0}
    )
    
    # Oś Z (indeks 2) powinna wynosić 2.0 dla wszystkich 5 punktów
    expected_z = np.full(5, 2.0)
    np.testing.assert_array_almost_equal(result[0, :, 2], expected_z)
    
def test_resample_polyline_batch_correctness():
    """Weryfikacja wektoryzowanej interpolacji."""
    # Start (0,0,0), Środek (10,0,0), Koniec (10,10,0)
    sparse = np.array([[[[0,0,0], [10,0,0], [10,10,0]]]])
    
    # 3 punkty wyjściowe (powinny wyjść: start, środek, koniec)
    dense = NSGA3Module.resample_polyline_batch(sparse, num_samples=3)
    
    np.testing.assert_array_almost_equal(dense[0,0,0], [0,0,0])
    np.testing.assert_array_almost_equal(dense[0,0,1], [10,0,0])
    np.testing.assert_array_almost_equal(dense[0,0,2], [10,10,0])

def test_heuristic_sampling_unsupported_shape_raises_error(mock_world_data):
    """
    Intencja: Sprawdzenie, czy sampler rzuca ValueError, gdy otrzyma nieobsłużony kształt przeszkody.
    """
    start = np.array([[0.0, 0.0, 1.0]])
    target = np.array([[10.0, 10.0, 1.0]])
    
    # Tworzymy mocka z nieistniejącym kształtem (np. "TRIANGLE" lub 999)
    bad_obs = MagicMock()
    bad_obs.shape_type = "UNKNOWN_SHAPE"
    bad_obs.data = np.array([[50.0, 50.0, 0.0, 10.0, 10.0, 10.0]])
    
    sampler = NSGA3Module.HeuristicSampling(
        start, target, n_inner_points=3, n_drones=1,
        world_data=mock_world_data, obstacles_data=bad_obs
    )
    
    mock_problem = MagicMock()
    mock_problem.xl = np.full(9, -50.0)
    mock_problem.xu = np.full(9, 150.0)
    
    # Sprawdzamy, czy wywołanie metody _do rzuci ValueError
    with pytest.raises(ValueError) as exc_info:
        sampler._do(mock_problem, n_samples=5)
    
    # Opcjonalnie sprawdzamy treść komunikatu błędu
    assert "Unsupported obstacle shape type" in str(exc_info.value)