import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.environments.obstacles.ObstacleShape import ObstacleShape
from src.algorithms.abstraction.trajectory.objective_constrains import VectorizedEvaluator

TARGET_MODULE = "src.algorithms.abstraction.trajectory.objective_constrains"

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def basic_params():
    return {
        "k_factor": 2.0,
        "absolute_min_node_dist": 5.0,
        "obstacle_safety_margin": 0.5,
        "min_drone_distance": 2.0,
        "max_accel_limit": 5.0
    }

@pytest.fixture
def mock_obstacles_cylinder():
    """Przeszkoda walcowa do testowania ekstrakcji promienia."""
    obs = MagicMock()
    obs.shape_type = ObstacleShape.CYLINDER
    # data: [x, y, z, promień, wysokość, ...]
    obs.data = np.array([[10.0, 10.0, 0.0, 2.0, 10.0, 0.0]])
    return obs

@pytest.fixture
def mock_obstacles_box():
    """Przeszkoda prostopadłościenna do testowania okręgu opisanego."""
    obs = MagicMock()
    obs.shape_type = ObstacleShape.BOX
    # data: [x, y, z, długość_x, szerokość_y, wysokość]
    obs.data = np.array([[20.0, 20.0, 0.0, 4.0, 3.0, 10.0]])
    return obs

# ==========================================
# TESTY INICJALIZACJI PRZESZKÓD
# ==========================================

@patch(f"{TARGET_MODULE}.calculate_dynamic_max_node_distance", return_value=10.0)
def test_evaluator_init_no_obstacles(mock_calc, basic_params):
    """Brak przeszkód nie powinien rzucać wyjątków, a tablice powinny być puste."""
    starts = np.array([[0.0, 0.0, 0.0]])
    targets = np.array([[100.0, 100.0, 0.0]])
    
    evaluator = VectorizedEvaluator(None, starts, targets, 3, basic_params)
    
    assert len(evaluator.obstacles_xy) == 0
    assert len(evaluator.obstacle_radii) == 0

@patch(f"{TARGET_MODULE}.calculate_dynamic_max_node_distance", return_value=10.0)
def test_evaluator_init_cylinder_obstacles(mock_calc, mock_obstacles_cylinder, basic_params):
    """Dla cylindra bierzemy jego promień i dodajemy margines bezpieczeństwa."""
    starts = np.array([[0.0, 0.0, 0.0]])
    targets = np.array([[100.0, 100.0, 0.0]])
    
    evaluator = VectorizedEvaluator(mock_obstacles_cylinder, starts, targets, 3, basic_params)
    
    assert evaluator.obstacles_xy.shape == (1, 2)
    np.testing.assert_array_almost_equal(evaluator.obstacles_xy[0], [10.0, 10.0])
    
    # Oczekiwany promień: 2.0 (z danych) + 0.5 (margines z parametrów) = 2.5
    np.testing.assert_array_almost_equal(evaluator.obstacle_radii, [2.5])

@patch(f"{TARGET_MODULE}.calculate_dynamic_max_node_distance", return_value=10.0)
def test_evaluator_init_box_obstacles(mock_calc, mock_obstacles_box, basic_params):
    """
    Dla pudełka liczymy promień okręgu opisanego na prostokącie XY, 
    a następnie dodajemy margines bezpieczeństwa.
    """
    starts = np.array([[0.0, 0.0, 0.0]])
    targets = np.array([[100.0, 100.0, 0.0]])
    
    evaluator = VectorizedEvaluator(mock_obstacles_box, starts, targets, 3, basic_params)
    
    # lx=4, wy=3 -> half_lx=2.0, half_wy=1.5
    # circumscribed_radius = sqrt(2^2 + 1.5^2) = sqrt(4 + 2.25) = sqrt(6.25) = 2.5
    # promień ostateczny: 2.5 + 0.5 (margines) = 3.0
    np.testing.assert_array_almost_equal(evaluator.obstacle_radii, [3.0])

# ==========================================
# TESTY FUNKCJI CELU I OGRANICZEŃ (EWALUACJA)
# ==========================================

@patch(f"{TARGET_MODULE}.calculate_dynamic_max_node_distance", return_value=20.0)
@patch(f"{TARGET_MODULE}.evaluate_bspline_trajectory_sync")
def test_evaluate_shapes_and_values(mock_sync, mock_calc, basic_params):
    """
    Sprawdza, czy funkcja poprawnie zespala wyniki własnej analizy kinematycznej
    (np.diff) z wynikami C-podobnej funkcji B-Spline.
    """
    starts = np.array([[0.0, 0.0, 0.0]])
    targets = np.array([[100.0, 100.0, 0.0]])
    
    pop_size = 2
    n_drones = 1
    n_inner = 3
    
    # Mockujemy wyniki z zewnętrznej, zsynchronizowanej funkcji Numbą:
    # 1. obs_collisions: shape (PopSize, NDrones)
    # 2. lengths: shape (PopSize, NDrones)
    # 3. swarm_collisions: shape (PopSize,)
    mock_sync.return_value = (
        np.array([[1.0], [0.0]]),       # Agent 0 zderzył się raz. Agent 1 czysty.
        np.array([[50.0], [45.0]]),     # Długości tras
        np.array([0.0, 2.0])            # Kolizje wewnątrz roju
    )
    
    evaluator = VectorizedEvaluator(None, starts, targets, n_inner, basic_params)
    
    # Zbudujmy płaską (zerową) trajektorię, żeby uniknąć kar kinematycznych z diff1 i diff2
    n_control_points = n_inner + 2 
    control_points = np.zeros((pop_size, n_drones, n_control_points, 3))
    
    out = {}
    evaluator.evaluate(control_points, out)
    
    # Weryfikacja wymiarowości (Zgodnie z protokołem Pymoo NSGA-III)
    assert "F" in out
    assert "G" in out
    assert out["F"].shape == (pop_size, 3)
    assert out["G"].shape == (pop_size, 3)
    
    F = out["F"]
    G = out["G"]
    
    # --- Cele (F) ---
    # Kolumna 0: Długość (suma dla drona)
    np.testing.assert_array_almost_equal(F[:, 0], [50.0, 45.0])
    # Kolumna 1: Smoothness (0 dla pustych trajektorii)
    np.testing.assert_array_almost_equal(F[:, 1], [0.0, 0.0])
    # Kolumna 2: Ryzyko kolizji międzydronowej
    np.testing.assert_array_almost_equal(F[:, 2], [0.0, 2.0])
    
    # --- Ograniczenia (G) ---
    # Kolumna 0: Kary za uderzenia w przeszkody
    np.testing.assert_array_almost_equal(G[:, 0], [1.0, 0.0])
    # Kolumna 1: Bezpieczny margines roju (swarm_collisions - 0.01)
    np.testing.assert_array_almost_equal(G[:, 1], [-0.01, 1.99])
    # Kolumna 2: Kary kinematyczne (dist + accel) -> 0 w tym teście
    np.testing.assert_array_almost_equal(G[:, 2], [0.0, 0.0])