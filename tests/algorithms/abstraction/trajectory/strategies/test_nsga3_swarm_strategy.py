import numpy as np
from unittest.mock import patch, MagicMock
import pytest

from src.environments.obstacles.ObstacleShape import ObstacleShape
from src.algorithms.abstraction.trajectory.strategies.nsga3_swarm_strategy import (
    nsga3_swarm_strategy,
    SwarmOptimizationProblem,
    calculate_n_partitions,
    MultiConditionTermination
)

TARGET_MODULE = "src.algorithms.abstraction.trajectory.strategies.nsga3_swarm_strategy"


@pytest.fixture
def mock_world_data():
    world = MagicMock()
    world.min_bounds = np.array([0.0, 0.0, 0.0])
    world.max_bounds = np.array([100.0, 100.0, 20.0])
    world.bounds = np.array([[0.0, 100.0], [0.0, 100.0], [0.0, 20.0]])
    return world


@pytest.fixture
def mock_obstacles_data():
    """
    Atrapa danych o przeszkodach.
    Walidator wymaga macierzy o kształcie (N, 6), gdzie N to liczba przeszkód.
    Używamy macierzy (0, 6) do symulacji środowiska bez przeszkód.
    """
    obstacles = MagicMock()
    # Zmieniamy kształt (shape) na oczekiwany przez walidator (0 wierszy, 6 kolumn)
    obstacles.data = np.empty((0, 6))
    return obstacles

def test_calculate_n_partitions():
    assert calculate_n_partitions(100, 3) == 13
    assert calculate_n_partitions(100, 2) == 12


def test_swarm_optimization_problem_bounds(mock_world_data):
    start = np.array([[10.0, 10.0, 1.0]])
    target = np.array([[90.0, 90.0, 5.0]])
    evaluator = MagicMock()

    problem = SwarmOptimizationProblem(
        n_drones=1,
        n_inner_points=3,
        world_data=mock_world_data,
        evaluator=evaluator,
        start_pos=start,
        target_pos=target
    )

    xl = problem.xl
    xu = problem.xu

    assert len(xl) == 9

    z_indices = [2, 5, 8]
    assert np.allclose(xl[z_indices], 0.8)
    assert np.allclose(xu[z_indices], 19.5)


@patch("hydra.core.hydra_config.HydraConfig")
@patch(f"{TARGET_MODULE}.minimize")
def test_nsga3_strategy_fallback_with_altitude(
    mock_minimize, mock_hydraconfig, mock_world_data, mock_obstacles_data, tmp_path
):
    mock_hydraconfig.get.return_value.runtime.output_dir = str(tmp_path)

    mock_res = MagicMock()
    mock_res.X = None
    mock_minimize.return_value = mock_res

    start = np.array([[0.0, 0.0, 0.0]])
    target = np.array([[10.0, 10.0, 0.0]])

    result = nsga3_swarm_strategy(
        start_positions=start,
        target_positions=target,
        obstacles_data=mock_obstacles_data,
        world_data=mock_world_data,
        number_of_waypoints=5,
        drone_swarm_size=1,
        algorithm_params={"min_safe_altitude": 2.0}
    )

    expected_z = np.full(5, 2.0)
    np.testing.assert_array_almost_equal(result[0, :, 2], expected_z)


@patch("src.algorithms.abstraction.trajectory.strategies.timing_utils.TimingCollector")
@patch(f"{TARGET_MODULE}.OptimizationHistoryWriter")
@patch("hydra.core.hydra_config.HydraConfig")
@patch(f"{TARGET_MODULE}.VectorizedEvaluator")
@patch(f"{TARGET_MODULE}.generate_bspline_batch")
@patch(f"{TARGET_MODULE}.minimize")
def test_nsga3_strategy_success(
    mock_minimize, 
    mock_bspline, 
    mock_evaluator, 
    mock_hydraconfig, 
    mock_writer, 
    mock_timing, 
    mock_world_data, 
    mock_obstacles_data
):
    # Ustawienie atrapy ścieżki (tylko po to, aby os.path.join w logice kodu nie wyrzucił TypeError)
    mock_hydraconfig.get.return_value.runtime.output_dir = "dummy_dir"

    # Symulacja udanej optymalizacji wielokryterialnej (Front Pareto z 1 rozwiązaniem)
    mock_res = MagicMock()
    mock_res.X = np.zeros((1, 9))
    mock_res.F = np.array([[10.0, 5.0, 0.0]])
    mock_res.G = np.array([[0.0]])
    mock_minimize.return_value = mock_res

    # Mockowanie post-processingu B-Spline
    n_waypoints = 10
    mock_bspline.return_value = np.zeros((1, 1, n_waypoints, 3))

    start = np.array([[0.0, 0.0, 1.0]])
    target = np.array([[10.0, 10.0, 1.0]])

    result = nsga3_swarm_strategy(
        start_positions=start,
        target_positions=target,
        obstacles_data=mock_obstacles_data,
        world_data=mock_world_data,
        number_of_waypoints=n_waypoints,
        drone_swarm_size=1,
        algorithm_params={"decision_mode": "safety", "n_inner_waypoints": 3, "pop_size": 10}
    )

    assert result.shape == (1, n_waypoints, 3)
    mock_bspline.assert_called_once()