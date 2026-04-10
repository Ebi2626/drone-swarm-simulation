import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from src.algorithms.abstraction.count_trajectories import count_trajectories
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.abstraction.generate_obstacles import ObstaclesData

STRATEGY_PATH = "src.algorithms.abstraction.trajectory.strategies.nsga3_swarm_strategy.nsga3_swarm_strategy"

@pytest.fixture
def mock_world_data():
    """Tworzy zaślepkę dla obiektu WorldData."""
    return MagicMock(spec=WorldData)

@pytest.fixture
def mock_obstacles_data():
    """Tworzy zaślepkę dla obiektu ObstaclesData."""
    return MagicMock(spec=ObstaclesData)

@pytest.fixture
def dummy_start_positions():
    """Przykładowe pozycje startowe dla 2 dronów."""
    return np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

@pytest.fixture
def dummy_target_positions():
    """Przykładowe pozycje docelowe dla 2 dronów."""
    return np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])

def test_count_trajectories_delegates_to_protocol(
    mock_world_data,
    mock_obstacles_data,
    dummy_start_positions,
    dummy_target_positions
):
    # --- ARRANGE (Przygotowanie danych) ---
    drone_swarm_size = 2
    number_of_waypoints = 5
    algorithm_params = {"max_iterations": 100, "c1": 2.0}

    # Definiujemy sztuczny wynik, który ma zostać zwrócony przez naszą strategię
    # Kształt (N, W, 3) = (2, 5, 3)
    expected_trajectories = np.ones((drone_swarm_size, number_of_waypoints, 3))

    # Tworzymy mocka dla TrajectoryStrategyProtocol
    mock_counting_protocol = Mock()
    mock_counting_protocol.return_value = expected_trajectories

    # --- ACT (Wykonanie funkcji) ---
    result = count_trajectories(
        world_data=mock_world_data,
        obstacles_data=mock_obstacles_data,
        counting_protocol=mock_counting_protocol,
        drone_swarm_size=drone_swarm_size,
        number_of_waypoints=number_of_waypoints,
        start_positions=dummy_start_positions,
        target_positions=dummy_target_positions,
        algorithm_params=algorithm_params
    )

    # --- ASSERT (Weryfikacja) ---
    # 1. Weryfikacja, czy strategia została wywołana dokładnie raz z odpowiednimi argumentami nazwanymi (kwargs)
    mock_counting_protocol.assert_called_once_with(
        start_positions=dummy_start_positions,
        target_positions=dummy_target_positions,
        obstacles_data=mock_obstacles_data,
        world_data=mock_world_data,
        number_of_waypoints=number_of_waypoints,
        drone_swarm_size=drone_swarm_size,
        algorithm_params=algorithm_params
    )

    # 2. Weryfikacja, czy funkcja count_trajectories zwraca dokładnie to, co zwróciła strategia
    np.testing.assert_array_equal(result, expected_trajectories)

def test_count_trajectories_with_none_algorithm_params(
    mock_world_data,
    mock_obstacles_data,
    dummy_start_positions,
    dummy_target_positions
):
    """Test sprawdzający zachowanie domyślnego parametru algorithm_params=None"""
    mock_counting_protocol = Mock()
    mock_counting_protocol.return_value = np.zeros((2, 5, 3))

    count_trajectories(
        world_data=mock_world_data,
        obstacles_data=mock_obstacles_data,
        counting_protocol=mock_counting_protocol,
        drone_swarm_size=2,
        number_of_waypoints=5,
        start_positions=dummy_start_positions,
        target_positions=dummy_target_positions
    )

    # Sprawdzamy, czy do protokołu faktycznie przekazano domyślne None
    mock_counting_protocol.assert_called_once()
    called_kwargs = mock_counting_protocol.call_args.kwargs
    assert called_kwargs["algorithm_params"] is None

@patch(STRATEGY_PATH, autospec=True)
def test_wrapper_cooperation_with_nsga3_strategy(
    mocked_nsga3_strategy,
    mock_world_data,
    mock_obstacles_data,
    dummy_start_positions,
    dummy_target_positions
):
    """
    Testuje, czy wrapper 'count_trajectories' prawidłowo komunikuje się 
    z rzeczywistą strategią 'nsga3_swarm_strategy', zachowując zgodność sygnatur.
    """
    # --- ARRANGE ---
    drone_swarm_size = 2
    number_of_waypoints = 15
    algorithm_params = {"pop_size": 50, "n_gen": 20}
    
    # Ustawiamy sztuczny wynik, który ma zwrócić zmockowana strategia
    expected_output = np.zeros((drone_swarm_size, number_of_waypoints, 3))
    mocked_nsga3_strategy.return_value = expected_output

    # --- ACT ---
    # Wywołujemy nasz wrapper, przekazując mu ZMOCKOWANĄ, ale sprawdzaną pod kątem typu strategię NSGA-III
    result = count_trajectories(
        world_data=mock_world_data,
        obstacles_data=mock_obstacles_data,
        counting_protocol=mocked_nsga3_strategy,
        drone_swarm_size=drone_swarm_size,
        number_of_waypoints=number_of_waypoints,
        start_positions=dummy_start_positions,
        target_positions=dummy_target_positions,
        algorithm_params=algorithm_params
    )

    # --- ASSERT ---
    # 1. Sprawdzamy, czy wrapper zwrócił wynik strategii
    np.testing.assert_array_equal(result, expected_output)

    # 2. Sprawdzamy, czy strategia została wywołana z poprawnymi parametrami nazwanymi.
    # Dzięki autospec=True, jeśli nsga3_swarm_strategy nie przyjmuje któregoś z tych argumentów, 
    # test natychmiast zgłosi błąd TypeError (tak jak w prawdziwym kodzie).
    mocked_nsga3_strategy.assert_called_once_with(
        start_positions=dummy_start_positions,
        target_positions=dummy_target_positions,
        obstacles_data=mock_obstacles_data,
        world_data=mock_world_data,
        number_of_waypoints=number_of_waypoints,
        drone_swarm_size=drone_swarm_size,
        algorithm_params=algorithm_params
    )