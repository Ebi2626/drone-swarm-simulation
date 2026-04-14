import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

from src.runner.GenerationDataStrategy import GenerationDataStrategy
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.obstacles.ObstacleShape import ObstacleShape

MODULE = "src.runner.GenerationDataStrategy"


@pytest.fixture
def strategy():
    return GenerationDataStrategy()


@pytest.fixture
def mock_runner():
    """Tworzy mocka ExperimentRunner z minimalnymi wymaganymi polami."""
    runner = MagicMock()
    runner.track_width = 100.0
    runner.track_length = 200.0
    runner.track_height = 50.0
    runner.ground_position = 0.0
    runner.placement_strategy_name = "strategy_random_uniform"
    runner.obstacles_number = 10
    runner.shape_type = ObstacleShape.BOX
    runner.obstacle_length = 5.0
    runner.obstacle_width = 3.0
    runner.obstacle_height = 10.0
    runner.start_positions = np.array([[0, 0, 1], [1, 1, 1]], dtype=np.float64)
    runner.end_positions = np.array([[90, 190, 5], [85, 185, 5]], dtype=np.float64)
    runner.number_of_waypoints = 5
    runner.num_drones = 2
    runner.cfg = OmegaConf.create({
        "optimizer": {"_target_": "some.Optimizer", "algorithm_params": {"n_inner_waypoints": 5}}
    })
    runner.logger = None
    runner.world_data = None
    runner.obstacles_data = None
    runner.trajectories = None
    return runner


@pytest.fixture
def fake_world_data():
    return WorldData(
        dimensions=np.array([100.0, 200.0, 50.0]),
        min_bounds=np.array([0.0, 0.0, 0.0]),
        max_bounds=np.array([100.0, 200.0, 50.0]),
        bounds=np.array([[0.0, 100.0], [0.0, 200.0], [0.0, 50.0]]),
        center=np.array([50.0, 100.0, 25.0]),
    )


@pytest.fixture
def fake_obstacles_data():
    data = np.zeros((10, 6), dtype=np.float64)
    return ObstaclesData(data=data, shape_type=ObstacleShape.BOX)


# --- Testy głównego przepływu ---

@patch(f"{MODULE}.instantiate")
@patch(f"{MODULE}.generate_obstacles")
@patch(f"{MODULE}.generate_world_boundaries")
@patch(f"{MODULE}.get_placement_strategy")
def test_prepare_data_full_flow(
    mock_get_strategy, mock_gen_world, mock_gen_obstacles, mock_instantiate,
    strategy, mock_runner, fake_world_data, fake_obstacles_data,
):
    """Sprawdza pełny przepływ: granice -> przeszkody -> optymalizator -> przypisanie."""
    mock_gen_world.return_value = fake_world_data
    mock_gen_obstacles.return_value = fake_obstacles_data
    mock_get_strategy.return_value = MagicMock()

    fake_trajectories = np.random.rand(2, 7, 3)
    mock_counting_strategy = MagicMock(return_value=fake_trajectories)
    mock_instantiate.return_value = mock_counting_strategy

    strategy.prepare_data(mock_runner)

    # 1. Granice świata
    mock_gen_world.assert_called_once_with(
        width=100.0, length=200.0, height=50.0, ground_height=0.0
    )
    assert mock_runner.world_data == fake_world_data

    # 2. Przeszkody
    mock_gen_obstacles.assert_called_once()
    assert mock_runner.obstacles_data == fake_obstacles_data

    # 3. Optymalizator
    mock_instantiate.assert_called_once_with(mock_runner.cfg.optimizer)
    mock_counting_strategy.assert_called_once()

    # 4. Trajektorie przypisane do runnera
    np.testing.assert_array_equal(mock_runner.trajectories, fake_trajectories)


@patch(f"{MODULE}.instantiate")
@patch(f"{MODULE}.generate_obstacles")
@patch(f"{MODULE}.generate_world_boundaries")
@patch(f"{MODULE}.get_placement_strategy")
def test_world_boundaries_params_forwarded(
    mock_get_strategy, mock_gen_world, mock_gen_obstacles, mock_instantiate,
    strategy, mock_runner, fake_world_data,
):
    """Sprawdza czy parametry granic świata są poprawnie przekazywane z runnera."""
    mock_gen_world.return_value = fake_world_data
    mock_gen_obstacles.return_value = MagicMock()
    mock_instantiate.return_value = MagicMock(return_value=np.zeros((2, 7, 3)))

    mock_runner.track_width = 42.0
    mock_runner.track_length = 84.0
    mock_runner.track_height = 21.0
    mock_runner.ground_position = 1.5

    strategy.prepare_data(mock_runner)

    mock_gen_world.assert_called_once_with(
        width=42.0, length=84.0, height=21.0, ground_height=1.5
    )


@patch(f"{MODULE}.instantiate")
@patch(f"{MODULE}.generate_obstacles")
@patch(f"{MODULE}.generate_world_boundaries")
@patch(f"{MODULE}.get_placement_strategy")
def test_obstacles_params_forwarded(
    mock_get_strategy, mock_gen_world, mock_gen_obstacles, mock_instantiate,
    strategy, mock_runner, fake_world_data, fake_obstacles_data,
):
    """Sprawdza czy parametry przeszkód są poprawnie przekazywane do generate_obstacles."""
    mock_gen_world.return_value = fake_world_data
    mock_gen_obstacles.return_value = fake_obstacles_data
    mock_placement_fn = MagicMock()
    mock_get_strategy.return_value = mock_placement_fn
    mock_instantiate.return_value = MagicMock(return_value=np.zeros((2, 7, 3)))

    strategy.prepare_data(mock_runner)

    mock_get_strategy.assert_called_once_with("strategy_random_uniform")
    mock_gen_obstacles.assert_called_once_with(
        fake_world_data,
        n_obstacles=10,
        shape_type=ObstacleShape.BOX,
        placement_strategy=mock_placement_fn,
        size_params={
            'length': 5.0,
            'width': 3.0,
            'height': 10.0,
        },
        start_positions=mock_runner.start_positions,
        target_positions=mock_runner.end_positions,
    )


@patch(f"{MODULE}.instantiate")
@patch(f"{MODULE}.generate_obstacles")
@patch(f"{MODULE}.generate_world_boundaries")
@patch(f"{MODULE}.get_placement_strategy")
def test_optimizer_receives_correct_args(
    mock_get_strategy, mock_gen_world, mock_gen_obstacles, mock_instantiate,
    strategy, mock_runner, fake_world_data, fake_obstacles_data,
):
    """Sprawdza, czy algorytm optymalizacyjny jest wywoływany z poprawnymi argumentami."""
    mock_gen_world.return_value = fake_world_data
    mock_gen_obstacles.return_value = fake_obstacles_data

    mock_counting_strategy = MagicMock(return_value=np.zeros((2, 7, 3)))
    mock_instantiate.return_value = mock_counting_strategy

    strategy.prepare_data(mock_runner)

    # functools.partial wywołuje counting_strategy z tymi parametrami, a potem ()
    _, kwargs = mock_counting_strategy.call_args
    assert np.array_equal(kwargs["start_positions"], mock_runner.start_positions)
    assert np.array_equal(kwargs["target_positions"], mock_runner.end_positions)
    assert kwargs["obstacles_data"] == fake_obstacles_data
    assert kwargs["world_data"] == fake_world_data
    assert kwargs["number_of_waypoints"] == 5
    assert kwargs["drone_swarm_size"] == 2


# --- Przypadki brzegowe ---

@patch(f"{MODULE}.instantiate")
@patch(f"{MODULE}.generate_obstacles")
@patch(f"{MODULE}.generate_world_boundaries")
def test_no_obstacles_when_placement_strategy_is_none(
    mock_gen_world, mock_gen_obstacles, mock_instantiate,
    strategy, mock_runner, fake_world_data,
):
    """Gdy placement_strategy_name jest None, przeszkody NIE powinny być generowane."""
    mock_gen_world.return_value = fake_world_data
    mock_instantiate.return_value = MagicMock(return_value=np.zeros((2, 7, 3)))
    mock_runner.placement_strategy_name = None

    strategy.prepare_data(mock_runner)

    mock_gen_obstacles.assert_not_called()


@patch(f"{MODULE}.instantiate")
@patch(f"{MODULE}.generate_obstacles")
@patch(f"{MODULE}.generate_world_boundaries")
@patch(f"{MODULE}.get_placement_strategy")
def test_logger_called_when_present(
    mock_get_strategy, mock_gen_world, mock_gen_obstacles, mock_instantiate,
    strategy, mock_runner, fake_world_data, fake_obstacles_data,
):
    """Sprawdza, że logger archiwizuje trajektorie, świat i przeszkody."""
    mock_gen_world.return_value = fake_world_data
    mock_gen_obstacles.return_value = fake_obstacles_data
    fake_trajectories = np.random.rand(2, 7, 3)
    mock_instantiate.return_value = MagicMock(return_value=fake_trajectories)

    mock_logger = MagicMock()
    mock_runner.logger = mock_logger

    strategy.prepare_data(mock_runner)

    mock_logger.log_chosen_trajectories.assert_called_once_with(fake_trajectories)
    mock_logger.log_world_dimensions.assert_called_once_with(fake_world_data)
    mock_logger.log_obstacles.assert_called_once_with(fake_obstacles_data)


@patch(f"{MODULE}.instantiate")
@patch(f"{MODULE}.generate_obstacles")
@patch(f"{MODULE}.generate_world_boundaries")
@patch(f"{MODULE}.get_placement_strategy")
def test_logger_not_called_when_none(
    mock_get_strategy, mock_gen_world, mock_gen_obstacles, mock_instantiate,
    strategy, mock_runner, fake_world_data, fake_obstacles_data,
):
    """Sprawdza, że brak loggera nie powoduje błędów."""
    mock_gen_world.return_value = fake_world_data
    mock_gen_obstacles.return_value = fake_obstacles_data
    mock_instantiate.return_value = MagicMock(return_value=np.zeros((2, 7, 3)))
    mock_runner.logger = None

    # Nie powinno wyrzucić wyjątku
    strategy.prepare_data(mock_runner)


@patch(f"{MODULE}.instantiate")
@patch(f"{MODULE}.generate_obstacles")
@patch(f"{MODULE}.generate_world_boundaries")
@patch(f"{MODULE}.get_placement_strategy")
def test_single_drone(
    mock_get_strategy, mock_gen_world, mock_gen_obstacles, mock_instantiate,
    strategy, mock_runner, fake_world_data, fake_obstacles_data,
):
    """Sprawdza poprawne działanie dla roju złożonego z jednego drona."""
    mock_gen_world.return_value = fake_world_data
    mock_gen_obstacles.return_value = fake_obstacles_data

    mock_runner.num_drones = 1
    mock_runner.start_positions = np.array([[0, 0, 1]], dtype=np.float64)
    mock_runner.end_positions = np.array([[90, 190, 5]], dtype=np.float64)

    fake_traj = np.random.rand(1, 7, 3)
    mock_counting_strategy = MagicMock(return_value=fake_traj)
    mock_instantiate.return_value = mock_counting_strategy

    strategy.prepare_data(mock_runner)

    _, kwargs = mock_counting_strategy.call_args
    assert kwargs["drone_swarm_size"] == 1
    np.testing.assert_array_equal(mock_runner.trajectories, fake_traj)


@patch(f"{MODULE}.instantiate")
@patch(f"{MODULE}.generate_obstacles")
@patch(f"{MODULE}.generate_world_boundaries")
@patch(f"{MODULE}.get_placement_strategy")
def test_world_data_assigned_before_obstacles(
    mock_get_strategy, mock_gen_world, mock_gen_obstacles, mock_instantiate,
    strategy, mock_runner, fake_world_data, fake_obstacles_data,
):
    """
    Sprawdza, że world_data jest przypisane do runnera PRZED wygenerowaniem przeszkód,
    ponieważ generate_obstacles korzysta z runner.world_data jako pierwszego argumentu.
    """
    call_order = []

    def track_gen_world(*args, **kwargs):
        call_order.append("gen_world")
        return fake_world_data

    def track_gen_obstacles(*args, **kwargs):
        call_order.append("gen_obstacles")
        # W tym momencie runner.world_data powinno już istnieć
        assert mock_runner.world_data is not None, (
            "world_data musi być przypisane przed wygenerowaniem przeszkód"
        )
        return fake_obstacles_data

    mock_gen_world.side_effect = track_gen_world
    mock_gen_obstacles.side_effect = track_gen_obstacles
    mock_instantiate.return_value = MagicMock(return_value=np.zeros((2, 7, 3)))

    strategy.prepare_data(mock_runner)

    assert call_order == ["gen_world", "gen_obstacles"]


@patch(f"{MODULE}.instantiate")
@patch(f"{MODULE}.generate_obstacles")
@patch(f"{MODULE}.generate_world_boundaries")
@patch(f"{MODULE}.get_placement_strategy")
def test_zero_obstacles(
    mock_get_strategy, mock_gen_world, mock_gen_obstacles, mock_instantiate,
    strategy, mock_runner, fake_world_data,
):
    """Sprawdza zachowanie przy zerowej liczbie przeszkód (ale z aktywną strategią)."""
    mock_gen_world.return_value = fake_world_data
    empty_obstacles = ObstaclesData(
        data=np.zeros((0, 6), dtype=np.float64),
        shape_type=ObstacleShape.BOX,
    )
    mock_gen_obstacles.return_value = empty_obstacles
    mock_instantiate.return_value = MagicMock(return_value=np.zeros((2, 7, 3)))

    mock_runner.obstacles_number = 0

    strategy.prepare_data(mock_runner)

    assert mock_runner.obstacles_data == empty_obstacles


def test_is_subclass_of_experiment_data_strategy():
    """Sprawdza, że GenerationDataStrategy implementuje interfejs ExperimentDataStrategy."""
    from src.runner.ExperimentDataStrategy import ExperimentDataStrategy
    assert issubclass(GenerationDataStrategy, ExperimentDataStrategy)


@patch(f"{MODULE}.instantiate")
@patch(f"{MODULE}.generate_obstacles")
@patch(f"{MODULE}.generate_world_boundaries")
@patch(f"{MODULE}.get_placement_strategy")
def test_obstacles_data_used_by_optimizer(
    mock_get_strategy, mock_gen_world, mock_gen_obstacles, mock_instantiate,
    strategy, mock_runner, fake_world_data, fake_obstacles_data,
):
    """
    Sprawdza, że dane przeszkód wygenerowane w kroku 2 trafiają do optymalizatora w kroku 3.
    """
    mock_gen_world.return_value = fake_world_data
    mock_gen_obstacles.return_value = fake_obstacles_data

    mock_counting_strategy = MagicMock(return_value=np.zeros((2, 7, 3)))
    mock_instantiate.return_value = mock_counting_strategy

    strategy.prepare_data(mock_runner)

    _, kwargs = mock_counting_strategy.call_args
    assert kwargs["obstacles_data"] is fake_obstacles_data
    assert kwargs["world_data"] is fake_world_data
