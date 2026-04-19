import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from omegaconf import OmegaConf
from pathlib import Path

from src.runner.ReplayDataStrategy import ReplayDataStrategy
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.obstacles.ObstacleShape import ObstacleShape


# --- Fixtures: dane CSV zapisywane na dysk (tmp_path) ---

@pytest.fixture
def world_csv(tmp_path):
    """Tworzy plik world_boundaries.csv zgodny z formatem SimulationLogger."""
    df = pd.DataFrame(
        {
            "Dimension": [100.0, 200.0, 50.0],
            "Min_Bound": [0.0, 0.0, 0.1],
            "Max_Bound": [100.0, 200.0, 50.0],
            "Center": [50.0, 100.0, 25.05],
        },
        index=["X", "Y", "Z"],
    )
    path = tmp_path / "world_boundaries.csv"
    df.to_csv(path, index=True, index_label="Axis", float_format="%.4f")
    return tmp_path


@pytest.fixture
def box_obstacles_csv(tmp_path):
    """Tworzy plik generated_obstacles.csv z przeszkodami BOX."""
    df = pd.DataFrame(
        {
            "x": [10.0, 50.0, 80.0],
            "y": [20.0, 100.0, 180.0],
            "z": [0.0, 0.0, 0.0],
            "length": [5.0, 5.0, 5.0],
            "width": [3.0, 3.0, 3.0],
            "height": [10.0, 10.0, 10.0],
        }
    )
    path = tmp_path / "generated_obstacles.csv"
    df.to_csv(path, index=False, float_format="%.4f")
    return tmp_path


@pytest.fixture
def cylinder_obstacles_csv(tmp_path):
    """Tworzy plik generated_obstacles.csv z przeszkodami CYLINDER (z unused_dim)."""
    df = pd.DataFrame(
        {
            "x": [10.0, 50.0],
            "y": [20.0, 100.0],
            "z": [0.0, 0.0],
            "radius": [1.0, 1.0],
            "height": [10.0, 10.0],
            "unused_dim": [0.0, 0.0],
        }
    )
    path = tmp_path / "generated_obstacles.csv"
    df.to_csv(path, index=False, float_format="%.4f")
    return tmp_path


@pytest.fixture
def trajectories_csv(tmp_path):
    """Tworzy counted_trajectories.csv dla 2 dronow, 3 waypointy kazdy."""
    df = pd.DataFrame(
        {
            "drone_id": [0, 0, 0, 1, 1, 1],
            "waypoint_id": [0, 1, 2, 0, 1, 2],
            "x": [0.0, 5.0, 10.0, 1.0, 6.0, 11.0],
            "y": [0.0, 5.0, 10.0, 1.0, 6.0, 11.0],
            "z": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    path = tmp_path / "counted_trajectories.csv"
    df.to_csv(path, index=False, float_format="%.4f")
    return tmp_path


def _make_all_csvs(tmp_path, shape_type="BOX"):
    """Pomocnicza funkcja tworzaca komplet plikow CSV w tmp_path."""
    # world_boundaries.csv
    world_df = pd.DataFrame(
        {
            "Dimension": [100.0, 200.0, 50.0],
            "Min_Bound": [0.0, 0.0, 0.1],
            "Max_Bound": [100.0, 200.0, 50.0],
            "Center": [50.0, 100.0, 25.05],
        },
        index=["X", "Y", "Z"],
    )
    world_df.to_csv(tmp_path / "world_boundaries.csv", index=True, index_label="Axis")

    # generated_obstacles.csv
    if shape_type == "BOX":
        obs_df = pd.DataFrame(
            {"x": [10.0], "y": [20.0], "z": [0.0], "length": [5.0], "width": [3.0], "height": [10.0]}
        )
    else:
        obs_df = pd.DataFrame(
            {"x": [10.0], "y": [20.0], "z": [0.0], "radius": [1.0], "height": [10.0], "unused_dim": [0.0]}
        )
    obs_df.to_csv(tmp_path / "generated_obstacles.csv", index=False)

    # counted_trajectories.csv
    traj_df = pd.DataFrame(
        {
            "drone_id": [0, 0, 0, 1, 1, 1],
            "waypoint_id": [0, 1, 2, 0, 1, 2],
            "x": [0.0, 5.0, 10.0, 1.0, 6.0, 11.0],
            "y": [0.0, 5.0, 10.0, 1.0, 6.0, 11.0],
            "z": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    traj_df.to_csv(tmp_path / "counted_trajectories.csv", index=False)
    return tmp_path


@pytest.fixture
def mock_runner():
    runner = MagicMock()
    runner.cfg = OmegaConf.create({
        "environment": {"params": {"shape_type": "BOX"}},
    })
    runner.world_data = None
    runner.obstacles_data = None
    runner.drones_trajectories = None
    runner.start_positions = None
    runner.end_positions = None
    return runner


# --- Testy _map_to_world_data ---

class TestMapToWorldData:
    def test_reconstructs_dimensions(self, world_csv):
        strategy = ReplayDataStrategy(world_csv)
        df = pd.read_csv(world_csv / "world_boundaries.csv")
        result = strategy._map_to_world_data(df)

        np.testing.assert_array_almost_equal(result.dimensions, [100.0, 200.0, 50.0])

    def test_reconstructs_bounds(self, world_csv):
        strategy = ReplayDataStrategy(world_csv)
        df = pd.read_csv(world_csv / "world_boundaries.csv")
        result = strategy._map_to_world_data(df)

        np.testing.assert_array_almost_equal(result.min_bounds, [0.0, 0.0, 0.1])
        np.testing.assert_array_almost_equal(result.max_bounds, [100.0, 200.0, 50.0])

    def test_reconstructs_center(self, world_csv):
        strategy = ReplayDataStrategy(world_csv)
        df = pd.read_csv(world_csv / "world_boundaries.csv")
        result = strategy._map_to_world_data(df)

        np.testing.assert_array_almost_equal(result.center, [50.0, 100.0, 25.05])

    def test_bounds_matrix_shape(self, world_csv):
        strategy = ReplayDataStrategy(world_csv)
        df = pd.read_csv(world_csv / "world_boundaries.csv")
        result = strategy._map_to_world_data(df)

        assert result.bounds.shape == (3, 2)
        np.testing.assert_array_almost_equal(
            result.bounds,
            [[0.0, 100.0], [0.0, 200.0], [0.1, 50.0]],
        )

    def test_returns_world_data_instance(self, world_csv):
        strategy = ReplayDataStrategy(world_csv)
        df = pd.read_csv(world_csv / "world_boundaries.csv")
        result = strategy._map_to_world_data(df)

        assert isinstance(result, WorldData)

    def test_axis_order_independent(self, tmp_path):
        """WorldData poprawna nawet gdy wiersze CSV sa w odwrotnej kolejnosci (Z, Y, X)."""
        df = pd.DataFrame(
            {
                "Dimension": [50.0, 200.0, 100.0],
                "Min_Bound": [0.1, 0.0, 0.0],
                "Max_Bound": [50.0, 200.0, 100.0],
                "Center": [25.05, 100.0, 50.0],
            },
            index=["Z", "Y", "X"],
        )
        df.to_csv(tmp_path / "world_boundaries.csv", index=True, index_label="Axis")

        strategy = ReplayDataStrategy(tmp_path)
        df_read = pd.read_csv(tmp_path / "world_boundaries.csv")
        result = strategy._map_to_world_data(df_read)

        np.testing.assert_array_almost_equal(result.dimensions, [100.0, 200.0, 50.0])


# --- Testy _map_to_obstacles_data ---

class TestMapToObstaclesData:
    def test_box_obstacles(self, box_obstacles_csv):
        strategy = ReplayDataStrategy(box_obstacles_csv)
        df = pd.read_csv(box_obstacles_csv / "generated_obstacles.csv")
        result = strategy._map_to_obstacles_data(df, "BOX")

        assert isinstance(result, ObstaclesData)
        assert result.shape_type == ObstacleShape.BOX
        assert result.data.shape == (3, 6)
        assert result.count == 3

    def test_box_data_values(self, box_obstacles_csv):
        strategy = ReplayDataStrategy(box_obstacles_csv)
        df = pd.read_csv(box_obstacles_csv / "generated_obstacles.csv")
        result = strategy._map_to_obstacles_data(df, "BOX")

        np.testing.assert_array_almost_equal(result.data[0], [10.0, 20.0, 0.0, 5.0, 3.0, 10.0])

    def test_cylinder_obstacles(self, cylinder_obstacles_csv):
        strategy = ReplayDataStrategy(cylinder_obstacles_csv)
        df = pd.read_csv(cylinder_obstacles_csv / "generated_obstacles.csv")
        result = strategy._map_to_obstacles_data(df, "CYLINDER")

        assert result.shape_type == ObstacleShape.CYLINDER
        assert result.data.shape == (2, 6)

    def test_cylinder_data_values(self, cylinder_obstacles_csv):
        strategy = ReplayDataStrategy(cylinder_obstacles_csv)
        df = pd.read_csv(cylinder_obstacles_csv / "generated_obstacles.csv")
        result = strategy._map_to_obstacles_data(df, "CYLINDER")

        # [x, y, z, radius, height, unused_dim]
        np.testing.assert_array_almost_equal(result.data[0], [10.0, 20.0, 0.0, 1.0, 10.0, 0.0])

    def test_case_insensitive_shape_type(self, box_obstacles_csv):
        strategy = ReplayDataStrategy(box_obstacles_csv)
        df = pd.read_csv(box_obstacles_csv / "generated_obstacles.csv")
        result = strategy._map_to_obstacles_data(df, "box")

        assert result.shape_type == ObstacleShape.BOX

    def test_missing_columns_raises_key_error(self, tmp_path):
        """Brak wymaganej kolumny powinien rzucic KeyError."""
        df = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [0.0]})
        df.to_csv(tmp_path / "generated_obstacles.csv", index=False)

        strategy = ReplayDataStrategy(tmp_path)
        df_read = pd.read_csv(tmp_path / "generated_obstacles.csv")

        with pytest.raises(KeyError, match="Brakuje następujących kolumn"):
            strategy._map_to_obstacles_data(df_read, "BOX")

    def test_data_dtype_is_float64(self, box_obstacles_csv):
        strategy = ReplayDataStrategy(box_obstacles_csv)
        df = pd.read_csv(box_obstacles_csv / "generated_obstacles.csv")
        result = strategy._map_to_obstacles_data(df, "BOX")

        assert result.data.dtype == np.float64


# --- Testy _map_to_trajectories ---

class TestMapToTrajectories:
    def test_tensor_shape(self, trajectories_csv):
        strategy = ReplayDataStrategy(trajectories_csv)
        df = pd.read_csv(trajectories_csv / "counted_trajectories.csv")
        result = strategy._map_to_trajectories(df)

        # 2 drony, 3 waypointy, 3 wspolrzedne
        assert result.shape == (2, 3, 3)

    def test_drone_0_values(self, trajectories_csv):
        strategy = ReplayDataStrategy(trajectories_csv)
        df = pd.read_csv(trajectories_csv / "counted_trajectories.csv")
        result = strategy._map_to_trajectories(df)

        expected_drone_0 = np.array([[0.0, 0.0, 1.0], [5.0, 5.0, 2.0], [10.0, 10.0, 3.0]])
        np.testing.assert_array_almost_equal(result[0], expected_drone_0)

    def test_drone_1_values(self, trajectories_csv):
        strategy = ReplayDataStrategy(trajectories_csv)
        df = pd.read_csv(trajectories_csv / "counted_trajectories.csv")
        result = strategy._map_to_trajectories(df)

        expected_drone_1 = np.array([[1.0, 1.0, 1.0], [6.0, 6.0, 2.0], [11.0, 11.0, 3.0]])
        np.testing.assert_array_almost_equal(result[1], expected_drone_1)

    def test_single_drone(self, tmp_path):
        df = pd.DataFrame(
            {
                "drone_id": [0, 0],
                "waypoint_id": [0, 1],
                "x": [0.0, 10.0],
                "y": [0.0, 10.0],
                "z": [1.0, 5.0],
            }
        )
        df.to_csv(tmp_path / "counted_trajectories.csv", index=False)

        strategy = ReplayDataStrategy(tmp_path)
        df_read = pd.read_csv(tmp_path / "counted_trajectories.csv")
        result = strategy._map_to_trajectories(df_read)

        assert result.shape == (1, 2, 3)

    def test_waypoints_sorted_correctly(self, tmp_path):
        """Waypointy powinny byc posortowane nawet gdy CSV nie jest chronologiczny."""
        df = pd.DataFrame(
            {
                "drone_id": [0, 0, 0],
                "waypoint_id": [2, 0, 1],
                "x": [30.0, 10.0, 20.0],
                "y": [30.0, 10.0, 20.0],
                "z": [3.0, 1.0, 2.0],
            }
        )
        df.to_csv(tmp_path / "counted_trajectories.csv", index=False)

        strategy = ReplayDataStrategy(tmp_path)
        df_read = pd.read_csv(tmp_path / "counted_trajectories.csv")
        result = strategy._map_to_trajectories(df_read)

        expected = np.array([[10.0, 10.0, 1.0], [20.0, 20.0, 2.0], [30.0, 30.0, 3.0]])
        np.testing.assert_array_almost_equal(result[0], expected)

    def test_dtype_is_float64(self, trajectories_csv):
        strategy = ReplayDataStrategy(trajectories_csv)
        df = pd.read_csv(trajectories_csv / "counted_trajectories.csv")
        result = strategy._map_to_trajectories(df)

        assert result.dtype == np.float64


# --- Testy prepare_data (pelny przeplyw) ---

class TestPrepareData:
    def test_full_flow_box(self, tmp_path, mock_runner):
        _make_all_csvs(tmp_path, shape_type="BOX")
        strategy = ReplayDataStrategy(tmp_path)

        strategy.prepare_data(mock_runner)

        assert isinstance(mock_runner.world_data, WorldData)
        assert isinstance(mock_runner.obstacles_data, ObstaclesData)
        assert mock_runner.drones_trajectories.shape == (2, 3, 3)

    def test_full_flow_cylinder(self, tmp_path, mock_runner):
        _make_all_csvs(tmp_path, shape_type="CYLINDER")
        mock_runner.cfg = OmegaConf.create({
            "environment": {"params": {"shape_type": "CYLINDER"}},
        })
        strategy = ReplayDataStrategy(tmp_path)

        strategy.prepare_data(mock_runner)

        assert mock_runner.obstacles_data.shape_type == ObstacleShape.CYLINDER

    def test_start_positions_from_first_waypoint(self, tmp_path, mock_runner):
        """Pozycje startowe powinny byc wyciagniete z pierwszego waypointu trajektorii."""
        _make_all_csvs(tmp_path)
        strategy = ReplayDataStrategy(tmp_path)

        strategy.prepare_data(mock_runner)

        # Dron 0: waypoint 0 -> [0.0, 0.0, 1.0]
        # Dron 1: waypoint 0 -> [1.0, 1.0, 1.0]
        np.testing.assert_array_almost_equal(
            mock_runner.start_positions, [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        )

    def test_end_positions_from_last_waypoint(self, tmp_path, mock_runner):
        """Pozycje koncowe powinny byc wyciagniete z ostatniego waypointu trajektorii."""
        _make_all_csvs(tmp_path)
        strategy = ReplayDataStrategy(tmp_path)

        strategy.prepare_data(mock_runner)

        # Dron 0: waypoint 2 -> [10.0, 10.0, 3.0]
        # Dron 1: waypoint 2 -> [11.0, 11.0, 3.0]
        np.testing.assert_array_almost_equal(
            mock_runner.end_positions, [[10.0, 10.0, 3.0], [11.0, 11.0, 3.0]]
        )

    def test_missing_world_csv_raises(self, tmp_path, mock_runner):
        """Brak pliku world_boundaries.csv powinien rzucic wyjatek."""
        strategy = ReplayDataStrategy(tmp_path)

        with pytest.raises(FileNotFoundError):
            strategy.prepare_data(mock_runner)

    def test_missing_obstacles_csv_raises(self, tmp_path, mock_runner):
        """Brak pliku generated_obstacles.csv powinien rzucic wyjatek po wczytaniu swiata."""
        # Tworzymy tylko world CSV
        world_df = pd.DataFrame(
            {
                "Dimension": [100.0, 200.0, 50.0],
                "Min_Bound": [0.0, 0.0, 0.0],
                "Max_Bound": [100.0, 200.0, 50.0],
                "Center": [50.0, 100.0, 25.0],
            },
            index=["X", "Y", "Z"],
        )
        world_df.to_csv(tmp_path / "world_boundaries.csv", index=True, index_label="Axis")

        strategy = ReplayDataStrategy(tmp_path)

        with pytest.raises(FileNotFoundError):
            strategy.prepare_data(mock_runner)

    def test_missing_trajectories_csv_raises(self, tmp_path, mock_runner):
        """Brak pliku counted_trajectories.csv powinien rzucic wyjatek."""
        # Tworzymy world + obstacles, ale bez trajektorii
        world_df = pd.DataFrame(
            {
                "Dimension": [100.0, 200.0, 50.0],
                "Min_Bound": [0.0, 0.0, 0.0],
                "Max_Bound": [100.0, 200.0, 50.0],
                "Center": [50.0, 100.0, 25.0],
            },
            index=["X", "Y", "Z"],
        )
        world_df.to_csv(tmp_path / "world_boundaries.csv", index=True, index_label="Axis")
        obs_df = pd.DataFrame(
            {"x": [10.0], "y": [20.0], "z": [0.0], "length": [5.0], "width": [3.0], "height": [10.0]}
        )
        obs_df.to_csv(tmp_path / "generated_obstacles.csv", index=False)

        strategy = ReplayDataStrategy(tmp_path)

        with pytest.raises(FileNotFoundError):
            strategy.prepare_data(mock_runner)

    def test_default_shape_type_when_missing_in_config(self, tmp_path):
        """Gdy shape_type brakuje w konfiguracji, uzywany jest domyslny CYLINDER."""
        _make_all_csvs(tmp_path, shape_type="CYLINDER")
        runner = MagicMock()
        runner.cfg = OmegaConf.create({
            "environment": {"params": {}},
        })

        strategy = ReplayDataStrategy(tmp_path)
        strategy.prepare_data(runner)

        assert runner.obstacles_data.shape_type == ObstacleShape.CYLINDER


# --- Testy konstruktora ---

class TestConstructor:
    def test_accepts_string_path(self):
        strategy = ReplayDataStrategy("/some/path")
        assert isinstance(strategy.results_path, Path)
        assert str(strategy.results_path) == "/some/path"

    def test_accepts_path_object(self, tmp_path):
        strategy = ReplayDataStrategy(tmp_path)
        assert isinstance(strategy.results_path, Path)
        assert strategy.results_path == tmp_path

    def test_is_subclass_of_experiment_data_strategy(self):
        from src.runner.ExperimentDataStrategy import ExperimentDataStrategy
        assert issubclass(ReplayDataStrategy, ExperimentDataStrategy)
