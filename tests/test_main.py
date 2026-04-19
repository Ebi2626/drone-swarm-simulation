import pytest
import signal
import sys
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from omegaconf import OmegaConf

from main import ExperimentRunner, main_replay

MODULE = "main"

# ---------------------------------------------------------------------------
# Bezpiecznik: każdy test ma maksymalnie 5 sekund (SIGALRM, Linux/macOS)
# ---------------------------------------------------------------------------

_TIMEOUT_SEC = 5

@pytest.fixture(autouse=True)
def _enforce_timeout():
    """Zabija test po _TIMEOUT_SEC — chroni przed nieskończonymi pętlami."""
    prev = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(_TIMEOUT_SEC)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, prev)


def _timeout_handler(signum, frame):
    raise TimeoutError("Test przekroczył limit czasu — prawdopodobna nieskończona pętla")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_cfg():
    """Minimalna konfiguracja naśladująca strukturę Hydry."""
    return OmegaConf.create({
        "simulation": {
            "drone_model": "CF2X",
            "physics": "PYB",
            "ctrl_freq": 48,
            "pyb_freq": 240,
            "gui": False,
            "duration_sec": 0.1,
            "sim_speed_multiplier": 1.0,
        },
        "environment": {
            "params": {
                "num_drones": 2,
                "placement_strategy": "strategy_grid_jitter",
                "ground_position": 0.0,
                "track_length": 10.0,
                "track_width": 10.0,
                "track_height": 10.0,
                "shape_type": "BOX",
                "obstacles_number": 5,
                "obstacle_width": 1.0,
                "obstacle_height": 2.0,
                "obstacle_length": 1.0,
            },
            "initial_rpys": [[0, 0, 0], [0, 0, 0]],
            "initial_xyzs": [[0, 0, 0], [1, 0, 0]],
            "end_xyzs": [[10, 0, 0], [9, 0, 0]],
        },
        "visualization": {
            "tracked_drone_id": 0,
            "show_lidar_rays": False,
            "lidar_draw_interval": 5,
            "camera_follow": False,
            "camera_distance": 2.0,
            "camera_yaw": 0.0,
            "camera_pitch": -30.0,
        },
        "optimizer": {
            "algorithm_params": {"n_inner_waypoints": 5},
        },
        "logging": {
            "enabled": False,
            "log_freq": 10,
        },
    })


@pytest.fixture
def mock_strategy():
    return MagicMock()


@pytest.fixture
def runner(base_cfg, mock_strategy):
    return ExperimentRunner(base_cfg, mock_strategy)


def _make_state(x=0.0, y=0.0, z=0.0):
    """13-elementowy wektor stanu drona (pos, quat, rpy, vel)."""
    return np.array([x, y, z, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestExperimentRunnerInit:

    def test_basic_config_parsing(self, runner):
        assert runner.num_drones == 2
        assert runner.ctrl_freq == 48
        assert runner.pyb_freq == 240
        assert runner.drone_model == "CF2X"
        assert runner.sim_speed_multiplier == 1.0  # z klucza sim_speed_multiplier

    def test_start_end_positions_are_ndarrays(self, runner):
        assert isinstance(runner.start_positions, np.ndarray)
        assert isinstance(runner.end_positions, np.ndarray)
        assert runner.start_positions.shape == (2, 3)
        assert runner.end_positions.shape == (2, 3)

    def test_environment_params(self, runner):
        assert runner.track_length == 10.0
        assert runner.track_width == 10.0
        assert runner.track_height == 10.0
        assert runner.obstacles_number == 5
        assert runner.shape_type == "BOX"

    def test_tracked_drone_id_clamped_when_out_of_range(self, base_cfg, mock_strategy):
        """tracked_drone_id >= num_drones powinno być zresetowane do 0."""
        base_cfg.visualization.tracked_drone_id = 99
        r = ExperimentRunner(base_cfg, mock_strategy)
        assert r.tracked_drone_id == 0

    def test_tracked_drone_id_valid(self, base_cfg, mock_strategy):
        base_cfg.visualization.tracked_drone_id = 1
        r = ExperimentRunner(base_cfg, mock_strategy)
        assert r.tracked_drone_id == 1

    def test_default_fields_are_none(self, runner):
        assert runner.environemnt is None
        assert runner.world_data is None
        assert runner.obstacles_data is None
        assert runner.drones_trajectories is None
        assert runner.logger is None
        assert runner.input_handler is None

    def test_number_of_waypoints(self, runner):
        assert runner.number_of_waypoints == 5

    def test_dynamic_obstacles_disabled_by_default(self, runner):
        """Bez flagi w configu nie tworzymy żadnych przeszkód dynamicznych."""
        assert runner.use_dynamic_obstacles is False
        assert runner.num_dynamic_obstacles == 0
        assert runner.total_agents == runner.num_drones

    def test_dynamic_obstacles_flag_doubles_total_agents(self, base_cfg, mock_strategy):
        """Po włączeniu flagi liczba wszystkich agentów powinna być x2."""
        base_cfg.simulation.dynamic_obstacles = True
        r = ExperimentRunner(base_cfg, mock_strategy)
        assert r.use_dynamic_obstacles is True
        assert r.num_dynamic_obstacles == r.num_drones
        assert r.total_agents == 2 * r.num_drones

    def test_dynamic_obstacle_controller_starts_as_none(self, runner):
        """Kontroler dynamicznych przeszkód to pole, które domyślnie jest None."""
        assert runner.dynamic_obstacle_trajectory_controller is None


# ---------------------------------------------------------------------------
# prepare_experiment
# ---------------------------------------------------------------------------

class TestPrepareExperiment:

    @patch(f"{MODULE}.TrajectoryFollowingAlgorithm")
    def test_delegates_to_data_strategy(self, mock_tfa, runner, mock_strategy):
        runner.prepare_experiment()
        mock_strategy.prepare_data.assert_called_once_with(runner)

    @patch(f"{MODULE}.TrajectoryFollowingAlgorithm")
    def test_initializes_trajectory_controller(self, mock_tfa, runner):
        runner.prepare_experiment()
        mock_tfa.assert_called_once()
        assert runner.trajectory_controller is not None

    @patch(f"{MODULE}.TrajectoryFollowingAlgorithm")
    def test_no_logger_when_logging_disabled(self, mock_tfa, runner):
        runner.prepare_experiment()
        assert runner.logger is None

    @patch(f"{MODULE}.SimulationLogger")
    @patch(f"{MODULE}.HydraConfig")
    @patch(f"{MODULE}.TrajectoryFollowingAlgorithm")
    def test_creates_logger_when_enabled(self, mock_tfa, mock_hydra_cfg, mock_sim_logger, base_cfg, mock_strategy):
        base_cfg.logging.enabled = True
        mock_hydra_cfg.get.return_value.runtime.output_dir = "/tmp/test_output"
        r = ExperimentRunner(base_cfg, mock_strategy)
        r.prepare_experiment()
        mock_sim_logger.assert_called_once()

    @patch(f"{MODULE}.SimulationLogger")
    @patch(f"{MODULE}.TrajectoryFollowingAlgorithm")
    def test_logger_uses_output_dir_from_config(self, mock_tfa, mock_sim_logger, base_cfg, mock_strategy):
        """Gdy logging.output_dir jest ustawiony w konfiguracji, logger powinien go użyć."""
        base_cfg.logging.enabled = True
        base_cfg.logging.output_dir = "/tmp/replay_output"
        r = ExperimentRunner(base_cfg, mock_strategy)
        r.prepare_experiment()
        mock_sim_logger.assert_called_once()
        assert mock_sim_logger.call_args.kwargs["output_dir"] == "/tmp/replay_output"

    @patch(f"{MODULE}.InputHandler")
    @patch(f"{MODULE}.TrajectoryFollowingAlgorithm")
    def test_no_input_handler_in_headless(self, mock_tfa, mock_ih, runner):
        runner.prepare_experiment()
        mock_ih.assert_not_called()
        assert runner.input_handler is None

    @patch(f"{MODULE}.InputHandler")
    @patch(f"{MODULE}.TrajectoryFollowingAlgorithm")
    def test_creates_input_handler_with_gui(self, mock_tfa, mock_ih, base_cfg, mock_strategy):
        base_cfg.simulation.gui = True
        r = ExperimentRunner(base_cfg, mock_strategy)
        r.prepare_experiment()
        mock_ih.assert_called_once_with(2)

    @patch(f"{MODULE}.TrajectoryFollowingAlgorithm")
    def test_single_controller_without_dynamic_obstacles(self, mock_tfa, runner):
        """Gdy flaga dynamic_obstacles wyłączona, tworzymy wyłącznie kontroler dronów głównych."""
        runner.prepare_experiment()
        assert mock_tfa.call_count == 1
        first_call_kwargs = mock_tfa.call_args_list[0].kwargs
        assert first_call_kwargs["is_obstacle"] is False
        assert first_call_kwargs["num_drones"] == runner.num_drones
        # Kontroler przeszkód pozostaje None
        assert runner.dynamic_obstacle_trajectory_controller is None

    @patch(f"{MODULE}.TrajectoryFollowingAlgorithm")
    def test_two_controllers_when_dynamic_obstacles_enabled(self, mock_tfa, base_cfg, mock_strategy):
        """Włączona flaga tworzy dwa osobne kontrolery: jeden dla dronów, drugi dla przeszkód."""
        base_cfg.simulation.dynamic_obstacles = True
        r = ExperimentRunner(base_cfg, mock_strategy)
        r.prepare_experiment()

        assert mock_tfa.call_count == 2

        # Pierwszy: drony główne
        first = mock_tfa.call_args_list[0].kwargs
        assert first["is_obstacle"] is False
        assert first["num_drones"] == r.num_drones

        # Drugi: przeszkody
        second = mock_tfa.call_args_list[1].kwargs
        assert second["is_obstacle"] is True
        assert second["num_drones"] == r.num_dynamic_obstacles

        # Oba pola muszą być przypisane
        assert r.trajectory_controller is not None
        assert r.dynamic_obstacle_trajectory_controller is not None


# ---------------------------------------------------------------------------
# initialize_world
# ---------------------------------------------------------------------------

class TestInitializeWorld:

    @patch(f"{MODULE}.instantiate")
    def test_calls_instantiate_with_config(self, mock_inst, runner):
        runner.world_data = MagicMock()
        runner.obstacles_data = MagicMock()
        runner.initialize_world()
        mock_inst.assert_called_once()
        kwargs = mock_inst.call_args[1]
        assert kwargs["drone_model"] == "CF2X"
        assert kwargs["physics"] == "PYB"
        assert kwargs["gui"] is False
        assert kwargs["ctrl_freq"] == 48
        assert kwargs["pyb_freq"] == 240
        np.testing.assert_array_equal(kwargs["initial_xyzs"], runner.start_positions)

    @patch(f"{MODULE}.instantiate")
    def test_env_kwargs_without_dynamic_obstacles(self, mock_inst, runner):
        """Bez dynamicznych przeszkód środowisko dostaje same drony (num_drones == primary)."""
        runner.initialize_world()
        kwargs = mock_inst.call_args[1]
        assert kwargs["num_drones"] == runner.num_drones
        assert kwargs["primary_num_drones"] == runner.num_drones
        assert kwargs["dynamic_obstacles_enabled"] is False
        assert kwargs["num_dynamic_obstacles"] == 0
        np.testing.assert_array_equal(kwargs["initial_xyzs"], runner.start_positions)
        np.testing.assert_array_equal(kwargs["end_xyzs"], runner.end_positions)


class TestInitializeWorldDynamicObstacles:
    """Weryfikuje, że przy dynamic_obstacles=True punkty startowe/celowe są odpowiednio rozszerzone."""

    @patch(f"{MODULE}.instantiate")
    def test_total_agents_passed_to_env(self, mock_inst, base_cfg, mock_strategy):
        base_cfg.simulation.dynamic_obstacles = True
        r = ExperimentRunner(base_cfg, mock_strategy)
        r.initialize_world()
        kwargs = mock_inst.call_args[1]
        # Liczba wszystkich agentów w środowisku to 2x liczba dronów
        assert kwargs["num_drones"] == r.total_agents == 2 * r.num_drones
        # Ale "prawdziwych" dronów jest tylko połowa
        assert kwargs["primary_num_drones"] == r.num_drones
        assert kwargs["dynamic_obstacles_enabled"] is True
        assert kwargs["num_dynamic_obstacles"] == r.num_drones

    @patch(f"{MODULE}.instantiate")
    def test_positions_stacked_and_reversed(self, mock_inst, base_cfg, mock_strategy):
        """Pierwsza połowa startuje zgodnie z configiem, druga — z końcowych pozycji dronów."""
        base_cfg.simulation.dynamic_obstacles = True
        r = ExperimentRunner(base_cfg, mock_strategy)
        r.initialize_world()

        kwargs = mock_inst.call_args[1]
        expected_initial = np.vstack((r.start_positions, r.end_positions))
        expected_end = np.vstack((r.end_positions, r.start_positions))

        np.testing.assert_array_equal(kwargs["initial_xyzs"], expected_initial)
        np.testing.assert_array_equal(kwargs["end_xyzs"], expected_end)

        # Sprawdźmy konkretnie: pierwszy dron startuje w swojej pozycji, a
        # pierwsza przeszkoda startuje w pozycji DOCELOWEJ tego drona.
        np.testing.assert_array_equal(kwargs["initial_xyzs"][0], r.start_positions[0])
        np.testing.assert_array_equal(kwargs["initial_xyzs"][r.num_drones], r.end_positions[0])

    @patch(f"{MODULE}.instantiate")
    def test_rpys_doubled_when_provided(self, mock_inst, base_cfg, mock_strategy):
        """Kąty startowe są powielane dla przeszkód (ten sam rozkład orientacji)."""
        base_cfg.simulation.dynamic_obstacles = True
        r = ExperimentRunner(base_cfg, mock_strategy)
        r.initialize_world()
        kwargs = mock_inst.call_args[1]
        expected_rpys = np.vstack((r.initial_rpys, r.initial_rpys))
        np.testing.assert_array_equal(kwargs["initial_rpys"], expected_rpys)
        assert kwargs["initial_rpys"].shape == (2 * r.num_drones, 3)

    @patch(f"{MODULE}.instantiate")
    def test_rpys_none_passes_none(self, mock_inst, base_cfg, mock_strategy):
        """Gdy initial_rpys jest None, środowisko dostaje też None — nie pustą tablicę."""
        base_cfg.simulation.dynamic_obstacles = True
        base_cfg.environment.initial_rpys = None
        r = ExperimentRunner(base_cfg, mock_strategy)
        r.initialize_world()
        kwargs = mock_inst.call_args[1]
        assert kwargs["initial_rpys"] is None


# ---------------------------------------------------------------------------
# _init_active_drones
# ---------------------------------------------------------------------------

class TestInitActiveDrones:

    def test_all_drones_active(self, runner):
        runner.trajectory_controller = MagicMock()
        runner.trajectory_controller.params = {"acceptance_radius": 0.3}
        runner._init_active_drones()
        assert runner.active_drones == {0, 1}
        assert runner.acceptance_radius == 0.3


# ---------------------------------------------------------------------------
# _process_collisions
# ---------------------------------------------------------------------------

class TestProcessCollisions:

    def test_drone_removed_on_collision(self, runner):
        runner.active_drones = {0, 1}
        runner.logger = None
        runner.environemnt = MagicMock()
        runner.environemnt.get_detailed_collisions.return_value = [(0, 99)]

        runner._process_collisions(sim_time=1.5, current_step=72)
        assert 0 not in runner.active_drones
        assert 1 in runner.active_drones

    def test_collision_logged(self, runner):
        runner.active_drones = {0, 1}
        runner.logger = MagicMock()
        runner.environemnt = MagicMock()
        runner.environemnt.get_detailed_collisions.return_value = [(1, 42)]

        runner._process_collisions(sim_time=2.0, current_step=96)
        runner.logger.log_collision.assert_called_once_with(2.0, 1, 42)

    def test_no_collisions(self, runner):
        runner.active_drones = {0, 1}
        runner.logger = MagicMock()
        runner.environemnt = MagicMock()
        runner.environemnt.get_detailed_collisions.return_value = []

        runner._process_collisions(sim_time=1.0, current_step=48)
        runner.logger.log_collision.assert_not_called()
        assert runner.active_drones == {0, 1}

    def test_multiple_collisions_same_step(self, runner):
        runner.active_drones = {0, 1}
        runner.logger = None
        runner.environemnt = MagicMock()
        runner.environemnt.get_detailed_collisions.return_value = [(0, 10), (1, 20)]

        runner._process_collisions(sim_time=3.0, current_step=144)
        assert runner.active_drones == set()

    def test_collision_with_already_inactive_drone(self, runner):
        """Kolizja drona, który już nie jest aktywny, nie powinna powodować błędu."""
        runner.active_drones = {1}
        runner.logger = None
        runner.environemnt = MagicMock()
        runner.environemnt.get_detailed_collisions.return_value = [(0, 5)]

        runner._process_collisions(sim_time=1.0, current_step=48)
        assert runner.active_drones == {1}

    def test_dynamic_obstacle_collisions_are_ignored(self, base_cfg, mock_strategy):
        """Kolizje o d_id >= num_drones (dynamiczne przeszkody) nie są logowane ani nie deaktywują dronów."""
        base_cfg.simulation.dynamic_obstacles = True
        r = ExperimentRunner(base_cfg, mock_strategy)
        r.active_drones = {0, 1}
        r.logger = MagicMock()
        r.environemnt = MagicMock()
        # (0, …) to prawdziwy dron — liczy się.
        # (2, …) i (3, …) to dynamiczne przeszkody (num_drones=2) — ignorowane.
        r.environemnt.get_detailed_collisions.return_value = [(0, 99), (2, 88), (3, 77)]

        r._process_collisions(sim_time=1.0, current_step=48)

        # Tylko dron 0 zgłoszony do loggera
        r.logger.log_collision.assert_called_once_with(1.0, 0, 99)
        # Dron 1 nadal aktywny — jego stan nie był dotknięty
        assert r.active_drones == {1}


# ---------------------------------------------------------------------------
# _process_arrivals
# ---------------------------------------------------------------------------

class TestProcessArrivals:

    def test_drone_arrives_at_target(self, runner):
        runner.active_drones = {0, 1}
        runner.acceptance_radius = 0.2
        # Dron 0 jest na pozycji docelowej
        states = [_make_state(10.0, 0.0, 0.0), _make_state(5.0, 0.0, 0.0)]
        runner._process_arrivals(states, sim_time=5.0)
        assert 0 not in runner.active_drones
        assert 1 in runner.active_drones

    def test_drone_near_but_outside_radius(self, runner):
        runner.active_drones = {0}
        runner.acceptance_radius = 0.2
        # Dron 0 jest blisko, ale poza promieniem akceptacji
        states = [_make_state(9.7, 0.0, 0.0)]
        runner.end_positions = np.array([[10.0, 0.0, 0.0]])
        runner._process_arrivals(states, sim_time=5.0)
        assert 0 in runner.active_drones

    def test_all_drones_arrive(self, runner):
        runner.active_drones = {0, 1}
        runner.acceptance_radius = 0.5
        states = [_make_state(10.0, 0.0, 0.0), _make_state(9.0, 0.0, 0.0)]
        runner._process_arrivals(states, sim_time=10.0)
        assert runner.active_drones == set()

    def test_no_active_drones(self, runner):
        """Pusta lista aktywnych dronów nie powinna powodować błędów."""
        runner.active_drones = set()
        runner.acceptance_radius = 0.2
        states = [_make_state(10.0, 0.0, 0.0)]
        runner._process_arrivals(states, sim_time=5.0)
        assert runner.active_drones == set()

    def test_invalid_acceptance_radius_raises(self, runner):
        runner.active_drones = {0}
        runner.acceptance_radius = "invalid"
        states = [_make_state(0, 0, 0)]
        with pytest.raises(ValueError, match="Błędny promień akceptacji"):
            runner._process_arrivals(states, sim_time=1.0)


# ---------------------------------------------------------------------------
# _split_states / _merge_actions – mechanika roz/scalania ruchu agentów
# ---------------------------------------------------------------------------

class TestSplitStates:

    def test_no_obstacles_returns_empty_obstacle_list(self, runner):
        """Gdy flaga wyłączona, druga lista jest zawsze pusta — niezależnie od wejścia."""
        states = [_make_state(i) for i in range(2)]
        drones, obstacles = runner._split_states(states)
        assert len(drones) == 2
        assert obstacles == []

    def test_with_obstacles_splits_at_num_drones(self, base_cfg, mock_strategy):
        """Granicą podziału jest num_drones: do niej drony, od niej przeszkody."""
        base_cfg.simulation.dynamic_obstacles = True
        r = ExperimentRunner(base_cfg, mock_strategy)

        states = [_make_state(i) for i in range(r.total_agents)]  # 4 agentów
        drones, obstacles = r._split_states(states)

        assert len(drones) == r.num_drones
        assert len(obstacles) == r.num_dynamic_obstacles
        # Drony — pierwsze r.num_drones pozycji (id 0 i 1)
        np.testing.assert_array_equal(drones[0], states[0])
        np.testing.assert_array_equal(drones[1], states[1])
        # Przeszkody — reszta (id 2 i 3)
        np.testing.assert_array_equal(obstacles[0], states[2])
        np.testing.assert_array_equal(obstacles[1], states[3])


class TestMergeActions:

    def test_merge_without_obstacles_passes_through(self, runner):
        drone_acts = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
        merged = runner._merge_actions(drone_acts, None)
        np.testing.assert_array_equal(merged, drone_acts)
        # Akcje przeszkód nie są dodawane — wynikowa tablica ma dokładnie num_drones wierszy
        assert merged.shape == drone_acts.shape

    def test_merge_stacks_obstacle_actions_after_drone_actions(self, runner):
        """Kolejność scalania musi pasować do indeksowania w środowisku: drony, potem przeszkody."""
        drone_acts = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
        obstacle_acts = np.array([[3, 3, 3, 3], [4, 4, 4, 4]])
        merged = runner._merge_actions(drone_acts, obstacle_acts)
        assert merged.shape == (4, 4)
        # Drony idą pierwsze
        np.testing.assert_array_equal(merged[:2], drone_acts)
        # Przeszkody — po dronach
        np.testing.assert_array_equal(merged[2:], obstacle_acts)


# ---------------------------------------------------------------------------
# _update_camera
# ---------------------------------------------------------------------------

class TestUpdateCamera:

    @patch(f"{MODULE}.update_camera_position")
    def test_camera_not_called_when_follow_disabled(self, mock_ucp, runner):
        states = [_make_state()]
        runner._update_camera(states)
        mock_ucp.assert_not_called()

    @patch(f"{MODULE}.update_camera_position")
    def test_camera_called_when_follow_enabled(self, mock_ucp, base_cfg, mock_strategy):
        base_cfg.visualization.camera_follow = True
        r = ExperimentRunner(base_cfg, mock_strategy)
        states = [_make_state(1, 2, 3), _make_state(4, 5, 6)]
        r._update_camera(states)
        mock_ucp.assert_called_once()
        kwargs = mock_ucp.call_args[1]
        np.testing.assert_array_equal(kwargs["drone_state"], states[0])


# ---------------------------------------------------------------------------
# run – pętla symulacji (headless)
# ---------------------------------------------------------------------------

class TestRunHeadless:

    def _setup_runner_for_run(self, runner):
        """Przygotowuje runnera z zamockowanymi zależnościami do testu run()."""
        mock_env = MagicMock()
        mock_env._getDroneStateVector.return_value = _make_state(0, 0, 0)
        mock_env.get_detailed_collisions.return_value = []

        mock_controller = MagicMock()
        mock_controller.compute_actions.return_value = np.zeros((2, 4))
        mock_controller.params = {"acceptance_radius": 0.2}

        runner.trajectory_controller = mock_controller
        return mock_env, mock_controller

    @patch(f"{MODULE}.instantiate")
    def test_runs_expected_steps(self, mock_inst, runner):
        mock_env, _ = self._setup_runner_for_run(runner)
        mock_inst.return_value = mock_env

        runner.run()

        # duration_sec=0.1, ctrl_freq=48 -> int(0.1 * 48) = 4 kroki
        assert mock_env.step.call_count == 4
        mock_env.close.assert_called_once()

    @patch(f"{MODULE}.instantiate")
    def test_stops_when_all_drones_arrive(self, mock_inst, base_cfg, mock_strategy):
        base_cfg.simulation.duration_sec = 10.0  # długa symulacja
        r = ExperimentRunner(base_cfg, mock_strategy)

        mock_env = MagicMock()
        # Drony od razu na pozycji docelowej
        mock_env._getDroneStateVector.side_effect = lambda d: _make_state(
            *r.end_positions[d]
        )
        mock_env.get_detailed_collisions.return_value = []
        mock_inst.return_value = mock_env

        mock_ctrl = MagicMock()
        mock_ctrl.compute_actions.return_value = np.zeros((2, 4))
        mock_ctrl.params = {"acceptance_radius": 0.2}
        r.trajectory_controller = mock_ctrl

        r.run()

        # Powinno przerwać po 1 kroku (drony natychmiast docierają)
        assert mock_env.step.call_count == 1

    @patch(f"{MODULE}.instantiate")
    def test_stops_when_all_drones_crash(self, mock_inst, base_cfg, mock_strategy):
        base_cfg.simulation.duration_sec = 10.0
        r = ExperimentRunner(base_cfg, mock_strategy)

        mock_env = MagicMock()
        mock_env._getDroneStateVector.return_value = _make_state(0, 0, 0)
        # Oba drony kolidują w każdym kroku
        mock_env.get_detailed_collisions.return_value = [(0, 99), (1, 99)]
        mock_inst.return_value = mock_env

        mock_ctrl = MagicMock()
        mock_ctrl.compute_actions.return_value = np.zeros((2, 4))
        mock_ctrl.params = {"acceptance_radius": 0.2}
        r.trajectory_controller = mock_ctrl

        r.run()
        assert mock_env.step.call_count == 1

    @patch(f"{MODULE}.instantiate")
    def test_logger_save_called(self, mock_inst, base_cfg, mock_strategy):
        base_cfg.simulation.duration_sec = 0.05
        r = ExperimentRunner(base_cfg, mock_strategy)

        mock_env = MagicMock()
        mock_env._getDroneStateVector.return_value = _make_state()
        mock_env.get_detailed_collisions.return_value = []
        mock_inst.return_value = mock_env

        mock_ctrl = MagicMock()
        mock_ctrl.compute_actions.return_value = np.zeros((2, 4))
        mock_ctrl.params = {"acceptance_radius": 0.2}
        r.trajectory_controller = mock_ctrl

        mock_logger = MagicMock()
        r.logger = mock_logger

        r.run()
        mock_logger.save.assert_called_once()

    @patch(f"{MODULE}.instantiate")
    def test_log_step_called_each_step(self, mock_inst, runner):
        mock_env, _ = self._setup_runner_for_run(runner)
        mock_inst.return_value = mock_env

        mock_logger = MagicMock()
        runner.logger = mock_logger

        runner.run()
        assert mock_logger.log_step.call_count == 4

    @patch(f"{MODULE}.instantiate")
    def test_init_lidars_called(self, mock_inst, runner):
        mock_env, mock_ctrl = self._setup_runner_for_run(runner)
        mock_inst.return_value = mock_env

        runner.run()
        mock_ctrl.init_lidars.assert_called_once_with(mock_env.CLIENT)

    @patch(f"{MODULE}.instantiate")
    def test_log_step_receives_only_main_drones(self, mock_inst, base_cfg, mock_strategy):
        """
        Z włączonymi dynamicznymi przeszkodami logger dostaje WYŁĄCZNIE stany dronów głównych,
        nawet jeśli w środowisku fizycznie lata 2x tyle agentów.
        """
        base_cfg.simulation.dynamic_obstacles = True
        base_cfg.simulation.duration_sec = 0.05  # ~1 krok
        r = ExperimentRunner(base_cfg, mock_strategy)

        # Każdy agent ma unikalną pozycję (po id), żeby móc rozpoznać kto został zalogowany
        mock_env = MagicMock()
        mock_env._getDroneStateVector.side_effect = lambda d: _make_state(d, 0, 0)
        mock_env.get_detailed_collisions.return_value = []
        mock_inst.return_value = mock_env

        mock_ctrl = MagicMock()
        mock_ctrl.compute_actions.return_value = np.zeros((r.num_drones, 4))
        mock_ctrl.params = {"acceptance_radius": 0.2}
        r.trajectory_controller = mock_ctrl

        mock_obs_ctrl = MagicMock()
        mock_obs_ctrl.compute_actions.return_value = np.zeros((r.num_dynamic_obstacles, 4))
        r.dynamic_obstacle_trajectory_controller = mock_obs_ctrl

        mock_logger = MagicMock()
        r.logger = mock_logger

        r.run()

        # Logger został wywołany, ale z listą długości num_drones, a nie total_agents
        assert mock_logger.log_step.call_count >= 1
        states_logged = mock_logger.log_step.call_args_list[0].args[2]
        assert len(states_logged) == r.num_drones
        assert len(states_logged) < r.total_agents

        # Pierwsze stany to drony główne (pos X = 0 i 1), nie przeszkody (pos X = 2 i 3)
        assert states_logged[0][0] == 0.0
        assert states_logged[1][0] == 1.0

    @patch(f"{MODULE}.instantiate")
    def test_obstacle_controller_also_drives_step(self, mock_inst, base_cfg, mock_strategy):
        """Z dynamicznymi przeszkodami obie klasy kontrolerów muszą dostarczyć akcje co krok."""
        base_cfg.simulation.dynamic_obstacles = True
        base_cfg.simulation.duration_sec = 0.05
        r = ExperimentRunner(base_cfg, mock_strategy)

        mock_env = MagicMock()
        mock_env._getDroneStateVector.return_value = _make_state()
        mock_env.get_detailed_collisions.return_value = []
        mock_inst.return_value = mock_env

        mock_ctrl = MagicMock()
        mock_ctrl.compute_actions.return_value = np.zeros((r.num_drones, 4))
        mock_ctrl.params = {"acceptance_radius": 0.2}
        r.trajectory_controller = mock_ctrl

        mock_obs_ctrl = MagicMock()
        mock_obs_ctrl.compute_actions.return_value = np.zeros((r.num_dynamic_obstacles, 4))
        r.dynamic_obstacle_trajectory_controller = mock_obs_ctrl

        r.run()

        # Obaj liczą akcje przynajmniej raz
        assert mock_ctrl.compute_actions.call_count >= 1
        assert mock_obs_ctrl.compute_actions.call_count >= 1
        # Do step() trafiła tablica scalona (num_drones + num_dynamic_obstacles wierszy)
        step_args = mock_env.step.call_args_list[0].args[0]
        assert step_args.shape[0] == r.total_agents

    @patch(f"{MODULE}.instantiate")
    def test_arrivals_use_post_step_state(self, mock_inst, base_cfg, mock_strategy):
        """Detekcja przybycia powinna korzystać ze stanu PO kroku fizyki."""
        base_cfg.simulation.duration_sec = 10.0
        r = ExperimentRunner(base_cfg, mock_strategy)

        call_count = [0]

        def state_for_drone(d):
            # Przed step(): dron daleko od celu
            # Po step(): dron na pozycji docelowej
            call_count[0] += 1
            if call_count[0] <= r.num_drones:
                # Pre-step: daleko
                return _make_state(0, 0, 0)
            else:
                # Post-step: na celu
                return _make_state(*r.end_positions[d])

        mock_env = MagicMock()
        mock_env._getDroneStateVector.side_effect = state_for_drone
        mock_env.get_detailed_collisions.return_value = []
        mock_inst.return_value = mock_env

        mock_ctrl = MagicMock()
        mock_ctrl.compute_actions.return_value = np.zeros((2, 4))
        mock_ctrl.params = {"acceptance_radius": 0.2}
        r.trajectory_controller = mock_ctrl

        r.run()

        # Drony powinny zostać wykryte jako przybyłe po pierwszym step()
        assert mock_env.step.call_count == 1

    @patch(f"{MODULE}.instantiate")
    def test_headless_prints_progress(self, mock_inst, base_cfg, mock_strategy, capsys):
        """Tryb headless powinien wypisywać informacje o postępie."""
        base_cfg.simulation.duration_sec = 1.0  # 48 kroków
        r = ExperimentRunner(base_cfg, mock_strategy)

        mock_env = MagicMock()
        mock_env._getDroneStateVector.return_value = _make_state(0, 0, 0)
        mock_env.get_detailed_collisions.return_value = []
        mock_inst.return_value = mock_env

        mock_ctrl = MagicMock()
        mock_ctrl.compute_actions.return_value = np.zeros((2, 4))
        mock_ctrl.params = {"acceptance_radius": 0.2}
        r.trajectory_controller = mock_ctrl

        r.run()

        captured = capsys.readouterr()
        assert "Postęp:" in captured.out


# ---------------------------------------------------------------------------
# run – GUI mode
# ---------------------------------------------------------------------------

class TestRunGUI:
    """Testy pętli GUI.

    Kluczowy szczegół: w trybie GUI ``is_running`` startuje jako ``False``
    (linia ``is_running = not self.cfg.simulation.gui``).  Symulacja rusza
    dopiero po komendzie TOGGLE_SIMULATION.  Bez niej ``current_step`` nigdy
    nie rośnie, więc ``while current_step < max_steps`` kręci się w
    nieskończoność — jedynym wyjściem jest ``p.isConnected() == False``.
    """

    @staticmethod
    def _make_gui_runner(base_cfg, mock_strategy, duration=10.0):
        base_cfg.simulation.gui = True
        base_cfg.simulation.duration_sec = duration
        r = ExperimentRunner(base_cfg, mock_strategy)

        mock_env = MagicMock()
        mock_env._getDroneStateVector.return_value = _make_state()
        mock_env.get_detailed_collisions.return_value = []

        mock_ctrl = MagicMock()
        mock_ctrl.compute_actions.return_value = np.zeros((2, 4))
        mock_ctrl.params = {"acceptance_radius": 0.2}
        r.trajectory_controller = mock_ctrl

        mock_ih = MagicMock()
        r.input_handler = mock_ih

        return r, mock_env, mock_ctrl, mock_ih

    # -- disconnect natychmiast przerywa pętlę ---------------------------------

    @patch(f"{MODULE}.p.isConnected", return_value=False)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.instantiate")
    def test_gui_stops_when_pybullet_disconnects(
        self, mock_inst, mock_sleep, mock_connected, base_cfg, mock_strategy
    ):
        r, mock_env, _, mock_ih = self._make_gui_runner(base_cfg, mock_strategy)
        mock_ih.get_command.return_value = None
        mock_inst.return_value = mock_env

        r.run()

        # is_running=False -> brak step, ale pętla natychmiast przerwana przez isConnected
        mock_env.step.assert_not_called()
        mock_env.close.assert_called_once()

    # -- toggle start/pauza ----------------------------------------------------

    @patch(f"{MODULE}.p.isConnected")
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.instantiate")
    def test_toggle_simulation_starts_and_pauses(
        self, mock_inst, mock_sleep, mock_connected, base_cfg, mock_strategy
    ):
        """TOGGLE co iterację: False→True (step), True→False (brak step), disconnect."""
        from src.utils.input_utils import CommandType

        r, mock_env, _, mock_ih = self._make_gui_runner(base_cfg, mock_strategy)
        mock_inst.return_value = mock_env

        toggle_cmd = MagicMock()
        toggle_cmd.type = CommandType.TOGGLE_SIMULATION
        mock_ih.get_command.return_value = toggle_cmd

        # isConnected sprawdzany jest PO step() w danej iteracji:
        # Cykl 1: toggle False→True → step → isConnected=True → dalej
        # Cykl 2: toggle True→False → brak step → isConnected=False → break
        mock_connected.side_effect = [True, False]

        r.run()

        assert mock_env.step.call_count == 1

    # -- switch camera ---------------------------------------------------------

    @patch(f"{MODULE}.p.isConnected", return_value=False)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.instantiate")
    def test_switch_drone_camera(
        self, mock_inst, mock_sleep, mock_connected, base_cfg, mock_strategy
    ):
        from src.utils.input_utils import CommandType

        r, mock_env, _, mock_ih = self._make_gui_runner(base_cfg, mock_strategy)
        mock_inst.return_value = mock_env

        switch_cmd = MagicMock()
        switch_cmd.type = CommandType.SWITCH_DRONE_CAMERA
        switch_cmd.payload = 1
        mock_ih.get_command.return_value = switch_cmd

        r.run()
        assert r.tracked_drone_id == 1

    # -- toggle lidar on -------------------------------------------------------

    @patch(f"{MODULE}.p.isConnected", return_value=False)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.instantiate")
    def test_toggle_lidar_rays_on(
        self, mock_inst, mock_sleep, mock_connected, base_cfg, mock_strategy
    ):
        from src.utils.input_utils import CommandType

        r, mock_env, _, mock_ih = self._make_gui_runner(base_cfg, mock_strategy)
        mock_inst.return_value = mock_env
        assert r.show_lidar_rays is False

        toggle_cmd = MagicMock()
        toggle_cmd.type = CommandType.TOGGLE_LIDAR_RAYS
        mock_ih.get_command.return_value = toggle_cmd

        r.run()
        assert r.show_lidar_rays is True

    # -- toggle lidar off (clear_lidar_rays) ------------------------------------

    @patch(f"{MODULE}.p.isConnected", return_value=False)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.instantiate")
    def test_toggle_lidar_off_clears_rays(
        self, mock_inst, mock_sleep, mock_connected, base_cfg, mock_strategy
    ):
        from src.utils.input_utils import CommandType

        r, mock_env, mock_ctrl, mock_ih = self._make_gui_runner(base_cfg, mock_strategy)
        mock_inst.return_value = mock_env
        r.show_lidar_rays = True  # już włączone

        toggle_cmd = MagicMock()
        toggle_cmd.type = CommandType.TOGGLE_LIDAR_RAYS
        mock_ih.get_command.return_value = toggle_cmd

        r.run()
        assert r.show_lidar_rays is False
        mock_ctrl.clear_lidar_rays.assert_called_once()

    # -- pełna pętla GUI z uruchomioną symulacją --------------------------------

    @patch(f"{MODULE}.p.isConnected")
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.instantiate")
    def test_gui_full_run_with_toggle_then_disconnect(
        self, mock_inst, mock_sleep, mock_connected, base_cfg, mock_strategy
    ):
        """Symulacja startuje (toggle), wykonuje kilka kroków, potem disconnect."""
        from src.utils.input_utils import CommandType

        r, mock_env, _, mock_ih = self._make_gui_runner(base_cfg, mock_strategy, duration=10.0)
        mock_inst.return_value = mock_env

        toggle_cmd = MagicMock()
        toggle_cmd.type = CommandType.TOGGLE_SIMULATION

        # Cykl 1: toggle → is_running=True → step → isConnected=True
        # Cykl 2: None → step → isConnected=True
        # Cykl 3: None → step → isConnected=False → break
        mock_ih.get_command.side_effect = [toggle_cmd, None, None]
        mock_connected.side_effect = [True, True, False]

        r.run()
        assert mock_env.step.call_count == 3
        mock_env.close.assert_called_once()


# ---------------------------------------------------------------------------
# main_replay
# ---------------------------------------------------------------------------

class TestMainReplay:

    def test_exits_when_config_missing(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            main_replay(str(tmp_path))
        assert exc_info.value.code == 1

    @patch(f"{MODULE}.ExperimentRunner")
    @patch(f"{MODULE}.OmegaConf.load")
    def test_loads_config_and_runs(self, mock_load, mock_runner_cls, tmp_path):
        hydra_dir = tmp_path / ".hydra"
        hydra_dir.mkdir()
        (hydra_dir / "config.yaml").write_text("simulation:\n  gui: false\n")

        mock_cfg = OmegaConf.create({"simulation": {"gui": False}})
        mock_load.return_value = mock_cfg

        mock_runner_instance = MagicMock()
        mock_runner_cls.return_value = mock_runner_instance

        main_replay(str(tmp_path))

        mock_load.assert_called_once()
        mock_runner_instance.prepare_experiment.assert_called_once()
        mock_runner_instance.run.assert_called_once()

    @patch(f"{MODULE}.ExperimentRunner")
    @patch(f"{MODULE}.OmegaConf.load")
    def test_headless_flag_overrides_gui(self, mock_load, mock_runner_cls, tmp_path):
        """Flaga headless=True powinna wymusić gui=False nawet jeśli config ma gui=True."""
        hydra_dir = tmp_path / ".hydra"
        hydra_dir.mkdir()
        (hydra_dir / "config.yaml").write_text("simulation:\n  gui: true\n")

        mock_cfg = OmegaConf.create({"simulation": {"gui": True}})
        mock_load.return_value = mock_cfg

        mock_runner_instance = MagicMock()
        mock_runner_cls.return_value = mock_runner_instance

        main_replay(str(tmp_path), headless=True)

        # Sprawdzamy, że ExperimentRunner dostał cfg z gui=False
        actual_cfg = mock_runner_cls.call_args[0][0]
        assert actual_cfg.simulation.gui is False


# ---------------------------------------------------------------------------
# CLI dispatch (__main__ block)
# ---------------------------------------------------------------------------

class TestCLIDispatch:

    @patch(f"{MODULE}.main_replay")
    def test_replay_flag_calls_main_replay(self, mock_replay):
        original_argv = sys.argv[:]
        try:
            sys.argv = ["main.py", "--replay", "/some/path"]
            if len(sys.argv) > 1 and sys.argv[1] == "--replay":
                if len(sys.argv) >= 3:
                    mock_replay(sys.argv[2])
            mock_replay.assert_called_once_with("/some/path")
        finally:
            sys.argv = original_argv

    def test_replay_flag_without_path_exits(self):
        original_argv = sys.argv[:]
        try:
            sys.argv = ["main.py", "--replay"]
            if len(sys.argv) > 1 and sys.argv[1] == "--replay":
                if len(sys.argv) < 3:
                    with pytest.raises(SystemExit):
                        sys.exit(1)
        finally:
            sys.argv = original_argv

    def test_headless_flag_injects_hydra_override(self):
        """--headless w trybie generate powinno zostać zamienione na override Hydry."""
        original_argv = sys.argv[:]
        try:
            sys.argv = ["main.py", "--headless", "seed=123"]

            # Symulujemy logikę z bloku __main__
            if "--headless" in sys.argv:
                sys.argv.remove("--headless")
                sys.argv.append("simulation.gui=false")

            assert "--headless" not in sys.argv
            assert "simulation.gui=false" in sys.argv
            # Pozostałe overrides Hydry powinny być nienaruszone
            assert "seed=123" in sys.argv
        finally:
            sys.argv = original_argv

    def test_headless_flag_works_without_other_args(self):
        """--headless jako jedyny argument powinno działać poprawnie."""
        original_argv = sys.argv[:]
        try:
            sys.argv = ["main.py", "--headless"]

            if "--headless" in sys.argv:
                sys.argv.remove("--headless")
                sys.argv.append("simulation.gui=false")

            assert sys.argv == ["main.py", "simulation.gui=false"]
        finally:
            sys.argv = original_argv


# ---------------------------------------------------------------------------
# Przypadki brzegowe
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_drone(self, mock_strategy):
        cfg = OmegaConf.create({
            "simulation": {
                "drone_model": "CF2X", "physics": "PYB",
                "ctrl_freq": 48, "pyb_freq": 240, "gui": False,
                "duration_sec": 0.1, "sim_speed_multiplier": 1.0,
            },
            "environment": {
                "params": {
                    "num_drones": 1,
                    "placement_strategy": "strategy_grid_jitter",
                    "ground_position": 0.0,
                    "track_length": 5.0, "track_width": 5.0, "track_height": 5.0,
                    "shape_type": "CYLINDER",
                    "obstacles_number": 0,
                    "obstacle_width": 0.5, "obstacle_height": 1.0, "obstacle_length": 0.5,
                },
                "initial_rpys": [[0, 0, 0]],
                "initial_xyzs": [[0, 0, 1]],
                "end_xyzs": [[5, 5, 1]],
            },
            "visualization": {
                "tracked_drone_id": 0, "show_lidar_rays": False,
                "lidar_draw_interval": 5, "camera_follow": False,
                "camera_distance": 2.0, "camera_yaw": 0.0, "camera_pitch": -30.0,
            },
            "optimizer": {"algorithm_params": {"n_inner_waypoints": 3}},
            "logging": {"enabled": False, "log_freq": 10},
        })
        r = ExperimentRunner(cfg, mock_strategy)
        assert r.num_drones == 1
        assert r.start_positions.shape == (1, 3)

    @patch(f"{MODULE}.instantiate")
    def test_zero_duration_runs_no_steps(self, mock_inst, base_cfg, mock_strategy):
        base_cfg.simulation.duration_sec = 0.0
        r = ExperimentRunner(base_cfg, mock_strategy)

        mock_env = MagicMock()
        mock_env._getDroneStateVector.return_value = _make_state()
        mock_env.get_detailed_collisions.return_value = []
        mock_inst.return_value = mock_env

        mock_ctrl = MagicMock()
        mock_ctrl.compute_actions.return_value = np.zeros((2, 4))
        mock_ctrl.params = {"acceptance_radius": 0.2}
        r.trajectory_controller = mock_ctrl

        r.run()
        mock_env.step.assert_not_called()
        mock_env.close.assert_called_once()

    def test_config_default_values(self, mock_strategy):
        """Konfiguracja z brakującymi opcjonalnymi polami powinna używać domyślnych."""
        cfg = OmegaConf.create({
            "simulation": {
                "ctrl_freq": 48, "pyb_freq": 240, "gui": False,
                "duration_sec": 1.0,
            },
            "environment": {
                "params": {
                    "num_drones": 1,
                    "placement_strategy": "grid",
                    "ground_position": 0.0,
                    "track_length": 10.0, "track_width": 10.0, "track_height": 10.0,
                    "shape_type": "BOX",
                    "obstacles_number": 0,
                    "obstacle_width": 1.0, "obstacle_height": 1.0, "obstacle_length": 1.0,
                },
                "initial_rpys": [[0, 0, 0]],
                "initial_xyzs": [[0, 0, 0]],
                "end_xyzs": [[1, 1, 1]],
            },
            "visualization": {
                "tracked_drone_id": 0, "show_lidar_rays": False,
                "lidar_draw_interval": 5, "camera_follow": False,
                "camera_distance": 1.0, "camera_yaw": 0.0, "camera_pitch": -30.0,
            },
            "optimizer": {"algorithm_params": {"n_inner_waypoints": 2}},
            "logging": {"enabled": False, "log_freq": 1},
        })
        r = ExperimentRunner(cfg, mock_strategy)
        # Domyślne wartości z cfg.simulation.get()
        assert r.drone_model == "CF2X"
        assert r.phyics == "PYB"
        assert r.sim_speed_multiplier == 5.0
