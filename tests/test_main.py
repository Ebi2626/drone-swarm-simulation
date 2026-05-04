import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from pathlib import Path
from dataclasses import dataclass

# Importujemy z głównego modułu
from main import (
    ExperimentRunner,
    _get_registry_job_key,
    main_generate,
    main_replay,
)
from src.runner.ExperimentDataStrategy import ExperimentDataStrategy
from src.environments.obstacles.ObstacleShape import ObstacleShape


# --- Podstawowa atrapa strategii danych ---
class DummyDataStrategy(ExperimentDataStrategy):
    def prepare_data(self, runner, seeds):
        runner.world_data = MagicMock()
        runner.obstacles_data = MagicMock()
        runner.drones_trajectories = np.zeros((1, 2, 3))


# --- FIXTURE DO SEEDÓW ---
@pytest.fixture
def mock_master_seed():
    """Mock dla SeedRegistry zwracający przewidywalne generatory losowości."""
    mock_registry = MagicMock()
    mock_registry.master_seed = 42
    
    # Tworzymy cache generatorów dla weryfikacji tożsamości obiektów w asercjach
    _generators = {}
    
    def mock_rng(namespace: str):
        if namespace not in _generators:
            _generators[namespace] = np.random.default_rng(42)
        return _generators[namespace]
        
    def mock_seed(namespace: str):
        # Pseudo int z wybranego namespace'u
        return 42
        
    mock_registry.rng.side_effect = mock_rng
    mock_registry.seed.side_effect = mock_seed
    return mock_registry


@pytest.fixture
def dummy_cfg():
    """Zwraca minimalną konfigurację potrzebną dla ExperimentRunner."""
    return OmegaConf.create({
        "seed": 42,
        "simulation": {
            "drone_model": "CF2X",
            "physics": "PYB",
            "ctrl_freq": 24,
            "pyb_freq": 240,
            "sim_speed_multiplier": 1.0,
            "dynamic_obstacles": False,
            "duration_sec": 0.1,  # Krótki czas = mało kroków w run()
            "gui": False,
            "dense_samples": 20,
        },
        "environment": {
            "name": "test_env",
            # Te 3 parametry w main.py wyciągane są bezpośrednio z 'environment'
            "initial_xyzs": [[0, 0, 1]],
            "end_xyzs": [[10, 10, 1]],
            "initial_rpys": None,
            "params": {
                "num_drones": 1,
                "placement_strategy": "test_strategy",
                "ground_position": 0.0,
                "track_length": 100.0,
                "track_width": 100.0,
                "track_height": 50.0,
                "shape_type": ObstacleShape.BOX,
                "obstacles_number": 5,
                "obstacle_width": 2.0,
                "obstacle_height": 2.0,
                "obstacle_length": 2.0,
                "safe_radius": 1.0,
            }
        },
        "visualization": {
            "tracked_drone_id": 0,
            "show_lidar_rays": False,
            "lidar_draw_interval": 5,
            "camera_follow": False,
        },
        "optimizer": {
            "_target_": "some.Optimizer",
            "algorithm_params": {
                "cruise_speed": 5.0,
                "max_accel": 1.0,
            }
        },
        "logging": {
            "enabled": False,
            "log_freq": 10,
        },
        "experiment_meta": {
            "id": "test_exp_id"
        },
        "avoidance": {
            "enable": False,
            "name": "none"
        }
    })


# --- TESTY: _get_registry_job_key ---

def test_get_registry_job_key_valid(dummy_cfg):
    """Sprawdza czy funkcja poprawnie mapuje klucze na format job_key."""
    dummy_cfg.optimizer._target_ = "src.algorithms.MSFFOAOptimizer"
    dummy_cfg.environment.name = "EnvWithTrees"
    dummy_cfg.avoidance.name = "MyAvoidance"
    dummy_cfg.seed = 999

    res = _get_registry_job_key(dummy_cfg)

    assert res is not None
    assert res["exp_id"] == "test_exp_id"
    assert res["optimizer"] == "msffoa"
    assert res["environment"] == "envwithtrees"
    assert res["avoidance"] == "myavoidance"
    assert res["seed"] == 999


def test_get_registry_job_key_missing_meta(dummy_cfg):
    """Brak experiment_meta.id powinien zwrócić None."""
    del dummy_cfg["experiment_meta"]
    assert _get_registry_job_key(dummy_cfg) is None


def test_get_registry_job_key_unknown_optimizer(dummy_cfg):
    """Jeśli w optimizerze nie ma znanej nazwy, powinno zmapować na 'unknown'."""
    dummy_cfg.optimizer._target_ = "some.weird.Class"
    res = _get_registry_job_key(dummy_cfg)
    assert res["optimizer"] == "unknown"


# --- TESTY: ExperimentRunner ---

class TestExperimentRunner:
    
    @patch("main.SwarmFlightController")
    def test_init_trajectory_following_algorithm_no_avoidance(
        self, mock_sfc, dummy_cfg, mock_master_seed
    ):
        """Upewnia się, że SwarmFlightController tworzy się poprawnie (bez unikania kolizji)."""
        runner = ExperimentRunner(dummy_cfg, DummyDataStrategy(), seeds=mock_master_seed)
        runner._init_trajectory_following_algorithm()

        assert runner.trajectory_controller is not None
        mock_sfc.assert_called_once()
        _, kwargs = mock_sfc.call_args
        assert kwargs["num_drones"] == 1
        assert kwargs["is_obstacle"] is False
        assert kwargs["avoidance_algorithm"] is None
        assert kwargs["params"]["enable_avoidance"] is False
        assert runner.dynamic_obstacle_trajectory_controller is None

    @patch("main.instantiate")
    @patch("main.SwarmFlightController")
    def test_init_trajectory_following_algorithm_with_avoidance(
        self, mock_sfc, mock_instantiate, dummy_cfg, mock_master_seed
    ):
        """Jeśli cfg.avoidance.enable=True, powinien się zinstancjonować avoidance_algo z SeedRegistry."""
        dummy_cfg.avoidance.enable = True
        mock_avoidance = MagicMock()
        mock_avoidance.name = "FakeAvoidance"
        mock_instantiate.return_value = mock_avoidance

        runner = ExperimentRunner(dummy_cfg, DummyDataStrategy(), seeds=mock_master_seed)
        runner._init_trajectory_following_algorithm()

        mock_instantiate.assert_called_once()
        # Wywołanie instantiation powinno zawierać seed z namespace 'avoidance'
        args, kwargs = mock_instantiate.call_args
        assert args[0] == dummy_cfg.avoidance
        assert "optimizer" in kwargs
        assert "rng" in kwargs["optimizer"]

        _, kwargs_sfc = mock_sfc.call_args
        assert kwargs_sfc["avoidance_algorithm"] is mock_avoidance
        assert kwargs_sfc["params"]["enable_avoidance"] is True

    @patch("main.instantiate")
    def test_initialize_world(self, mock_instantiate, dummy_cfg, mock_master_seed):
        """Sprawdza czy środowisko jest instancjonowane z poprawnymi argumentami z Hydra configs."""
        runner = ExperimentRunner(dummy_cfg, DummyDataStrategy(), seeds=mock_master_seed)
        runner.world_data = "FAKE_WORLD"
        runner.obstacles_data = "FAKE_OBSTACLES"

        mock_env = MagicMock()
        mock_instantiate.return_value = mock_env

        runner.initialize_world()

        mock_instantiate.assert_called_once()
        _, kwargs = mock_instantiate.call_args
        assert kwargs["world_data"] == "FAKE_WORLD"
        assert kwargs["obstacles_data"] == "FAKE_OBSTACLES"
        assert kwargs["num_drones"] == 1
        np.testing.assert_array_equal(kwargs["initial_xyzs"], [[0, 0, 1]])
        np.testing.assert_array_equal(kwargs["end_xyzs"], [[10, 10, 1]])

        assert runner.environemnt is mock_env

    @patch("main.SwarmFlightController")
    @patch("main.ExperimentRunner.initialize_world")
    @patch("main.time.time", side_effect=[0.0, 0.1, 0.2, 0.3])
    @patch("main.p.isConnected", return_value=True)
    def test_run_headless(
        self, mock_is_connected, mock_time, mock_init_world, mock_sfc_class, dummy_cfg, mock_master_seed
    ):
        """Prosty przebieg run() w trybie headless, aby sprawdzić czy pętla się wykonuje bez crasha."""
        runner = ExperimentRunner(dummy_cfg, DummyDataStrategy(), seeds=mock_master_seed)
        runner.environemnt = MagicMock()
        
        # Symulacja stanu dronów (tylko 1 dron)
        runner.environemnt._getDroneStateVector.return_value = [0.0, 0.0, 1.0] * 7  # sztuczny stan drona
        
        mock_controller = MagicMock()
        mock_controller.compute_actions.return_value = np.array([[1.0, 1.0, 1.0, 1.0]])
        runner.trajectory_controller = mock_controller

        # Omijamy _init_trajectory_following_algorithm żeby móc wstrzyknąć mocka wyżej
        runner._init_trajectory_following_algorithm = MagicMock()
        
        # Upewniamy się, że active_drones są poprawnie sprawdzane w `run()`
        runner.active_drones = {0}
        
        # mock_process_arrivals żeby odznaczył wszystkie drony po 1 kroku
        def mock_process_arrivals(states, t):
            runner.active_drones.clear()
            
        runner._process_arrivals = mock_process_arrivals

        # _process_collisions nie powinna wpływać na test
        runner._process_collisions = MagicMock()

        # RUN!
        runner.run()

        mock_init_world.assert_called_once()
        mock_controller.init_lidars.assert_called_once()
        runner.environemnt.step.assert_called()


# --- TESTY PUNKÓW WEJŚCIA (main_generate, main_replay) ---
@patch("main.Path.mkdir")
@patch("main.p.disconnect")
@patch("main.p.isConnected", return_value=True)
@patch("main.ExperimentRunner")
@patch("main.RunRegistry")
@patch("main.bootstrap_global_seed")
@patch("main.seed_numba")
def test_main_generate_new_job(
    mock_seed_numba, mock_bootstrap, mock_registry_class, mock_runner_class, 
    mock_is_connected, mock_disconnect, mock_mkdir, dummy_cfg
):
    """Sprawdza, że dla nowego zadania `main_generate` rejestruje start i wykonuje run()."""
    mock_registry = MagicMock()
    mock_registry.should_run.return_value = True  # Tak, trzeba odpalić
    mock_registry_class.return_value = mock_registry

    mock_runner_instance = MagicMock()
    mock_runner_class.return_value = mock_runner_instance

    main_generate(dummy_cfg)

    # Rejestr był sprawdzony
    mock_registry.should_run.assert_called_once()
    mock_registry.mark_started.assert_called_once()
    
    # Wykonanie eksperymentu
    mock_runner_instance.prepare_experiment.assert_called_once()
    mock_runner_instance.run.assert_called_once()
    
    # Zamknięcie
    mock_registry.mark_completed.assert_called_once()
    mock_disconnect.assert_called_once()

@patch("main.Path.mkdir")
@patch("main.p.disconnect")
@patch("main.p.isConnected", return_value=True)
@patch("main.ExperimentRunner")
@patch("main.RunRegistry")
@patch("main.bootstrap_global_seed")
@patch("main.seed_numba")
def test_main_generate_skip_job(
    mock_seed_numba, mock_bootstrap, mock_registry_class, mock_runner_class, 
    mock_is_connected, mock_disconnect, mock_mkdir, dummy_cfg
):
    """Sprawdza, że jeśli `should_run` = False, eksperyment jest pomijany."""
    mock_registry = MagicMock()
    mock_registry.should_run.return_value = False  # Nie uruchamiaj, zrobione!
    mock_registry_class.return_value = mock_registry

    mock_runner_instance = MagicMock()
    mock_runner_class.return_value = mock_runner_instance

    main_generate(dummy_cfg)

    mock_registry.should_run.assert_called_once()
    mock_registry.mark_started.assert_not_called()
    mock_runner_instance.prepare_experiment.assert_not_called()


@patch("main.OmegaConf.load")
@patch("main.Path.exists", return_value=True)
@patch("main.Path.mkdir")
@patch("main.ExperimentRunner")
@patch("main.bootstrap_global_seed")
@patch("main.seed_numba")
def test_main_replay(
    mock_seed_numba, mock_bootstrap, mock_runner_class, 
    mock_mkdir, mock_exists, mock_load, dummy_cfg
):
    """Sprawdza czy main_replay poprawnie ładuje config i włącza tryb headless."""
    mock_load.return_value = dummy_cfg
    mock_runner_instance = MagicMock()
    mock_runner_class.return_value = mock_runner_instance

    main_replay("/fake/dir", headless=True)

    mock_load.assert_called_once()
    assert dummy_cfg.logging.enabled is False
    assert dummy_cfg.simulation.gui is False
    
    mock_runner_instance.prepare_experiment.assert_called_once()
    mock_runner_instance.run.assert_called_once()
