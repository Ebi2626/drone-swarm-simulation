import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
import numpy as np

from ExperimentRunner import ExperimentRunner

@pytest.fixture
def dummy_cfg():
    """Tworzy bezpieczną, sztuczną konfigurację, która naśladuje tę z Hydry."""
    return OmegaConf.create({
        "simulation": {
            "drone_model": "CF2X",
            "physics": "PYB",
            "ctrl_freq": 48,
            "pyb_freq": 240,
            "gui": False,           # Wyłączamy GUI do testów
            "duration_sec": 0.1,    # Bardzo krótki czas symulacji (np. 4-5 kroków)
            "speed_multiplier": 1.0
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
                "obstacle_length": 1.0
            },
            "initial_rpys": [[0, 0, 0], [0, 0, 0]],
            "initial_xyzs": [[0, 0, 0], [1, 1, 1]],
            "end_xyzs": [[10, 10, 10], [9, 9, 9]]
        },
        "visualization": {
            "tracked_drone_id": 0,
            "camera_follow": False,
            "camera_distance": 2.0,
            "camera_yaw": 0.0,
            "camera_pitch": -30.0
        },
        "optimizer": {
            "algorithm_params": {
                "n_inner_waypoints": 5
            }
        },
        "logging": {
            "enabled": False,       # Wyłączamy faktyczne pisanie po dysku
            "log_freq": 10
        }
    })

# Znowu upewnij się, że patchujesz właściwą nazwę pliku, z którego importujesz
MODULE_PATH = "ExperimentRunner" 

def test_runner_initialization(dummy_cfg):
    """Sprawdza, czy Runner poprawnie wyciąga dane z konfiguracji Hydry."""
    runner = ExperimentRunner(dummy_cfg)
    
    assert runner.num_drones == 2
    assert runner.ctrl_freq == 48
    assert runner.track_length == 10.0
    assert runner.tracked_drone_id == 0
    # Weryfikacja rzutowania list na wektory
    assert isinstance(runner.start_positions, np.ndarray)
    assert runner.start_positions.shape == (2, 3)

@patch(f"{MODULE_PATH}.instantiate")
@patch(f"{MODULE_PATH}.generate_world_boundaries")
@patch(f"{MODULE_PATH}.generate_obstacles")
def test_offline_trajectory_counting(
    mock_gen_obstacles, 
    mock_gen_world, 
    mock_instantiate, 
    dummy_cfg
):
    """Testuje pierwszą fazę działania (obliczenia offline), sprawdzając przepływ danych."""
    runner = ExperimentRunner(dummy_cfg)
    
    # Mockujemy strategię liczenia (to co zwraca Hydra instantiate dla optimizer)
    mock_counting_strategy = MagicMock()
    # Zwracana trajektoria: (Liczba dronów, Liczba waypointów, 3)
    mock_counting_strategy.return_value = np.zeros((2, 5, 3)) 
    mock_instantiate.return_value = mock_counting_strategy

    # Odpalamy fazę offline
    runner.offilne_trajectory_counting()

    # 1. Sprawdzamy czy poproszono o wygenerowanie świata
    mock_gen_world.assert_called_once()
    
    # 2. Sprawdzamy czy poproszono o wygenerowanie przeszkód
    mock_gen_obstacles.assert_called_once()
    
    # 3. Sprawdzamy, czy przypisało wygenerowane trajektorie do pola w klasie
    assert runner.trajectories is not None
    assert runner.trajectories.shape == (2, 5, 3)
    
    # 4. Sprawdzamy czy uruchomiono pod-kontroler
    assert hasattr(runner, "trajectory_controller")

@patch(f"{MODULE_PATH}.p.isConnected")
@patch(f"{MODULE_PATH}.time.sleep")
@patch(f"{MODULE_PATH}.instantiate")
def test_run_simulation_loop_headless(
    mock_instantiate, 
    mock_sleep, 
    mock_pybullet_connected, 
    dummy_cfg
):
    """
    Testuje główną pętlę symulacyjną run().
    Używamy trybu Headless (gui=False), aby przetestować przechodzenie pętli czasu.
    """
    runner = ExperimentRunner(dummy_cfg)
    
    # Sztuczne środowisko
    mock_environment = MagicMock()
    mock_environment._getDroneStateVector.return_value = (0, 0, 0)
    mock_environment.get_detailed_collisions.return_value = []
    
    # Instancjowanie środowiska zwraca naszego mocka
    mock_instantiate.return_value = mock_environment
    
    # Podpinamy atrapę kontrolera, żeby nie sypało błędami przy compute_actions
    mock_controller = MagicMock()
    mock_controller.compute_actions.return_value = np.array([1, 1])
    runner.trajectory_controller = mock_controller

    # Odpalamy!
    runner.run()

    # W trybie headless (0.1 sekundy przy ctrl_freq=48) pętla powinna 
    # wykonać się około 4 razy (0.1 * 48 = 4.8 -> int = 4).
    assert mock_environment.step.call_count == 4
    
    # Środowisko powinno zostać zamknięte po symulacji
    mock_environment.close.assert_called_once()