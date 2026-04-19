import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.abstraction.generate_obstacles import ObstaclesData

import importlib
TARGET_MODULE = "src.environments.EmptyWorld"
EmptyWorld = importlib.import_module(TARGET_MODULE).EmptyWorld

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_world_data():
    return MagicMock(spec=WorldData)

@pytest.fixture
def mock_obstacles_data():
    return MagicMock(spec=ObstaclesData)

# ==========================================
# TESTY
# ==========================================

@patch(f"{TARGET_MODULE}.sanitize_init_params")
@patch(f"{TARGET_MODULE}.SwarmBaseWorld.__init__", return_value=None)
def test_empty_world_initialization(mock_super_init, mock_sanitize, mock_world_data, mock_obstacles_data):
    """
    Intencja: Sprawdzenie, czy EmptyWorld poprawnie sanitizuje parametry,
    wylicza num_drones na podstawie initial_xyzs i przekazuje wszystko do SwarmBaseWorld.
    """
    # 1. Konfigurujemy mock dla sanitize_init_params.
    # Musi on zwrócić krotkę 5 elementów, zgodnie z Twoim kodem:
    # (drone_model, physics, initial_xyzs, end_xyzs, initial_rpys)
    fake_start_pos = np.array([[0, 0, 1], [0, 1, 1]]) # 2 drony
    fake_end_pos = np.array([[10, 10, 1], [10, 11, 1]])
    fake_rpys = np.array([[0, 0, 0], [0, 0, 0]])
    
    mock_sanitize.return_value = (
        DroneModel.RACE, 
        Physics.DYN, 
        fake_start_pos, 
        fake_end_pos, 
        fake_rpys
    )
    
    # 2. Inicjalizujemy EmptyWorld
    world = EmptyWorld(
        world_data=mock_world_data,
        obstacles_data=mock_obstacles_data,
        drone_model="RACE", # symulujemy przekazanie stringa
        physics="DYN",
        initial_xyzs=[[0,0,1], [0,1,1]], # Niesanitizowane dane
        end_xyzs=[[10,10,1], [10,11,1]],
        custom_kwarg="test_value" # Sprawdzenie przekazywania kwargs
    )
    
    # 3. Sprawdzamy czy sanitize_init_params zostało wywołane z oryginalnymi argumentami
    mock_sanitize.assert_called_once_with(
        "RACE", "DYN", 
        [[0,0,1], [0,1,1]], 
        [[10,10,1], [10,11,1]], 
        None
    )
    
    # 4. Upewniamy się, że klasa zapisała docelowe pozycje
    np.testing.assert_array_equal(world.end_xyzs, fake_end_pos)
    
    # 5. Najważniejsze: sprawdzamy, czy do super().__init__ trafiły zsanitizowane 
    # parametry i poprawnie wyliczone num_drones!
    mock_super_init.assert_called_once_with(
        world_data=mock_world_data,
        obstacles_data=mock_obstacles_data,
        primary_num_drones=None,
        dynamic_obstacles_enabled=False,
        num_dynamic_obstacles=0,
        drone_model=DroneModel.RACE,
        physics=Physics.DYN,
        num_drones=2, # <- To udowadnia, że len(initial_xyzs) zadziałało!
        initial_xyzs=fake_start_pos,
        initial_rpys=fake_rpys,
        obstacles=True,
        custom_kwarg="test_value"
    )

@patch(f"{TARGET_MODULE}.sanitize_init_params")
@patch(f"{TARGET_MODULE}.SwarmBaseWorld.__init__", return_value=None)
def test_empty_world_draw_obstacles(mock_super_init, mock_sanitize, mock_world_data, mock_obstacles_data, capsys):
    """
    Intencja: Upewnienie się, że metoda draw_obstacles rzuca odpowiedni print 
    w konsoli, spełniając kontrakt z klasy abstrakcyjnej.
    """
    # Inicjalizacja atrapy
    mock_sanitize.return_value = (DroneModel.CF2X, Physics.PYB, np.zeros((1,3)), None, None)
    world = EmptyWorld(mock_world_data, mock_obstacles_data, initial_xyzs=[[0,0,0]])
    
    # Uruchamiamy testowaną metodę
    world.draw_obstacles()
    
    # Przechwytujemy i sprawdzamy logi w konsoli
    captured = capsys.readouterr()
    assert "[DEBUG] Generating empty environment..." in captured.out