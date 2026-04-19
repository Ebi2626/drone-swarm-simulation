import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.abstraction.generate_obstacles import ObstaclesData

import importlib
TARGET_MODULE = "src.environments.ForestWorld"
ForestWorld = importlib.import_module(TARGET_MODULE).ForestWorld

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_world_data():
    return MagicMock(spec=WorldData)

@pytest.fixture
def mock_obstacles_data():
    # Przygotowujemy atrapę danych o przeszkodach (2 drzewa)
    # Format wiersza: [x, y, z, radius, height, 0.0]
    fake_data = np.array([
        [1.0, 2.0, 0.0, 0.5, 10.0, 0.0],  # Drzewo 1: wys = 10, promień = 0.5
        [-5.0, 3.0, 2.0, 1.2, 20.0, 0.0]  # Drzewo 2: wys = 20, promień = 1.2, stoi na podwyższeniu (z=2)
    ])
    
    obs_mock = MagicMock(spec=ObstaclesData)
    obs_mock.data = fake_data
    return obs_mock

# ==========================================
# TESTY INICJALIZACJI
# ==========================================

@patch(f"{TARGET_MODULE}.sanitize_init_params")
@patch(f"{TARGET_MODULE}.SwarmBaseWorld.__init__", return_value=None)
def test_forest_world_initialization(mock_super_init, mock_sanitize, mock_world_data, mock_obstacles_data):
    """
    Intencja: Sprawdzenie, czy parametry są sanitizowane i poprawnie przekazywane do SwarmBaseWorld.
    (Test bardzo podobny do EmptyWorld, aby zachować pewność przy refaktoryzacji).
    """
    fake_start_pos = np.array([[0, 0, 1]]) 
    
    mock_sanitize.return_value = (DroneModel.RACE, Physics.DYN, fake_start_pos, None, None)
    
    world = ForestWorld(
        world_data=mock_world_data,
        obstacles_data=mock_obstacles_data,
        initial_xyzs=[[0,0,1]]
    )
    
    mock_super_init.assert_called_once_with(
        world_data=mock_world_data,
        obstacles_data=mock_obstacles_data,
        primary_num_drones=None,
        dynamic_obstacles_enabled=False,
        num_dynamic_obstacles=0,
        drone_model=DroneModel.RACE,
        physics=Physics.DYN,
        num_drones=1,
        initial_xyzs=fake_start_pos,
        initial_rpys=None,
        obstacles=True
    )

# ==========================================
# TESTY RYSOWANIA GEOMETRII W PYBULLET
# ==========================================

@patch(f"{TARGET_MODULE}.sanitize_init_params")
@patch(f"{TARGET_MODULE}.SwarmBaseWorld.__init__", return_value=None)
@patch(f"{TARGET_MODULE}.p") # Mockujemy API pybulleta w testowanym pliku
def test_draw_obstacles_creates_trees(mock_p, mock_super_init, mock_sanitize, mock_world_data, mock_obstacles_data):
    """
    Intencja: Upewnienie się, że pętla generująca las prawidłowo wywołuje funkcje PyBulleta
    oraz poprawnie wylicza środek ciężkości (base_z = z + height/2).
    """
    # Ustawiamy atrapy zwracające stałe ID (żeby MultiBody miało poprawne indeksy)
    mock_p.GEOM_CYLINDER = 3 # Zastępujemy ewentualne stałe z pybulleta
    mock_p.createCollisionShape.return_value = 101
    mock_p.createVisualShape.return_value = 202

    # Inicjalizujemy pusty stan dla sanitize (żeby klasa nie krzyczała o błędach w init)
    mock_sanitize.return_value = (DroneModel.CF2X, Physics.PYB, np.zeros((1,3)), None, None)
    world = ForestWorld(mock_world_data, mock_obstacles_data, initial_xyzs=[[0,0,0]])
    world.obstacles = mock_obstacles_data
    
    # Odpalamy główną funkcję
    world.draw_obstacles()

    # Mamy 2 drzewa w fixture, więc każda metoda powinna być wywołana dokładnie 2 razy
    assert mock_p.createCollisionShape.call_count == 2
    assert mock_p.createVisualShape.call_count == 2
    assert mock_p.createMultiBody.call_count == 2

    # --- Analiza wywołań (call_args_list) ---
    multi_body_calls = mock_p.createMultiBody.call_args_list

    # Sprawdzenie DRZEWA 1 [1.0, 2.0, 0.0, 0.5, 10.0, 0.0]
    # base_z = 0.0 + 10.0/2 = 5.0
    _, kwargs_tree1 = multi_body_calls[0]
    assert kwargs_tree1['baseCollisionShapeIndex'] == 101
    assert kwargs_tree1['baseVisualShapeIndex'] == 202
    np.testing.assert_array_equal(kwargs_tree1['basePosition'], [1.0, 2.0, 5.0])

    # Sprawdzenie DRZEWA 2 [-5.0, 3.0, 2.0, 1.2, 20.0, 0.0]
    # base_z = 2.0 + 20.0/2 = 12.0
    _, kwargs_tree2 = multi_body_calls[1]
    np.testing.assert_array_equal(kwargs_tree2['basePosition'], [-5.0, 3.0, 12.0])

    # Opcjonalnie sprawdzamy, czy przekazano poprawne parametry geometrii do createVisualShape
    visual_calls = mock_p.createVisualShape.call_args_list
    
    # Drzewo 1: radius=0.5, length=10.0 (zwróć uwagę, że pybullet wymaga parametru 'length' przy VisualShape, a 'height' przy CollisionShape)
    _, v_kwargs_tree1 = visual_calls[0]
    assert v_kwargs_tree1['radius'] == 0.5
    assert v_kwargs_tree1['length'] == 10.0