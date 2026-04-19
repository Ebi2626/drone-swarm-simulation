import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.abstraction.generate_obstacles import ObstaclesData

import importlib
TARGET_MODULE = "src.environments.UrbanWorld"
UrbanWorld = importlib.import_module(TARGET_MODULE).UrbanWorld

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_world_data():
    return MagicMock(spec=WorldData)

@pytest.fixture
def mock_obstacles_data():
    # Atrapa danych o przeszkodach (2 budynki)
    # Format wiersza dla BOX: [x, y, z, length, width, height]
    fake_data = np.array([
        [10.0, 20.0, 0.0, 4.0, 6.0, 20.0],   # Budynek 1: wys = 20, podstawa 4x6
        [-5.0, -5.0, 2.0, 10.0, 10.0, 50.0]  # Budynek 2: wys = 50, podstawa 10x10, stoi na wzniesieniu (z=2)
    ])
    
    obs_mock = MagicMock(spec=ObstaclesData)
    obs_mock.data = fake_data
    return obs_mock

# ==========================================
# TESTY INICJALIZACJI
# ==========================================

@patch(f"{TARGET_MODULE}.sanitize_init_params")
@patch(f"{TARGET_MODULE}.SwarmBaseWorld.__init__", return_value=None)
def test_urban_world_initialization(mock_super_init, mock_sanitize, mock_world_data, mock_obstacles_data):
    """
    Intencja: Sprawdzenie sanitizacji i propagacji argumentów do konstruktora klasy bazowej.
    """
    fake_start_pos = np.array([[0, 0, 1]]) 
    fake_end_pos = np.array([[10, 10, 10]])
    
    mock_sanitize.return_value = (DroneModel.RACE, Physics.DYN, fake_start_pos, fake_end_pos, None)
    
    world = UrbanWorld(
        world_data=mock_world_data,
        obstacles_data=mock_obstacles_data,
        initial_xyzs=[[0,0,1]],
        end_xyzs=[[10,10,10]]
    )
    
    # UrbanWorld ma własne przypisanie self.end_xyzs, sprawdźmy to:
    np.testing.assert_array_equal(world.end_xyzs, fake_end_pos)
    
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
@patch(f"{TARGET_MODULE}.np.random.uniform", return_value=0.75) # "Zamrażamy" losowy kolor
@patch(f"{TARGET_MODULE}.p") # Mockujemy API pybulleta
def test_draw_obstacles_creates_buildings(mock_p, mock_random, mock_super_init, mock_sanitize, mock_world_data, mock_obstacles_data):
    """
    Intencja: Upewnienie się, że geometria budynków (BOX) otrzymuje prawidłowe parametry, 
    w szczególności weryfikacja logiki dzielenia na pół (halfExtents) i podnoszenia osi Z (base_z).
    """
    mock_p.GEOM_BOX = 4 
    mock_p.createCollisionShape.return_value = 101
    mock_p.createVisualShape.return_value = 202

    mock_sanitize.return_value = (DroneModel.CF2X, Physics.PYB, np.zeros((1,3)), None, None)
    world = UrbanWorld(mock_world_data, mock_obstacles_data, initial_xyzs=[[0,0,0]])
    
    # Wstrzykujemy dane manualnie (poprawka dla pominiętego super().__init__)
    world.obstacles = mock_obstacles_data
    
    # Odpalamy generowanie miasta
    world.draw_obstacles()

    # Mamy 2 budynki w fixture
    assert mock_p.createCollisionShape.call_count == 2
    assert mock_p.createVisualShape.call_count == 2
    assert mock_p.createMultiBody.call_count == 2
    assert mock_random.call_count == 2 # Zostało wywołane dla każdego budynku

    # --- Analiza parametrów BUDYNKU 1 ---
    # Wejście: [10.0, 20.0, 0.0, 4.0, 6.0, 20.0]
    
    # --- Analiza parametrów BUDYNKU 1 ---
    # 1. Kolizja
    _, kwargs_col1 = mock_p.createCollisionShape.call_args_list[0]
    np.testing.assert_array_equal(kwargs_col1['halfExtents'], [2.0, 3.0, 10.0])
    
    # 2. Wygląd
    _, kwargs_vis1 = mock_p.createVisualShape.call_args_list[0]
    np.testing.assert_array_equal(kwargs_vis1['halfExtents'], [2.0, 3.0, 10.0])
    np.testing.assert_array_equal(kwargs_vis1['rgbaColor'], [0.75, 0.75, 0.75, 1.0])
    
    # 3. Pozycja (MultiBody)
    # W UrbanWorld parametry podano pozycyjnie: (0, col_shape, vis_shape, [x, y, base_z])
    # Dlatego rozpakowujemy `args` (pierwszy element krotki wywołania), a `kwargs` ignorujemy
    args_multi1, _ = mock_p.createMultiBody.call_args_list[0]
    # args_multi1[3] to nasz czwarty argument, czyli [x, y, base_z]
    np.testing.assert_array_equal(args_multi1[3], [10.0, 20.0, 10.0])

    # --- Analiza parametrów BUDYNKU 2 ---
    _, kwargs_col2 = mock_p.createCollisionShape.call_args_list[1]
    np.testing.assert_array_equal(kwargs_col2['halfExtents'], [5.0, 5.0, 25.0])
    
    args_multi2, _ = mock_p.createMultiBody.call_args_list[1]
    np.testing.assert_array_equal(args_multi2[3], [-5.0, -5.0, 27.0])