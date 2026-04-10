import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.utils.SimulationLogger import SimulationLogger
from src.environments.obstacles.ObstacleShape import ObstacleShape

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def logger(tmp_path):
    """
    Inicjalizuje loggera z tymczasowym katalogiem (tmp_path).
    Wszystkie pliki CSV zapiszą się w bezpiecznym folderze w /tmp.
    freq: log = 10Hz, ctrl = 100Hz -> log_step_interval = 10
    """
    return SimulationLogger(
        output_dir=str(tmp_path), 
        log_freq=10, 
        ctrl_freq=100, 
        num_drones=2
    )

@pytest.fixture
def dummy_state():
    """ 
    Atrapa stanu drona. Wymaga co najmniej 13 elementów, 
    aby logger mógł pobrać indeks 12 bez błędu IndexError. 
    (np. x, y, z, qx, qy, qz, qw, r, p, y, vx, vy, vz)
    """
    return [1.1119, 2.2229, 3.3339, 0, 0, 0, 0, 4.4449, 5.5559, 6.6669, 7.7779, 8.8889, 9.9999]

# ==========================================
# TESTY BUFOROWANIA W RAM
# ==========================================

def test_log_step_interval_and_rounding(logger, dummy_state):
    """ 
    Intencja: Logger powinien zapisywać stan tylko co 'log_step_interval' 
    (tutaj co 10 krok) oraz poprawnie zaokrąglać do 3 miejsc po przecinku.
    """
    # Krok 0 (powinien zostać zalogowany)
    logger.log_step(step_idx=0, current_time=0.1234, all_states=[dummy_state, dummy_state])
    assert len(logger.trajectory_buffer) == 2
    
    # Krok 1 do 9 (powinny zostać zignorowane)
    logger.log_step(step_idx=5, current_time=0.15, all_states=[dummy_state, dummy_state])
    assert len(logger.trajectory_buffer) == 2
    
    # Krok 10 (znowu zalogowany)
    logger.log_step(step_idx=10, current_time=0.20, all_states=[dummy_state, dummy_state])
    assert len(logger.trajectory_buffer) == 4

    # Sprawdzenie poprawnego zaokrąglenia i cięcia parametrów z dummy_state
    record = logger.trajectory_buffer[0]
    expected_record = (0.123, 0, 1.112, 2.223, 3.334, 4.445, 5.556, 6.667, 7.778, 8.889, 10.0)
    assert record == expected_record

def test_log_collision_logic(logger):
    """
    Intencja: Kolizje poniżej 1. sekundy są ignorowane. Dron może zderzyć się
    tylko raz (trafia do 'crashed_drones').
    """
    # 1. Kolizja w czasie < 1s (ignorowana)
    logger.log_collision(current_time=0.5, drone_id=0, other_body_id=99)
    assert len(logger.collision_buffer) == 0
    
    # 2. Prawidłowa kolizja
    logger.log_collision(current_time=1.5, drone_id=0, other_body_id=99)
    assert len(logger.collision_buffer) == 1
    assert 0 in logger.crashed_drones
    
    # 3. Kolejna kolizja TEGO SAMEGO drona (powinna zostać zignorowana)
    logger.log_collision(current_time=2.0, drone_id=0, other_body_id=100)
    assert len(logger.collision_buffer) == 1

def test_crashed_drones_are_not_logged_in_trajectory(logger, dummy_state):
    """
    Intencja: Jeśli dron uderzył, logger w log_step powinien go całkowicie pomijać.
    """
    logger.crashed_drones.add(0) # Symulujemy rozbicie drona o ID=0
    
    # Próbujemy zalogować flotę dwóch dronów
    logger.log_step(step_idx=0, current_time=1.0, all_states=[dummy_state, dummy_state])
    
    # W buforze powinien być tylko jeden zapis (dla drona o ID=1)
    assert len(logger.trajectory_buffer) == 1
    assert logger.trajectory_buffer[0][1] == 1  # Sprawdzamy pole drone_id

# ==========================================
# TESTY KONWERSJI DO DATAFRAME
# ==========================================

def test_trajectory_to_dataframe(logger):
    """
    Intencja: Poprawne spłaszczenie (flatten/ravel) trójwymiarowego tensora 
    NumPy (N, Waypoints, 3) do płaskiego DataFrame.
    """
    # Tensor: 2 drony, 3 waypointy, 3 koordynaty (x,y,z)
    traj = np.zeros((2, 3, 3))
    traj[1, 2] = [10.5, 20.5, 30.5]  # Wpisujemy dane dla Drona 1, Waypoint 2
    
    df = logger._trajectory_to_dataframe(traj)
    
    assert len(df) == 6  # 2 drony * 3 waypointy
    assert list(df.columns) == ["drone_id", "waypoint_id", "x", "y", "z"]
    
    # Weryfikacja czy poprawne wartości trafiły w poprawne miejsce
    row = df[(df['drone_id'] == 1) & (df['waypoint_id'] == 2)].iloc[0]
    assert row['x'] == 10.5
    assert row['y'] == 20.5
    assert row['z'] == 30.5

def test_obstacles_to_dataframe_cylinder_bug(logger):
    """
    Ten test będzie świecił na żółto (XFAIL) dopóki nie naprawisz literówki
    przy usuwaniu kolumn dla CYLINDER w głównym pliku.
    """
    mock_obs = MagicMock()
    mock_obs.shape_type = ObstacleShape.CYLINDER
    mock_obs.data = np.array([[1.0, 2.0, 3.0, 0.5, 2.0, 0.0]])
    
    df = logger._obstacles_to_dataframe(mock_obs)
    
    assert "unused_dim" not in df.columns
    assert "dim3_unused" not in df.columns

def test_world_to_dataframe(logger):
    """ Sprawdza poprawne przypisanie indeksów osi X, Y, Z """
    mock_world = MagicMock()
    mock_world.dimensions = [10, 20, 30]
    mock_world.min_bounds = [-5, -10, -15]
    mock_world.max_bounds = [5, 10, 15]
    mock_world.center = [0, 0, 0]
    
    df = logger._world_to_dataframe(mock_world)
    
    assert list(df.index) == ['X', 'Y', 'Z']
    assert df.loc['Y', 'Dimension'] == 20

# ==========================================
# TESTY FIZYCZNEGO ZAPISU (I/O)
# ==========================================

def test_save_writes_to_disk_and_clears_buffers(logger, tmp_path):
    """
    Intencja: Sprawdzenie, czy po wywołaniu `save()` tworzą się fizyczne
    pliki CSV we wskazanym folderze, a następnie bufory pamięci zostają wyczyszczone.
    """
    # Symulujemy zapełnienie buforów
    logger.trajectory_buffer.append((1.0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0))
    logger.collision_buffer.append((1.5, 0, 99))
    
    logger.save()
    
    # tmp_path działa jak obiekt Path z pathlib
    traj_file = tmp_path / "trajectories.csv"
    coll_file = tmp_path / "collisions.csv"
    
    assert traj_file.exists()
    assert coll_file.exists()
    
    # Bufory po udanym zapisie powinny być puste
    assert len(logger.trajectory_buffer) == 0
    assert len(logger.collision_buffer) == 0

    # Sprawdzenie zawartości (opcjonalnie)
    saved_traj = pd.read_csv(traj_file)
    assert len(saved_traj) == 1
    assert "drone_id" in saved_traj.columns

def test_save_does_not_create_empty_collision_file(logger, tmp_path):
    """
    Edge case: Jeśli nie było kolizji, plik `collisions.csv` nie powinien zostać utworzony.
    """
    logger.trajectory_buffer.append((1.0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0))
    # Celowo nie dodajemy nic do collision_buffer
    
    logger.save()
    
    assert (tmp_path / "trajectories.csv").exists()
    assert not (tmp_path / "collisions.csv").exists()