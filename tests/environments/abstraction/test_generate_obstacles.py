import pytest
import numpy as np
from unittest.mock import MagicMock

from src.environments.obstacles.ObstacleShape import ObstacleShape
from src.environments.abstraction.generate_obstacles import (
    ObstaclesData,
    strategy_empty,
    strategy_random_uniform,
    strategy_grid_jitter,
    generate_obstacles
)

# ==========================================
# TESTY STRUKTURY DANYCH
# ==========================================

def test_obstacles_data_property():
    """ Intencja: Właściwość .count powinna poprawnie odczytywać liczbę rzędów w macierzy. """
    mock_matrix = np.zeros((15, 6))
    obs_data = ObstaclesData(data=mock_matrix, shape_type=ObstacleShape.BOX)
    
    assert obs_data.count == 15

# ==========================================
# TESTY STRATEGII POZYCJONOWANIA
# ==========================================

def test_strategy_empty():
    """ Intencja: Strategia pustego świata musi ignorować count i zwracać pustą macierz 3D. """
    min_b = np.array([0.0, 0.0, 0.0])
    max_b = np.array([10.0, 10.0, 10.0])
    
    result = strategy_empty(min_b, max_b, count=100)
    
    assert result.shape == (0, 3)

def test_strategy_random_uniform_safe_zones():
    """
    Intencja: Sprawdzenie Rejection Sampling. Wygenerowane przeszkody 
    muszą znajdować się poza promieniem bezpiecznym wokół Startu i Celu.
    """
    min_b = np.array([0.0, 0.0, 0.0])
    max_b = np.array([50.0, 50.0, 0.0])
    
    start_pos = np.array([[10.0, 10.0, 0.0]])
    target_pos = np.array([[40.0, 40.0, 0.0]])
    safe_rad = 5.0
    
    positions = strategy_random_uniform(
        min_b, max_b, count=50, 
        start_positions=start_pos, target_positions=target_pos, 
        safe_radius=safe_rad
    )
    
    assert positions.shape == (50, 3)
    
    # Obliczamy odległości wszystkich wygenerowanych punktów do Startu i Celu
    dist_to_start = np.linalg.norm(positions - start_pos[0], axis=1)
    dist_to_target = np.linalg.norm(positions - target_pos[0], axis=1)
    
    # Udowadniamy, że ŻADNA przeszkoda nie złamała strefy bezpiecznej
    assert np.all(dist_to_start >= safe_rad)
    assert np.all(dist_to_target >= safe_rad)

def test_strategy_grid_jitter_fallback(capsys):
    """
    Edge case: Oversampling w siatce. Co się stanie, gdy zażądamy 100 przeszkód 
    na małej powierzchni, gdzie ogromną część zajmuje strefa bezpieczna?
    Oczekujemy: Mniejszej liczby przeszkód oraz ostrzeżenia w konsoli.
    """
    min_b = np.array([0.0, 0.0, 0.0])
    max_b = np.array([10.0, 10.0, 0.0]) # Mała powierzchnia (10x10)
    
    start_pos = np.array([[5.0, 5.0, 0.0]]) # Start na samym środku
    safe_rad = 8.0 # Promień na tyle duży, że pożre prawie całą planszę
    
    positions = strategy_grid_jitter(
        min_b, max_b, count=100, 
        start_positions=start_pos, safe_radius=safe_rad
    )
    
    # Udowadniamy, że zwrócił mniej niż żądano (zapobiega nieskończonej pętli)
    assert len(positions) < 100
    
    # Upewniamy się, że to co zwrócił, jest bezpieczne w osiach XY
    if len(positions) > 0:
        dist_to_start_xy = np.linalg.norm(positions[:, :2] - start_pos[0, :2], axis=1)
        assert np.all(dist_to_start_xy >= safe_rad)
        
    # Przechwytujemy log z konsoli i upewniamy się, że algorytm poinformował o problemie
    captured = capsys.readouterr()
    assert "[WARN] Nie udało się wygenerować" in captured.out

# ==========================================
# TESTY GENERATORA MACIERZY DANYCH (6D)
# ==========================================

@pytest.fixture
def mock_world():
    """ Atrapa WorldData. """
    world = MagicMock()
    # Z-oś celowo ustawiona z offsetem, aby sprawdzić, czy funkcja "wbija" je w ziemię
    world.min_bounds = np.array([-10.0, -10.0, -5.0])
    world.max_bounds = np.array([10.0, 10.0, 5.0])
    return world

def test_generate_obstacles_cylinder(mock_world):
    """ 
    Intencja: Sprawdzenie pakowania danych dla CYLINDRA.
    Wymiary mają być równe: [radius(width), height, 0.0]
    """
    params = {'width': 2.5, 'height': 8.0} # width = radius
    
    obs_data = generate_obstacles(
        world=mock_world, n_obstacles=3,
        shape_type=ObstacleShape.CYLINDER, 
        size_params=params
    )
    
    assert obs_data.shape_type == ObstacleShape.CYLINDER
    assert obs_data.count == 3
    
    # Ekstrakcja 3 ostatnich kolumn (wymiarów) z macierzy 6D
    dimensions = obs_data.data[:, 3:6]
    
    np.testing.assert_array_equal(dimensions[:, 0], 2.5) # radius
    np.testing.assert_array_equal(dimensions[:, 1], 8.0) # height
    np.testing.assert_array_equal(dimensions[:, 2], 0.0) # 3. wymiar = 0
    
    # Weryfikacja wymuszenia na poziomie gruntu (Z=0)
    z_positions = obs_data.data[:, 2]
    np.testing.assert_array_equal(z_positions, 0.0)

def test_generate_obstacles_box(mock_world):
    """ 
    Intencja: Sprawdzenie pakowania danych dla BOXA.
    Wymiary mają być równe: [length, width, height]
    """
    params = {'length': 1.0, 'width': 2.0, 'height': 3.0}
    
    obs_data = generate_obstacles(
        world=mock_world, n_obstacles=2,
        shape_type=ObstacleShape.BOX, 
        size_params=params
    )
    
    assert obs_data.shape_type == ObstacleShape.BOX
    
    dimensions = obs_data.data[:, 3:6]
    
    np.testing.assert_array_equal(dimensions[:, 0], 1.0) # length
    np.testing.assert_array_equal(dimensions[:, 1], 2.0) # width
    np.testing.assert_array_equal(dimensions[:, 2], 3.0) # height