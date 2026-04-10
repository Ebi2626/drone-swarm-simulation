import numpy as np

from src.environments.abstraction.generate_world_boundaries import (
    WorldData, 
    generate_world_boundaries
)

# ==========================================
# TESTY
# ==========================================

def test_generate_world_boundaries_standard_values():
    """
    Intencja: Sprawdzenie poprawności wyliczania wszystkich parametrów świata
    (wymiary, granice, środek) dla standardowych, dodatnich wartości.
    """
    # Dane wejściowe
    w, l, h = 100.0, 200.0, 50.0
    ground = 5.0
    
    # Generowanie świata
    world = generate_world_boundaries(width=w, length=l, height=h, ground_height=ground)
    
    # 1. Sprawdzenie przypisania podstawowych wymiarów (dimensions)
    np.testing.assert_array_equal(world.dimensions, [100.0, 200.0, 50.0])
    
    # 2. Sprawdzenie granic (min i max)
    # Zwróć uwagę, że oś Z minimalna zaczyna się od ground_height (5.0), a nie 0.0
    np.testing.assert_array_equal(world.min_bounds, [0.0, 0.0, 5.0])
    np.testing.assert_array_equal(world.max_bounds, [100.0, 200.0, 50.0])
    
    # 3. Sprawdzenie poprawnego wyliczenia środka ciężkości (center)
    # X: (0 + 100)/2 = 50, Y: (0 + 200)/2 = 100, Z: (5 + 50)/2 = 27.5
    np.testing.assert_array_equal(world.center, [50.0, 100.0, 27.5])
    
    # 4. Sprawdzenie poprawnego złożenia macierzy bounds za pomocą column_stack
    # Oczekiwany kształt: 3 wiersze (X, Y, Z), 2 kolumny (min, max)
    expected_bounds = np.array([
        [0.0, 100.0],
        [0.0, 200.0],
        [5.0, 50.0]
    ])
    np.testing.assert_array_equal(world.bounds, expected_bounds)

def test_generate_world_boundaries_data_types():
    """
    Intencja: Upewnienie się, że funkcja bezwzględnie rzutuje dane na typ float64,
    nawet jeśli użytkownik przekaże liczby całkowite (int).
    Jest to ważne dla stabilności obliczeń w silniku fizycznym.
    """
    # Podajemy wartości typu int
    world = generate_world_boundaries(width=10, length=20, height=30, ground_height=0)
    
    # Weryfikujemy, czy instancja to dataclass WorldData
    assert isinstance(world, WorldData)
    
    # Weryfikujemy typy wszystkich pól numpy
    assert world.dimensions.dtype == np.float64
    assert world.min_bounds.dtype == np.float64
    assert world.max_bounds.dtype == np.float64
    assert world.center.dtype == np.float64
    assert world.bounds.dtype == np.float64

def test_generate_world_boundaries_zero_values():
    """
    Edge case: Generowanie "płaskiego" lub punktowego świata (same zera).
    Test sprawdza, czy kod nie rzuca błędów dzielenia przez zero przy liczeniu środka.
    """
    world = generate_world_boundaries(width=0.0, length=0.0, height=0.0, ground_height=0.0)
    
    np.testing.assert_array_equal(world.dimensions, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(world.center, [0.0, 0.0, 0.0])
    
    expected_bounds = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ])
    np.testing.assert_array_equal(world.bounds, expected_bounds)

def test_generate_world_boundaries_negative_ground():
    """
    Edge case: Ground height jest wartością ujemną (np. symulacja w depresji, pod poziomem 0).
    Z matematycznego punktu widzenia funkcja powinna to obsłużyć bez problemu.
    """
    world = generate_world_boundaries(width=10.0, length=10.0, height=20.0, ground_height=-5.0)
    
    np.testing.assert_array_equal(world.min_bounds, [0.0, 0.0, -5.0])
    np.testing.assert_array_equal(world.max_bounds, [10.0, 10.0, 20.0])
    
    # Z: (-5 + 20) / 2 = 7.5
    np.testing.assert_array_equal(world.center, [5.0, 5.0, 7.5])