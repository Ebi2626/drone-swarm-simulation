import pytest
import numpy as np

# UWAGA: Podmień ścieżkę importu na właściwą dla Twojego projektu
from src.utils.postions_to_tensor import positions_to_tensor

# ==========================================
# TESTY POPRAWNYCH DANYCH (Happy Path)
# ==========================================

def test_positions_to_tensor_valid_input():
    """
    Intencja: Poprawne przekształcenie listy 2D na tensor NumPy z wymiarem (N, 3).
    """
    positions = [
        [0.0, 0.0, 1.0],
        [1.5, 2.5, 3.5],
        [-1.0, -2.0, -3.0]
    ]
    
    tensor = positions_to_tensor(positions)
    
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (3, 3)
    assert tensor.dtype == np.float64
    np.testing.assert_array_equal(tensor, np.array(positions))

def test_positions_to_tensor_type_conversion():
    """
    Intencja: Upewnienie się, że podanie wartości całkowitych (int) zostanie
    poprawnie rzutowane na zmiennoprzecinkowe (float64) przez funkcję.
    """
    positions_int = [[1, 2, 3], [4, 5, 6]]
    
    tensor = positions_to_tensor(positions_int)
    
    assert tensor.dtype == np.float64
    # Sprawdzamy, czy wartości to faktycznie 1.0, 2.0 itd.
    np.testing.assert_array_equal(tensor, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

# ==========================================
# TESTY EDGE CASES I WALIDACJI (Błędy)
# ==========================================

def test_invalid_inner_dimension():
    """
    Edge case: Podano punkty 2D zamiast 3D (brak osi Z).
    Oczekiwany wyjątek: ValueError.
    """
    positions_2d = [
        [0.0, 1.0],
        [2.0, 3.0]
    ]
    
    # Przechwytujemy wyjątek za pomocą pytest.raises
    with pytest.raises(ValueError) as exc_info:
        positions_to_tensor(positions_2d)
        
    # Opcjonalnie: możemy sprawdzić, czy komunikat błędu zawiera kluczowe słowa
    assert "where N is a number of drones, and 3 represents coordinates" in str(exc_info.value)
    assert "Current shape: (2, 2)" in str(exc_info.value)

def test_invalid_1d_list():
    """
    Edge case: Podano płaską listę (1D) zamiast zagnieżdżonej (List[List]).
    Dla drona to tylko [x, y, z], a powinno być [[x, y, z]].
    Oczekiwany wyjątek: ValueError.
    """
    positions_1d = [1.0, 2.0, 3.0]
    
    with pytest.raises(ValueError) as exc_info:
        positions_to_tensor(positions_1d)
        
    assert "Current shape: (3,)" in str(exc_info.value)

def test_empty_list():
    """
    Edge case: Przekazano pustą listę.
    Wymiar zwrócony przez numpy to (0,), co powinno odpalić walidację.
    """
    with pytest.raises(ValueError):
        positions_to_tensor([])

def test_ragged_list_creation():
    """
    Edge case: Przekazano "poszarpaną" listę (różna liczba współrzędnych dla różnych dronów).
    W nowszych wersjach NumPy samo tworzenie tablicy z nieregularnych kształtów rzuca ValueError,
    zanim jeszcze kod dojdzie do Twojej własnej walidacji 'tensor_positions.ndim'.
    """
    ragged_positions = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0]        # Brakuje Z dla drugiego drona
    ]
    
    with pytest.raises(ValueError):
        positions_to_tensor(ragged_positions)