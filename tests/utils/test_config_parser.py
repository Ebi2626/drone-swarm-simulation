import pytest
import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# UWAGA: Podmień "src.utils.sanitizer" na ścieżkę do Twojego pliku
from src.utils.config_parser import (
    _parse_drone_model,
    _parse_physics,
    _parse_coordinates,
    sanitize_init_params
)

# ==========================================
# TESTY HELPERÓW
# ==========================================

def test_parse_drone_model_valid_inputs():
    """ Intencja: Sprawdzenie poprawnych typów (Enum lub String) dla modelu drona. """
    # 1. Przekazanie gotowego enuma
    assert _parse_drone_model(DroneModel.CF2X) == DroneModel.CF2X
    
    # 2. Przekazanie poprawnego stringa
    # Założyłem, że DroneModel posiada typ 'RACE' (powszechny w tej bibliotece)
    assert _parse_drone_model("RACE") == DroneModel.RACE

def test_parse_drone_model_edge_cases(capsys):
    """ Edge cases: Złe stringi, literówki, None. Oczekiwany fallback to CF2X. """
    # 1. Nieznany string (literówka)
    assert _parse_drone_model("UNDEFINED_DRONE") == DroneModel.CF2X
    
    # Sprawdzenie, czy wypisało się ostrzeżenie w konsoli
    captured = capsys.readouterr()
    assert "[WARN] Unknown drone model 'UNDEFINED_DRONE', using default CF2X." in captured.out
    
    # 2. Zupełnie błędny typ (np. liczba albo None)
    assert _parse_drone_model(None) == DroneModel.CF2X
    assert _parse_drone_model(123) == DroneModel.CF2X

def test_parse_physics_valid_inputs():
    """ Intencja: Sprawdzenie poprawnych typów (Enum lub String) dla fizyki. """
    assert _parse_physics(Physics.DYN) == Physics.DYN
    assert _parse_physics("DYN") == Physics.DYN

def test_parse_physics_edge_cases(capsys):
    """ Edge cases: Złe stringi, literówki, None. Oczekiwany fallback to PYB. """
    assert _parse_physics("MAGIC_PHYSICS") == Physics.PYB
    
    # Sprawdzenie, czy ostrzeżenie zadziałało
    captured = capsys.readouterr()
    assert "[WARN] Unknown physics 'MAGIC_PHYSICS', using default PYB." in captured.out
    
    assert _parse_physics(None) == Physics.PYB
    assert _parse_physics(3.14) == Physics.PYB

def test_parse_coordinates():
    """ 
    Intencja: Sprawdzenie konwersji różnych struktur na ujednolicony np.ndarray, 
    co np. naprawia znany błąd Hydra ListConfig (zgodnie z komentarzem w kodzie).
    """
    # 1. Podano None - brak parametrów
    assert _parse_coordinates(None) is None
    
    # 2. Standardowa zagnieżdżona lista Pythonowa
    list_coords = [[0, 0, 0], [1.5, 2.0, -1.0]]
    result = _parse_coordinates(list_coords)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([[0.0, 0.0, 0.0], [1.5, 2.0, -1.0]]))
    
    # 3. Krotka (Tuple)
    tuple_coords = (1, 2, 3)
    result = _parse_coordinates(tuple_coords)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))
    
    # 4. Przekazanie gotowego numpy arraya - upewniamy się, że nie zepsuje
    np_coords = np.array([10, 20, 30])
    result = _parse_coordinates(np_coords)
    np.testing.assert_array_equal(result, np_coords)

# ==========================================
# TEST INTEGRACYJNY GŁÓWNEJ FUNKCJI
# ==========================================

def test_sanitize_init_params():
    """
    Intencja: Sprawdzenie czy główna funkcja poprawnie deleguje zadania do 
    helperów i zwraca zsanitizowaną krotkę 5 elementów w odpowiedniej kolejności.
    """
    raw_start = [[0, 0, 1]]
    raw_end = [[10, 10, 5]]
    raw_rpys = [[0, 0, 0]]
    
    model, physics, start, end, rpys = sanitize_init_params(
        drone_model="RACE",
        physics="DYN",
        start_xyzs=raw_start,
        end_xyzs=raw_end,
        initial_rpys=raw_rpys
    )
    
    # Assert Enums
    assert model == DroneModel.RACE
    assert physics == Physics.DYN
    
    # Assert Numpy conversions
    assert isinstance(start, np.ndarray)
    assert isinstance(end, np.ndarray)
    assert isinstance(rpys, np.ndarray)
    
    np.testing.assert_array_equal(start, np.array(raw_start))
    np.testing.assert_array_equal(end, np.array(raw_end))
    np.testing.assert_array_equal(rpys, np.array(raw_rpys))