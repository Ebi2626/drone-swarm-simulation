import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import importlib
TARGET_MODULE = "src.algorithms.abstraction.trajectory.strategies.nsga3_utils.core_math"
MathEngine = importlib.import_module(TARGET_MODULE)

# ==========================================
# FIXTURES (Symulacja CuPy)
# ==========================================

@pytest.fixture
def mock_cupy_env(mocker):
    """
    Fixture symulujący obecność biblioteki CuPy.
    Tworzy atrapy modułu 'cp', tablicy 'cp.ndarray' oraz funkcji 'cp.asarray'/'cp.asnumpy'.
    """
    mock_cp = MagicMock()
    
    # Symulacja klasy bazowej dla tablic CuPy
    class DummyCuPyArray:
        pass
    mock_cp.ndarray = DummyCuPyArray
    
    # Symulacja funkcji przenoszących
    mock_cp.asarray = MagicMock(side_effect=lambda x: DummyCuPyArray())
    mock_cp.asnumpy = MagicMock(side_effect=lambda x: np.array([1, 2, 3])) # Zwraca stałą by uprościć test

    # Zastępujemy flagę HAS_CUPY i moduł cp w testowanym pliku
    mocker.patch(f"{TARGET_MODULE}.HAS_CUPY", True)
    mocker.patch(f"{TARGET_MODULE}.cp", mock_cp)
    
    return mock_cp, DummyCuPyArray

@pytest.fixture
def no_cupy_env(mocker):
    """
    Fixture symulujący CAŁKOWITY brak CuPy (fallback do CPU).
    """
    mocker.patch(f"{TARGET_MODULE}.HAS_CUPY", False)
    mocker.patch(f"{TARGET_MODULE}.cp", None)

# ==========================================
# TESTY: get_xp() (Dynamiczny wybór)
# ==========================================

def test_get_xp_returns_numpy_by_default(no_cupy_env):
    """
    Intencja: Kiedy CuPy nie ma w systemie, get_xp MUSI zawsze zwrócić NumPy,
    niezależnie od tego, co mu przekażemy.
    """
    array = np.array([1, 2, 3])
    xp = MathEngine.get_xp(array)
    
    assert xp is np

def test_get_xp_returns_cupy_when_array_is_cupy(mock_cupy_env):
    """
    Intencja: Mając CuPy, jeśli wejściem jest gpu_array (cp.ndarray),
    get_xp powinno wyłapać typ i zwrócić moduł CuPy.
    """
    mock_cp, DummyCuPyArray = mock_cupy_env
    gpu_array = DummyCuPyArray()
    
    xp = MathEngine.get_xp(gpu_array)
    
    assert xp is mock_cp

def test_get_xp_returns_numpy_for_regular_array_even_with_cupy(mock_cupy_env):
    """
    Edge case: Nawet jeśli CuPy JEST zainstalowane, jeśli podamy zwykłą
    tablicę Numpy, get_xp powinien zwrócić numpy.
    """
    cpu_array = np.array([1, 2, 3])
    xp = MathEngine.get_xp(cpu_array)
    
    assert xp is np

# ==========================================
# TESTY: to_device() (Transfer danych CPU <-> GPU)
# ==========================================

def test_to_device_cpu_to_cpu(no_cupy_env):
    """ Intencja: Przeniesienie z numpy do numpy powinno zostawić tablicę jako numpy. """
    input_list = [1, 2, 3]
    result = MathEngine.to_device(input_list, np)
    
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 2, 3])

def test_to_device_fallback_when_cupy_missing(no_cupy_env):
    """
    Edge case: Użytkownik żąda przeniesienia na GPU (docelowy xp=cp), ale
    w systemie nie ma CuPy. Zgodnie z kodem ma to cicho spaść (fallback) do np.asarray.
    """
    # Testujemy za pomocą fałszywego wskaźnika 'cp', mimo że HAS_CUPY=False
    # Normalnie 'cp' mogłoby tu być przekazane jako zwykły obiekt
    dummy_target = MagicMock() 
    
    # Nadpisujemy logikę Twojej funkcji, byśmy mogli wywołać target_xp == cp 
    # (podmieniamy cp w module lokalnie tylko na czas tego testu, żeby if zadziałał)
    with patch(f"{TARGET_MODULE}.cp", dummy_target):
        input_list = [1, 2, 3]
        result = MathEngine.to_device(input_list, dummy_target)
        
        # Oczekujemy cichego fallbacku do Numpy
        assert isinstance(result, np.ndarray)

def test_to_device_numpy_to_cupy(mock_cupy_env):
    """
    Intencja: Transfer CPU -> GPU przy zainstalowanym środowisku.
    Powinno odpalić cp.asarray().
    """
    mock_cp, DummyCuPyArray = mock_cupy_env
    cpu_array = np.array([1, 2, 3])
    
    result = MathEngine.to_device(cpu_array, mock_cp)
    
    # Sprawdzamy czy wywołano cp.asarray z naszą tablicą
    mock_cp.asarray.assert_called_once()
    # Sprawdzamy, czy funkcja zwróciła atrapę tablicy GPU
    assert isinstance(result, DummyCuPyArray)

def test_to_device_cupy_to_numpy(mock_cupy_env):
    """
    Intencja: Transfer GPU -> CPU.
    Jeśli podamy tablicę CuPy i poprosimy o NumPy, powinno odpalić cp.asnumpy().
    """
    mock_cp, DummyCuPyArray = mock_cupy_env
    gpu_array = DummyCuPyArray()
    
    result = MathEngine.to_device(gpu_array, np)
    
    mock_cp.asnumpy.assert_called_once_with(gpu_array)
    # W fixture ustawiliśmy, że mock_cp.asnumpy zawsze zwraca np.array([1,2,3])
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 2, 3])