import pytest
import numpy as np
from unittest.mock import patch
from src.utils.pybullet_utils import update_camera_position

# ==========================================
# TESTY FUNKCJI KAMERY
# ==========================================

@patch('src.utils.pybullet_utils.p')
def test_update_camera_position_with_standard_list(mock_p):
    """
    Intencja: Sprawdzenie, czy funkcja poprawnie wycina pierwsze 3 elementy (x, y, z)
    ze standardowej listy Pythonowej i wywołuje PyBulleta z nazwanymi argumentami.
    """
    # Symulujemy stan drona, który zazwyczaj ma więcej niż 3 parametry
    # (np. [x, y, z, qx, qy, qz, qw, vx, vy, vz, ...])
    drone_state = [10.0, 20.0, 30.0, 0.5, 0.5, 0.5, 0.5, 1.0, 2.0, 3.0]
    
    update_camera_position(
        drone_state=drone_state, 
        distance=5.0, 
        yaw_offset=45.0, 
        pitch=-30.0
    )
    
    # Udowadniamy, że PyBullet dostał tylko pozycję docelową [10.0, 20.0, 30.0]
    mock_p.resetDebugVisualizerCamera.assert_called_once_with(
        cameraDistance=5.0,
        cameraYaw=45.0,
        cameraPitch=-30.0,
        cameraTargetPosition=[10.0, 20.0, 30.0]
    )

@patch('src.utils.pybullet_utils.p')
def test_update_camera_position_with_numpy_array(mock_p):
    """
    Edge case: Biblioteka gym_pybullet_drones niemal wszędzie operuje na tablicach NumPy.
    Upewniamy się, że podanie tablicy numpy również zachowa się poprawnie bez rzucania błędów.
    """
    drone_state_np = np.array([-5.0, 15.0, 10.0, 0.0, 0.0, 0.0])
    
    update_camera_position(
        drone_state=drone_state_np, 
        distance=2.5, 
        yaw_offset=90.0, 
        pitch=-45.0
    )
    
    # Pobieramy argumenty, z jakimi została faktycznie wywołana zmockowana funkcja
    args, kwargs = mock_p.resetDebugVisualizerCamera.call_args
    
    # Sprawdzamy zwykłe argumenty
    assert kwargs['cameraDistance'] == 2.5
    assert kwargs['cameraYaw'] == 90.0
    assert kwargs['cameraPitch'] == -45.0
    
    # Gdy operujemy na mockach i NumPy, nie możemy użyć zwykłego assert_called_once_with,
    # ponieważ porównanie macierzy (array == array) wyrzuca błąd "The truth value of an array...".
    # Zamiast tego używamy dedykowanego narzędzia z numpy.testing:
    np.testing.assert_array_equal(kwargs['cameraTargetPosition'], np.array([-5.0, 15.0, 10.0]))

@patch('src.utils.pybullet_utils.p')
def test_update_camera_position_exact_size(mock_p):
    """
    Edge case: Co jeśli ktoś przekaże do funkcji tylko pozycję [x, y, z] zamiast całego stanu?
    Slicing [0:3] w Pythonie poradzi sobie z tym bez błędu "IndexError".
    """
    short_state = [1.0, 2.0, 3.0]
    
    update_camera_position(short_state, 1.0, 0.0, 0.0)
    
    mock_p.resetDebugVisualizerCamera.assert_called_once_with(
        cameraDistance=1.0,
        cameraYaw=0.0,
        cameraPitch=0.0,
        cameraTargetPosition=[1.0, 2.0, 3.0]
    )