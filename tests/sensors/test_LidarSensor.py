import numpy as np
import pytest
from unittest.mock import patch

from src.sensors.LidarSensor import LidarSensor

@pytest.fixture(autouse=True)
def reset_lidar_sensor():
    """Zapewnia czysty stan globalny klasy przed każdym testem."""
    LidarSensor._base_ray_directions = None
    LidarSensor._num_rays = 0
    yield

@pytest.fixture
def mock_pybullet():
    """Tworzy mocka dla funkcji z pybullet używanych w klasie."""
    with patch("src.sensors.LidarSensor.p") as p_mock:
        # Zabezpieczenie dla getBaseVelocity, by zwracało liniową i kątową prędkość
        p_mock.getBaseVelocity.return_value = ([2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        # Fałszywy błąd dla bloków try-except (gdy obiekt to siatka statyczna)
        p_mock.error = Exception
        yield p_mock


def test_compute_ray_directions():
    """Weryfikuje poprawność generowania wektorów na podstawie koncentrycznych pierścieni."""
    sensor = LidarSensor(physics_client_id=0)
    
    # 1+4+10+18+26+34+42+50+58+66+74+84 = 467
    expected_num_rays = 467 
    
    assert sensor._num_rays == expected_num_rays
    assert sensor._base_ray_directions.shape == (expected_num_rays, 3)
    
    # Sprawdzenie czy wektory bazowe są unormowane
    norms = np.linalg.norm(sensor._base_ray_directions, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_scan_single_drone(mock_pybullet):
    """Weryfikuje czy pojedynczy skan prawidłowo obraca promienie i parsuje trafienia."""
    drone_pos = np.array([10.0, 10.0, 5.0])
    sensor = LidarSensor(physics_client_id=1)
    
    # Symulacja odpowiedzi PyBullet (objectUniqueId, linkIndex, hitFraction, hitPosition, hitNormal)
    # 1. ray: trafienie w obiekt o ID 42 (fraction=0.5 -> 50m)
    mock_results = [(42, -1, 0.5, [10.0, 60.0, 5.0], [0.0, -1.0, 0.0])]
    # Uzupełnijmy resztę promieniami (pudłami) by zachować spójność z ilością 467 promieni
    mock_results.extend([(-1, -1, 1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])] * (sensor._num_rays - 1))
    
    mock_pybullet.rayTestBatch.return_value = mock_results
    
    hits = sensor.scan(drone_pos)
    
    assert len(hits) == 1
    hit = hits[0]
    assert hit.object_id == 42
    assert hit.distance == 50.0  # 0.5 * MAX_RANGE
    np.testing.assert_array_equal(hit.hit_position, np.array([10.0, 60.0, 5.0]))
    
    # Sprawdzenie czy pobrano prędkość z silnika
    mock_pybullet.getBaseVelocity.assert_called_once_with(42, physicsClientId=1)
    np.testing.assert_array_equal(hit.velocity, np.array([2.0, 0.0, 0.0]))
    
    mock_pybullet.rayTestBatch.assert_called_once()


def test_batch_ray_test_multiple_drones(mock_pybullet):
    """Weryfikacja wektorów generowanych do rayTestBatch dla wielu dronów z uwzględnieniem obrotu."""
    sensor = LidarSensor(physics_client_id=0)

    # 3 drony w różnych pozycjach
    positions = np.array([
        [0.0, 0.0, 1.0],
        [10.0, 0.0, 2.0],
        [20.0, 0.0, 3.0]
    ])

    # Zerowe kwaterniony dla uproszczenia
    quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (3, 1))

    mock_pybullet.rayTestBatch.return_value = []

    LidarSensor.batch_ray_test(positions, physics_client_id=0, orientations_quat=quats)

    mock_pybullet.rayTestBatch.assert_called_once()
    _, kwargs = mock_pybullet.rayTestBatch.call_args

    ray_from = np.array(kwargs["rayFromPositions"])
    ray_to = np.array(kwargs["rayToPositions"])

    expected_total_rays = 3 * sensor._num_rays
    assert ray_from.shape == (expected_total_rays, 3)
    assert ray_to.shape == (expected_total_rays, 3)

    # Weryfikacja punktów początkowych pierwszego i trzeciego drona
    expected_ray_from_0 = np.tile(positions[0], (sensor._num_rays, 1))
    np.testing.assert_array_equal(ray_from[:sensor._num_rays], expected_ray_from_0)

    expected_ray_from_2 = np.tile(positions[2], (sensor._num_rays, 1))
    np.testing.assert_array_equal(ray_from[2 * sensor._num_rays:], expected_ray_from_2)


def test_draw_debug_lines_first_draw(mock_pybullet):
    """Testuje rysowanie promieni LiDARa dla pierwszej iteracji."""
    sensor = LidarSensor(physics_client_id=0)
    drone_pos = np.array([0.0, 0.0, 0.0])
    
    # Tworzymy wektory kierunkowe aby nie rysować w kosmosie
    sensor._last_rotated_offsets = sensor._base_ray_directions * sensor.MAX_RANGE
    
    # Symulacja 2 promieni: 1 trafienie, 1 pudło
    mock_raw_results = [
        (-1, -1, 1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        (10, -1, 0.2, [5.0, 5.0, 0.0], [0.0, 0.0, 0.0])
    ]
    sensor._last_raw_results = mock_raw_results
    
    # Zastąp fałszywym id zwracanym przez pybullet
    mock_pybullet.addUserDebugLine.side_effect = lambda *args, **kwargs: 100 + len(sensor._debug_ray_ids)
    
    sensor.draw_debug_lines(drone_pos)
    
    assert mock_pybullet.addUserDebugLine.call_count == 2
    assert len(sensor._debug_ray_ids) == 2
    
    # Sprawdzenie kolorów (zielony dla pudła, czerwony dla trafienia)
    calls = mock_pybullet.addUserDebugLine.call_args_list
    assert calls[0].kwargs["lineColorRGB"] == [0.0, 1.0, 0.0]
    assert calls[1].kwargs["lineColorRGB"] == [1.0, 0.0, 0.0]
    assert "replaceItemUniqueId" not in calls[0].kwargs


def test_draw_debug_lines_replace(mock_pybullet):
    """Zastępowanie starych promieni podczas poruszania (optymalizacja p.addUserDebugLine)."""
    sensor = LidarSensor(physics_client_id=0)
    drone_pos = np.array([0.0, 0.0, 0.0])
    
    sensor._last_rotated_offsets = sensor._base_ray_directions * sensor.MAX_RANGE
    sensor._last_raw_results = [(-1, -1, 1.0, [0, 0, 0], [0, 0, 0])]
    sensor._debug_ray_ids = [999]  
    
    sensor.draw_debug_lines(drone_pos)
    
    mock_pybullet.addUserDebugLine.assert_called_once()
    assert "replaceItemUniqueId" in mock_pybullet.addUserDebugLine.call_args.kwargs
    assert mock_pybullet.addUserDebugLine.call_args.kwargs["replaceItemUniqueId"] == 999


def test_clear_debug_lines(mock_pybullet):
    """Sprawdzenie czyszczenia zasobów z C++ API."""
    sensor = LidarSensor(physics_client_id=0)
    sensor._debug_ray_ids = [10, 11, 12]
    
    sensor.clear_debug_lines()
    
    assert mock_pybullet.removeUserDebugItem.call_count == 3
    assert len(sensor._debug_ray_ids) == 0