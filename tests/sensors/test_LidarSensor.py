import numpy as np
import pytest
from unittest.mock import patch

from src.sensors.LidarSensor import LidarSensor

@pytest.fixture(autouse=True)
def reset_lidar_sensor():
    """Zapewnia czysty stan globalny klasy przed każdym testem."""
    LidarSensor._ray_directions = None
    LidarSensor._ray_offsets = None
    LidarSensor._num_rays = 0
    yield


def test_compute_ray_directions():
    """Weryfikuje poprawność generowania wektorów na podstawie zadanych kątów sferycznych."""
    sensor = LidarSensor(physics_client_id=0)
    
    expected_num_rays = LidarSensor.NUM_HORIZONTAL * len(LidarSensor.ELEVATION_LAYERS_DEG)
    assert sensor._num_rays == expected_num_rays
    assert sensor._ray_directions.shape == (expected_num_rays, 3)
    assert sensor._ray_offsets.shape == (expected_num_rays, 3)
    
    # Sprawdzenie czy wektory są unormowane
    norms = np.linalg.norm(sensor._ray_directions, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    # Sprawdzenie czy offsety są przeskalowane do MAX_RANGE
    offsets_norms = np.linalg.norm(sensor._ray_offsets, axis=1)
    np.testing.assert_allclose(offsets_norms, LidarSensor.MAX_RANGE, rtol=1e-5)


@patch("src.sensors.LidarSensor.p.rayTestBatch")
def test_scan_single_drone(mock_ray_test_batch):
    """Weryfikuje czy pojedynczy skan prawidłowo filtruje pudła i parsuje trafienia."""
    drone_pos = np.array([10.0, 10.0, 5.0])
    sensor = LidarSensor(physics_client_id=1)
    
    # Symulacja odpowiedzi PyBullet:
    # 1. ray: pudło (-1)
    # 2. ray: trafienie w obiekt o ID 42 (fraction=0.5 -> 50m)
    mock_results = [
        (-1, -1, 1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        (42, -1, 0.5, [10.0, 60.0, 5.0], [0.0, -1.0, 0.0])
    ] 
    # Uzupełnijmy resztę promieniami (pudłami) by zachować spójność z ilością 108 promieni
    mock_results.extend([(-1, -1, 1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])] * (sensor._num_rays - 2))
    
    mock_ray_test_batch.return_value = mock_results
    
    hits = sensor.scan(drone_pos)
    
    assert len(hits) == 1
    hit = hits[0]
    assert hit.object_id == 42
    assert hit.distance == 50.0  # 0.5 * MAX_RANGE
    np.testing.assert_array_equal(hit.hit_position, np.array([10.0, 60.0, 5.0]))
    
    mock_ray_test_batch.assert_called_once()
    _, kwargs = mock_ray_test_batch.call_args
    assert "physicsClientId" in kwargs
    assert kwargs["physicsClientId"] == 1
    assert len(kwargs["rayFromPositions"]) == sensor._num_rays


@patch("src.sensors.LidarSensor.p.rayTestBatch")
def test_batch_ray_test_multiple_drones(mock_ray_test_batch):
    """Corner-case: weryfikacja shape'ów danych wejściowych w operacjach macierzowych batch-a."""
    sensor = LidarSensor(physics_client_id=0)  # inicjalizuje offsety

    # 3 drony na różnych wysokościach
    positions = np.array([
        [0.0, 0.0, 1.0],
        [10.0, 0.0, 2.0],
        [20.0, 0.0, 3.0]
    ])

    mock_ray_test_batch.return_value = []

    LidarSensor.batch_ray_test(positions, physics_client_id=0)

    mock_ray_test_batch.assert_called_once()
    _, kwargs = mock_ray_test_batch.call_args

    ray_from = np.array(kwargs["rayFromPositions"])
    ray_to = np.array(kwargs["rayToPositions"])

    expected_total_rays = 3 * sensor._num_rays
    assert ray_from.shape == (expected_total_rays, 3)
    assert ray_to.shape == (expected_total_rays, 3)

    # Weryfikacja broadcastingu - pierwszy dron (powielamy 1D wektor do macierzy (108, 3))
    expected_ray_from_0 = np.tile(positions[0], (sensor._num_rays, 1))
    np.testing.assert_array_equal(ray_from[:sensor._num_rays], expected_ray_from_0)

    # Weryfikacja broadcastingu - trzeci dron
    expected_ray_from_2 = np.tile(positions[2], (sensor._num_rays, 1))
    np.testing.assert_array_equal(ray_from[2 * sensor._num_rays:], expected_ray_from_2)

    # Weryfikacja offsetów
    vector_diffs = ray_to - ray_from
    repeated_offsets = np.tile(sensor._ray_offsets, (3, 1))
    np.testing.assert_allclose(vector_diffs, repeated_offsets, rtol=1e-5, atol=1e-7)

@patch("src.sensors.LidarSensor.p.addUserDebugLine")
def test_draw_debug_lines_first_draw(mock_add_line):
    """Testuje rysowanie promieni LiDARa dla pierwszej iteracji."""
    sensor = LidarSensor(physics_client_id=0)
    drone_pos = np.array([0.0, 0.0, 0.0])
    
    # Sztuczne ustawienie wyników - 1 pudło, 1 trafienie
    mock_raw_results = [
        (-1, -1, 1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        (10, -1, 0.2, [5.0, 5.0, 0.0], [0.0, 0.0, 0.0])
    ]
    sensor._last_raw_results = mock_raw_results
    
    # Zastąp fałszywym id zwracanym przez pybullet
    mock_add_line.side_effect = lambda *args, **kwargs: 100 + len(sensor._debug_ray_ids)
    
    sensor.draw_debug_lines(drone_pos)
    
    assert mock_add_line.call_count == 2
    assert len(sensor._debug_ray_ids) == 2
    
    # Sprawdzenie kolorów (zielony dla pudła, czerwony dla trafienia)
    calls = mock_add_line.call_args_list
    assert calls[0].kwargs["lineColorRGB"] == [0.0, 1.0, 0.0]
    assert calls[1].kwargs["lineColorRGB"] == [1.0, 0.0, 0.0]
    # W pierwszej iteracji nie wysyłamy replaceItemUniqueId
    assert "replaceItemUniqueId" not in calls[0].kwargs


@patch("src.sensors.LidarSensor.p.addUserDebugLine")
def test_draw_debug_lines_replace(mock_add_line):
    """Corner-case: zastępowanie starych promieni podczas poruszania (optymalizacja p.addUserDebugLine)."""
    sensor = LidarSensor(physics_client_id=0)
    drone_pos = np.array([0.0, 0.0, 0.0])
    
    sensor._last_raw_results = [(-1, -1, 1.0, [0, 0, 0], [0, 0, 0])]
    sensor._debug_ray_ids = [999]  # Symulacja że już raz narysowano promienie
    
    sensor.draw_debug_lines(drone_pos)
    
    mock_add_line.assert_called_once()
    assert "replaceItemUniqueId" in mock_add_line.call_args.kwargs
    assert mock_add_line.call_args.kwargs["replaceItemUniqueId"] == 999


@patch("src.sensors.LidarSensor.p.removeUserDebugItem")
def test_clear_debug_lines(mock_remove_item):
    """Sprawdzenie czyszczenia zasobów z C++ API."""
    sensor = LidarSensor(physics_client_id=0)
    sensor._debug_ray_ids = [10, 11, 12]
    
    sensor.clear_debug_lines()
    
    assert mock_remove_item.call_count == 3
    assert len(sensor._debug_ray_ids) == 0