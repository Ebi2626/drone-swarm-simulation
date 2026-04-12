from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import pybullet as p


@dataclass(slots=True)
class LidarHit:
    object_id: int
    distance: float
    hit_position: NDArray[np.float64]
    ray_direction: NDArray[np.float64]


class LidarSensor:
    """Lidar 3D oparty wyłącznie na pybullet.rayTestBatch().

    Parametry skanowania:
        - 36 promieni w poziomie (co 10°)
        - 3 warstwy pionowe: -30°, 0°, +30°
        - Łącznie 108 promieni
        - Zasięg maksymalny: 100 m
    """

    NUM_HORIZONTAL: int = 36
    HORIZONTAL_STEP_DEG: float = 10.0
    ELEVATION_LAYERS_DEG: tuple[float, ...] = (-30.0, 0.0, 30.0)
    MAX_RANGE: float = 100.0

    # Kierunki i offsety promieni — wspólne dla wszystkich instancji,
    # obliczane raz przy pierwszej konstrukcji.
    _ray_directions: NDArray[np.float64] | None = None
    _ray_offsets: NDArray[np.float64] | None = None
    _num_rays: int = 0

    def __init__(self, physics_client_id: int) -> None:
        self._client_id: int = physics_client_id

        if LidarSensor._ray_directions is None:
            LidarSensor._ray_directions = self._compute_ray_directions()
            LidarSensor._ray_offsets = LidarSensor._ray_directions * self.MAX_RANGE
            LidarSensor._num_rays = LidarSensor._ray_directions.shape[0]

        self._last_raw_results: list[tuple] = []
        self._debug_ray_ids: list[int] = []

    # -------------------------------------------------------------- #
    #  Prekomputacja kierunków (sferyczne → kartezjańskie)            #
    # -------------------------------------------------------------- #

    @staticmethod
    def _compute_ray_directions() -> NDArray[np.float64]:
        azimuths_deg = np.arange(LidarSensor.NUM_HORIZONTAL) * LidarSensor.HORIZONTAL_STEP_DEG
        elevations_deg = np.array(LidarSensor.ELEVATION_LAYERS_DEG)

        azim_rad = np.deg2rad(azimuths_deg)
        elev_rad = np.deg2rad(elevations_deg)

        elev_grid, azim_grid = np.meshgrid(elev_rad, azim_rad, indexing="ij")
        elev_flat = elev_grid.ravel()
        azim_flat = azim_grid.ravel()

        cos_elev = np.cos(elev_flat)
        return np.column_stack([
            cos_elev * np.cos(azim_flat),
            cos_elev * np.sin(azim_flat),
            np.sin(elev_flat),
        ])

    # -------------------------------------------------------------- #
    #  Skanowanie                                                     #
    # -------------------------------------------------------------- #

    def scan(self, drone_position: NDArray[np.float64]) -> list[LidarHit]:
        """Skan lidarowy z jednego drona (standalone, bez batchingu)."""
        origin = np.asarray(drone_position, dtype=np.float64)

        ray_from = np.broadcast_to(origin, (self._num_rays, 3))
        ray_to = origin + self._ray_offsets

        results = p.rayTestBatch(
            rayFromPositions=ray_from.tolist(),
            rayToPositions=ray_to.tolist(),
            physicsClientId=self._client_id,
        )
        return self._parse_raw_results(results)

    def process_batch_results(self, raw_results: list[tuple]) -> list[LidarHit]:
        """Przetwarza wcześniej pobrane surowe wyniki z zewnętrznego batcha."""
        return self._parse_raw_results(raw_results)

    def _parse_raw_results(self, results: list[tuple]) -> list[LidarHit]:
        self._last_raw_results = results

        hits: list[LidarHit] = []
        for i, result in enumerate(results):
            if result[0] == -1:
                continue
            hits.append(LidarHit(
                object_id=result[0],
                distance=result[2] * self.MAX_RANGE,
                hit_position=np.array(result[3], dtype=np.float64),
                ray_direction=self._ray_directions[i],
            ))
        return hits

    # -------------------------------------------------------------- #
    #  Batch helper — jedno wywołanie rayTestBatch dla N dronów       #
    # -------------------------------------------------------------- #

    @staticmethod
    def batch_ray_test(
        positions: NDArray[np.float64],
        physics_client_id: int,
    ) -> list[tuple]:
        """Jedno wywołanie rayTestBatch dla wielu dronów naraz.

        Args:
            positions: Tablica (N, 3) pozycji dronów.
            physics_client_id: ID klienta PyBullet.

        Returns:
            Surowe wyniki — lista N * num_rays krotek z PyBullet.
        """
        n_drones = positions.shape[0]
        num_rays = LidarSensor._num_rays
        offsets = LidarSensor._ray_offsets

        # (N, 1, 3) + (1, 108, 3) → (N, 108, 3) → (N*108, 3)
        origins = positions[:, np.newaxis, :]
        all_from = np.broadcast_to(origins, (n_drones, num_rays, 3)).reshape(-1, 3)
        all_to = (origins + offsets[np.newaxis, :, :]).reshape(-1, 3)

        return p.rayTestBatch(
            rayFromPositions=all_from.tolist(),
            rayToPositions=all_to.tolist(),
            physicsClientId=physics_client_id,
        )

    # -------------------------------------------------------------- #
    #  Wizualizacja debug                                             #
    # -------------------------------------------------------------- #

    def draw_debug_lines(
        self,
        drone_position: NDArray[np.float64],
    ) -> None:
        """Rysuje/podmienia promienie lidaru w GUI PyBullet.

        Trafienia (hit) → czerwone, do punktu kolizji.
        Pudła (miss) → zielone, do pełnego zasięgu 100 m.
        """
        if not self._last_raw_results:
            return

        origin = np.asarray(drone_position, dtype=np.float64)
        ray_from = origin.tolist()

        # Wektory końcowe dla pudłowanych promieni — jedna alokacja
        miss_endpoints = origin + self._ray_offsets

        miss_color: list[float] = [0.0, 1.0, 0.0]
        hit_color: list[float] = [1.0, 0.0, 0.0]
        first_draw = len(self._debug_ray_ids) == 0

        for ray_idx, result in enumerate(self._last_raw_results):
            if result[0] != -1:
                color = hit_color
                ray_to = list(result[3])
            else:
                color = miss_color
                ray_to = miss_endpoints[ray_idx].tolist()

            if first_draw:
                line_id = p.addUserDebugLine(
                    ray_from, ray_to,
                    lineColorRGB=color,
                    lineWidth=1.0,
                    lifeTime=0,
                    physicsClientId=self._client_id,
                )
                self._debug_ray_ids.append(line_id)
            else:
                p.addUserDebugLine(
                    ray_from, ray_to,
                    lineColorRGB=color,
                    lineWidth=1.0,
                    lifeTime=0,
                    replaceItemUniqueId=self._debug_ray_ids[ray_idx],
                    physicsClientId=self._client_id,
                )

    def clear_debug_lines(self) -> None:
        """Usuwa promienie z GUI PyBullet (przydatne przy zmianie śledzonego drona)."""
        for line_id in self._debug_ray_ids:
            p.removeUserDebugItem(line_id, physicsClientId=self._client_id)
        self._debug_ray_ids.clear()
