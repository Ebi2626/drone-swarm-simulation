from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import pybullet as p
from scipy.spatial.transform import Rotation


@dataclass(slots=True)
class LidarHit:
    object_id: int
    distance: float
    hit_position: NDArray[np.float64]
    ray_direction: NDArray[np.float64]
    velocity: NDArray[np.float64]  # Dodany wektor prędkości [vx, vy, vz]


class LidarSensor:
    """
    Lidar 3D oparty wyłącznie na pybullet.rayTestBatch().
    
    Parametry skanowania (Gęsty, wypełniony stożek - Spotlight FOV):
        - Całkowity kąt rozwarcia stożka: 30 stopni (maksymalne odchylenie 15 stopni).
        - Dystrybucja: 7 koncentrycznych pierścieni o rosnącej gęstości.
        - Łączna liczba promieni: 123.
        - Zasięg maksymalny: 100 m.
    """

    MAX_RANGE: float = 100.0

    _base_ray_directions: NDArray[np.float64] | None = None
    _num_rays: int = 0

    def __init__(self, physics_client_id: int) -> None:
        self._client_id: int = physics_client_id

        if LidarSensor._base_ray_directions is None:
            LidarSensor._base_ray_directions = self._compute_ray_directions()
            LidarSensor._num_rays = LidarSensor._base_ray_directions.shape[0]

        self._last_raw_results: list[tuple] = []
        self._debug_ray_ids: list[int] = []
        
        # Zapamiętujemy obrócone wektory dla debugowania i parsowania
        self._last_rotated_directions: NDArray[np.float64] | None = None
        self._last_rotated_offsets: NDArray[np.float64] | None = None

    @staticmethod
    def _compute_ray_directions() -> NDArray[np.float64]:
        # Definicja gęstych, koncentrycznych pierścieni dla litego stożka.
        # Szczyt stożka ma 30 stopni rozwarcia (maksymalnie 15 stopni od osi centralnej).
        # Taki układ pozwala na gładkie skalowanie sił odpychających w metodzie APF.
        rings = [
            (0.0, 1),     # Promień centralny (w sam środek)
            (0.5, 4),     # Ekstremalnie gęsty rdzeń - chroni przed czołowym uderzeniem małych obiektów
            (1.5, 10),    # Rozdzielczość 1.0 stopnia względem poprzedniego pierścienia
            (3.0, 18),    # Od tego miejsca stały krok co 1.5 stopnia na promieniu
            (4.5, 26),    # Liczba promieni na obwodzie rośnie proporcjonalnie do obwodu pierścienia,
            (6.0, 34),    # co zapewnia stałą gęstość siatki (ok. 1.0 - 1.5 stopnia odstępu) w każdym miejscu.
            (7.5, 42),
            (9.0, 50),
            (10.5, 58),
            (12.0, 66),
            (13.5, 74),
            (15.0, 84)    # Granica FOV (15 stopni odchylenia = 30 stopni całkowitego rozwarcia)
        ]
        # Sumarycznie: 467 promieni na drona. 
        # Dla roju 5 dronów daje to ok. 2335 promieni na krok symulacji.
        # pybullet.rayTestBatch jest wysoce zoptymalizowany pod C++ i przetwarza takie wartości w <1 ms.
        
        # Całkowita liczba promieni: 1 + 6 + 10 + 16 + 24 + 30 + 36 = 123 promienie.
        
        azimuths_deg = []
        elevations_deg = []
        
        # Oś Y to "przód" modelu CF2X. 
        # Przekręcamy bazowy stożek o 90 stopni, by idealnie zgrać go z układem współrzędnych drona.
        FORWARD_AZIMUTH = 90.0 
        
        for radius_deg, num_rays in rings:
            if num_rays == 1:
                azimuths_deg.append(FORWARD_AZIMUTH)
                elevations_deg.append(0.0)
            else:
                # Rozkładamy promienie równomiernie na obwodzie danego pierścienia
                angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
                for angle in angles:
                    azimuths_deg.append(FORWARD_AZIMUTH + radius_deg * np.cos(angle))
                    elevations_deg.append(radius_deg * np.sin(angle))
                    
        azim_rad = np.deg2rad(azimuths_deg)
        elev_rad = np.deg2rad(elevations_deg)

        cos_elev = np.cos(elev_rad)
        return np.column_stack([
            cos_elev * np.cos(azim_rad),  # X
            cos_elev * np.sin(azim_rad),  # Y
            np.sin(elev_rad),             # Z
        ])

    def scan(self, drone_position: NDArray[np.float64], drone_orientation_quat: NDArray[np.float64] | None = None) -> list[LidarHit]:
        """Skan lidarowy z jednego drona uwzględniający jego orientację."""
        origin = np.asarray(drone_position, dtype=np.float64)
        
        # Fallback na wypadek braku podanej orientacji (zachowuje wsteczną kompatybilność API)
        if drone_orientation_quat is None:
            drone_orientation_quat = np.array([0.0, 0.0, 0.0, 1.0])

        # Obracamy bazowe wektory zgodnie z orientacją drona (lub wirtualnego gimbala)
        rot = Rotation.from_quat(drone_orientation_quat)
        self._last_rotated_directions = rot.apply(self._base_ray_directions)
        self._last_rotated_offsets = self._last_rotated_directions * self.MAX_RANGE

        ray_from = np.broadcast_to(origin, (self._num_rays, 3))
        ray_to = origin + self._last_rotated_offsets

        results = p.rayTestBatch(
            rayFromPositions=ray_from.tolist(),
            rayToPositions=ray_to.tolist(),
            physicsClientId=self._client_id,
        )
        return self._parse_raw_results(results)

    def process_batch_results(
        self, 
        raw_results: list[tuple],
        logger=None, 
        current_time: float = 0.0, 
        drone_id: int = -1
    ) -> list[LidarHit]:
        """
        Przetwarza surowe wyniki z zewnętrznego batcha.
        Opcjonalnie automatycznie loguje trafienia do SimulationLogger.
        """
        hits = self._parse_raw_results(raw_results)
        
        if logger is not None:
            for hit in hits:
                logger.log_lidar_hit(current_time, drone_id, hit)
                
        return hits

    def _parse_raw_results(self, results: list[tuple]) -> list[LidarHit]:
        self._last_raw_results = results
        hits: list[LidarHit] = []
        
        dirs_to_use = self._last_rotated_directions if self._last_rotated_directions is not None else self._base_ray_directions

        for i, result in enumerate(results):
            obj_id = result[0]
            if obj_id == -1:
                continue
            
            dir_idx = i % self._num_rays 
            
            # Pobieranie prędkości z silnika fizycznego
            try:
                # p.getBaseVelocity zwraca (linear_velocity, angular_velocity)
                linear_vel, _ = p.getBaseVelocity(obj_id, physicsClientId=self._client_id)
                velocity_vector = np.array(linear_vel, dtype=np.float64)
            except p.error:
                # Zabezpieczenie dla obiektów czysto statycznych (np. siatek bez ciał sztywnych)
                velocity_vector = np.zeros(3, dtype=np.float64)
            
            hits.append(LidarHit(
                object_id=obj_id,
                distance=result[2] * self.MAX_RANGE,
                hit_position=np.array(result[3], dtype=np.float64),
                ray_direction=dirs_to_use[dir_idx],
                velocity=velocity_vector
            ))
        return hits

    @staticmethod
    def batch_ray_test(
        positions: NDArray[np.float64],
        physics_client_id: int,
        orientations_quat: NDArray[np.float64] | None = None
    ) -> list[tuple]:
        n_drones = positions.shape[0]
        num_rays = LidarSensor._num_rays
        origins = positions[:, np.newaxis, :]
        
        # Jeśli brak argumentu, zakłada orientację zerową dla wszystkich dronów
        if orientations_quat is None:
            orientations_quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n_drones, 1))

        all_from = np.broadcast_to(origins, (n_drones, num_rays, 3)).reshape(-1, 3)
        all_to = np.zeros_like(all_from)

        # Obliczamy obrócenie wiązki dla każdego drona osobno
        for i in range(n_drones):
            rot = Rotation.from_quat(orientations_quat[i])
            rotated_dirs = rot.apply(LidarSensor._base_ray_directions)
            offsets = rotated_dirs * LidarSensor.MAX_RANGE
            
            start_idx = i * num_rays
            end_idx = start_idx + num_rays
            all_to[start_idx:end_idx] = all_from[start_idx:end_idx] + offsets

        return p.rayTestBatch(
            rayFromPositions=all_from.tolist(),
            rayToPositions=all_to.tolist(),
            physicsClientId=physics_client_id,
        )

    def draw_debug_lines(self, drone_position: NDArray[np.float64]) -> None:
        if not self._last_raw_results:
            return

        origin = np.asarray(drone_position, dtype=np.float64)
        ray_from = origin.tolist()
        
        # Rysujemy promienie na podstawie poprawnie zrotowanych offsetów
        if self._last_rotated_offsets is not None:
            offsets = self._last_rotated_offsets
        else:
            offsets = self._base_ray_directions * self.MAX_RANGE
            
        miss_endpoints = origin + offsets

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
        for line_id in self._debug_ray_ids:
            p.removeUserDebugItem(line_id, physicsClientId=self._client_id)
        self._debug_ray_ids.clear()