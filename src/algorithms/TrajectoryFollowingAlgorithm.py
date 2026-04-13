import numpy as np
from src.algorithms.BaseAlgorithm import BaseAlgorithm
from src.sensors.LidarSensor import LidarSensor, LidarHit
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
import matplotlib.pyplot as plt

# Pamiętaj, aby upewnić się, że ścieżka importu jest poprawna w Twoim środowisku
from src.trajectory.BSplineTrajectory import BSplineTrajectory

class TrajectoryFollowingAlgorithm(BaseAlgorithm):
    _debug_file = open("drone_debug.log", "w")

    def __init__(self, parent, num_drones, params=None):
        super().__init__(parent, num_drones, params)
        self.controllers = [
            DSLPIDControl(drone_model=DroneModel.CF2X)
            for _ in range(num_drones)
        ]
        self._cached_trajectories = None
        self._lidars: list[LidarSensor] | None = None
        self.latest_scans: list[list[LidarHit]] = [[] for _ in range(num_drones)]

        # Parametry ogólne
        self._ctrl_timestep = 1.0 / self.params.get("ctrl_freq", 48)
        self.hover_duration = self.params.get("hover_duration", 3.0)
        self.finish_radius = self.params.get("finish_radius", 0.5)

        # Nowe parametry dla B-Spline i Profilu Trapezowego
        self.cruise_speed = self.params.get("cruise_speed", 8.0)
        self.max_accel = self.params.get("max_accel", 2.0)
        self.safe_radius = self.params.get("safe_radius", 0.4)

    def init_lidars(self, physics_client_id: int) -> None:
        """Inicjalizuje sensory lidar po utworzeniu świata PyBullet."""
        self._physics_client_id: int = physics_client_id
        self._lidars = [
            LidarSensor(physics_client_id) for _ in range(self.num_drones)
        ]

    def draw_lidar_rays(self, current_states: list, tracked_drone_idx: int) -> None:
        """Rysuje promienie lidaru tylko dla śledzonego drona w GUI PyBullet.
        
        Args:
            current_states: Lista stanów wszystkich dronów.
            tracked_drone_idx: Indeks drona, którego lidar ma być wizualizowany.
        """
        if self._lidars is None:
            return
            
        # Zabezpieczenie przed błędnym indeksem
        if not (0 <= tracked_drone_idx < self.num_drones):
            return

        # Rysujemy promienie tylko dla wybranego drona
        self._lidars[tracked_drone_idx].draw_debug_lines(current_states[tracked_drone_idx][0:3])
        
        # Czyścimy promienie dla pozostałych dronów, aby uniknąć "zamrożonych" 
        # linii w przypadku zmiany śledzonego drona w trakcie symulacji
        for i in range(self.num_drones):
            if i != tracked_drone_idx:
                self._lidars[i].clear_debug_lines()

    def _prepare_trajectories(self):
        """Pobiera waypointy, generuje splajny i sprawdza kolizje w pętli."""
        trajectories_from_nsga = self.parent.trajectories
        if trajectories_from_nsga is None:
            raise ValueError("Brak wyników optymalizacji w parent.trajectories!")
        
        # Przekształcenie na listę, ponieważ podczas naprawy drony mogą mieć różną liczbę waypointów
        raw_waypoints = [np.copy(path) for path in trajectories_from_nsga]
        
        max_retries = 5
        spline_trajectories = []
        
        for attempt in range(max_retries):
            # 1. Budowa wstępnych krzywych B-Spline dla każdego drona
            spline_trajectories = [
                BSplineTrajectory(raw_waypoints[i], self.cruise_speed, self.max_accel)
                for i in range(self.num_drones)
            ]
            
            # 2. Weryfikacja bezkolizyjności w dziedzinie czasu
            collision = self._verify_trajectories(spline_trajectories)
            
            if not collision:
                print(f"[Info] Trajektorie bezpieczne po {attempt} poprawkach.")
                break
            else:
                print(f"[Ostrzeżenie] Wykryto kolizję w próbie {attempt}. Naprawiam...")
                # 3. Naprawa (dodanie dodatkowych węzłów w miejscu kolizji)
                raw_waypoints = self._repair_waypoints(raw_waypoints, collision)
                
                if attempt == max_retries - 1:
                    print("[Ostrzeżenie] Osiągnięto limit poprawek, trajektoria może nie być w 100% bezpieczna.")

        # Wizualizacja końcowego wyniku
        self._visualize_trajectories(spline_trajectories)

        return spline_trajectories 

    def _verify_trajectories(self, splines) -> tuple:
        """Próbkuje czas i sprawdza odległości między wszystkimi dronami."""
        max_time = max(spline.total_duration for spline in splines)
        dt = 0.1  # Próbkowanie sprawdzania kolizji co 0.1s
        times = np.arange(0, max_time + dt, dt)
        
        for t in times:
            positions = [splines[i].get_state_at_time(t)[0] for i in range(self.num_drones)]
            
            for i in range(self.num_drones):
                for j in range(i + 1, self.num_drones):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < self.safe_radius:
                        # Zwracamy pierwszą znalezioną kolizję: (id_drona_1, id_drona_2, pos_1, pos_2)
                        return (i, j, positions[i], positions[j])
        return ()

    def _repair_waypoints(self, raw_waypoints, collision) -> list:
        """Znajduje odcinek najbliżej miejsca kolizji i wstawia punkt pośrodku (midpoint)."""
        new_waypoints = [path.copy() for path in raw_waypoints]
        i, j, pos_i, pos_j = collision

        # Naprawa trasy dla drona i
        new_waypoints[i] = self._insert_midpoint_near(new_waypoints[i], pos_i)
        # Naprawa trasy dla drona j
        new_waypoints[j] = self._insert_midpoint_near(new_waypoints[j], pos_j)
        
        return new_waypoints

    def _insert_midpoint_near(self, waypoints, target_pos):
        """Wstawia nowy waypoint dokładnie w połowie odcinka najbliższego kolizji."""
        dists = np.linalg.norm(waypoints - target_pos, axis=1)
        closest_idx = int(np.argmin(dists))
        
        if closest_idx == 0:
            idx_to_split = 0
        elif closest_idx == len(waypoints) - 1:
            idx_to_split = len(waypoints) - 2
        else:
            dist_prev = np.linalg.norm(waypoints[closest_idx - 1] - target_pos)
            dist_next = np.linalg.norm(waypoints[closest_idx + 1] - target_pos)
            idx_to_split = closest_idx - 1 if dist_prev < dist_next else closest_idx
            
        midpoint = (waypoints[idx_to_split] + waypoints[idx_to_split + 1]) / 2.0
        return np.insert(waypoints, idx_to_split + 1, midpoint, axis=0)

    def _visualize_trajectories(self, splines):
        """Rysuje naprawione punkty kontrolne oraz wygładzoną trajektorię."""
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        
        for i, spline in enumerate(splines):
            color = colors[i % len(colors)]
            
            # Punkty kontrolne (waypointy po naprawie)
            wp = spline.waypoints
            ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], color=color, alpha=0.3, s=10)
            ax.scatter(*wp[0], marker='o', color=color) # Start
            ax.scatter(*wp[-1], marker='x', color=color) # Cel
            
            # Wygładzona krzywa
            times = np.linspace(0, spline.total_duration, 200)
            smooth_path = np.array([spline.get_state_at_time(t)[0] for t in times])
            ax.plot(smooth_path[:, 0], smooth_path[:, 1], smooth_path[:, 2], 
                    label=f'Dron {i} (B-Spline)', color=color)

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend()
        plt.savefig("trajectory_preview_bspline.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def compute_actions(self, current_states, current_time):
        if self._cached_trajectories is None:
            self._cached_trajectories = self._prepare_trajectories()

        # Zbiorczy skan lidarowy — jedno wywołanie rayTestBatch dla N dronów
        if self._lidars is not None:
            positions = np.array([s[0:3] for s in current_states], dtype=np.float64)
            num_rays = LidarSensor._num_rays
            all_results = LidarSensor.batch_ray_test(positions, self._physics_client_id)
            for i in range(self.num_drones):
                chunk = all_results[i * num_rays : (i + 1) * num_rays]
                self.latest_scans[i] = self._lidars[i].process_batch_results(chunk)

        actions = []
        for i in range(self.num_drones):
            state = current_states[i]
            spline_traj = self._cached_trajectories[i]

            # Czas liczony od startu misji (po zakończeniu fazy hover)
            flight_time = current_time - self.hover_duration

            if flight_time < 0:
                # --- Hover przed rozpoczęciem ruchu ---
                target_pos, _ = spline_traj.get_state_at_time(0.0)
                target_vel = np.zeros(3)
            else:
                # --- Lot ciągły z wykorzystaniem profilu trapezowego ---
                target_pos, target_vel = spline_traj.get_state_at_time(flight_time)

            action, _, _ = self.controllers[i].computeControlFromState(
                control_timestep=self._ctrl_timestep,
                state=state,
                target_pos=target_pos,
                target_vel=target_vel,
                target_rpy=np.array([0, 0, 0])
            )
            actions.append(action)

        return np.array(actions)

    @property
    def all_finished(self) -> bool:
        if self._cached_trajectories is None:
            return False
            
        # Zakończenie misji jest weryfikowane względem ostatniego waypointu na ścieżce
        return all(
            np.linalg.norm(
                self._cached_trajectories[i].waypoints[-1] - self.parent.current_states[i][0:3]
            ) < self.finish_radius
            for i in range(self.num_drones)
        )