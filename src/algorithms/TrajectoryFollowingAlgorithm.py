import numpy as np
from src.algorithms.BaseAlgorithm import BaseAlgorithm

# POPRAWKA: Wszystkie zunifikowane struktury pobieramy z BaseAvoidance
from src.algorithms.avoidance.EvasionContextBuilder import EvasionContextBuilder

from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import KinematicState, ThreatAlert
from src.sensors.LidarSensor import LidarSensor, LidarHit
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
import matplotlib.pyplot as plt
from src.trajectory.BSplineTrajectory import BSplineTrajectory
from scipy.spatial.transform import Rotation

class TrajectoryFollowingAlgorithm(BaseAlgorithm):
    _debug_file = open("drone_debug.log", "w")

    MODE_TRACKING = 0
    MODE_EVASION = 1

    def __init__(self, parent, num_drones, is_obstacle: bool, avoidance_algorithm=None, params=None):
        super().__init__(parent, num_drones, params)
        self._trajectory_start_times = np.zeros(num_drones)
        self.is_obstacle = is_obstacle
        self.avoidance_algorithm = avoidance_algorithm

        self.controllers = [
            DSLPIDControl(drone_model=DroneModel.CF2X)
            for _ in range(num_drones)
        ]

        # Bazowe trajektorie (offline, nigdy zmieniane)
        self._base_trajectories: list[BSplineTrajectory] | None = None
        # Trajektorie uniku (lokalne, niezmiennicze w czasie trwania uniku)
        self._evasion_trajectories: list[BSplineTrajectory | None] = [None] * num_drones

        self._lidars: list[LidarSensor] | None = None
        self.latest_scans: list[list[LidarHit]] = [[] for _ in range(num_drones)]

        # Stan maszyny
        self._flight_modes = np.full(num_drones, self.MODE_TRACKING, dtype=int)
        self._evasion_start_times = np.full(num_drones, -np.inf)
        self._evasion_cooldown = float(self.params.get("evasion_cooldown", 1.0))
        self._last_threat_positions: list[np.ndarray | None] = [None] * num_drones
        self._rejoin_base_arcs = np.zeros(num_drones)
        self._rejoin_points: list[np.ndarray | None] = [None] * num_drones
        # Bazowy czas startu (po hover lub po rejoin) — używany do tracking
        self._tracking_start_times = np.zeros(num_drones)

        # Parametry ruchu
        self._ctrl_timestep = 1.0 / self.params.get("ctrl_freq", 48)
        self.hover_duration = self.params.get("hover_duration", 3.0)
        self.finish_radius = self.params.get("finish_radius", 0.5)
        self.cruise_speed = self.params.get("cruise_speed", 8.0)
        self.max_accel = self.params.get("max_accel", 2.0)
        self.collision_radius = self.params.get("collision_radius", 0.4)
        self.enable_avoidance = self.params.get("enable_avoidance", False)

        # Detekcja zagrożenia
        self._critical_distance = float(self.params.get("critical_distance", 25.0))
        self._trigger_ttc = float(self.params.get("trigger_ttc", 1.5))
        self._trigger_distance = float(self.params.get("trigger_distance", 6.0))
        self._forward_cone_cos = float(self.params.get("forward_cone_cos", 0.5))

        # Rejoin
        av = self.avoidance_algorithm
        self._rejoin_switch_radius = float(
            (av.params.get("rejoin_switch_radius_m", 1.5) if av is not None else 1.5)
        )

    # =========================================================================
    # Inicjalizacja
    # =========================================================================

    def init_lidars(self, physics_client_id: int) -> None:
        if self.is_obstacle:
            return
        self._physics_client_id: int = physics_client_id
        self._lidars = [
            LidarSensor(physics_client_id) for _ in range(self.num_drones)
        ]

    def draw_lidar_rays(self, current_states: list, tracked_drone_idx: int) -> None:
        if self.is_obstacle or self._lidars is None:
            return
        if not (0 <= tracked_drone_idx < self.num_drones):
            return
        self._lidars[tracked_drone_idx].draw_debug_lines(current_states[tracked_drone_idx][0:3])
        for i in range(self.num_drones):
            if i != tracked_drone_idx:
                self._lidars[i].clear_debug_lines()

    def clear_lidar_rays(self) -> None:
        if self._lidars is None:
            return
        for lidar in self._lidars:
            lidar.clear_debug_lines()

    # =========================================================================
    # Przygotowanie bazowych trajektorii
    # =========================================================================

    def _prepare_trajectories(self):
        if self.is_obstacle:
            source = self.parent.drones_trajectories
            if source is None:
                raise ValueError("Brak trajektorii dronów w parent.drones_trajectories!")
            raw_waypoints = [np.flipud(np.copy(path)) for path in source]
            spline_trajectories = [
                BSplineTrajectory(raw_waypoints[i], self.cruise_speed, self.max_accel)
                for i in range(self.num_drones)
            ]
            self._visualize_trajectories(spline_trajectories)
            return spline_trajectories

        source = self.parent.drones_trajectories
        if source is None:
            raise ValueError("Brak wyników optymalizacji w parent.drones_trajectories!")
        raw_waypoints = [np.copy(path) for path in source]
        max_retries = 5
        spline_trajectories = []
        for attempt in range(max_retries):
            spline_trajectories = [
                BSplineTrajectory(raw_waypoints[i], self.cruise_speed, self.max_accel)
                for i in range(self.num_drones)
            ]
            collision = self._verify_trajectories(spline_trajectories)
            if not collision:
                print(f"[Info] Trajektorie bezpieczne po {attempt} poprawkach.")
                break
            print(f"[Ostrzeżenie] Wykryto kolizję w próbie {attempt}. Naprawiam...")
            raw_waypoints = self._repair_waypoints(raw_waypoints, collision)
            if attempt == max_retries - 1:
                print("[Ostrzeżenie] Osiągnięto limit poprawek.")
        self._visualize_trajectories(spline_trajectories)
        return spline_trajectories

    def _verify_trajectories(self, splines) -> tuple:
        max_time = max(spline.total_duration for spline in splines)
        dt = 0.1
        times = np.arange(0, max_time + dt, dt)
        safe_dist = self.collision_radius * 3.0
        for t in times:
            positions = [splines[i].get_state_at_time(t)[0] for i in range(self.num_drones)]
            for i in range(self.num_drones):
                for j in range(i + 1, self.num_drones):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < safe_dist:
                        return (i, j, positions[i], positions[j])
        return ()

    def _repair_waypoints(self, raw_waypoints, collision) -> list:
        new_waypoints = [path.copy() for path in raw_waypoints]
        i, j, pos_i, pos_j = collision
        repel_vec = pos_i - pos_j
        norm = np.linalg.norm(repel_vec)
        if norm < 1e-3:
            repel_vec = np.array([np.random.rand(), np.random.rand(), 0.0])
            norm = np.linalg.norm(repel_vec)
        repel_dir = repel_vec / norm
        push_distance = 2.0
        new_waypoints[i] = self._insert_midpoint_with_offset(new_waypoints[i], pos_i, repel_dir * push_distance)
        new_waypoints[j] = self._insert_midpoint_with_offset(new_waypoints[j], pos_j, -repel_dir * push_distance)
        return new_waypoints

    def _insert_midpoint_with_offset(self, waypoints, target_pos, offset):
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
        midpoint += offset
        return np.insert(waypoints, idx_to_split + 1, midpoint, axis=0)

    def _visualize_trajectories(self, splines):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        label_prefix = "Przeszkoda" if self.is_obstacle else "Dron"
        for i, spline in enumerate(splines):
            color = colors[i % len(colors)]
            wp = spline.waypoints
            ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], color=color, alpha=0.3, s=10)
            ax.scatter(*wp[0], marker='o', color=color)
            ax.scatter(*wp[-1], marker='x', color=color)
            times = np.linspace(0, spline.total_duration, 200)
            smooth_path = np.array([spline.get_state_at_time(t)[0] for t in times])
            ax.plot(smooth_path[:, 0], smooth_path[:, 1], smooth_path[:, 2],
                    label=f'{label_prefix} {i} (B-Spline)', color=color)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend()
        filename = "trajectory_preview_obstacles.png" if self.is_obstacle else "trajectory_preview_bspline.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # =========================================================================
    # Detekcja zagrożeń (LiDAR + dynamiczne przeszkody)
    # =========================================================================

    def _analyze_lidar_for_threat(
        self,
        hits: list[LidarHit],
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        floor_z: float,
        ceiling_z: float,
    ) -> dict | None:
        if not hits:
            return None
        speed = float(np.linalg.norm(current_vel))
        forward_dir = current_vel / speed if speed > 0.5 else np.array([0.0, 1.0, 0.0])

        dangerous = []
        for hit in hits:
            if hit.distance > self._critical_distance:
                continue
            hz = hit.hit_position[2]
            if hz <= floor_z or hz >= ceiling_z:
                continue
            vec = hit.hit_position - current_pos
            d = float(np.linalg.norm(vec))
            if d < 1e-3:
                continue
            alignment = float(np.dot(forward_dir, vec / d))
            if alignment > self._forward_cone_cos:
                dangerous.append(hit)

        if not dangerous:
            return None
            
        closest = min(dangerous, key=lambda h: h.distance)
        
        return {
            "position": np.asarray(closest.hit_position, dtype=np.float64),
            "velocity": np.asarray(closest.velocity, dtype=np.float64),
            "distance": float(closest.distance),
            "radius": 0.5,
        }
    
    # =========================================================================
    # Rezolucja bazowego łuku (arc) i odwrotność profilu trapezoidalnego
    # =========================================================================

    def _base_flight_time(self, drone_id: int, current_time: float) -> float:
        return max(0.0, current_time - self._tracking_start_times[drone_id])

    def _base_arc_progress(self, drone_id: int, current_time: float) -> float:
        if self._base_trajectories is None:
            return 0.0
        spline = self._base_trajectories[drone_id]
        flight_time = self._base_flight_time(drone_id, current_time)
        dist, _ = spline.profile.get_state(flight_time)
        return float(dist)

    @staticmethod
    def _invert_profile_arc_to_time(spline: BSplineTrajectory, target_arc: float) -> float:
        profile = spline.profile
        if target_arc <= 0.0:
            return 0.0
        if target_arc >= profile.total_distance:
            return profile.total_duration

        if target_arc <= profile.s_a:
            if profile.max_accel > 1e-6:
                return float(np.sqrt(2.0 * target_arc / profile.max_accel))
            return 0.0

        s_end_cruise = profile.s_a + profile.s_c
        if target_arc <= s_end_cruise:
            if profile.v_peak > 1e-6:
                return float(profile.t_a + (target_arc - profile.s_a) / profile.v_peak)
            return profile.t_a

        remaining = target_arc - s_end_cruise
        a = profile.max_accel
        v = profile.v_peak
        if a < 1e-6:
            return profile.t_a + profile.t_c
        disc = v * v - 2.0 * a * remaining
        disc = max(0.0, disc)
        dt = (v - np.sqrt(disc)) / a
        return float(profile.t_a + profile.t_c + dt)

    # =========================================================================
    # Główna pętla
    # =========================================================================

    def compute_actions(self, current_states, current_time):
        if self._base_trajectories is None:
            self._base_trajectories = self._prepare_trajectories()
            self._trajectory_start_times.fill(self.hover_duration)
            self._tracking_start_times.fill(self.hover_duration)

        if not self.is_obstacle and self._lidars is not None:
            self._run_lidar_and_detect(current_states, current_time)

        actions = []
        for i in range(self.num_drones):
            state = current_states[i]

            if self._flight_modes[i] == self.MODE_EVASION:
                target_pos, target_vel = self._step_evasion(i, state, current_time)
            else:
                target_pos, target_vel = self._step_tracking(i, current_time)

            action, _, _ = self.controllers[i].computeControlFromState(
                control_timestep=self._ctrl_timestep,
                state=state,
                target_pos=target_pos,
                target_vel=target_vel,
                target_rpy=np.array([0, 0, 0])
            )
            actions.append(action)

        return np.array(actions)

    # =========================================================================
    # Tryb TRACKING
    # =========================================================================

    def _step_tracking(self, drone_id: int, current_time: float) -> tuple[np.ndarray, np.ndarray]:
        spline = self._base_trajectories[drone_id]
        flight_time = current_time - self._trajectory_start_times[drone_id]
        if flight_time < 0:
            return spline.get_state_at_time(0.0)
        return spline.get_state_at_time(flight_time)

    # =========================================================================
    # Tryb EVASION
    # =========================================================================

    def _step_evasion(self, drone_id: int, state: np.ndarray, current_time: float) -> tuple[np.ndarray, np.ndarray]:
        ev_spline = self._evasion_trajectories[drone_id]
        if ev_spline is None:
            return self._step_tracking(drone_id, current_time)

        ev_time = current_time - self._evasion_start_times[drone_id]
        if ev_time < 0:
            ev_time = 0.0

        target_pos, target_vel = ev_spline.get_state_at_time(ev_time)

        drone_pos = np.asarray(state[0:3], dtype=np.float64)
        rejoin_pt = self._rejoin_points[drone_id]

        dist_to_rejoin = float(np.linalg.norm(drone_pos - rejoin_pt)) if rejoin_pt is not None else np.inf
        evasion_finished = ev_time >= ev_spline.total_duration
        close_to_rejoin = dist_to_rejoin <= self._rejoin_switch_radius

        if evasion_finished or close_to_rejoin:
            self._switch_to_tracking_after_evasion(drone_id, current_time)
            return self._step_tracking(drone_id, current_time)

        return target_pos, target_vel

    def _switch_to_tracking_after_evasion(self, drone_id: int, current_time: float) -> None:
        base_spline = self._base_trajectories[drone_id]
        rejoin_arc = float(self._rejoin_base_arcs[drone_id])
        t_ref = self._invert_profile_arc_to_time(base_spline, rejoin_arc)
        self._trajectory_start_times[drone_id] = current_time - t_ref
        self._tracking_start_times[drone_id] = current_time - t_ref
        self._flight_modes[drone_id] = self.MODE_TRACKING
        self._evasion_trajectories[drone_id] = None
        self._rejoin_points[drone_id] = None
        print(
            f"[INFO] Dron {drone_id} t={current_time:.2f}s powrót do TRACKING "
            f"(rejoin_arc={rejoin_arc:.2f}m, t_ref={t_ref:.2f}s)"
        )

    # =========================================================================
    # Lidar + wykrywanie zagrożeń + trigger
    # =========================================================================

    def _run_lidar_and_detect(self, current_states: list, current_time: float) -> None:
        all_positions = np.array([s[0:3] for s in current_states], dtype=np.float64)
        raw_quats = np.array([s[3:7] for s in current_states], dtype=np.float64)

        stabilized_quats = np.zeros_like(raw_quats)
        for idx, q in enumerate(raw_quats):
            vel = current_states[idx][10:13]
            speed_2d = float(np.linalg.norm(vel[0:2]))
            speed_3d = float(np.linalg.norm(vel))
            if speed_2d > 0.5:
                yaw = float(np.arctan2(vel[1], vel[0])) - np.pi / 2.0
            else:
                euler = Rotation.from_quat(q).as_euler('xyz', degrees=False)
                yaw = float(euler[2])
            if speed_3d > 0.5:
                pitch = float(np.arcsin(np.clip(vel[2] / speed_3d, -1.0, 1.0)))
            else:
                pitch = 0.0
            stabilized_quats[idx] = Rotation.from_euler('xyz', [0.0, pitch, yaw]).as_quat()

        num_rays = LidarSensor._num_rays
        all_results = LidarSensor.batch_ray_test(
            positions=all_positions,
            physics_client_id=self._physics_client_id,
            orientations_quat=stabilized_quats,
        )

        sim_logger = getattr(self.parent, "logger", None)
        env = getattr(self.parent, "environemnt", getattr(self.parent, "environment", None))
        dynamic_ids = set()
        if env is not None and hasattr(env, "DRONE_IDS") and len(env.DRONE_IDS) > self.num_drones:
            dynamic_ids = set(env.DRONE_IDS[self.num_drones:])

        env_floor_z, env_ceiling_z = self._resolve_env_z_bounds(env)

        for i in range(self.num_drones):
            chunk = all_results[i * num_rays: (i + 1) * num_rays]
            raw_hits = self._lidars[i].process_batch_results(
                raw_results=chunk, logger=sim_logger, current_time=current_time, drone_id=i
            )
            self.latest_scans[i] = [h for h in raw_hits if h.object_id in dynamic_ids]

            if not self.enable_avoidance or self.avoidance_algorithm is None:
                continue
            if self._flight_modes[i] == self.MODE_EVASION:
                continue 

            current_velocity = current_states[i][10:13]
            threat = self._analyze_lidar_for_threat(
                hits=self.latest_scans[i],
                current_pos=all_positions[i],
                current_vel=current_velocity,
                floor_z=env_floor_z + 0.3,
                ceiling_z=env_ceiling_z - 0.3,
            )
            if threat is None:
                continue

            self._maybe_trigger_evasion(
                drone_id=i,
                drone_state=current_states[i],
                threat=threat,
                current_time=current_time,
                env_bounds=self._resolve_env_xy_bounds(env, env_floor_z, env_ceiling_z),
            )

    def _maybe_trigger_evasion(
        self,
        drone_id: int,
        drone_state: list,
        threat: dict,
        current_time: float,
        env_bounds: tuple[np.ndarray, np.ndarray],
    ) -> None:
        current_pos = np.asarray(drone_state[0:3], dtype=np.float64)
        current_vel = np.asarray(drone_state[10:13], dtype=np.float64)
        obs_vel = threat.get("velocity", np.zeros(3, dtype=np.float64))

        vec = threat["position"] - current_pos
        dist = float(np.linalg.norm(vec))
        if dist < 1e-3:
            return
        dir_to_threat = vec / dist
        
        rel_vel = current_vel - obs_vel
        closing_speed = float(np.dot(rel_vel, dir_to_threat))
        
        ttc = dist / closing_speed if closing_speed > 0.1 else np.inf

        should_trigger = (ttc < self._trigger_ttc) or (dist < self._trigger_distance)

        time_since = current_time - self._evasion_start_times[drone_id]
        last_threat = self._last_threat_positions[drone_id]
        same_threat = (
            last_threat is not None
            and np.linalg.norm(threat["position"] - last_threat) < 10.0
        )
        threat_receding = same_threat and closing_speed < 0.0
        if same_threat and time_since < self._evasion_cooldown and not threat_receding:
            should_trigger = False

        if not should_trigger:
            return

        base_spline = self._base_trajectories[drone_id]
        base_arc = self._base_arc_progress(drone_id, current_time)

        print(
            f"[WARN] Dron {drone_id} t={current_time:.2f}s triggers evasion "
            f"(d={dist:.2f}m, rel_TTC={ttc:.2f}s) via {self.avoidance_algorithm.name}"
        )

        drone_kinematic = KinematicState(
            position=current_pos,
            velocity=current_vel,
            radius=self.collision_radius
        )
        
        threat_alert = ThreatAlert(
            obstacle_state=KinematicState(
                position=threat["position"],
                velocity=obs_vel,
                radius=threat.get("radius", 0.5)
            ),
            distance=dist,
            time_to_collision=ttc,
            relative_velocity=rel_vel
        )

        t_min = float(self.avoidance_algorithm.params.get("evasion_time_min", 2.0))
        t_max = float(self.avoidance_algorithm.params.get("evasion_time_max", 4.0))
        builder = EvasionContextBuilder(t_min=t_min, t_max=t_max)
        
        context = builder.build(
            drone_id=drone_id,
            current_time=current_time,
            drone_state=drone_kinematic,
            threat=threat_alert,
            base_spline=base_spline,
            base_arc_progress=base_arc,
            env_bounds=env_bounds
        )

        plan = self.avoidance_algorithm.compute_evasion_plan(context)

        if plan is None:
            print(f"[WARN] Plan uniku nie powstał dla drona {drone_id} — pozostajemy w TRACKING")
            return

        self._evasion_trajectories[drone_id] = plan.evasion_spline
        self._evasion_start_times[drone_id] = current_time
        self._rejoin_base_arcs[drone_id] = plan.rejoin_base_arc
        self._rejoin_points[drone_id] = np.asarray(plan.rejoin_point, dtype=np.float64)
        self._last_threat_positions[drone_id] = np.array(threat["position"], dtype=np.float64)
        self._flight_modes[drone_id] = self.MODE_EVASION

    # =========================================================================
    # Granice świata (z cfg / world_data)
    # =========================================================================

    def _resolve_env_z_bounds(self, env) -> tuple[float, float]:
        parent = self.parent
        floor_z = 0.1
        ceiling_z = 10.0
        world_data = getattr(parent, "world_data", None)
        if world_data is not None and hasattr(world_data, "dimensions"):
            try:
                ceiling_z = float(world_data.dimensions[2])
            except Exception:
                pass
        if hasattr(parent, "ground_position") and parent.ground_position is not None:
            try:
                floor_z = float(parent.ground_position)
            except Exception:
                pass
        return floor_z, ceiling_z

    def _resolve_env_xy_bounds(
        self, env, floor_z: float, ceiling_z: float
    ) -> tuple[np.ndarray, np.ndarray]:
        parent = self.parent
        x_min, x_max = 0.0, 60.0
        y_min, y_max = 0.0, 600.0
        world_data = getattr(parent, "world_data", None)
        if world_data is not None and hasattr(world_data, "dimensions"):
            try:
                x_max = float(world_data.dimensions[0])
                y_max = float(world_data.dimensions[1])
            except Exception:
                pass
        elif hasattr(parent, "track_width") and hasattr(parent, "track_length"):
            try:
                x_max = float(parent.track_width)
                y_max = float(parent.track_length)
            except Exception:
                pass
        return (
            np.array([x_min, y_min, floor_z], dtype=np.float64),
            np.array([x_max, y_max, ceiling_z], dtype=np.float64),
        )

    # =========================================================================
    # Stan końcowy
    # =========================================================================

    @property
    def all_finished(self) -> bool:
        if self._base_trajectories is None:
            return False
        return all(
            np.linalg.norm(
                self._base_trajectories[i].waypoints[-1] - self.parent.current_states[i][0:3]
            ) < self.finish_radius
            for i in range(self.num_drones)
        )