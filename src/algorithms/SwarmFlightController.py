import numpy as np
from src.algorithms.abstraction.trajectory.strategies.shared.NumbaTrajectoryProfile import NumbaTrajectoryProfile

# POPRAWKA: Wszystkie zunifikowane struktury pobieramy z BaseAvoidance
from src.algorithms.avoidance.EvasionContextBuilder import EvasionContextBuilder

from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import KinematicState, ThreatAlert
from src.sensors.LidarSensor import LidarSensor, LidarHit
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from .numba_optimizers import check_collisions_njit, calculate_repulsion_njit

class SwarmFlightController():
    _debug_file = open("drone_debug.log", "w")

    MODE_TRACKING = 0
    MODE_EVASION = 1
    MODE_REJOIN_BLEND = 2  # Faza 3.2: mostek EVASION→TRACKING (mieszanie komend PID)

    def __init__(self, parent, num_drones, is_obstacle: bool, avoidance_algorithm=None, params=None):
        self._trajectory_start_times = np.zeros(num_drones)
        self.is_obstacle = is_obstacle
        self.avoidance_algorithm = avoidance_algorithm
        self.parent = parent
        self.num_drones = num_drones
        self.params = params or {}

        self.controllers = [
            DSLPIDControl(drone_model=DroneModel.CF2X)
            for _ in range(num_drones)
        ]

        # Bazowe trajektorie (offline, nigdy zmieniane)
        self._base_trajectories: list[NumbaTrajectoryProfile] | None = None
        # Trajektorie uniku (lokalne, niezmiennicze w czasie trwania uniku)
        self._evasion_trajectories: list[NumbaTrajectoryProfile | None] = [None] * num_drones

        self._lidars: list[LidarSensor] | None = None
        self.latest_scans: list[list[LidarHit]] = [[] for _ in range(num_drones)]

        # Stan maszyny
        self._flight_modes = np.full(num_drones, self.MODE_TRACKING, dtype=int)
        self._evasion_start_times = np.full(num_drones, -np.inf)
        self._evasion_cooldown = float(self.params.get("evasion_cooldown", 1.0))
        # Minimalne okno realizacji poprzedniego planu — nawet przy imminent
        # (ttc < cooldown/2) nie replanujemy częściej niż co `min_dt`. Bez tego
        # throttla w korytarzu z wieloma przeszkodami oracle skakał między
        # celami co 20 ms, PID nie nadążał wykonać żadnego z kolejnych planów.
        # Wartość 1.5 s ≈ pełna długość typowego planu uniku po skróceniu
        # `evasion_time_max` z 6→3 s (Bug #2 plan, Krok 3a). Plan ma realnie
        # zdążyć skończyć (BLEND→TRACKING) zanim go wymienimy. Dla scenariuszy
        # z dużą dynamiką zagrożeń tłumimy replan dodatkowym kryterium
        # `_lateral_progress_min_m` (patrz `_maybe_trigger_evasion`).
        self._imminent_replan_min_dt = float(self.params.get("imminent_replan_min_dt", 1.5))
        # Próg lateralnego odjazdu drona od trasy bazowej — jeśli dron już
        # zdążył odlecieć od bazy o ≥ ten próg [m] dzięki bieżącemu planowi,
        # tłumimy `imminent` re-trigger (plan działa, nie zaorajmy go).
        # Bypass: high_divergence (zagrożenie ewidentnie inne niż przewidywane)
        # i new_obstacle (oracle wskazał inny obiekt) zawsze przepuszczają.
        self._lateral_progress_min_m = float(self.params.get("lateral_progress_min_m", 1.0))
        self._last_threat_positions: list[np.ndarray | None] = [None] * num_drones
        self._last_threat_velocities: list[np.ndarray | None] = [None] * num_drones
        self._last_threat_times: list[float | None] = [None] * num_drones
        # Identyfikator (obs_idx) przeszkody, na którą zbudowano aktualny plan —
        # potrzebny żeby oracle przełączający cel w trakcie uniku (dron na evasion-
        # spline zbliża się do innego obiektu niż ten, dla którego planowaliśmy)
        # nie wywoływał fałszywego `high_divergence` co tick.
        self._last_threat_obs_idx: list[int | None] = [None] * num_drones
        # Sticky axis: replany w trakcie uniku dziedziczą oś z poprzedniego planu,
        # co zapobiega flip-floppingowi up↔right↔left, gdy oracle przełącza cel
        # między przeszkodami w tym samym korytarzu. Resetowane przy końcu BLEND.
        self._last_preferred_axis: list[str | None] = [None] * num_drones
        # Re-trigger cooldown po `compute_evasion_plan → None` (refactor 2026-05-02
        # Single-Arc Deflection): bez tego drone re-triggeruje co tick (~10 ms),
        # każdy trigger trwa do 1 s — symulacja staje się non-realtime. Cooldown
        # daje drone'owi prosty oddech zanim spróbuje znowu. Czytamy z
        # avoidance.params (yaml configs/avoidance/*.yaml `no_plan_cooldown_s`).
        _av_params_cooldown = (
            self.avoidance_algorithm.params if self.avoidance_algorithm is not None else {}
        )
        self._no_plan_cooldown_s = float(
            _av_params_cooldown.get("no_plan_cooldown_s", 0.5)
        )
        self._no_plan_until: np.ndarray = np.zeros(num_drones, dtype=np.float64)
        self._rejoin_base_arcs = np.zeros(num_drones)
        self._rejoin_points: list[np.ndarray | None] = [None] * num_drones
        # Bazowy czas startu (po hover lub po rejoin) — używany do tracking
        self._tracking_start_times = np.zeros(num_drones)

        # Faza 3.2: mostek EVASION→TRACKING (MODE_REJOIN_BLEND).
        # W oknie [_blend_start, _blend_start + blend_duration] liczymy target jako
        # mieszankę: α·evasion + (1-α)·tracking, α ∈ [1, 0]. Po wygaśnięciu okna
        # przełączamy na MODE_TRACKING, a evasion_spline jest zwalniany.
        self._blend_start_times = np.full(num_drones, -np.inf)
        av = self.avoidance_algorithm
        av_params_blend = av.params if av is not None else {}
        self._blend_duration = float(av_params_blend.get("blend_duration", 0.6))

        # Ostatni znany "ground-truth yaw/pitch" wyliczony z wektora prędkości.
        # Stabilizacja LiDARu używa tego bufora gdy `speed_2d <= 0.5` (hamowanie,
        # hover, rejoin), zamiast fallbacku do kwaternionu z silnika fizyki.
        self._last_heading_yaw: list[float | None] = [None] * num_drones
        self._last_heading_pitch: list[float | None] = [None] * num_drones
        self._last_heading_time: list[float] = [-np.inf] * num_drones
        self._heading_hold_time = float(self.params.get("heading_hold_time", 2.0))

        # Parametry ruchu
        self._ctrl_timestep = 1.0 / self.params.get("ctrl_freq", 48)
        self.hover_duration = self.params.get("hover_duration", 3.0)
        self.finish_radius = self.params.get("finish_radius", 0.5)
        self.cruise_speed = self.params.get("cruise_speed", 8.0)
        self.max_accel = self.params.get("max_accel", 2.0)
        self.collision_radius = self.params.get("collision_radius", 0.4)
        self.enable_avoidance = self.params.get("enable_avoidance", False)

        # Detekcja zagrożenia — progi czytamy preferencyjnie z avoidance_algorithm.params
        # (sekcja `trigger_*` w configs/avoidance/*.yaml), z fallbackiem na self.params
        # i bezpieczne wartości domyślne. Progi dystansowe/krytyczne są *skalowane*
        # do |rel_vel| w runtime (_maybe_trigger_evasion) — tu trzymamy same bazy.
        av = self.avoidance_algorithm
        av_params = av.params if av is not None else {}

        def _read(key: str, default: float) -> float:
            val = av_params.get(key, self.params.get(key, default))
            return float(val)

        self._trigger_ttc = _read("trigger_ttc", 3.5)
        self._trigger_distance_base = _read("trigger_distance_base", 6.0)
        self._trigger_distance_ttc_factor = _read("trigger_distance_ttc_factor", 1.0)
        self._critical_distance_base = _read("critical_distance_base", 25.0)
        self._critical_distance_ttc_factor = _read("critical_distance_ttc_factor", 1.5)
        self._forward_cone_cos = _read("forward_cone_cos", 0.5)

        # Niezmiennik budżetowy: trigger_ttc musi być >= evasion_time_min,
        # inaczej plan uniku jest projektowany na dłuższy horyzont niż pozostały
        # czas do kolizji (gwarantowana kolizja dla head-on).
        evasion_time_min = float(av_params.get("evasion_time_min", 2.0))
        planning_overhead_margin = 0.5  # [s] rezerwa na A* + rozbieg kontrolera
        if self._trigger_ttc < evasion_time_min + planning_overhead_margin:
            print(
                f"[WARN] Niezmiennik budżetu uniku naruszony: "
                f"trigger_ttc={self._trigger_ttc:.2f}s < evasion_time_min"
                f"({evasion_time_min:.2f}s)+{planning_overhead_margin:.2f}s. "
                f"Plan manewru może się nie zmieścić przed kolizją."
            )

        # Rejoin
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
                NumbaTrajectoryProfile(raw_waypoints[i], self.cruise_speed, self.max_accel)
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
                NumbaTrajectoryProfile(raw_waypoints[i], self.cruise_speed, self.max_accel)
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

    def _verify_trajectories(self, splines):
        max_time = max([spline.total_duration for spline in splines])
        dt = 0.1
        times = np.arange(0, max_time + dt, dt)
        safe_dist = self.collision_radius * 3.0
        
        # Ekstrakcja danych obiektowych do macierzy przed wywołaniem JIT
        positions = np.zeros((len(times), self.num_drones, 3))
        for t_idx, t in enumerate(times):
            for i in range(self.num_drones):
                positions[t_idx, i] = splines[i].get_state_at_time(t)[0]

        # Wywołanie zoptymalizowanej funkcji
        i, j, t_idx = check_collisions_njit(positions, safe_dist)
        
        if i != -1:
            return i, j, positions[t_idx, i], positions[t_idx, j]
        return None

    def _repair_waypoints(self, raw_waypoints, collision):
        new_waypoints = [path.copy() for path in raw_waypoints]
        i, j, pos_i, pos_j = collision
        
        push_distance = 2.0
        
        # Wywołanie zoptymalizowanej naprawy ścieżek
        new_wp_i, new_wp_j = calculate_repulsion_njit(
            new_waypoints[i], new_waypoints[j], pos_i, pos_j, push_distance
        )
        
        new_waypoints[i] = new_wp_i
        new_waypoints[j] = new_wp_j
        
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

    def _oracle_threat_check(
        self,
        drone_id: int,
        current_time: float,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        lookahead: float | None = None,
        dt: float = 0.1,
    ) -> dict | None:
        """
        Deterministyczna predykcja kolizji na bazie *znanych* splajnów obu roi
        (głównego + dynamicznych przeszkód). Zwraca threat-dict jeśli najbliższa
        wzajemna odległość w oknie `[now, now+lookahead]` jest poniżej
        `2*collision_radius + margin`.

        Predykcja pozycji drona (kluczowe dla poprawności TTC):
            dt_ahead=0  → `current_pos` (stan faktyczny z silnika fizyki).
            dt_ahead>0  → splajn, którego dron *aktualnie* się trzyma:
                          * `evasion_spline` w MODE_EVASION / MODE_REJOIN_BLEND,
                          * `base_spline` w MODE_TRACKING.
            Po wyczerpaniu splajnu ekstrapolujemy liniowo z bieżącej prędkości
            (to bezpieczny fallback — oracle sumarycznie *zaniża* TTC, a nie
            zawyża, co lepsze dla bezpieczeństwa).

        Motywacja: bez tej korekty oracle w trakcie/po uniku czytał pozycję
        z base-spline'a (gdzie dron *powinien być*, nie gdzie *jest*) —
        co dawało fałszywe TTC=0 mimo faktycznej odległości >8m i zapętlało
        re-triggery.

        Scenariusz head-on (ten sam korytarz) — LiDAR widzi przeszkodę dopiero
        w stożku; oracle wyzwala unik zanim to nastąpi (Faza 1 planu).
        """
        obs_ctrl = getattr(self.parent, "dynamic_obstacle_trajectory_controller", None)
        if obs_ctrl is None or obs_ctrl._base_trajectories is None:
            return None
        if self._base_trajectories is None:
            return None

        if lookahead is None:
            # Lookahead minimum = trigger_ttc; z marginesem by trigger miał budżet.
            lookahead = max(4.0, self._trigger_ttc + 1.0)

        threshold = 2.0 * self.collision_radius + 1.0

        # Wybór splajnu referencyjnego dla predykcji pozycji drona.
        mode = int(self._flight_modes[drone_id])
        if mode in (self.MODE_EVASION, self.MODE_REJOIN_BLEND):
            my_spline = self._evasion_trajectories[drone_id]
            my_t0 = float(self._evasion_start_times[drone_id])
        else:
            my_spline = None
            my_t0 = 0.0
        if my_spline is None:
            my_spline = self._base_trajectories[drone_id]
            my_t0 = float(self._trajectory_start_times[drone_id])

        best: tuple[float, float, int] | None = None
        ts = np.arange(0.0, lookahead + 1e-9, dt)

        for obs_idx, obs_spline in enumerate(obs_ctrl._base_trajectories):
            obs_t0 = float(obs_ctrl._trajectory_start_times[obs_idx])
            for dt_ahead in ts:
                t = current_time + float(dt_ahead)
                obs_ft = max(0.0, t - obs_t0)
                obs_pos_t, _ = obs_spline.get_state_at_time(obs_ft)

                my_pos_t = self._predict_drone_position(
                    my_spline, my_t0, t, current_pos, current_vel, float(dt_ahead)
                )

                d = float(np.linalg.norm(my_pos_t - obs_pos_t))
                if d < threshold:
                    if best is None or d < best[0]:
                        best = (d, float(dt_ahead), obs_idx)
                    break  # kolejne próbki tej pary — pierwszy hit wystarczy

        if best is None:
            return None

        _, ttc_oracle, obs_idx = best
        obs_spline = obs_ctrl._base_trajectories[obs_idx]
        obs_t0 = float(obs_ctrl._trajectory_start_times[obs_idx])
        obs_ft_now = max(0.0, current_time - obs_t0)
        obs_pos_now, obs_vel_now = obs_spline.get_state_at_time(obs_ft_now)
        dist_now = float(np.linalg.norm(np.asarray(obs_pos_now) - current_pos))

        # Promień: oracle nie ma ID obiektu w PyBullet (pracuje na splajnach),
        # ale `self.collision_radius` pochodzi z configu i jest spójny z tym,
        # czym jest drone-przeszkoda (CF2X). Bufor bezpieczeństwa jest dodatkowo
        # egzekwowany przez `threshold` (2*collision_radius + 1.0).
        return {
            "position": np.asarray(obs_pos_now, dtype=np.float64),
            "velocity": np.asarray(obs_vel_now, dtype=np.float64),
            "distance": dist_now,
            "radius": float(self.collision_radius),
            "ttc_override": float(ttc_oracle),
            "source": "oracle",
            "obs_idx": int(obs_idx),
        }

    @staticmethod
    def _predict_drone_position(
        my_spline,
        my_t0: float,
        t_absolute: float,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        dt_ahead: float,
    ) -> np.ndarray:
        """
        Pozycja drona w czasie `t_absolute` dla potrzeb oracle.

        * `dt_ahead == 0` → fizyczna pozycja drona (current_pos).
        * `dt_ahead > 0`:
            - preferujemy wartość splajnu w lokalnym czasie `t_absolute - my_t0`,
              o ile mieści się w jego zakresie — splajn modeluje przyszłe
              zakręty lepiej niż ekstrapolacja liniowa;
            - jeśli lokalny czas wykracza poza splajn, stosujemy ekstrapolację
              liniową `current_pos + current_vel * dt_ahead` jako bezpieczny
              fallback.
        """
        if dt_ahead <= 0.0:
            return np.asarray(current_pos, dtype=np.float64)

        if my_spline is not None:
            my_ft = t_absolute - my_t0
            if 0.0 <= my_ft <= my_spline.total_duration + 1e-6:
                pos_spline, _ = my_spline.get_state_at_time(max(0.0, my_ft))
                return np.asarray(pos_spline, dtype=np.float64)

        return np.asarray(current_pos, dtype=np.float64) + np.asarray(
            current_vel, dtype=np.float64
        ) * float(dt_ahead)

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
            # Skaluj próg krytyczny do prędkości *względnej* (head-on → ~2×cruise).
            rel_vel_hit = current_vel - hit.velocity
            rel_speed_hit = float(np.linalg.norm(rel_vel_hit))
            critical_distance_hit = max(
                self._critical_distance_base,
                self._critical_distance_ttc_factor * rel_speed_hit * self._trigger_ttc,
            )
            if hit.distance > critical_distance_hit:
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
            "radius": float(getattr(closest, "obstacle_radius", 0.5)),
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
        return self._arc_at_time(spline, flight_time)

    @staticmethod
    def _arc_at_time(spline: NumbaTrajectoryProfile, t: float) -> float:
        # Analityczna pozycja po krzywej (skalar) wg trapezoidalnego profilu prędkości.
        # NumbaTrajectoryProfile eksponuje get_state_at_time() zwracające (pos, vel) jako
        # wektory 3D — tu potrzebujemy tylko skalarnego "arc" (dystans wzdłuż trasy).
        if t <= 0.0:
            return 0.0
        if t >= spline.total_duration:
            return float(spline.total_distance)
        if t < spline.ta:
            return float(0.5 * spline.max_accel * t * t)
        if t < spline.ta + spline.tc:
            return float(spline.sa + spline.v_peak * (t - spline.ta))
        t_dec = t - spline.ta - spline.tc
        return float(
            spline.sa + spline.sc
            + spline.v_peak * t_dec
            - 0.5 * spline.max_accel * t_dec * t_dec
        )

    @staticmethod
    def _invert_profile_arc_to_time(spline: NumbaTrajectoryProfile, target_arc: float) -> float:
        if target_arc <= 0.0:
            return 0.0
        if target_arc >= spline.total_distance:
            return float(spline.total_duration)

        if target_arc <= spline.sa:
            if spline.max_accel > 1e-6:
                return float(np.sqrt(2.0 * target_arc / spline.max_accel))
            return 0.0

        s_end_cruise = spline.sa + spline.sc
        if target_arc <= s_end_cruise:
            if spline.v_peak > 1e-6:
                return float(spline.ta + (target_arc - spline.sa) / spline.v_peak)
            return float(spline.ta)

        remaining = target_arc - s_end_cruise
        a = spline.max_accel
        v = spline.v_peak
        if a < 1e-6:
            return float(spline.ta + spline.tc)
        disc = v * v - 2.0 * a * remaining
        disc = max(0.0, disc)
        dt = (v - np.sqrt(disc)) / a
        return float(spline.ta + spline.tc + dt)

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
            elif self._flight_modes[i] == self.MODE_REJOIN_BLEND:
                # Faza 3.2: mostek mieszający komendę EVASION→TRACKING.
                target_pos, target_vel = self._step_blend(i, current_time)
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
            drone_vel = np.asarray(state[10:13], dtype=np.float64)
            # Faza 3.2/3.3: nie przełączamy od razu na TRACKING — najpierw BLEND.
            # Dopasowanie czasowe bazowego splinu do rzeczywistej pozycji drona.
            self._start_rejoin_blend(drone_id, current_time, drone_pos, drone_vel)
            return self._step_blend(drone_id, current_time)

        return target_pos, target_vel

    # --- Faza 3.3: dopasowanie fazy czasowej do aktualnej pozycji drona ---

    @staticmethod
    def _fit_tracking_time_to_drone(
        base_spline: NumbaTrajectoryProfile,
        t_nominal: float,
        drone_pos: np.ndarray,
        search_window_s: float = 0.3,
        samples: int = 31,
    ) -> float:
        """
        Zamiast sztywnego `t_nominal = invert_profile_arc_to_time(rejoin_arc)`,
        szukamy t* = argmin_t ||base_spline.pos(t) - drone_pos|| w oknie
        [t_nominal - Δ, t_nominal + Δ]. Eliminuje "teleportację" komendy PID
        w chwili przełączenia EVASION→TRACKING: bierzemy ten moment bazowej
        trajektorii, który jest geometrycznie *najbliżej* drona.

        Uzasadnienie: `invert_profile_arc_to_time` mapuje łuk → czas przy
        założeniu, że dron przebył dokładnie `rejoin_arc` od startu —
        co po uniku nie jest prawdą (dron jest w *innym miejscu* niż idealny
        tracking-point). Argmin koryguje tę niespójność.
        """
        t_lo = float(max(0.0, t_nominal - search_window_s))
        t_hi = float(min(base_spline.total_duration, t_nominal + search_window_s))
        if t_hi <= t_lo + 1e-6:
            return float(np.clip(t_nominal, 0.0, base_spline.total_duration))

        candidates = np.linspace(t_lo, t_hi, samples)
        errors = np.array([
            float(np.linalg.norm(base_spline.get_state_at_time(float(t))[0] - drone_pos))
            for t in candidates
        ])
        return float(candidates[int(np.argmin(errors))])

    def _start_rejoin_blend(
        self,
        drone_id: int,
        current_time: float,
        drone_pos: np.ndarray,
        drone_vel: np.ndarray,
    ) -> None:
        """Uruchamia mostek MODE_REJOIN_BLEND (Faza 3.2).

        Ustawia bazowy tracking-time tak, by pasował do aktualnej pozycji drona
        (Faza 3.3), po czym `_step_blend` miesza komendy przez `blend_duration`.
        """
        base_spline = self._base_trajectories[drone_id]
        rejoin_arc = float(self._rejoin_base_arcs[drone_id])
        t_nominal = self._invert_profile_arc_to_time(base_spline, rejoin_arc)
        t_ref = self._fit_tracking_time_to_drone(base_spline, t_nominal, drone_pos)

        self._trajectory_start_times[drone_id] = current_time - t_ref
        self._tracking_start_times[drone_id] = current_time - t_ref
        self._flight_modes[drone_id] = self.MODE_REJOIN_BLEND
        self._blend_start_times[drone_id] = current_time

        # Pomiar nieciągłości w chwili wejścia w BLEND.
        target_pos_ref, target_vel_ref = base_spline.get_state_at_time(t_ref)
        pos_err = float(np.linalg.norm(drone_pos - target_pos_ref))
        vel_err = float(np.linalg.norm(drone_vel - target_vel_ref))

        logger = getattr(self.parent, "logger", None)
        if logger is not None and hasattr(logger, "log_evasion_event"):
            logger.log_evasion_event(
                current_time=current_time,
                drone_id=drone_id,
                event_type="rejoin",
                mode=int(self.MODE_REJOIN_BLEND),
                rejoin_point=self._rejoin_points[drone_id],
                rejoin_arc=rejoin_arc,
                pos_error_at_rejoin=pos_err,
                vel_error_at_rejoin=vel_err,
                notes=f"t_nominal={t_nominal:.3f}s t_ref={t_ref:.3f}s blend={self._blend_duration:.2f}s",
            )

        print(
            f"[INFO] Dron {drone_id} t={current_time:.2f}s START BLEND "
            f"(t_nominal={t_nominal:.2f}s, t_ref={t_ref:.2f}s, "
            f"pos_err={pos_err:.2f}m, vel_err={vel_err:.2f}m/s)"
        )

    def _step_blend(self, drone_id: int, current_time: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Faza 3.2: liniowe mieszanie komend EVASION i TRACKING przez
        okres `blend_duration`. α(t): 1 → 0.
            target = α·evasion_state + (1-α)·tracking_state
        Po wygaśnięciu okna przechodzimy na MODE_TRACKING i zwalniamy
        evasion_spline/rejoin_point. Eliminuje impuls komendy PID.
        """
        ev_spline = self._evasion_trajectories[drone_id]
        blend_start = float(self._blend_start_times[drone_id])
        dt_blend = current_time - blend_start

        # Koniec okna lub brak evasion_spline (awaryjnie) — finalizacja BLEND.
        if ev_spline is None or dt_blend >= self._blend_duration:
            self._flight_modes[drone_id] = self.MODE_TRACKING
            self._evasion_trajectories[drone_id] = None
            self._rejoin_points[drone_id] = None
            # Zrzut stanu threata — nowe zagrożenie po powrocie do TRACKING
            # nie powinno dziedziczyć predykcji z poprzedniego uniku.
            self._last_threat_positions[drone_id] = None
            self._last_threat_velocities[drone_id] = None
            self._last_threat_times[drone_id] = None
            self._last_threat_obs_idx[drone_id] = None
            self._last_preferred_axis[drone_id] = None
            return self._step_tracking(drone_id, current_time)

        alpha = float(np.clip(1.0 - dt_blend / self._blend_duration, 0.0, 1.0))

        ev_time = current_time - float(self._evasion_start_times[drone_id])
        ev_time = max(0.0, min(ev_time, ev_spline.total_duration))
        ev_pos, ev_vel = ev_spline.get_state_at_time(ev_time)

        tr_pos, tr_vel = self._step_tracking(drone_id, current_time)

        blended_pos = alpha * ev_pos + (1.0 - alpha) * tr_pos
        blended_vel = alpha * ev_vel + (1.0 - alpha) * tr_vel
        return blended_pos, blended_vel

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
                self._last_heading_yaw[idx] = yaw
                self._last_heading_time[idx] = current_time
            else:
                # Podczas hamowania / zawisu nie skaczemy na yaw z kwaternionu fizyki
                # (dron CF2X często rotuje wokół osi Z w miejscu → LiDAR traci cel).
                held = self._last_heading_yaw[idx]
                hold_age = current_time - self._last_heading_time[idx]
                if held is not None and hold_age <= self._heading_hold_time:
                    yaw = float(held)
                else:
                    euler = Rotation.from_quat(q).as_euler('xyz', degrees=False)
                    yaw = float(euler[2])

            if speed_3d > 0.5:
                pitch = float(np.arcsin(np.clip(vel[2] / speed_3d, -1.0, 1.0)))
                self._last_heading_pitch[idx] = pitch
            else:
                held_pitch = self._last_heading_pitch[idx]
                hold_age = current_time - self._last_heading_time[idx]
                if held_pitch is not None and hold_age <= self._heading_hold_time:
                    pitch = float(held_pitch)
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
            # Faza 2: NIE pomijamy MODE_EVASION — decyzja o re-triggerze
            # zapada w `_maybe_trigger_evasion` po sprawdzeniu divergencji zagrożenia.

            current_velocity = current_states[i][10:13]

            # Faza 1: Ground-truth predykcja TTC na bazie znanych splajnów.
            # Uruchamiamy oracle PRZED LiDARem — wyzwala unik wcześniej niż stożek.
            threat = self._oracle_threat_check(
                drone_id=i,
                current_time=current_time,
                current_pos=all_positions[i],
                current_vel=np.asarray(current_velocity, dtype=np.float64),
            )

            # LiDAR jako niezależny sensor potwierdzający (defense-in-depth).
            if threat is None:
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
                all_drone_states=current_states,
            )

    def _maybe_trigger_evasion(
        self,
        drone_id: int,
        drone_state: list,
        threat: dict,
        current_time: float,
        env_bounds: tuple[np.ndarray, np.ndarray],
        all_drone_states: list | None = None,
    ) -> None:
        # No-plan cooldown (refactor 2026-05-02): drone w cooldown po niedawnym
        # `compute_evasion_plan → None` — silent skip żeby nie spamować
        # optymalizatora 100×/s tym samym infeasible problemem. Krok 5 fairness
        # (2026-05-03): logujemy `event_type="cooldown_skip"` dla analizy w pracy
        # — algorytmy z większą liczbą no_plan dostają więcej skipów (bias).
        if (
            self._flight_modes[drone_id] == self.MODE_TRACKING
            and current_time < float(self._no_plan_until[drone_id])
        ):
            logger = getattr(self.parent, "logger", None)
            if logger is not None and hasattr(logger, "log_evasion_event"):
                try:
                    logger.log_evasion_event(
                        current_time=current_time,
                        drone_id=drone_id,
                        event_type="cooldown_skip",
                        mode=int(self._flight_modes[drone_id]),
                        ttc=float("nan"),
                        dist_to_threat=float("nan"),
                        threat_pos=np.asarray(threat["position"], dtype=np.float64),
                        threat_vel=np.asarray(
                            threat.get("velocity", np.zeros(3)), dtype=np.float64
                        ),
                        notes=f"cooldown until {float(self._no_plan_until[drone_id]):.3f}s",
                    )
                except Exception:
                    pass  # log best-effort; nie blokujemy unikania
            return

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

        # Preferuj TTC z oracle (deterministyczna predykcja po splajnach)
        # jeśli dostępne — obejmuje przyszły ruch obu stron, nie tylko chwilowy stan.
        ttc_override = threat.get("ttc_override") if isinstance(threat, dict) else None
        if ttc_override is not None and np.isfinite(ttc_override):
            ttc = float(ttc_override)
        else:
            ttc = dist / closing_speed if closing_speed > 0.1 else np.inf

        # Skalowanie progu dystansowego do rzeczywistej prędkości względnej
        # — zapobiega spóźnionemu wyzwoleniu przy head-on (rel_vel ≈ 2×cruise).
        rel_speed = float(np.linalg.norm(rel_vel))
        trigger_distance_dynamic = max(
            self._trigger_distance_base,
            self._trigger_distance_ttc_factor * rel_speed * self._trigger_ttc,
        )
        should_trigger = (ttc < self._trigger_ttc) or (dist < trigger_distance_dynamic)

        time_since = current_time - self._evasion_start_times[drone_id]
        last_threat = self._last_threat_positions[drone_id]
        last_threat_vel = self._last_threat_velocities[drone_id]
        last_threat_t = self._last_threat_times[drone_id]
        last_obs_idx = self._last_threat_obs_idx[drone_id]
        current_obs_idx = threat.get("obs_idx")

        # Błąd predykcji tylko *tego samego* obiektu — inaczej oracle przełączający
        # cel (dron na evasion-spline zbliża się do nowego obiektu) generowałby
        # ogromny pred_err_pos i wyzwalał replan co tick.
        same_obstacle = (
            last_obs_idx is not None
            and current_obs_idx is not None
            and int(current_obs_idx) == int(last_obs_idx)
        )

        if (
            same_obstacle
            and last_threat is not None
            and last_threat_vel is not None
            and last_threat_t is not None
        ):
            dt_since = max(0.0, current_time - last_threat_t)
            expected_pos = last_threat + last_threat_vel * dt_since
            pred_err_pos = float(np.linalg.norm(threat["position"] - expected_pos))
            pred_err_vel = float(np.linalg.norm(obs_vel - last_threat_vel))
        else:
            pred_err_pos = float("inf")
            pred_err_vel = float("inf")

        # Tolerancja pozycji skalowana do rel_speed: 0.5s względnego ruchu to
        # granica na szum predykcji/sensora, minimum 3m dla niskich prędkości.
        same_threat_pos_tol = max(3.0, 0.5 * rel_speed)
        same_threat = same_obstacle and pred_err_pos < same_threat_pos_tol
        threat_receding = same_threat and closing_speed < 0.0
        if same_threat and time_since < self._evasion_cooldown and not threat_receding:
            should_trigger = False

        # Faza 2: Re-trigger w trakcie EVASION tylko jeśli zagrożenie
        # znacząco różni się od tego, co już unikamy (nowy obiekt, zmieniony
        # wektor prędkości) albo jest na tyle blisko, że stary plan nie
        # zdąży (ttc < pół cooldown).
        # BLEND też traktujemy jak "w trakcie uniku" — przerwanie blendu na rzecz
        # re-triggera jest dopuszczalne tylko przy znaczącej dywergencji zagrożenia.
        in_evasion = self._flight_modes[drone_id] in (
            self.MODE_EVASION,
            self.MODE_REJOIN_BLEND,
        )
        if in_evasion:
            # Próg dywergencji ≈ 0.3s rel_speed (np. 5m dla head-on 16 m/s),
            # min. 3m — obstacle na zaplanowanym torze nie wyzwala re-triggera.
            divergence_tol = max(3.0, 0.3 * rel_speed)
            high_divergence = pred_err_pos > divergence_tol or pred_err_vel > 2.0
            new_obstacle = not same_obstacle  # oracle wskazał INNY obiekt
            imminent = np.isfinite(ttc) and ttc < (self._evasion_cooldown / 2.0)
            if not (high_divergence or imminent or new_obstacle):
                return  # bieżący plan powinien obsłużyć to zagrożenie

            # Lateral progress check: jeśli bieżący plan już odsunął drona od
            # trasy bazowej o ≥ `_lateral_progress_min_m`, tłumimy replan na
            # bazie samego `imminent` — plan działa, nie zaorajmy go w trakcie.
            # Bypass: high_divergence (zagrożenie radykalnie inne niż przewidywane)
            # i new_obstacle (oracle wskazał inny cel) zawsze przepuszczają.
            if imminent and not (high_divergence or new_obstacle):
                base_pos_now, _ = self._base_trajectories[drone_id].get_state_at_time(
                    self._base_flight_time(drone_id, current_time)
                )
                lateral_disp = float(np.linalg.norm(current_pos - np.asarray(base_pos_now)))
                if lateral_disp >= self._lateral_progress_min_m:
                    return  # bieżący plan już efektywnie wymanewrowuje drona

            # Zawsze wymagamy minimum okna realizacji planu, niezależnie od
            # powodu replanowania. Bez tego throttla przy wielu przeszkodach
            # oracle skakał między celami i PID nie nadążał wykonać żadnego
            # z kolejnych planów → narastający pos_err i crash.
            if time_since < self._imminent_replan_min_dt:
                return
            should_trigger = True  # wymuszamy przebudowę

        if not should_trigger:
            return

        logger = getattr(self.parent, "logger", None)
        if logger is not None and hasattr(logger, "log_evasion_event"):
            logger.log_evasion_event(
                current_time=current_time,
                drone_id=drone_id,
                event_type="trigger",
                mode=int(self._flight_modes[drone_id]),
                ttc=float(ttc) if np.isfinite(ttc) else float("inf"),
                dist_to_threat=float(dist),
                threat_pos=np.asarray(threat["position"], dtype=np.float64),
                threat_vel=np.asarray(obs_vel, dtype=np.float64),
            )

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
        # Faza 2.5: pad bbox adaptacyjny do |rel_vel|. Pobierany z configu A*
        # by obie struktury (pad w Context) używały
        # tej samej stałej — spójność buforów.
        margin_gain = float(self.avoidance_algorithm.params.get("margin_velocity_gain", 0.05))
        rejoin_arc_m = float(self.avoidance_algorithm.params.get("rejoin_arc_distance_m", 8.0))
        rejoin_flyby_safety_m = float(
            self.avoidance_algorithm.params.get("rejoin_flyby_safety_m", 3.0)
        )
        lateral_max_offset_m = float(
            self.avoidance_algorithm.params.get("lateral_max_offset_m", 8.0)
        )
        builder = EvasionContextBuilder(
            t_min=t_min,
            t_max=t_max,
            rejoin_arc_m=rejoin_arc_m,
            margin_velocity_gain=margin_gain,
            rejoin_flyby_safety_m=rejoin_flyby_safety_m,
            lateral_max_offset_m=lateral_max_offset_m,
        )
        
        # Sticky axis: w trybie EVASION/BLEND przekazujemy oś z poprzedniego planu
        # jako hint; w TRACKING (pierwszy trigger) hint = None — planner wybiera oś
        # od zera na bazie geometrii i obs_vel.
        axis_hint = (
            self._last_preferred_axis[drone_id]
            if self._flight_modes[drone_id] in (self.MODE_EVASION, self.MODE_REJOIN_BLEND)
            else None
        )

        # Sequential cooperative planning (2026-05-01): zbierz pozostałe drony
        # (≠ drone_id) jako secondary_threats z DOKŁADNĄ TRAJEKTORIĄ (base spline
        # w MODE_TRACKING, evasion spline w MODE_EVASION/REJOIN_BLEND). Bo pętla
        # `_run_lidar_and_detect` przetwarza drony sekwencyjnie 0..N, drony
        # planujące się POŹNIEJ w tym samym ticku widzą NOWO ZBUDOWANE evasion
        # plany wcześniejszych dronów (sequential coordination prioritized
        # planning a la Erdmann & Lozano-Pérez 1987, Van den Berg & Overmars 2005).
        # Filtrujemy do max 30 m promienia (w 3s window dalsze i tak nie kolidują).
        secondary_threats: list[ThreatAlert] = []
        if all_drone_states is not None and len(all_drone_states) > 1:
            primary_obs_pos = np.asarray(threat["position"], dtype=np.float64)
            for j, st in enumerate(all_drone_states):
                if j == drone_id:
                    continue
                other_pos = np.asarray(st[0:3], dtype=np.float64)
                # Pomijamy drony bardzo blisko primary threat (są albo same primary,
                # albo wyrównane z nim na lidarze i już uwzględnione).
                if np.linalg.norm(other_pos - primary_obs_pos) < 1.5:
                    continue
                rel_d = float(np.linalg.norm(other_pos - current_pos))
                if rel_d > 30.0:
                    continue
                other_vel = np.asarray(st[10:13], dtype=np.float64)

                # Determine drone j's active trajectory + offset on it.
                # Priorytet: evasion spline (jeśli aktywne) → base spline → linear.
                j_traj: object | None = None
                j_offset: float = 0.0
                j_mode = self._flight_modes[j]
                if (
                    j_mode in (self.MODE_EVASION, self.MODE_REJOIN_BLEND)
                    and self._evasion_trajectories[j] is not None
                ):
                    j_traj = self._evasion_trajectories[j]
                    j_offset = float(current_time - self._evasion_start_times[j])
                elif self._base_trajectories is not None:
                    j_traj = self._base_trajectories[j]
                    j_offset = float(current_time - self._tracking_start_times[j])

                secondary_threats.append(
                    ThreatAlert(
                        obstacle_state=KinematicState(
                            position=other_pos,
                            velocity=other_vel,
                            radius=self.collision_radius,
                        ),
                        distance=rel_d,
                        time_to_collision=float("inf"),
                        relative_velocity=current_vel - other_vel,
                        trajectory=j_traj,
                        trajectory_start_offset=j_offset,
                    )
                )

        context = builder.build(
            drone_id=drone_id,
            current_time=current_time,
            drone_state=drone_kinematic,
            threat=threat_alert,
            base_spline=base_spline,
            base_arc_progress=base_arc,
            env_bounds=env_bounds,
            preferred_axis_hint=axis_hint,
            secondary_threats=secondary_threats,
        )

        plan = self.avoidance_algorithm.compute_evasion_plan(context)

        if plan is None:
            # Cooldown: nie próbuj znowu przez `_no_plan_cooldown_s` sekund.
            # Zapobiega re-trigger storm gdy geometria jest aktualnie infeasible.
            self._no_plan_until[drone_id] = current_time + self._no_plan_cooldown_s
            print(f"[WARN] Plan uniku nie powstał dla drona {drone_id} — pozostajemy w TRACKING")
            if logger is not None and hasattr(logger, "log_evasion_event"):
                logger.log_evasion_event(
                    current_time=current_time,
                    drone_id=drone_id,
                    event_type="no_plan",
                    mode=int(self._flight_modes[drone_id]),
                    ttc=float(ttc) if np.isfinite(ttc) else float("inf"),
                    dist_to_threat=float(dist),
                    threat_pos=np.asarray(threat["position"], dtype=np.float64),
                    threat_vel=np.asarray(obs_vel, dtype=np.float64),
                    notes=f"compute_evasion_plan returned None; cooldown {self._no_plan_cooldown_s}s",
                )
            return

        self._evasion_trajectories[drone_id] = plan.evasion_spline
        self._evasion_start_times[drone_id] = current_time
        self._rejoin_base_arcs[drone_id] = plan.rejoin_base_arc
        self._rejoin_points[drone_id] = np.asarray(plan.rejoin_point, dtype=np.float64)
        self._last_threat_positions[drone_id] = np.array(threat["position"], dtype=np.float64)
        self._last_threat_velocities[drone_id] = np.array(obs_vel, dtype=np.float64)
        self._last_threat_times[drone_id] = float(current_time)
        obs_idx_value = threat.get("obs_idx") if isinstance(threat, dict) else None
        self._last_threat_obs_idx[drone_id] = int(obs_idx_value) if obs_idx_value is not None else None
        self._last_preferred_axis[drone_id] = plan.preferred_axis
        self._flight_modes[drone_id] = self.MODE_EVASION

        if logger is not None and hasattr(logger, "log_evasion_event"):
            logger.log_evasion_event(
                current_time=current_time,
                drone_id=drone_id,
                event_type="plan_built",
                mode=int(self._flight_modes[drone_id]),
                ttc=float(ttc) if np.isfinite(ttc) else float("inf"),
                dist_to_threat=float(dist),
                threat_pos=np.asarray(threat["position"], dtype=np.float64),
                threat_vel=np.asarray(obs_vel, dtype=np.float64),
                rejoin_point=np.asarray(plan.rejoin_point, dtype=np.float64),
                rejoin_arc=float(plan.rejoin_base_arc),
                astar_success=plan.astar_success,
                fallback_used=plan.fallback_used,
                planning_wall_time_s=plan.planning_wall_time_s,
                notes=f"axis={plan.preferred_axis}",
            )

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