import os
import numpy as np
from scipy.interpolate import splev
import matplotlib.pyplot as plt

from src.algorithms.avoidance.AStar.UAV3DGridSearch import UAV3DGridSearch
from src.algorithms.avoidance.BaseAvoidance import BaseAvoidance, EvasionPlan, EvasionContext
from src.trajectory.BSplineTrajectory import BSplineTrajectory


class AStarAvoidance(BaseAvoidance):
    """
    Lokalny A* zaimplementowany na zunifikowanej architekturze EvasionContext.
    
    Wykorzystuje granice przestrzeni poszukiwań zdefiniowane przez ContextBuilder
    (uwzględniające Velocity Obstacles - predykcję ruchu przeszkód), delegując
    skomplikowane wyliczenia geometrii analitycznej poza algorytm.
    """

    def __init__(self, **kwargs):
        algo_name = kwargs.pop("name", "A* Local Planner (Context-Aware)")
        super().__init__(name=algo_name, **kwargs)

        # --- A* ---
        self.grid_res = float(self.params.get("grid_resolution", 0.5))
        self.margin = float(self.params.get("margin_multiplier", 1.8))
        self.max_depth = int(self.params.get("max_search_depth", 2500))

        # --- Heurystyka kierunku ---
        self.prefer_axis_order = list(
            self.params.get("prefer_axis_order", ["up", "down", "right", "left"])
        )
        self.bias_preferred = float(self.params.get("bias_preferred", 1.0))
        self.bias_perpendicular = float(self.params.get("bias_perpendicular", 1.4))
        self.bias_oppose = float(self.params.get("bias_oppose", 2.5))

        # --- Downsampling ---
        self.min_waypoints = int(self.params.get("min_waypoints", 4))
        self.max_waypoints = int(self.params.get("max_waypoints", 8))
        self.dp_epsilon = float(self.params.get("douglas_peucker_epsilon_m", 0.6))

        # --- Prędkość ---
        self.speed_multiplier = float(self.params.get("evasion_speed_multiplier", 0.85))

        # --- Wizualizacja ---
        self.visualize = bool(self.params.get("visualize", True))
        self.viz_dir = os.path.join(os.getcwd(), "evasion_plots")
        if self.visualize:
            os.makedirs(self.viz_dir, exist_ok=True)

    # =========================================================================
    # Główne API
    # =========================================================================

    def compute_evasion_plan(self, context: EvasionContext) -> EvasionPlan | None:
        # Rozpakowanie zunifikowanego stanu
        current_pos = context.drone_state.position
        current_vel = context.drone_state.velocity
        current_speed = float(np.linalg.norm(current_vel))

        obs_pos = context.threat.obstacle_state.position
        obs_radius = context.threat.obstacle_state.radius * self.margin

        rejoin_point = context.rejoin_point
        bbox_min = context.search_space_min
        bbox_max = context.search_space_max

        world_min, world_max = context.world_bounds
        floor_z = float(world_min[2])
        ceiling_z = float(world_max[2])

        # --- Prędkość i kierunek ruchu ---
        cruise = float(getattr(context.base_spline.profile, "cruise_speed", 8.0))
        ref_speed = max(current_speed, cruise * 0.5)

        forward_3d = self._compute_forward_direction(
            current_vel, context.base_spline, context.rejoin_base_arc
        )
        forward_xy = np.array([forward_3d[0], forward_3d[1], 0.0])
        fnorm = float(np.linalg.norm(forward_xy))
        forward_xy = forward_xy / fnorm if fnorm > 1e-6 else np.array([1.0, 0.0, 0.0])
        lateral_xy = np.array([-forward_xy[1], forward_xy[0], 0.0])

        # --- Preferowana oś ---
        axis_name, preferred_dir = self._pick_preferred_axis(
            current_pos=current_pos,
            obs_pos=obs_pos,
            forward_xy=forward_xy,
            lateral_xy=lateral_xy,
            floor_z=floor_z,
            ceiling_z=ceiling_z,
            world_min=world_min,
            world_max=world_max,
        )

        # --- Start/goal snapped do gridu ---
        start_snapped = self._snap_inside_bbox(current_pos, bbox_min, bbox_max)
        goal_snapped = self._snap_inside_bbox(rejoin_point, bbox_min, bbox_max)

        start_t = tuple(np.round(start_snapped / self.grid_res) * self.grid_res)
        goal_t = tuple(np.round(goal_snapped / self.grid_res) * self.grid_res)

        # --- A* ---
        searcher = UAV3DGridSearch(
            obs_pos=obs_pos,
            obs_radius=obs_radius,
            grid_res=self.grid_res,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            preferred_dir=preferred_dir,
            bias_preferred=self.bias_preferred,
            bias_perpendicular=self.bias_perpendicular,
            bias_oppose=self.bias_oppose,
        )

        astar_raw = None
        used_fallback = False
        try:
            path_iter = searcher.astar(start_t, goal_t)
            if path_iter is not None:
                astar_list = list(path_iter)
                if len(astar_list) >= 2:
                    astar_raw = np.array(astar_list, dtype=np.float64)
        except Exception as e:
            print(f"[WARN] A* exception drona {context.drone_id}: {e}")

        if astar_raw is None:
            print(f"[WARN] A* nie znalazł ścieżki dla drona {context.drone_id} — fallback")
            astar_raw = self._fallback_path(
                current_pos, rejoin_point, preferred_dir, obs_pos, obs_radius,
                floor_z, ceiling_z,
            )
            used_fallback = True

        # --- Downsample: Douglas-Peucker ---
        waypoints = self._douglas_peucker(astar_raw, self.dp_epsilon)

        # Zawsze zachowuj dokładne pozycje krańcowe (start/rejoin)
        waypoints[0] = current_pos
        waypoints[-1] = rejoin_point

        # Dołóż punkty pośrednie jeśli za mało
        if len(waypoints) < self.min_waypoints:
            waypoints = self._resample_uniform(astar_raw, self.min_waypoints)
            waypoints[0] = current_pos
            waypoints[-1] = rejoin_point

        # Przytnij do max_waypoints
        if len(waypoints) > self.max_waypoints:
            waypoints = self._resample_uniform(waypoints, self.max_waypoints)
            waypoints[0] = current_pos
            waypoints[-1] = rejoin_point

        # --- Waypointy kontynuacji stycznej na brzegach ---
        base_tangent_at_rejoin = self._base_tangent_at_arc(context.base_spline, context.rejoin_base_arc)
        waypoints = self._insert_tangent_leads(
            waypoints=waypoints,
            current_pos=current_pos,
            forward_3d=forward_3d,
            rejoin_point=rejoin_point,
            base_tangent_at_rejoin=base_tangent_at_rejoin,
            ref_speed=ref_speed,
            obs_pos=obs_pos,
            obs_radius=obs_radius,
        )

        waypoints[:, 2] = np.clip(waypoints[:, 2], floor_z, ceiling_z)

        # --- Budowa lokalnego splinu ---
        evasion_cruise = max(current_speed, cruise * self.speed_multiplier)
        max_accel = float(getattr(context.base_spline.profile, "max_accel", 2.0))
        try:
            evasion_spline = BSplineTrajectory(
                waypoints=waypoints,
                cruise_speed=evasion_cruise,
                max_accel=max_accel,
                constant_speed=True,
            )
        except Exception as e:
            print(f"[ERROR] BSpline build fail dron {context.drone_id}: {e}")
            return None

        plan = EvasionPlan(
            evasion_spline=evasion_spline,
            rejoin_point=rejoin_point,
            rejoin_base_arc=context.rejoin_base_arc,
            preferred_axis=axis_name,
        )

        if self.visualize:
            self._visualize(
                context=context,
                evasion_spline=evasion_spline,
                waypoints=waypoints,
                astar_raw=astar_raw,
                axis_name=axis_name,
                used_fallback=used_fallback,
            )

        return plan

    # =========================================================================
    # Heurystyka wyboru kierunku
    # =========================================================================

    def _pick_preferred_axis(
        self,
        current_pos: np.ndarray,
        obs_pos: np.ndarray,
        forward_xy: np.ndarray,
        lateral_xy: np.ndarray,
        floor_z: float,
        ceiling_z: float,
        world_min: np.ndarray,
        world_max: np.ndarray,
    ) -> tuple[str, np.ndarray]:
        space = {
            "up":    ceiling_z - current_pos[2],
            "down":  current_pos[2] - floor_z,
            "right": self._space_in_xy_dir(current_pos, lateral_xy, world_min, world_max),
            "left":  self._space_in_xy_dir(current_pos, -lateral_xy, world_min, world_max),
        }

        min_required = 2.5

        dir_map = {
            "up":    np.array([0.0, 0.0, 1.0]),
            "down":  np.array([0.0, 0.0, -1.0]),
            "right": lateral_xy.copy(),
            "left":  -lateral_xy.copy(),
        }

        for axis in self.prefer_axis_order:
            if axis in space and space[axis] >= min_required:
                return axis, dir_map[axis]

        axis = max(space.keys(), key=lambda a: space[a])
        return axis, dir_map[axis]

    @staticmethod
    def _space_in_xy_dir(
        pos: np.ndarray,
        direction: np.ndarray,
        world_min: np.ndarray,
        world_max: np.ndarray,
    ) -> float:
        dir_xy = np.array([direction[0], direction[1]])
        norm = float(np.linalg.norm(dir_xy))
        if norm < 1e-6:
            return 0.0
        dir_xy = dir_xy / norm

        dists = []
        for i in range(2):
            if dir_xy[i] > 1e-6:
                dists.append((world_max[i] - pos[i]) / dir_xy[i])
            elif dir_xy[i] < -1e-6:
                dists.append((world_min[i] - pos[i]) / dir_xy[i])
        return float(min(dists)) if dists else 0.0

    # =========================================================================
    # Geometria bazowego splinu
    # =========================================================================

    @staticmethod
    def _base_tangent_at_arc(spline: BSplineTrajectory, arc: float) -> np.ndarray:
        if spline.arc_length <= 1e-6:
            return np.array([1.0, 0.0, 0.0])
        u = float(np.clip(arc / spline.arc_length, 0.0, 1.0))
        tangent = np.array(splev(u, spline.tck, der=1), dtype=np.float64)
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return tangent / norm

    def _insert_tangent_leads(
        self,
        waypoints: np.ndarray,
        current_pos: np.ndarray,
        forward_3d: np.ndarray,
        rejoin_point: np.ndarray,
        base_tangent_at_rejoin: np.ndarray,
        ref_speed: float,
        obs_pos: np.ndarray,
        obs_radius: float,
    ) -> np.ndarray:
        if len(waypoints) < 2:
            return waypoints

        lead_dist = float(np.clip(ref_speed * 0.1, 0.3, 0.8))

        fwd = np.asarray(forward_3d, dtype=np.float64)
        fwd_norm = float(np.linalg.norm(fwd))
        lead_in = None
        if fwd_norm > 1e-6:
            fwd = fwd / fwd_norm
            candidate = current_pos + fwd * lead_dist
            if self._is_point_safe(candidate, obs_pos, obs_radius):
                lead_in = candidate

        back = np.asarray(base_tangent_at_rejoin, dtype=np.float64)
        back_norm = float(np.linalg.norm(back))
        lead_out = None
        if back_norm > 1e-6:
            back = back / back_norm
            candidate = rejoin_point - back * lead_dist
            if self._is_point_safe(candidate, obs_pos, obs_radius):
                lead_out = candidate

        head = [current_pos]
        if lead_in is not None:
            head.append(lead_in)

        tail = [rejoin_point]
        if lead_out is not None:
            tail.insert(0, lead_out)

        middle = [waypoints[i] for i in range(1, len(waypoints) - 1)]

        combined = np.array(head + middle + tail, dtype=np.float64)
        if len(combined) < 4:
            mid = 0.5 * (combined[0] + combined[-1])
            combined = np.vstack([combined[:1], mid, combined[1:]])
        return combined

    @staticmethod
    def _is_point_safe(point: np.ndarray, obs_pos: np.ndarray, obs_radius: float) -> bool:
        return float(np.linalg.norm(point - obs_pos)) >= obs_radius + 0.1

    @staticmethod
    def _compute_forward_direction(
        current_vel: np.ndarray,
        base_spline: BSplineTrajectory,
        base_arc_progress: float,
    ) -> np.ndarray:
        speed = float(np.linalg.norm(current_vel))
        if speed > 0.5:
            return current_vel / speed

        if base_spline.arc_length > 1e-6:
            u = float(np.clip(base_arc_progress / base_spline.arc_length, 0.0, 1.0))
        else:
            u = 0.0
        tangent = np.array(splev(u, base_spline.tck, der=1), dtype=np.float64)
        tnorm = float(np.linalg.norm(tangent))
        if tnorm > 1e-6:
            return tangent / tnorm
        return np.array([1.0, 0.0, 0.0])

    @staticmethod
    def _snap_inside_bbox(
        point: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
    ) -> np.ndarray:
        return np.minimum(np.maximum(point, bbox_min), bbox_max)

    # =========================================================================
    # Path post-processing
    # =========================================================================

    @staticmethod
    def _douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
        if len(points) < 3:
            return points.copy()

        def rdp(pts: np.ndarray) -> np.ndarray:
            if len(pts) < 3:
                return pts
            a = pts[0]
            b = pts[-1]
            ab = b - a
            ab_sq = float(np.dot(ab, ab))
            if ab_sq < 1e-12:
                return np.vstack([a, b])
            d = pts - a
            t = np.clip(d @ ab / ab_sq, 0.0, 1.0)
            closest = a + t[:, None] * ab
            errors = np.linalg.norm(pts - closest, axis=1)
            idx = int(np.argmax(errors))
            if errors[idx] < epsilon:
                return np.vstack([a, b])
            left = rdp(pts[: idx + 1])
            right = rdp(pts[idx:])
            return np.vstack([left[:-1], right])

        return rdp(points)

    @staticmethod
    def _resample_uniform(points: np.ndarray, n: int) -> np.ndarray:
        if len(points) <= 2 or n <= 2:
            if len(points) >= 2:
                return np.vstack([points[0], points[-1]])
            return points.copy()

        seg_lens = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
        total = float(cum[-1])
        if total < 1e-6:
            return np.tile(points[0], (n, 1))

        targets = np.linspace(0.0, total, n)
        out = []
        j = 0
        for s in targets:
            while j < len(cum) - 2 and cum[j + 1] < s:
                j += 1
            s0, s1 = cum[j], cum[j + 1]
            ratio = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
            out.append(points[j] + ratio * (points[j + 1] - points[j]))
        return np.array(out, dtype=np.float64)

    # =========================================================================
    # Fallback ścieżka
    # =========================================================================

    def _fallback_path(
        self,
        current_pos: np.ndarray,
        rejoin_point: np.ndarray,
        preferred_dir: np.ndarray,
        obs_pos: np.ndarray,
        obs_radius: float,
        floor_z: float,
        ceiling_z: float,
    ) -> np.ndarray:
        mid = 0.5 * (current_pos + rejoin_point)
        offset_m = max(obs_radius * 1.5, 2.5)
        apex = mid + preferred_dir * offset_m
        apex[2] = float(np.clip(apex[2], floor_z, ceiling_z))

        if np.linalg.norm(apex - obs_pos) < obs_radius * 1.2:
            apex = mid + preferred_dir * (offset_m * 2.0)
            apex[2] = float(np.clip(apex[2], floor_z, ceiling_z))

        return np.vstack([current_pos, apex, rejoin_point])

    # =========================================================================
    # Wizualizacja
    # =========================================================================

    def _visualize(
        self,
        context: EvasionContext,
        evasion_spline: BSplineTrajectory,
        waypoints: np.ndarray,
        astar_raw: np.ndarray,
        axis_name: str,
        used_fallback: bool,
    ) -> None:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(projection='3d')

        base_spline = context.base_spline
        drone_id = context.drone_id
        current_time = context.current_time
        current_pos = context.drone_state.position
        obs_pos = context.threat.obstacle_state.position
        obs_radius = context.threat.obstacle_state.radius * self.margin
        rejoin_point = context.rejoin_point
        bbox_min = context.search_space_min
        bbox_max = context.search_space_max

        # Bazowy spline
        t_base = np.linspace(0.0, base_spline.total_duration, 250)
        base_xyz = np.array([base_spline.get_state_at_time(float(t))[0] for t in t_base])
        ax.plot(base_xyz[:, 0], base_xyz[:, 1], base_xyz[:, 2],
                color='grey', ls='--', alpha=0.5, label='Bazowa trasa')

        # Spline uniku
        t_ev = np.linspace(0.0, evasion_spline.total_duration, 200)
        ev_xyz = np.array([evasion_spline.get_state_at_time(float(t))[0] for t in t_ev])
        ax.plot(ev_xyz[:, 0], ev_xyz[:, 1], ev_xyz[:, 2],
                color='tab:blue', lw=2.2, label='Spline uniku')

        # Raw A*
        if astar_raw is not None and len(astar_raw) > 0:
            ax.scatter(astar_raw[:, 0], astar_raw[:, 1], astar_raw[:, 2],
                       color='lightgrey', s=6, alpha=0.5, label='A* raw')

        # Waypointy splinu uniku
        ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                   color='tab:orange', s=55, edgecolors='black',
                   label='Waypointy (fallback)' if used_fallback else 'Waypointy (DP)')

        # Bounding box
        self._plot_bbox(ax, bbox_min, bbox_max)

        # Przeszkoda i Predykcja VO (Wektor prędkości)
        self._plot_sphere(ax, obs_pos, obs_radius, color='red', alpha=0.25)
        ax.scatter(*obs_pos, color='red', s=90, marker='x', label='Przeszkoda (Start)')
        
        # Rysowanie wektora prędkości przeszkody
        obs_future = obs_pos + (context.threat.obstacle_state.velocity * 2.0) # predykcja 2s
        ax.plot([obs_pos[0], obs_future[0]], [obs_pos[1], obs_future[1]], [obs_pos[2], obs_future[2]], 
                color='darkred', lw=2, linestyle=':', label='Wektor prędkości (VO)')

        ax.scatter(*current_pos, color='green', s=110, marker='^', label='Dron')
        ax.scatter(*rejoin_point, color='magenta', s=110, marker='*', label='Rejoin')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(
            f'Unik d{drone_id} t={current_time:.2f}s oś={axis_name} '
            f'{"[fallback]" if used_fallback else "[A*]"} '
            f'|wp|={len(waypoints)}'
        )
        ax.legend(loc='upper left', fontsize=8)

        fname = os.path.join(
            self.viz_dir,
            f'evasion_d{drone_id}_t{current_time:07.2f}.png',
        )
        plt.savefig(fname, dpi=140, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _plot_sphere(ax, center, radius, color='red', alpha=0.25):
        u = np.linspace(0.0, 2.0 * np.pi, 24)
        v = np.linspace(0.0, np.pi, 18)
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        x = center[0] + radius * np.outer(cu, sv)
        y = center[1] + radius * np.outer(su, sv)
        z = center[2] + radius * np.outer(np.ones_like(u), cv)
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    @staticmethod
    def _plot_bbox(ax, bbox_min, bbox_max):
        xmin, ymin, zmin = bbox_min
        xmax, ymax, zmax = bbox_max
        edges = [
            [(xmin, ymin, zmin), (xmax, ymin, zmin)],
            [(xmax, ymin, zmin), (xmax, ymax, zmin)],
            [(xmax, ymax, zmin), (xmin, ymax, zmin)],
            [(xmin, ymax, zmin), (xmin, ymin, zmin)],
            [(xmin, ymin, zmax), (xmax, ymin, zmax)],
            [(xmax, ymin, zmax), (xmax, ymax, zmax)],
            [(xmax, ymax, zmax), (xmin, ymax, zmax)],
            [(xmin, ymax, zmax), (xmin, ymin, zmax)],
            [(xmin, ymin, zmin), (xmin, ymin, zmax)],
            [(xmax, ymin, zmin), (xmax, ymin, zmax)],
            [(xmax, ymax, zmin), (xmax, ymax, zmax)],
            [(xmin, ymax, zmin), (xmin, ymax, zmax)],
        ]
        for (a, b) in edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color='black', lw=0.5, alpha=0.35)