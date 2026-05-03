import numpy as np
from numba import njit
from scipy.interpolate import splev
from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import KinematicState, ThreatAlert
from src.trajectory.BSplineTrajectory import BSplineTrajectory
from src.algorithms.avoidance.BaseAvoidance import EvasionContext

# --------------------------------------------------------------------------- #
# KERNELS NUMBA                                                                #
# --------------------------------------------------------------------------- #

@njit(cache=True, fastmath=True)
def jit_compute_flyby_arc(drone_pos_x: float, drone_pos_y: float, drone_pos_z: float,
                           drone_vel_x: float, drone_vel_y: float, drone_vel_z: float,
                           obs_pos_x: float, obs_pos_y: float, obs_pos_z: float,
                           obs_vel_x: float, obs_vel_y: float, obs_vel_z: float,
                           obs_radius: float, ref_speed: float,
                           t_max: float, rejoin_flyby_safety_m: float) -> float:

    speed = (drone_vel_x**2 + drone_vel_y**2 + drone_vel_z**2) ** 0.5
    if speed < 1e-3:
        return 0.0

    inv_speed = 1.0 / speed
    f_x = drone_vel_x * inv_speed
    f_y = drone_vel_y * inv_speed
    f_z = drone_vel_z * inv_speed

    rp_x = obs_pos_x - drone_pos_x
    rp_y = obs_pos_y - drone_pos_y
    rp_z = obs_pos_z - drone_pos_z

    rv_x = drone_vel_x - obs_vel_x
    rv_y = drone_vel_y - obs_vel_y
    rv_z = drone_vel_z - obs_vel_z

    rel_pos_fwd = rp_x * f_x + rp_y * f_y + rp_z * f_z
    rel_vel_fwd = rv_x * f_x + rv_y * f_y + rv_z * f_z

    if rel_vel_fwd < 0.3 or rel_pos_fwd <= 0.0:
        return 0.0

    t_pass = rel_pos_fwd / rel_vel_fwd
    limit = 2.0 * t_max
    if t_pass > limit:
        t_pass = limit

    return ref_speed * t_pass + rejoin_flyby_safety_m + obs_radius


@njit(cache=True, fastmath=True)
def jit_build_dynamic_search_space(current_pos: np.ndarray, rejoin_point: np.ndarray,
                                    obs_pos: np.ndarray, obs_vel: np.ndarray,
                                    relative_velocity: np.ndarray, obs_radius: float,
                                    forward_xy: np.ndarray, lateral_xy: np.ndarray,
                                    ref_speed: float, t_max: float, lateral_margin: float,
                                    margin_velocity_gain: float,
                                    floor_z: float, ceiling_z: float,
                                    world_min: np.ndarray, world_max: np.ndarray):

    forward_margin = ref_speed * t_max

    obs_future_x = obs_pos[0] + obs_vel[0] * t_max
    obs_future_y = obs_pos[1] + obs_vel[1] * t_max
    obs_future_z = obs_pos[2] + obs_vel[2] * t_max

    extra_fwd_x = current_pos[0] + forward_xy[0] * forward_margin
    extra_fwd_y = current_pos[1] + forward_xy[1] * forward_margin
    extra_fwd_z = current_pos[2] + forward_xy[2] * forward_margin

    extra_lat_px = current_pos[0] + lateral_xy[0] * lateral_margin
    extra_lat_py = current_pos[1] + lateral_xy[1] * lateral_margin
    extra_lat_pz = current_pos[2] + lateral_xy[2] * lateral_margin

    extra_lat_nx = current_pos[0] - lateral_xy[0] * lateral_margin
    extra_lat_ny = current_pos[1] - lateral_xy[1] * lateral_margin
    extra_lat_nz = current_pos[2] - lateral_xy[2] * lateral_margin

    # Ręczne min/max - brak alokacji pośrednich tablic przez np.array([...])
    bbox_min = np.empty(3, dtype=np.float64)
    bbox_max = np.empty(3, dtype=np.float64)

    all_x = (current_pos[0], rejoin_point[0], obs_pos[0], obs_future_x,
             extra_fwd_x, extra_lat_px, extra_lat_nx)
    all_y = (current_pos[1], rejoin_point[1], obs_pos[1], obs_future_y,
             extra_fwd_y, extra_lat_py, extra_lat_ny)
    all_z = (current_pos[2], rejoin_point[2], obs_pos[2], obs_future_z,
             extra_fwd_z, extra_lat_pz, extra_lat_nz)

    mn_x = all_x[0]; mx_x = all_x[0]
    mn_y = all_y[0]; mx_y = all_y[0]
    mn_z = all_z[0]; mx_z = all_z[0]

    for i in range(1, 7):
        if all_x[i] < mn_x: mn_x = all_x[i]
        if all_x[i] > mx_x: mx_x = all_x[i]
        if all_y[i] < mn_y: mn_y = all_y[i]
        if all_y[i] > mx_y: mx_y = all_y[i]
        if all_z[i] < mn_z: mn_z = all_z[i]
        if all_z[i] > mx_z: mx_z = all_z[i]

    rv_mag = (relative_velocity[0]**2 + relative_velocity[1]**2 + relative_velocity[2]**2) ** 0.5
    pad = obs_radius + 1.0 + margin_velocity_gain * rv_mag

    bbox_min[0] = mn_x - pad
    bbox_min[1] = mn_y - pad
    bbox_min[2] = mn_z - pad

    bbox_max[0] = mx_x + pad
    bbox_max[1] = mx_y + pad
    bbox_max[2] = mx_z + pad

    if bbox_min[2] < floor_z: bbox_min[2] = floor_z
    if bbox_max[2] > ceiling_z: bbox_max[2] = ceiling_z

    w_min_0 = world_min[0] + pad
    w_min_1 = world_min[1] + pad
    if bbox_min[0] < w_min_0: bbox_min[0] = w_min_0
    if bbox_min[1] < w_min_1: bbox_min[1] = w_min_1

    w_max_0 = world_max[0] - pad
    w_max_1 = world_max[1] - pad
    if bbox_max[0] > w_max_0: bbox_max[0] = w_max_0
    if bbox_max[1] > w_max_1: bbox_max[1] = w_max_1

    return bbox_min, bbox_max

# --------------------------------------------------------------------------- #

class EvasionContextBuilder:
    """
    Preprocesor akademicki dla algorytmów unikania kolizji.
    Zoptymalizowany numerycznie: ciężkie jądra matematyczne skompilowane przez
    Numba JIT, splev (FITPACK/Fortran) pozostaje po stronie Pythona.
    """
    def __init__(self, t_min=2.0, t_max=4.0, rejoin_arc_m=8.0,
                 floor_margin=1.0, ceiling_margin=1.0, lateral_margin=4.0,
                 margin_velocity_gain: float = 0.05,
                 rejoin_flyby_safety_m: float = 3.0,
                 lateral_max_offset_m: float = 8.0):
        self.t_min = t_min
        self.t_max = t_max
        self.rejoin_arc_m = rejoin_arc_m
        self.floor_margin = floor_margin
        self.ceiling_margin = ceiling_margin
        self.lateral_margin = lateral_margin
        self.margin_velocity_gain = margin_velocity_gain
        self.rejoin_flyby_safety_m = rejoin_flyby_safety_m
        # Hard cap na lateralną odchyłkę BBOX-u uniku od drona (Bug #2 plan,
        # Krok 3c). Bez tego BBOX rośnie z VO + obs_future, AStar wybiera
        # waypointy odlegające o dziesiątki metrów → ostre zakrzywienia.
        # Cap działa w osi lateralnej (XY ⊥ forward) i osi Z. Forward NIE
        # jest cap'owany — drone musi sięgnąć rejoin pointu. `<= 0` wyłącza.
        self.lateral_max_offset_m = float(lateral_max_offset_m)

    def build(self,
              drone_id: int,
              current_time: float,
              drone_state: KinematicState,
              threat: ThreatAlert,
              base_spline: BSplineTrajectory,
              base_arc_progress: float,
              env_bounds: tuple[np.ndarray, np.ndarray],
              preferred_axis_hint: str | None = None,
              secondary_threats: list[ThreatAlert] | None = None) -> EvasionContext:

        world_min, world_max = env_bounds
        floor_z = float(world_min[2]) + self.floor_margin
        ceiling_z = float(world_max[2]) - self.ceiling_margin

        v = drone_state.velocity
        current_speed = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) ** 0.5
        cruise = float(getattr(base_spline, "cruise_speed", 8.0))
        ref_speed = current_speed if current_speed > cruise * 0.5 else cruise * 0.5

        t_pos = threat.obstacle_state.position
        t_vel = threat.obstacle_state.velocity

        flyby_arc = jit_compute_flyby_arc(
            float(drone_state.position[0]), float(drone_state.position[1]), float(drone_state.position[2]),
            float(v[0]), float(v[1]), float(v[2]),
            float(t_pos[0]), float(t_pos[1]), float(t_pos[2]),
            float(t_vel[0]), float(t_vel[1]), float(t_vel[2]),
            float(threat.obstacle_state.radius), ref_speed,
            self.t_max, self.rejoin_flyby_safety_m
        )

        min_arc = current_speed * self.t_min
        chosen_arc_offset = self.rejoin_arc_m if self.rejoin_arc_m > min_arc else min_arc
        chosen_arc_offset = flyby_arc if flyby_arc > chosen_arc_offset else chosen_arc_offset

        target_arc = base_arc_progress + chosen_arc_offset
        if target_arc > base_spline.arc_length:
            target_arc = base_spline.arc_length

        # splev pozostaje w Pythonie - niedostępny dla @njit
        rejoin_point = self._sample_base_at_arc(base_spline, target_arc)
        if rejoin_point[2] < floor_z: rejoin_point[2] = floor_z
        elif rejoin_point[2] > ceiling_z: rejoin_point[2] = ceiling_z

        forward_3d = self._compute_forward_direction(drone_state.velocity, base_spline, base_arc_progress)

        fnorm = (forward_3d[0]*forward_3d[0] + forward_3d[1]*forward_3d[1]) ** 0.5
        if fnorm > 1e-6:
            inv_fnorm = 1.0 / fnorm
            forward_xy = np.array([forward_3d[0] * inv_fnorm, forward_3d[1] * inv_fnorm, 0.0])
        else:
            forward_xy = np.array([1.0, 0.0, 0.0])

        lateral_xy = np.array([-forward_xy[1], forward_xy[0], 0.0])

        search_min, search_max = jit_build_dynamic_search_space(
            np.asarray(drone_state.position, dtype=np.float64),
            rejoin_point,
            np.asarray(threat.obstacle_state.position, dtype=np.float64),
            np.asarray(threat.obstacle_state.velocity, dtype=np.float64),
            np.asarray(threat.relative_velocity, dtype=np.float64),
            float(threat.obstacle_state.radius),
            forward_xy, lateral_xy,
            ref_speed, self.t_max, self.lateral_margin,
            self.margin_velocity_gain,
            floor_z, ceiling_z,
            np.asarray(world_min, dtype=np.float64),
            np.asarray(world_max, dtype=np.float64)
        )

        # Hard cap na lateralną odchyłkę BBOX-u (Bug #2 Krok 3c). Cap'ujemy:
        #  - oś Z (zawsze prostopadła do forward),
        #  - dominującą oś XY lateralnej (X jeśli |lateral_xy.x| > |lateral_xy.y|,
        #    inaczej Y) — przybliżenie axis-aligned bbox-u do pełnej rotacji.
        # Forward NIE cap'ujemy. Rejoin point i drone muszą zostać wewnątrz —
        # rozszerzamy z powrotem jeśli cap by je odciął.
        cap = self.lateral_max_offset_m
        if cap > 0.0:
            cur = np.asarray(drone_state.position, dtype=np.float64)
            search_min = np.asarray(search_min, dtype=np.float64).copy()
            search_max = np.asarray(search_max, dtype=np.float64).copy()
            # Z cap (axis 2)
            search_min[2] = max(search_min[2], cur[2] - cap)
            search_max[2] = min(search_max[2], cur[2] + cap)
            # Lateral XY cap — wybieramy dominującą oś
            lat_axis = 0 if abs(lateral_xy[0]) > abs(lateral_xy[1]) else 1
            search_min[lat_axis] = max(search_min[lat_axis], cur[lat_axis] - cap)
            search_max[lat_axis] = min(search_max[lat_axis], cur[lat_axis] + cap)
            # Re-inkluzja kluczowych punktów (rejoin/threat) z marginesem 1m,
            # żeby AStar miał gdzie zakończyć ścieżkę.
            for keep_pt in (cur, rejoin_point):
                search_min = np.minimum(search_min, keep_pt - 1.0)
                search_max = np.maximum(search_max, keep_pt + 1.0)

        return EvasionContext(
            drone_id=drone_id,
            current_time=current_time,
            drone_state=drone_state,
            threat=threat,
            base_spline=base_spline,
            rejoin_point=rejoin_point,
            rejoin_base_arc=target_arc,
            world_bounds=env_bounds,
            search_space_min=search_min,
            search_space_max=search_max,
            preferred_axis_hint=preferred_axis_hint,
            secondary_threats=list(secondary_threats) if secondary_threats else [],
        )

    def _build_dynamic_search_space(self, current_pos, rejoin_point, threat: ThreatAlert,
                                    forward_xy, lateral_xy, ref_speed,
                                    floor_z, ceiling_z, world_min, world_max):
        return jit_build_dynamic_search_space(
            np.asarray(current_pos, dtype=np.float64),
            rejoin_point,
            np.asarray(threat.obstacle_state.position, dtype=np.float64),
            np.asarray(threat.obstacle_state.velocity, dtype=np.float64),
            np.asarray(threat.relative_velocity, dtype=np.float64),
            float(threat.obstacle_state.radius),
            forward_xy, lateral_xy,
            ref_speed, self.t_max, self.lateral_margin,
            self.margin_velocity_gain,
            floor_z, ceiling_z,
            np.asarray(world_min, dtype=np.float64),
            np.asarray(world_max, dtype=np.float64)
        )

    def _compute_flyby_arc(self, drone_pos, drone_vel, threat: ThreatAlert, ref_speed: float) -> float:
        t_pos = threat.obstacle_state.position
        t_vel = threat.obstacle_state.velocity
        return jit_compute_flyby_arc(
            float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2]),
            float(drone_vel[0]), float(drone_vel[1]), float(drone_vel[2]),
            float(t_pos[0]), float(t_pos[1]), float(t_pos[2]),
            float(t_vel[0]), float(t_vel[1]), float(t_vel[2]),
            float(threat.obstacle_state.radius), ref_speed,
            self.t_max, self.rejoin_flyby_safety_m
        )

    @staticmethod
    def _sample_base_at_arc(spline: BSplineTrajectory, arc: float) -> np.ndarray:
        if spline.arc_length <= 1e-6:
            u = 1.0
        else:
            u = arc / spline.arc_length
            if u < 0.0: u = 0.0
            elif u > 1.0: u = 1.0
        return np.array(splev(u, spline.tck), dtype=np.float64)

    @staticmethod
    def _compute_forward_direction(current_vel, base_spline, base_arc_progress) -> np.ndarray:
        speed = (current_vel[0]*current_vel[0] + current_vel[1]*current_vel[1] + current_vel[2]*current_vel[2]) ** 0.5
        if speed > 0.5:
            inv_speed = 1.0 / speed
            return current_vel * inv_speed

        if base_spline.arc_length > 1e-6:
            u = base_arc_progress / base_spline.arc_length
            if u < 0.0: u = 0.0
            elif u > 1.0: u = 1.0
        else:
            u = 0.0

        tangent = np.array(splev(u, base_spline.tck, der=1), dtype=np.float64)
        tnorm = (tangent[0]*tangent[0] + tangent[1]*tangent[1] + tangent[2]*tangent[2]) ** 0.5
        if tnorm > 1e-6:
            inv_tnorm = 1.0 / tnorm
            return tangent * inv_tnorm

        return np.array([1.0, 0.0, 0.0])