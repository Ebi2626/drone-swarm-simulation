"""Numba-jitted kernele wspólne dla ścieżki avoidance.

Wszystkie funkcje są bezstanowe, deterministyczne, `@njit(cache=True, fastmath=True)`.
Używane przez `AStarOptimizer` (fallback path, space-in-direction) i `BSplineSmoother`
(DP simplification, tangent leads, uniform resampling).

Kontrakt:
  - `jit_douglas_peucker`: redukcja gęstości waypointów A* (preserves endpoints).
  - `jit_resample_uniform`: równomierne resampling po długości łuku do `n` punktów.
  - `jit_fallback_path`: 5-waypointowa awaryjna trajektoria objazdu (gdy A* zawiódł).
  - `jit_insert_tangent_leads`: lead-in / lead-out styczne do v_drone i base-spline.
  - `jit_space_in_xy_dir`: dystans od pos do najbliższej ściany BBOX-u w XY.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def jit_douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
    n = len(points)
    if n < 3:
        return points.copy()

    keep = np.zeros(n, dtype=np.bool_)
    keep[0] = True
    keep[-1] = True

    # Iteracyjny stos (stack) omija narzut rekursji w Pythonie
    stack = np.empty((n, 2), dtype=np.int32)
    stack[0, 0] = 0
    stack[0, 1] = n - 1
    stack_ptr = 1

    while stack_ptr > 0:
        stack_ptr -= 1
        start_idx = stack[stack_ptr, 0]
        end_idx = stack[stack_ptr, 1]

        a = points[start_idx]
        b = points[end_idx]
        ab_x = b[0] - a[0]; ab_y = b[1] - a[1]; ab_z = b[2] - a[2]
        ab_sq = ab_x**2 + ab_y**2 + ab_z**2

        max_dist = -1.0
        max_idx = -1

        for i in range(start_idx + 1, end_idx):
            p = points[i]
            if ab_sq < 1e-12:
                dist = np.sqrt((p[0]-a[0])**2 + (p[1]-a[1])**2 + (p[2]-a[2])**2)
            else:
                ap_x = p[0] - a[0]; ap_y = p[1] - a[1]; ap_z = p[2] - a[2]
                t = (ap_x*ab_x + ap_y*ab_y + ap_z*ab_z) / ab_sq
                if t < 0.0: t = 0.0
                elif t > 1.0: t = 1.0
                closest_x = a[0] + t * ab_x
                closest_y = a[1] + t * ab_y
                closest_z = a[2] + t * ab_z
                dist = np.sqrt((p[0]-closest_x)**2 + (p[1]-closest_y)**2 + (p[2]-closest_z)**2)

            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > epsilon:
            keep[max_idx] = True
            stack[stack_ptr, 0] = max_idx
            stack[stack_ptr, 1] = end_idx
            stack_ptr += 1
            stack[stack_ptr, 0] = start_idx
            stack[stack_ptr, 1] = max_idx
            stack_ptr += 1

    out_count = int(np.sum(keep))
    out = np.empty((out_count, 3), dtype=np.float64)
    idx = 0
    for i in range(n):
        if keep[i]:
            out[idx] = points[i]
            idx += 1
    return out


@njit(cache=True, fastmath=True)
def jit_resample_uniform(points: np.ndarray, n: int) -> np.ndarray:
    n_pts = len(points)
    if n_pts <= 2 or n <= 2:
        if n_pts >= 2:
            out = np.empty((2, 3), dtype=np.float64)
            out[0] = points[0]
            out[1] = points[-1]
            return out
        return points.copy()

    seg_lens = np.empty(n_pts - 1, dtype=np.float64)
    total = 0.0
    for i in range(n_pts - 1):
        d = np.sqrt((points[i+1, 0] - points[i, 0])**2 +
                    (points[i+1, 1] - points[i, 1])**2 +
                    (points[i+1, 2] - points[i, 2])**2)
        seg_lens[i] = d
        total += d

    if total < 1e-6:
        out = np.empty((n, 3), dtype=np.float64)
        for i in range(n):
            out[i] = points[0]
        return out

    cum = np.empty(n_pts, dtype=np.float64)
    cum[0] = 0.0
    for i in range(n_pts - 1):
        cum[i+1] = cum[i] + seg_lens[i]

    targets = np.linspace(0.0, total, n)
    out = np.empty((n, 3), dtype=np.float64)
    j = 0
    for i in range(n):
        s = targets[i]
        while j < n_pts - 2 and cum[j + 1] < s:
            j += 1
        s0 = cum[j]
        s1 = cum[j + 1]
        ratio = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
        out[i, 0] = points[j, 0] + ratio * (points[j+1, 0] - points[j, 0])
        out[i, 1] = points[j, 1] + ratio * (points[j+1, 1] - points[j, 1])
        out[i, 2] = points[j, 2] + ratio * (points[j+1, 2] - points[j, 2])
    return out


@njit(cache=True, fastmath=True)
def jit_fallback_path(current_pos: np.ndarray, rejoin_point: np.ndarray,
                      preferred_dir: np.ndarray, obs_pos: np.ndarray,
                      obs_radius: float, floor_z: float, ceiling_z: float,
                      obs_vel: np.ndarray) -> np.ndarray:
    diag = rejoin_point - current_pos
    mid = 0.5 * (current_pos + rejoin_point)
    offset_m = obs_radius * 1.5
    if offset_m < 2.5: offset_m = 2.5

    esc_vec = np.zeros(3, dtype=np.float64)
    v_norm = np.sqrt(obs_vel[0]**2 + obs_vel[1]**2 + obs_vel[2]**2)
    if v_norm > 0.1:
        obs_vel_hat = obs_vel / v_norm
        escape_gain = v_norm * 0.5
        if escape_gain < 1.0: escape_gain = 1.0
        esc_vec = -obs_vel_hat * escape_gain

    safe_rad_sq = (obs_radius + 0.1)**2

    p0 = current_pos.copy()
    p4 = rejoin_point.copy()
    p1 = np.empty(3, dtype=np.float64)
    p2 = np.empty(3, dtype=np.float64)
    p3 = np.empty(3, dtype=np.float64)

    for attempt in range(3):
        mult = 1.0 + 0.8 * attempt
        off = offset_m * mult

        for i in range(3):
            p1[i] = current_pos[i] + 0.25 * diag[i] + 0.5 * preferred_dir[i] * off
            p2[i] = mid[i] + preferred_dir[i] * off + esc_vec[i] * mult
            p3[i] = current_pos[i] + 0.75 * diag[i] + 0.5 * preferred_dir[i] * off

        for p in (p1, p2, p3):
            if p[2] < floor_z: p[2] = floor_z
            elif p[2] > ceiling_z: p[2] = ceiling_z

        safe = True
        for p in (p1, p2, p3):
            d_sq = (p[0] - obs_pos[0])**2 + (p[1] - obs_pos[1])**2 + (p[2] - obs_pos[2])**2
            if d_sq < safe_rad_sq:
                safe = False
                break

        if safe:
            out = np.empty((5, 3), dtype=np.float64)
            out[0]=p0; out[1]=p1; out[2]=p2; out[3]=p3; out[4]=p4
            return out

    out = np.empty((5, 3), dtype=np.float64)
    out[0]=p0; out[1]=p1; out[2]=p2; out[3]=p3; out[4]=p4
    return out


@njit(cache=True, fastmath=True)
def jit_insert_tangent_leads(waypoints: np.ndarray, current_pos: np.ndarray,
                             forward_3d: np.ndarray, rejoin_point: np.ndarray,
                             base_tangent_at_rejoin: np.ndarray,
                             ref_speed: float, obs_pos: np.ndarray,
                             obs_radius: float, lead_mult: float) -> np.ndarray:
    n_wp = len(waypoints)
    if n_wp < 2:
        return waypoints.copy()

    lead_dist = ref_speed * 0.1
    if lead_dist < 0.3: lead_dist = 0.3
    elif lead_dist > 0.8: lead_dist = 0.8
    lead_dist *= lead_mult

    safe_rad_sq = (obs_radius + 0.1)**2

    fwd_norm = np.sqrt(forward_3d[0]**2 + forward_3d[1]**2 + forward_3d[2]**2)
    has_lead_in = False
    lead_in = np.zeros(3, dtype=np.float64)
    if fwd_norm > 1e-6:
        cx = current_pos[0] + (forward_3d[0] / fwd_norm) * lead_dist
        cy = current_pos[1] + (forward_3d[1] / fwd_norm) * lead_dist
        cz = current_pos[2] + (forward_3d[2] / fwd_norm) * lead_dist
        if (cx - obs_pos[0])**2 + (cy - obs_pos[1])**2 + (cz - obs_pos[2])**2 >= safe_rad_sq:
            has_lead_in = True
            lead_in[0] = cx; lead_in[1] = cy; lead_in[2] = cz

    back_norm = np.sqrt(base_tangent_at_rejoin[0]**2 + base_tangent_at_rejoin[1]**2 + base_tangent_at_rejoin[2]**2)
    has_lead_out = False
    lead_out = np.zeros(3, dtype=np.float64)
    if back_norm > 1e-6:
        cx = rejoin_point[0] - (base_tangent_at_rejoin[0] / back_norm) * lead_dist
        cy = rejoin_point[1] - (base_tangent_at_rejoin[1] / back_norm) * lead_dist
        cz = rejoin_point[2] - (base_tangent_at_rejoin[2] / back_norm) * lead_dist
        if (cx - obs_pos[0])**2 + (cy - obs_pos[1])**2 + (cz - obs_pos[2])**2 >= safe_rad_sq:
            has_lead_out = True
            lead_out[0] = cx; lead_out[1] = cy; lead_out[2] = cz

    out_len = n_wp
    if has_lead_in: out_len += 1
    if has_lead_out: out_len += 1

    out = np.empty((out_len, 3), dtype=np.float64)
    idx = 0
    out[idx] = current_pos
    idx += 1
    if has_lead_in:
        out[idx] = lead_in
        idx += 1
    for i in range(1, n_wp - 1):
        out[idx] = waypoints[i]
        idx += 1
    if has_lead_out:
        out[idx] = lead_out
        idx += 1
    out[idx] = rejoin_point

    if len(out) < 4:
        mid = np.empty(3, dtype=np.float64)
        mid[0] = 0.5 * (out[0, 0] + out[-1, 0])
        mid[1] = 0.5 * (out[0, 1] + out[-1, 1])
        mid[2] = 0.5 * (out[0, 2] + out[-1, 2])
        new_out = np.empty((len(out) + 1, 3), dtype=np.float64)
        new_out[0] = out[0]
        new_out[1] = mid
        for i in range(1, len(out)):
            new_out[i+1] = out[i]
        return new_out

    return out


@njit(cache=True, fastmath=True)
def jit_space_in_xy_dir(pos_x: float, pos_y: float, dir_x: float, dir_y: float,
                        wmin_x: float, wmin_y: float, wmax_x: float, wmax_y: float) -> float:
    norm = np.sqrt(dir_x**2 + dir_y**2)
    if norm < 1e-6: return 0.0
    dx = dir_x / norm; dy = dir_y / norm
    min_dist = 1e9
    found = False

    if dx > 1e-6:
        d = (wmax_x - pos_x) / dx
        if d < min_dist: min_dist = d; found = True
    elif dx < -1e-6:
        d = (wmin_x - pos_x) / dx
        if d < min_dist: min_dist = d; found = True

    if dy > 1e-6:
        d = (wmax_y - pos_y) / dy
        if d < min_dist: min_dist = d; found = True
    elif dy < -1e-6:
        d = (wmin_y - pos_y) / dy
        if d < min_dist: min_dist = d; found = True

    return min_dist if found else 0.0
