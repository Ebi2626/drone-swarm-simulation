import numpy as np
from numba import njit


@njit(cache=True)
def bspline_basis_cubic(t: float):
    it = 1.0 - t
    b0 = (it ** 3) / 6.0
    b1 = (3.0 * (t ** 3) - 6.0 * (t ** 2) + 4.0) / 6.0
    b2 = (-3.0 * (t ** 3) + 3.0 * (t ** 2) + 3.0 * t + 1.0) / 6.0
    b3 = (t ** 3) / 6.0
    return b0, b1, b2, b3

@njit(cache=True)
def build_clamped_control_points(free_cp, start_pos, goal_pos):
    # free_cp: (PopSize, NDrones, N_free, 3)
    pop_size, n_drones, n_free, _ = free_cp.shape
    rep = 4  # cubic B-spline, gdy start/cel nie są w free_cp

    cp = np.empty((pop_size, n_drones, n_free + 2 * rep, 3), dtype=np.float64)

    for p in range(pop_size):
        for d in range(n_drones):
            for i in range(rep):
                cp[p, d, i, :] = start_pos[d]
                cp[p, d, n_free + rep + i, :] = goal_pos[d]
            cp[p, d, rep:rep + n_free, :] = free_cp[p, d]

    return cp


@njit(cache=True)
def clamp_control_points_batch(control_points):
    """
    Klamruje wektory węzłów przez trzykrotne powtórzenie pierwszego
    i ostatniego punktu kontrolnego dla każdego drona w populacji.

    Dla B-Spline stopnia 3 (cubic) wymagana krotność końcowych węzłów = 3,
    co wymusza interpolację krzywej przez P[0] i P[-1].

    Wejście:  (PopSize, NDrones, N_ctrl, 3)
    Wyjście:  (PopSize, NDrones, N_ctrl + 4, 3)
              [P0, P0, P0, P1, ..., Pn-1, Pn, Pn, Pn]
    """
    pop_size, n_drones, n_ctrl, dims = control_points.shape
    # Każdy koniec powtarzamy 2x extra (raz już istnieje), łącznie krotność = 3
    n_clamped = n_ctrl + 4
    out = np.zeros((pop_size, n_drones, n_clamped, dims), dtype=np.float64)

    for p in range(pop_size):
        for d in range(n_drones):
            # Pierwsze 3 pozycje = P[0] (krotność 3)
            for rep in range(3):
                for dim in range(dims):
                    out[p, d, rep, dim] = control_points[p, d, 0, dim]
            # Środkowe punkty bez zmian
            for i in range(1, n_ctrl - 1):
                for dim in range(dims):
                    out[p, d, 2 + i, dim] = control_points[p, d, i, dim]
            # Ostatnie 3 pozycje = P[-1] (krotność 3)
            for rep in range(3):
                for dim in range(dims):
                    out[p, d, n_clamped - 1 - rep, dim] = control_points[p, d, n_ctrl - 1, dim]

    return out


@njit(cache=True)
def check_aabb_cylinder_collision(min_x, min_y, max_x, max_y, cyl_x, cyl_y, cyl_radius):
    closest_x = max(min_x, min(cyl_x, max_x))
    closest_y = max(min_y, min(cyl_y, max_y))
    dist_sq = (closest_x - cyl_x)**2 + (closest_y - cyl_y)**2
    return dist_sq <= (cyl_radius ** 2)

@njit(cache=True)
def point_to_segment_dist_sq(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    den = abx * abx + aby * aby
    if den < 1e-12:
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy

    t = ((px - ax) * abx + (py - ay) * aby) / den
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    qx = ax + t * abx
    qy = ay + t * aby
    dx = px - qx
    dy = py - qy
    return dx * dx + dy * dy


@njit(cache=True)
def evaluate_bspline_trajectory(
    control_points,
    start_pos,
    goal_pos,
    obstacles_xy,
    obstacle_radii,
    drone_radius=0.25,
    safety_margin=0.10,
    min_samples_per_seg=30,
    min_z=-1.0e30,
    max_z=1.0e30,
):
    # Klamrowanie 3x (tożsame z rekonstrukcją w generate_bspline_batch).
    # control_points to pełny wielobok [Start, Inner_1..n, Target] — start_pos/goal_pos
    # pozostają w sygnaturze dla zgodności z wcześniejszym API, ale nie są tu już
    # wykorzystywane (interpolacja przez pierwszy/ostatni punkt jest zapewniona
    # przez krotność 3 w clamp_control_points_batch).
    cp = clamp_control_points_batch(control_points)

    pop_size, n_drones, n_ctrl, _ = cp.shape
    n_segments = n_ctrl - 3

    collisions = np.zeros((pop_size, n_drones), dtype=np.float64)
    lengths = np.zeros((pop_size, n_drones), dtype=np.float64)
    z_violations = np.zeros((pop_size, n_drones), dtype=np.float64)

    for p in range(pop_size):
        for d in range(n_drones):
            for i in range(n_segments):
                seg_pts = cp[p, d, i:i+4]

                ctrl_poly_len = (
                    np.sqrt(np.sum((seg_pts[1] - seg_pts[0]) ** 2)) +
                    np.sqrt(np.sum((seg_pts[2] - seg_pts[1]) ** 2)) +
                    np.sqrt(np.sum((seg_pts[3] - seg_pts[2]) ** 2))
                )

                samples = min_samples_per_seg
                if ctrl_poly_len > 1.0:
                    samples = max(min_samples_per_seg, int(ctrl_poly_len * 4.0))

                b0, b1, b2, b3 = bspline_basis_cubic(0.0)
                prev = np.empty(3, dtype=np.float64)
                prev[0] = b0*seg_pts[0,0] + b1*seg_pts[1,0] + b2*seg_pts[2,0] + b3*seg_pts[3,0]
                prev[1] = b0*seg_pts[0,1] + b1*seg_pts[1,1] + b2*seg_pts[2,1] + b3*seg_pts[3,1]
                prev[2] = b0*seg_pts[0,2] + b1*seg_pts[1,2] + b2*seg_pts[2,2] + b3*seg_pts[3,2]

                # Naruszenie Z na starcie segmentu (kara proporcjonalna do głębokości
                # zanurzenia pod podłogę / ponad sufit).
                if prev[2] < min_z:
                    z_violations[p, d] += (min_z - prev[2])
                elif prev[2] > max_z:
                    z_violations[p, d] += (prev[2] - max_z)

                for step in range(1, samples + 1):
                    t = step / samples
                    b0, b1, b2, b3 = bspline_basis_cubic(t)

                    cur = np.empty(3, dtype=np.float64)
                    cur[0] = b0*seg_pts[0,0] + b1*seg_pts[1,0] + b2*seg_pts[2,0] + b3*seg_pts[3,0]
                    cur[1] = b0*seg_pts[0,1] + b1*seg_pts[1,1] + b2*seg_pts[2,1] + b3*seg_pts[3,1]
                    cur[2] = b0*seg_pts[0,2] + b1*seg_pts[1,2] + b2*seg_pts[2,2] + b3*seg_pts[3,2]

                    lengths[p, d] += np.sqrt(np.sum((cur - prev) ** 2))

                    # Kara Z — sumujemy po wszystkich próbkach, by naruszenie
                    # ciągnące się przez dłuższy odcinek trasy miało większą wagę.
                    if cur[2] < min_z:
                        z_violations[p, d] += (min_z - cur[2])
                    elif cur[2] > max_z:
                        z_violations[p, d] += (cur[2] - max_z)

                    for obs_idx in range(len(obstacles_xy)):
                        ox = obstacles_xy[obs_idx, 0]
                        oy = obstacles_xy[obs_idx, 1]
                        r = obstacle_radii[obs_idx] + drone_radius + safety_margin

                        dist_sq = point_to_segment_dist_sq(
                            ox, oy,
                            prev[0], prev[1],
                            cur[0], cur[1]
                        )

                        if dist_sq < r * r:
                            collisions[p, d] += (r - np.sqrt(dist_sq))

                    prev = cur

    return collisions, lengths, z_violations


@njit(cache=True)
def generate_bspline_batch(control_points, num_samples):
    """
    Generuje gładkie trajektorie B-Spline z węzłów kontrolnych.
    Automatycznie klamruje końce — krzywa interpoluje P[0] i P[-1].

    Wejście:  (PopSize, NDrones, N_ctrl, 3)
    Wyjście:  (PopSize, NDrones, num_samples, 3)
              Pierwszy i ostatni punkt == odpowiednio P[0] i P[-1].
    """
    # Centralne klamrowanie
    cp = clamp_control_points_batch(control_points)

    pop_size, n_drones, n_ctrl_c, dims = cp.shape
    n_segments = n_ctrl_c - 3

    # Łączna liczba punktów wyjściowych: próbkujemy każdy segment,
    # dzielimy num_samples równomiernie między segmenty
    samples_per_seg = max(2, num_samples // n_segments)
    total_samples = n_segments * samples_per_seg

    out_points = np.zeros((pop_size, n_drones, total_samples, dims), dtype=np.float64)

    for p in range(pop_size):
        for d in range(n_drones):
            idx = 0
            for i in range(n_segments):
                # Ostatni segment próbkujemy do t=1 włącznie
                is_last = (i == n_segments - 1)
                for step in range(samples_per_seg):
                    if is_last and step == samples_per_seg - 1:
                        t = 1.0
                    else:
                        t = step / samples_per_seg

                    b0, b1, b2, b3 = bspline_basis_cubic(t)
                    seg_pts = cp[p, d, i:i+4]

                    for dim in range(dims):
                        out_points[p, d, idx, dim] = (
                            b0 * seg_pts[0, dim] + b1 * seg_pts[1, dim]
                            + b2 * seg_pts[2, dim] + b3 * seg_pts[3, dim]
                        )
                    idx += 1

    return out_points


# ---------------------------------------------------------------------------
# Funkcje dynamiki lotu — bez zmian
# ---------------------------------------------------------------------------

@njit(cache=True)
def calculate_trapezoidal_profile(total_distance, cruise_speed, max_accel):
    ta = cruise_speed / max_accel
    sa = 0.5 * max_accel * ta**2

    if 2 * sa > total_distance:
        sa = total_distance / 2.0
        ta = np.sqrt(2.0 * sa / max_accel)
        v_peak = max_accel * ta
        tc = 0.0
        sc = 0.0
    else:
        sc = total_distance - 2 * sa
        tc = sc / cruise_speed
        v_peak = cruise_speed

    td = ta
    return ta, tc, td, sa, sc, v_peak, ta + tc + td


@njit(cache=True)
def get_state_at_time_numba(waypoints, distances, cumulative_distances,
                            t, ta, tc, td, sa, sc, v_peak, max_accel):
    total_time = ta + tc + td
    if t <= 0.0:
        return waypoints[0], np.zeros(3, dtype=np.float64)
    if t >= total_time:
        return waypoints[-1], np.zeros(3, dtype=np.float64)

    current_dist = 0.0
    current_speed = 0.0

    if t < ta:
        current_dist = 0.5 * max_accel * t**2
        current_speed = max_accel * t
    elif t < ta + tc:
        t_cruise = t - ta
        current_dist = sa + v_peak * t_cruise
        current_speed = v_peak
    else:
        t_dec = t - ta - tc
        current_dist = sa + sc + (v_peak * t_dec - 0.5 * max_accel * t_dec**2)
        current_speed = v_peak - max_accel * t_dec

    idx = np.searchsorted(cumulative_distances, current_dist) - 1
    if idx < 0:
        idx = 0
    if idx >= len(waypoints) - 1:
        idx = len(waypoints) - 2

    seg_dist = distances[idx]
    if seg_dist < 1e-6:
        ratio = 0.0
    else:
        dist_in_seg = current_dist - cumulative_distances[idx]
        ratio = dist_in_seg / seg_dist

    pos = waypoints[idx] + ratio * (waypoints[idx + 1] - waypoints[idx])

    direction = waypoints[idx + 1] - waypoints[idx]
    if seg_dist > 1e-6:
        direction = direction / seg_dist

    velocity = direction * current_speed
    return pos, velocity

@njit(cache=True)
def calculate_dynamic_max_node_distance(
    start_pos: np.ndarray, 
    target_pos: np.ndarray, 
    n_inner_points: int, 
    k_factor: float = 2.0,
    absolute_min: float = 5.0
) -> float:
    """
    Oblicza dynamiczny, maksymalny dozwolony dystans między węzłami kontrolnymi (Numba).
    Działa ekstremalnie szybko unikając wywołań osiowych numpy.
    
    start_pos, target_pos: tablice (NDrones, 3)
    n_inner_points: liczba węzłów pomiędzy hover_start a hover_target
    """
    n_drones = start_pos.shape[0]
    
    # Szukamy najdłuższej trasy euklidesowej
    max_route_length = 0.0
    
    for d in range(n_drones):
        dx = target_pos[d, 0] - start_pos[d, 0]
        dy = target_pos[d, 1] - start_pos[d, 1]
        dz = target_pos[d, 2] - start_pos[d, 2]
        
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        if dist > max_route_length:
            max_route_length = dist

    # Liczba segmentów = (węzły wewnętrzne) + (hover_start) + (hover_target) + (cel) - 1
    # Ponieważ po start idzie hover, a przed cel hover, łącznie węzłów zależy od 
    # implementacji doklejania. W Rozwiązaniu A mieliśmy:
    # starts, starts_hover, inner, targets_hover, targets 
    # czyli n_inner_points + 4 punkty -> n_inner_points + 3 segmenty
    n_segments = n_inner_points + 3
    
    # Aby uniknąć dzielenia przez zero przy dziwnych parametrach:
    if n_segments < 1:
        n_segments = 1
        
    avg_segment_length = max_route_length / n_segments
    dynamic_limit = avg_segment_length * k_factor
    
    return max(absolute_min, dynamic_limit)


@njit(cache=True)
def evaluate_bspline_trajectory_sync(
    control_points,
    obstacles_xy,
    obstacle_radii,
    min_drone_dist=2.0,
    drone_radius=0.25,
    safety_margin=0.10,
    min_samples_per_seg=30,
):
    cp = clamp_control_points_batch(control_points)

    pop_size, n_drones, n_ctrl, _ = cp.shape
    n_segments = n_ctrl - 3

    obstacle_collisions = np.zeros((pop_size, n_drones), dtype=np.float64)
    lengths = np.zeros((pop_size, n_drones), dtype=np.float64)
    swarm_collisions = np.zeros(pop_size, dtype=np.float64)

    # Numba-friendly: brak przeszkód = puste tablice o rozmiarze 0
    n_obstacles = obstacles_xy.shape[0]
    
    use_obstacles = n_obstacles > 0

    for p in range(pop_size):
        for i in range(n_segments):

            max_ctrl_len = 0.0
            for d in range(n_drones):
                seg_pts = cp[p, d, i:i+4]
                l = (
                    np.sqrt(np.sum((seg_pts[1] - seg_pts[0]) ** 2)) +
                    np.sqrt(np.sum((seg_pts[2] - seg_pts[1]) ** 2)) +
                    np.sqrt(np.sum((seg_pts[3] - seg_pts[2]) ** 2))
                )
                if l > max_ctrl_len:
                    max_ctrl_len = l

            samples = min_samples_per_seg
            if max_ctrl_len > 1.0:
                samples = max(min_samples_per_seg, int(max_ctrl_len * 4.0))

            prev_positions = np.empty((n_drones, 3), dtype=np.float64)
            cur_positions = np.empty((n_drones, 3), dtype=np.float64)

            b0, b1, b2, b3 = bspline_basis_cubic(0.0)
            for d in range(n_drones):
                seg_pts = cp[p, d, i:i+4]
                prev_positions[d, 0] = b0*seg_pts[0,0] + b1*seg_pts[1,0] + b2*seg_pts[2,0] + b3*seg_pts[3,0]
                prev_positions[d, 1] = b0*seg_pts[0,1] + b1*seg_pts[1,1] + b2*seg_pts[2,1] + b3*seg_pts[3,1]
                prev_positions[d, 2] = b0*seg_pts[0,2] + b1*seg_pts[1,2] + b2*seg_pts[2,2] + b3*seg_pts[3,2]

            for step in range(1, samples + 1):
                t = step / samples
                b0, b1, b2, b3 = bspline_basis_cubic(t)

                for d in range(n_drones):
                    seg_pts = cp[p, d, i:i+4]

                    cx = b0*seg_pts[0,0] + b1*seg_pts[1,0] + b2*seg_pts[2,0] + b3*seg_pts[3,0]
                    cy = b0*seg_pts[0,1] + b1*seg_pts[1,1] + b2*seg_pts[2,1] + b3*seg_pts[3,1]
                    cz = b0*seg_pts[0,2] + b1*seg_pts[1,2] + b2*seg_pts[2,2] + b3*seg_pts[3,2]

                    cur_positions[d, 0] = cx
                    cur_positions[d, 1] = cy
                    cur_positions[d, 2] = cz

                    px = prev_positions[d, 0]
                    py = prev_positions[d, 1]
                    pz = prev_positions[d, 2]

                    lengths[p, d] += np.sqrt((cx - px)**2 + (cy - py)**2 + (cz - pz)**2)

                    # Kolizje z przeszkodami liczymy tylko, gdy przeszkody istnieją
                    if use_obstacles:
                        for obs_idx in range(n_obstacles):
                            ox = obstacles_xy[obs_idx, 0]
                            oy = obstacles_xy[obs_idx, 1]
                            r = obstacle_radii[obs_idx] + drone_radius + safety_margin

                            dist_sq = point_to_segment_dist_sq(ox, oy, px, py, cx, cy)
                            if dist_sq < r * r:
                                obstacle_collisions[p, d] += (r - np.sqrt(dist_sq))

                for d1 in range(n_drones):
                    for d2 in range(d1 + 1, n_drones):
                        diff_x = cur_positions[d1, 0] - cur_positions[d2, 0]
                        diff_y = cur_positions[d1, 1] - cur_positions[d2, 1]
                        diff_z = cur_positions[d1, 2] - cur_positions[d2, 2]

                        dist_sq = diff_x**2 + diff_y**2 + diff_z**2
                        if dist_sq < min_drone_dist**2:
                            swarm_collisions[p] += (min_drone_dist - np.sqrt(dist_sq))

                for d in range(n_drones):
                    prev_positions[d, 0] = cur_positions[d, 0]
                    prev_positions[d, 1] = cur_positions[d, 1]
                    prev_positions[d, 2] = cur_positions[d, 2]

    return obstacle_collisions, lengths, swarm_collisions