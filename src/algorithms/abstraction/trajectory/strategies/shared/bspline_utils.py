"""Numba-skompilowane jądra dla B-spline'ów i profilu trapezoidalnego prędkości.

Funkcje publiczne dzielą się na cztery grupy:
- Bazowe pomocnicze: `bspline_basis_cubic`, `point_to_segment_dist_sq`,
  `check_aabb_cylinder_collision`.
- Manipulacja punktami kontrolnymi: `build_clamped_control_points`,
  `clamp_control_points_batch`.
- Próbkowanie i ewaluacja krzywych: `evaluate_bspline_trajectory`,
  `evaluate_bspline_trajectory_sync`, `generate_bspline_batch`.
- Profil trapezoidalny: `calculate_trapezoidal_profile`,
  `get_state_at_time_numba`, `compute_max_observed_acceleration`,
  `calculate_dynamic_max_node_distance`.

Wszystkie kernele kompilowane z `@njit(cache=True)`.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def bspline_basis_cubic(t: float):
    """Wartości czterech bazowych funkcji kubicznego B-spline'a w punkcie `t ∈ [0, 1]`.

    Returns:
        Krotka `(b0, b1, b2, b3)` współczynników do liniowej kombinacji
        4 sąsiednich punktów kontrolnych segmentu.
    """
    it = 1.0 - t
    b0 = (it ** 3) / 6.0
    b1 = (3.0 * (t ** 3) - 6.0 * (t ** 2) + 4.0) / 6.0
    b2 = (-3.0 * (t ** 3) + 3.0 * (t ** 2) + 3.0 * t + 1.0) / 6.0
    b3 = (t ** 3) / 6.0
    return b0, b1, b2, b3

@njit(cache=True)
def build_clamped_control_points(free_cp, start_pos, goal_pos):
    """Doklej `start_pos` / `goal_pos` jako 4× powtórzone końce wieloboku kontrolnego.

    `rep = 4` dla kubicznego B-spline'a daje końcom krotność 3, więc krzywa
    interpoluje endpointy.

    Args:
        free_cp: `(PopSize, NDrones, N_free, 3)` wewnętrzne punkty kontrolne.
        start_pos: `(NDrones, 3)` pozycje startowe per dron.
        goal_pos: `(NDrones, 3)` pozycje docelowe per dron.

    Returns:
        `(PopSize, NDrones, N_free + 8, 3)` pełny wielobok kontrolny.
    """
    pop_size, n_drones, n_free, _ = free_cp.shape
    rep = 4

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
    """Powiel pierwszy i ostatni punkt kontrolny każdego drona do krotności 3.

    Wymóg dla kubicznego B-spline'a, by krzywa interpolowała `P[0]` i `P[-1]`.

    Args:
        control_points: `(PopSize, NDrones, N_ctrl, 3)` wielobok kontrolny.

    Returns:
        `(PopSize, NDrones, N_ctrl + 4, 3)` z układem
        `[P0, P0, P0, P1, …, Pn-1, Pn, Pn, Pn]`.
    """
    pop_size, n_drones, n_ctrl, dims = control_points.shape
    # Krotność końców = 3 → P[0] i P[-1] dwukrotnie ekstra (raz już istnieją).
    n_clamped = n_ctrl + 4
    out = np.zeros((pop_size, n_drones, n_clamped, dims), dtype=np.float64)

    for p in range(pop_size):
        for d in range(n_drones):
            for rep in range(3):
                for dim in range(dims):
                    out[p, d, rep, dim] = control_points[p, d, 0, dim]
            for i in range(1, n_ctrl - 1):
                for dim in range(dims):
                    out[p, d, 2 + i, dim] = control_points[p, d, i, dim]
            for rep in range(3):
                for dim in range(dims):
                    out[p, d, n_clamped - 1 - rep, dim] = control_points[p, d, n_ctrl - 1, dim]

    return out


@njit(cache=True)
def check_aabb_cylinder_collision(min_x, min_y, max_x, max_y, cyl_x, cyl_y, cyl_radius):
    """Sprawdź, czy okrąg `(cyl_x, cyl_y, cyl_radius)` przecina prostokąt AABB w XY.

    Returns:
        `True`, gdy najbliższy punkt prostokąta leży nie dalej niż
        `cyl_radius` od środka okręgu.
    """
    closest_x = max(min_x, min(cyl_x, max_x))
    closest_y = max(min_y, min(cyl_y, max_y))
    dist_sq = (closest_x - cyl_x)**2 + (closest_y - cyl_y)**2
    return dist_sq <= (cyl_radius ** 2)


@njit(cache=True)
def point_to_segment_dist_sq(px, py, ax, ay, bx, by):
    """Kwadrat odległości euklidesowej w XY od punktu `(px, py)` do odcinka `AB`.

    Returns:
        Kwadrat dystansu — bez `sqrt` dla wydajności (porównania `<` można
        robić na kwadratach).
    """
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
    """Próbkuj B-spline'y i policz kolizje, długości oraz naruszenia Z dla całej populacji.

    Args:
        control_points: `(PopSize, NDrones, N_ctrl, 3)` pełny wielobok
            kontrolny `[Start, Inner_1…n, Target]`.
        start_pos: Zachowane dla zgodności starszego API — nie używane,
            bo klamrowanie zapewnia interpolację przez `P[0]`.
        goal_pos: Zachowane dla zgodności — patrz wyżej.
        obstacles_xy: `(N_obs, 2)` pozycje cylindrycznych przeszkód.
        obstacle_radii: `(N_obs,)` promienie przeszkód [m].
        drone_radius: Promień drona [m] dodawany do bufora kolizji.
        safety_margin: Dodatkowy margines bezpieczeństwa [m].
        min_samples_per_seg: Minimalna gęstość próbkowania jednego segmentu.
        min_z, max_z: Granice korytarza w osi Z; przekroczenia akumulowane
            w `z_violations`.

    Returns:
        Krotka `(collisions, lengths, z_violations)` — wszystkie
        `(PopSize, NDrones)`:
        - `collisions` — sumaryczna głębokość penetracji buforów przeszkód.
        - `lengths` — długości łuku 3D w metrach.
        - `z_violations` — sumaryczne wyjścia poza `[min_z, max_z]`.
    """
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
    """Wygeneruj gładkie trajektorie B-spline z węzłów kontrolnych dla całej populacji.

    Końce wieloboku są klamrowane automatycznie, więc krzywa interpoluje
    `P[0]` i `P[-1]`.

    Args:
        control_points: `(PopSize, NDrones, N_ctrl, 3)` wielobok kontrolny.
        num_samples: Docelowa liczba punktów wyjściowych (zostanie
            zaokrąglona w dół do wielokrotności liczby segmentów).

    Returns:
        `(PopSize, NDrones, total_samples, 3)` próbki krzywych —
        pierwszy i ostatni punkt to `P[0]` i `P[-1]`.
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


@njit(cache=True)
def calculate_trapezoidal_profile(total_distance, cruise_speed, max_accel):
    """Wylicz fazy profilu trapezoidalnego dla danego dystansu.

    Args:
        total_distance: Całkowita długość trasy [m].
        cruise_speed: Docelowa prędkość przelotowa [m/s].
        max_accel: Maksymalne przyspieszenie/hamowanie [m/s²].

    Returns:
        Krotka `(ta, tc, td, sa, sc, v_peak, total_duration)`:
        - `ta, tc, td` — czasy faz przyspieszania, cruise i hamowania [s].
        - `sa, sc` — dystanse przebyte w fazach przyspieszania i cruise [m].
        - `v_peak` — faktycznie osiągnięta prędkość szczytowa [m/s]
          (mniejsza niż `cruise_speed` dla bardzo krótkich tras).
        - `total_duration = ta + tc + td` [s].
    """
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
    """Pomocnik fallbackowy: stan trasy w chwili `t` z interpolacją liniową.

    Używane gdy `NumbaTrajectoryProfile` nie mógł zafitować B-spline'a
    (degeneracja: < 4 waypointów albo zerowa długość trasy).

    Args:
        waypoints: `(W, 3)` punkty trasy.
        distances: `(W-1,)` długości segmentów.
        cumulative_distances: `(W,)` skumulowane długości od startu.
        t: Czas od startu trasy [s].
        ta, tc, td, sa, sc, v_peak, max_accel: Parametry profilu z
            `calculate_trapezoidal_profile`.

    Returns:
        Para `(pos, velocity)` — `(3,)` w metrach i `(3,)` w m/s.
    """
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
    """Wylicz dynamiczny limit odległości sąsiednich węzłów kontrolnych.

    Wartość = `max(absolute_min, k_factor × avg_segment_length)`, gdzie
    `avg_segment_length = max_route_length / (n_inner_points + 3)`. Liczba
    segmentów odpowiada konwencji doklejania
    `[starts, starts_hover, inner…, targets_hover, targets]`.

    Args:
        start_pos: `(NDrones, 3)` pozycje startowe [m].
        target_pos: `(NDrones, 3)` pozycje docelowe [m].
        n_inner_points: Liczba wewnętrznych węzłów kontrolnych.
        k_factor: Mnożnik średniej długości segmentu.
        absolute_min: Twardy dolny limit [m].

    Returns:
        Maksymalny dozwolony dystans między sąsiednimi węzłami [m].
    """
    n_drones = start_pos.shape[0]

    max_route_length = 0.0
    for d in range(n_drones):
        dx = target_pos[d, 0] - start_pos[d, 0]
        dy = target_pos[d, 1] - start_pos[d, 1]
        dz = target_pos[d, 2] - start_pos[d, 2]
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        if dist > max_route_length:
            max_route_length = dist

    n_segments = n_inner_points + 3
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
    """Wariant zsynchronizowany — wszystkie drony porównywane w tym samym kroku czasowym.

    Dodatkowo do `evaluate_bspline_trajectory` liczy `swarm_collisions`:
    sumę naruszeń dystansu pomiędzy parami dronów w tym samym indeksie
    próbki (synchroniczna ocena bezpieczeństwa wewnątrz roju).

    Args:
        control_points: `(PopSize, NDrones, N_ctrl, 3)` wielobok kontrolny.
        obstacles_xy: `(N_obs, 2)` pozycje przeszkód.
        obstacle_radii: `(N_obs,)` promienie przeszkód [m].
        min_drone_dist: Minimalny dozwolony dystans pary dronów [m].
        drone_radius: Promień drona [m].
        safety_margin: Margines bezpieczeństwa wokół przeszkód [m].
        min_samples_per_seg: Minimalna gęstość próbkowania segmentu.

    Returns:
        Krotka `(obstacle_collisions, lengths, swarm_collisions)`:
        - `obstacle_collisions` `(PopSize, NDrones)` — naruszenia buforów
          przeszkód statycznych.
        - `lengths` `(PopSize, NDrones)` — długości łuku 3D [m].
        - `swarm_collisions` `(PopSize,)` — naruszenia dystansu w roju.
    """
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


@njit(cache=True)
def compute_max_observed_acceleration(
    control_points,
    cruise_speed: float,
    samples_per_segment: int = 30,
    boundary_segments_skip: int = 0,
):
    """Twardy fizyczny constraint na **LATERAL** acceleration drona podążającego
    za B-spline'em.

    Mierzy wyłącznie składową lateralną (perpendicular do kierunku ruchu) bo:
    - **Tangencjalna** acceleration (along velocity) jest bounded przez
      trapezoidal velocity profile w `NumbaTrajectoryProfile` (≤ max_accel
      by construction). Mierzenie tej składowej daje **false positives**
      dla geometrycznie prostych trajektorii (np. linia prosta ma niezerową
      drugą pochodną parametryczną z powodu klampowania, ale lateral=0).
    - **Lateralna** = `v² × κ` (κ = krzywizna). To jest faktyczne wymaganie
      siły bocznej od drona — to chcemy ograniczyć przez max_accel.

    Implementacja:
    1. Sample B-spline parametrycznie (samples_per_segment per segment)
    2. Central-difference v_param i a_param (parametryczne)
    3. Lateral component: a_lat_param = a_param - (a_param · v_unit)·v_unit
    4. Mapping fizyczne: |a_lat_phys| = |a_lat_param| × (cruise_speed/|v_param|)²
       (chain rule przy zamianie zmiennej; assume worst case v_phys=cruise_speed)
    5. Track max przez całą trajektorię (per drone, per individual)

    Z lateral-only check, `boundary_segments_skip=0` jest bezpieczne —
    klampowanie produkuje tangential acceleration (ramp-in/ramp-out)
    która nie jest karana. Tylko realne zakręty (lateral) trafiają.

    Args:
        control_points: shape (PopSize, NDrones, NControl, 3)
        cruise_speed: max physical speed (m/s)
        samples_per_segment: gęstość samplingu (default 30)
        boundary_segments_skip: ile pierwszych/ostatnich segmentów pominąć
            (default 0 — z lateral-only nie ma artefaktów klampowania)

    Returns:
        max_acc: shape (PopSize, NDrones) — max observed |a_lat_phys|.
    """
    cp = clamp_control_points_batch(control_points)
    pop_size, n_drones, n_ctrl, _ = cp.shape
    n_segments = n_ctrl - 3

    max_acc = np.zeros((pop_size, n_drones), dtype=np.float64)

    if n_segments < 1:
        return max_acc

    # Core segments po skipowaniu boundary. Dla bardzo krótkich trajektorii
    # (n_seg ≤ 2*skip) — fallback na całą trajektorię (lepiej coś niż nic).
    if n_segments > 2 * boundary_segments_skip:
        seg_start = boundary_segments_skip
        seg_end = n_segments - boundary_segments_skip
    else:
        seg_start = 0
        seg_end = n_segments

    dt_param = 1.0 / samples_per_segment
    dt_param_sq = dt_param * dt_param
    v_phys_sq = cruise_speed * cruise_speed

    # Minimum ||v_param||² żeby a_phys mapping był sensowny. Powolne
    # ruchy w param space (clustered control points) → nieproporcjonalna
    # eksplozja a_phys. Threshold = (cruise_speed × dt_param)² = oczekiwana
    # parametryczna prędkość gdy drone leci cruise. Min: 0.01 × cruise_speed²
    # (drone w hover/ramp).
    min_v_param_sq = 0.01 * v_phys_sq * dt_param_sq

    for p in range(pop_size):
        for d in range(n_drones):
            # Reset sliding window per (p, d)
            ppx = ppy = ppz = 0.0
            px = py = pz = 0.0
            cx = cy = cz = 0.0
            n_collected = 0

            for seg in range(seg_start, seg_end):
                seg_p0x = cp[p, d, seg, 0]
                seg_p0y = cp[p, d, seg, 1]
                seg_p0z = cp[p, d, seg, 2]
                seg_p1x = cp[p, d, seg+1, 0]
                seg_p1y = cp[p, d, seg+1, 1]
                seg_p1z = cp[p, d, seg+1, 2]
                seg_p2x = cp[p, d, seg+2, 0]
                seg_p2y = cp[p, d, seg+2, 1]
                seg_p2z = cp[p, d, seg+2, 2]
                seg_p3x = cp[p, d, seg+3, 0]
                seg_p3y = cp[p, d, seg+3, 1]
                seg_p3z = cp[p, d, seg+3, 2]

                start_step = 0 if seg == seg_start else 1
                for step in range(start_step, samples_per_segment + 1):
                    t = step * dt_param
                    b0, b1, b2, b3 = bspline_basis_cubic(t)
                    new_x = b0*seg_p0x + b1*seg_p1x + b2*seg_p2x + b3*seg_p3x
                    new_y = b0*seg_p0y + b1*seg_p1y + b2*seg_p2y + b3*seg_p3y
                    new_z = b0*seg_p0z + b1*seg_p1z + b2*seg_p2z + b3*seg_p3z

                    ppx, ppy, ppz = px, py, pz
                    px, py, pz = cx, cy, cz
                    cx, cy, cz = new_x, new_y, new_z
                    n_collected += 1

                    if n_collected < 3:
                        continue

                    # Central difference velocity (parametric)
                    vx = (cx - ppx) / (2.0 * dt_param)
                    vy = (cy - ppy) / (2.0 * dt_param)
                    vz = (cz - ppz) / (2.0 * dt_param)
                    v_param_sq = vx*vx + vy*vy + vz*vz

                    # Skip slow regions (stagnant lub clustered control points)
                    # — physical mapping a_phys eksploduje, ale faktycznie
                    # to jest ramp/hover gdzie v_phys też → 0.
                    if v_param_sq < min_v_param_sq:
                        continue

                    # Central difference acceleration (parametric)
                    ax = (cx - 2.0*px + ppx) / dt_param_sq
                    ay = (cy - 2.0*py + ppy) / dt_param_sq
                    az = (cz - 2.0*pz + ppz) / dt_param_sq

                    # Decompose into tangential + lateral (perpendicular do v)
                    # v_unit = v_param / ||v_param||
                    # a_tang = (a_param · v_unit) * v_unit
                    # a_lat = a_param - a_tang
                    # ||v_param|| = sqrt(v_param_sq)
                    inv_v = 1.0 / np.sqrt(v_param_sq)
                    vux = vx * inv_v
                    vuy = vy * inv_v
                    vuz = vz * inv_v
                    a_dot_v = ax*vux + ay*vuy + az*vuz
                    a_lat_x = ax - a_dot_v * vux
                    a_lat_y = ay - a_dot_v * vuy
                    a_lat_z = az - a_dot_v * vuz
                    a_lat_param_sq = a_lat_x*a_lat_x + a_lat_y*a_lat_y + a_lat_z*a_lat_z

                    # Physical lateral: |a_lat_phys| = |a_lat_param| × (v_phys/v_param)²
                    a_lat_phys_sq = a_lat_param_sq * (v_phys_sq / v_param_sq) ** 2
                    if a_lat_phys_sq > max_acc[p, d] * max_acc[p, d]:
                        max_acc[p, d] = np.sqrt(a_lat_phys_sq)

    return max_acc