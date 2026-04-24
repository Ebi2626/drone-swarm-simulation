import numpy as np
from numba import njit

@njit
def check_collisions_njit(positions_matrix, safe_dist):
    # Gwarancja typu float64 dla poprawności np.linalg.norm w Numbie
    positions_matrix = positions_matrix.astype(np.float64)
    n_times, n_drones, _ = positions_matrix.shape
    
    for t in range(n_times):
        for i in range(n_drones):
            for j in range(i + 1, n_drones):
                dist = np.linalg.norm(positions_matrix[t, i] - positions_matrix[t, j])
                if dist < safe_dist:
                    return i, j, t
    return -1, -1, -1

@njit
def insert_midpoint_njit(waypoints, target_pos, offset):
    # Wymuszenie typu float64 dla wszystkich wektorów wejściowych
    waypoints = waypoints.astype(np.float64)
    target_pos = target_pos.astype(np.float64)
    offset = offset.astype(np.float64)
    
    n = waypoints.shape[0]
    dists = np.empty(n, dtype=np.float64)
    for k in range(n):
        dists[k] = np.linalg.norm(waypoints[k] - target_pos)

    closest_idx = np.argmin(dists)

    if closest_idx == 0:
        idx_to_split = 0
    elif closest_idx == n - 1:
        idx_to_split = n - 2
    else:
        dist_prev = np.linalg.norm(waypoints[closest_idx - 1] - target_pos)
        dist_next = np.linalg.norm(waypoints[closest_idx + 1] - target_pos)
        if dist_prev < dist_next:
            idx_to_split = closest_idx - 1
        else:
            idx_to_split = closest_idx

    midpoint = (waypoints[idx_to_split] + waypoints[idx_to_split + 1]) / 2.0 + offset

    out = np.empty((n + 1, waypoints.shape[1]), dtype=np.float64)
    out[:idx_to_split + 1] = waypoints[:idx_to_split + 1]
    out[idx_to_split + 1] = midpoint
    out[idx_to_split + 2:] = waypoints[idx_to_split + 1:]
    
    return out

@njit
def calculate_repulsion_njit(waypoints_i, waypoints_j, pos_i, pos_j, push_distance):
    # Wymuszenie typu float64 pozwala uniknąć błędów TypeMismatch dla np.linalg.norm
    pos_i = pos_i.astype(np.float64)
    pos_j = pos_j.astype(np.float64)
    waypoints_i = waypoints_i.astype(np.float64)
    waypoints_j = waypoints_j.astype(np.float64)
    
    repel_vec = pos_i - pos_j
    norm = np.linalg.norm(repel_vec)
    
    if norm < 1e-3:
        # Generowanie deterministycznego i zmiennoprzecinkowego wektora
        repel_vec = np.array([np.random.rand(), np.random.rand(), 0.0], dtype=np.float64)
        norm = np.linalg.norm(repel_vec)
        
    repel_dir = repel_vec / norm
    
    new_i = insert_midpoint_njit(waypoints_i, pos_i, repel_dir * push_distance)
    new_j = insert_midpoint_njit(waypoints_j, pos_j, -repel_dir * push_distance)
    
    return new_i, new_j