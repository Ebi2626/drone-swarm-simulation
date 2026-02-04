import numpy as np
from typing import Any, Dict, Optional
from numpy.typing import NDArray
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.abstraction.generate_obstacles import ObstaclesData

def linear_trajectory(
    *, 
    start_positions: NDArray[np.float64],
    target_positions: NDArray[np.float64],
    obstacles_data: ObstaclesData,
    world_data: WorldData,
    number_of_waypoints: int,
    drone_swarm_size: int, 
    algorithm_params: Optional[Dict[str, Any]] = None
) -> NDArray[np.float64]:
    """
    Baseline strategy: Generates straight-line trajectories from start to target.
    Does not consider obstacles (can be used to test collision detection).
    """
    
    # 1. Walidacja zgodności wymiarów (Dobra praktyka)
    if start_positions.shape[0] != drone_swarm_size or target_positions.shape[0] != drone_swarm_size:
        raise ValueError(f"Mismatch: Start/Target positions count vs swarm size ({drone_swarm_size})")

    # 2. Prealokacja tensora wynikowego (N, W, 3)
    # trajectories[dron_id, krok_czasowy, współrzędna]
    trajectories = np.zeros((drone_swarm_size, number_of_waypoints, 3), dtype=np.float64)

    # 3. Generowanie trajektorii dla każdego drona
    # Używamy wektoryzacji NumPy tam gdzie się da, ale linspace działa per oś/wektor
    
    # Metoda A: Pętla (Czytelniejsza)
    for i in range(drone_swarm_size):
        start = start_positions[i]   # (3,)
        end = target_positions[i]    # (3,)
        
        # Generujemy linię w 3D: x(t), y(t), z(t)
        # np.linspace zwraca (number_of_waypoints,) dla każdej osi
        # vstack -> (3, W), transpozycja .T -> (W, 3)
        trajectory_3d = np.vstack([
            np.linspace(start[0], end[0], number_of_waypoints),
            np.linspace(start[1], end[1], number_of_waypoints),
            np.linspace(start[2], end[2], number_of_waypoints)
        ]).T
        
        trajectories[i] = trajectory_3d

    # Metoda B: Pełna wektoryzacja (Szybsza dla dużych rojów, N > 1000)
    # alpha = np.linspace(0, 1, number_of_waypoints)  # (W,)
    # # Broadcasting: Start + alpha * (Target - Start)
    # # (N, 1, 3) + (1, W, 1) * (N, 1, 3) -> (N, W, 3)
    # vector = target_positions - start_positions
    # trajectories = start_positions[:, np.newaxis, :] + alpha[np.newaxis, :, np.newaxis] * vector[:, np.newaxis, :]

    return trajectories
