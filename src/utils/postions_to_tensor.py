

import numpy as np
from typing import List

def positions_to_tensor(positions: List[List[float]]) -> np.ndarray:
    """
    Transform positions from List[List[x,y,z]] to tensor NDArray(N, 3).
    
    Args:
        positions (List[List[float]]): List of positions (e.g. start/target) for drones.

    Returns:
        np.ndarray: Tensor array of drone positions with shape (N, 3) [x, y, z].
    """

    tensor_positions = np.array(positions, dtype=np.float64)
    
    if tensor_positions.ndim != 2 or tensor_positions.shape[1] != 3:
        raise ValueError(
            f"Dimensional error: expected tensor shape (N, 3), "
            f"where N is a number of drones, and 3 represents coordinates [x, y, z]. "
            f"Current shape: {tensor_positions.shape}"
        )
        
    return tensor_positions
