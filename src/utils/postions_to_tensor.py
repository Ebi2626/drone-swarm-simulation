

import numpy as np
from typing import List

def positions_to_tensor(positions: List[List[float]]) -> np.ndarray:
    """Zamień `List[List[x, y, z]]` na `np.ndarray (N, 3)` z walidacją kształtu.

    Args:
        positions: Lista list pozycji dronów (start lub target).

    Returns:
        `(N, 3)` macierz float64 `[x, y, z]`.

    Raises:
        ValueError: Gdy końcowy kształt nie jest `(N, 3)`.
    """

    tensor_positions = np.array(positions, dtype=np.float64)
    
    if tensor_positions.ndim != 2 or tensor_positions.shape[1] != 3:
        raise ValueError(
            f"Dimensional error: expected tensor shape (N, 3), "
            f"where N is a number of drones, and 3 represents coordinates [x, y, z]. "
            f"Current shape: {tensor_positions.shape}"
        )
        
    return tensor_positions
