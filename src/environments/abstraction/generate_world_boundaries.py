from dataclasses import dataclass
import numpy as np

@dataclass
class WorldData:
    """Dane geometryczne świata: wymiary, granice min/max, bounds (`(3, 2)`) i środek.

    Attributes:
        dimensions: `(3,)` `[x, y, z]` w metrach.
        min_bounds: `(3,)` `[min_x, min_y, min_z]`.
        max_bounds: `(3,)` `[max_x, max_y, max_z]`.
        bounds: `(3, 2)` `[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`.
        center: `(3,)` środek świata.
    """
    dimensions: np.ndarray
    min_bounds: np.ndarray
    max_bounds: np.ndarray
    bounds: np.ndarray
    center: np.ndarray[np.float64]



def generate_world_boundaries(width: float, length: float, height: float, ground_height: float) -> WorldData:
    """Wygeneruj granice świata `(0, 0, ground_height) → (width, length, height)`.

    Args:
        width: Szerokość obszaru [m].
        length: Długość obszaru [m].
        height: Wysokość świata [m].
        ground_height: Wysokość podłogi [m] (overlay nad domyślnym ground PyBullet).

    Returns:
        `WorldData` z policzonymi `dimensions / min_bounds / max_bounds / bounds / center`.
    """

    dtype = np.float64
    dims = np.array([width, length, height], dtype=dtype)
    min_b = np.array([0.0, 0.0, ground_height], dtype=dtype)
    max_b = np.array([width, length, height], dtype=dtype)
    center = (min_b + max_b) / 2.0
    bounds = np.column_stack((min_b, max_b))

    return WorldData(
        dimensions=dims, 
        min_bounds=min_b, 
        max_bounds=max_b,
        bounds=bounds,
        center=center
        )
