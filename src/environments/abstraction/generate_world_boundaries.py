from dataclasses import dataclass
import numpy as np

@dataclass
class WorldData:
    """
    World data class.

    Args:
       dimensions (np.ndarray): world dimensions (x, y, z)
       min_bounds (np.ndarray): minimum bounds of the world (x, y, z)
       max_bounds (np.ndarray): maximum bounds of the world (x, y, z)
       bounds (np.ndarray): world bounds [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
       center (np.ndarray[np.float64]): center of the world
    Returns:
        WordlData: (dimensions: np.ndarray, min_bounds: np.ndarray, max_bounds: np.ndarray, center: np.ndarray[np.float64])
    """
    dimensions: np.ndarray # x, y, z
    min_bounds: np.ndarray # min x, min y, min z
    max_bounds: np.ndarray # max x, max y, max z
    bounds: np.ndarray # World bounds [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    center: np.ndarray[np.float64]



def generate_world_boundaries(width: float, length: float, height: float, ground_height: float) -> WorldData:
    """ 
    Ground, ceiling and side world boundaries generation.
    World should start from (0, 0, 0) and end at (width, length, height). 

    Args:
        width (float): area width in meters (float)
        length (float): area length in meters (float)
        height (float): area height in meters (float)
        ground_height (float): ground location height in meters (float) - assuming we are creating overlay above default pybullet ground

    Returns:
        WorldData: (dimensions: np.ndarray, min_bounds: np.ndarray, max_bounds: np.ndarray)
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
