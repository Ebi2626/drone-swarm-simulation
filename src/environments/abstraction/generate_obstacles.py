import numpy as np
from typing import Callable, NamedTuple, Literal
from numpy.typing import NDArray
from src.environments.abstraction.generate_world_boundaries import WorldData

class ObstaclesData(NamedTuple):
    """
    Math abstraction of obstacles
    
    Attributes:
        data (np.ndarray): Matrix (N, 6)  conatining coordinates of obstacles and their dimensions.\n
            One obstacles row: [x, y, z, dim1, dim2, dim3]\n
            For CYLINDER: [x, y, z, radius, height, 0.0]\n
            For BOX:      [x, y, z, length_x, width_y, height_z]
        shape_type (str): 'CYLINDER' or 'BOX'
    """
    data: NDArray[np.float64] 
    shape_type: Literal['CYLINDER', 'BOX']

    @property
    def count(self) -> int:
        return self.data.shape[0]

PlacementStrategy = Callable[[NDArray[np.float64], NDArray[np.float64], int], NDArray[np.float64]]
"""
    Strategy for obstacles generation. It takes minimum and maximum bounds and amount of obstacles and returns positions of obstacles.

    Args:
        min_b (np.ndarray): minimal values of x, y, z - np.ndarray [min_x, min_y, min_z]
        max_b (np.ndarray): maximum values of x, y, z - np.ndarray [max_x, max_y, max_z]
        count (int): int - number of obstacles
    
    Returns:
        np.ndarray: positions of obstacles (N, 3) [x, y, z]
"""

def strategy_random_uniform(min_b: np.ndarray, max_b: np.ndarray, count: int) -> np.ndarray:
    """
    Simple strategy to generate obstacles in completly random way

    Args:
        min_b (np.ndarray): minimal values of x, y, z - np.ndarray [min_x, min_y, min_z]
        max_b (np.ndarray): maximum values of x, y, z - np.ndarray [max_x, max_y, max_z]
        count (int): int - number of obstacles

    Returns:
        np.ndarray: positions of obstacles (N, 3) [x, y, z]
    """
    return np.random.uniform(low=min_b, high=max_b, size=(count, 3))


def strategy_grid_jitter(min_b: np.ndarray, max_b: np.ndarray, count: int) -> np.ndarray:
    """
    Generates obstacles in a grid pattern attempting to preserve square block aspect ratio.
    Guarantees exactly 'count' obstacles are returned by oversampling grid and selecting subset.
    
    Args:
        min_b (np.ndarray): [min_x, min_y, min_z]
        max_b (np.ndarray): [max_x, max_y, max_z]
        count (int): exact number of obstacles required
    """
    # 1. Obliczenie wymiarów obszaru
    lx = max_b[0] - min_b[0]
    ly = max_b[1] - min_b[1]
    
    # Zabezpieczenie dla płaskich lub zerowych światów
    if lx <= 0 or ly <= 0:
        return np.zeros((count, 3))

    # 2. Oblicz optymalne proporcje siatki (nx, ny)
    # Dążymy do tego, by komórka była kwadratowa (lx/nx approx ly/ny)
    ratio = lx / ly
    
    # Wstępne szacowanie nx, ny
    nx = np.sqrt(count * ratio)
    ny = np.sqrt(count / ratio)
    
    nx = max(1, int(np.round(nx)))
    ny = max(1, int(np.round(ny)))
    
    # 3. Korekta: Upewnij się, że siatka ma co najmniej 'count' punktów
    # Jeśli nx*ny < count, powiększamy wymiar, który najmniej zaburzy aspect ratio
    while nx * ny < count:
        if (lx / (nx + 1)) / (ly / ny) > (lx / nx) / (ly / (ny + 1)): 
            # Sprawdzenie, który ruch utrzymuje ratio bliżej 1.0 (uproszczona heurystyka: powiększamy mniejszy bok lub proporcjonalnie)
            # Tutaj prościej: inkrementuj ten wymiar, który jest "niedoszacowany" względem idealnego ratio
            if nx / ny < ratio:
                nx += 1
            else:
                ny += 1
        else:
             # Fallback dla prostych przypadków
             if nx < ny: nx += 1
             else: ny += 1
    
    # 4. Generowanie pełnej siatki
    x = np.linspace(min_b[0], max_b[0], nx)
    y = np.linspace(min_b[1], max_b[1], ny)
    xv, yv = np.meshgrid(x, y)
    
    positions = np.column_stack([xv.ravel(), yv.ravel(), np.zeros(nx*ny)])
    
    # 5. Losowy wybór dokładnie 'count' pozycji (bez powtórzeń)
    # To symuluje miasto, w którym nie każda działka jest zabudowana
    indices = np.random.choice(positions.shape[0], count, replace=False)
    final_positions = positions[indices]
    
    # 6. Dodanie szumu (Jitter)
    # Skalujemy szum względem wielkości działki, by budynki na siebie nie wchodziły
    avg_block_size = (lx/nx + ly/ny) / 2
    noise_scale = avg_block_size * 0.15 # 15% wielkości bloku
    
    noise = np.random.normal(0, noise_scale, final_positions.shape)
    noise[:, 2] = 0 # Zerujemy szum w Z (budynki stoją na ziemi)
    
    return final_positions + noise

def generate_obstacles(
    world: WorldData,
    n_obstacles: int,
    shape_type: Literal['CYLINDER', 'BOX'] = 'CYLINDER',
    placement_strategy: PlacementStrategy = strategy_random_uniform,
    size_params: dict = {'radius': 5.0, 'height': 20.0, 'width': 5.0, 'length': 5.0}
) -> ObstaclesData:
    """
    Generating obstacles of given type, size and amount with given strategy 

    Args:
        world (WorldData): WorldData - named tuple with information about world boundaries
        n_obstacles (int): amount of obstacles to generate
        shape_type (Literal['CYLINDER', 'BOX'], optional): type of obstacle - cylinder or box. Defaults to 'CYLINDER'.
        placement_strategy (PlacementStrategy, optional): strategy for placing obstacles in the world. Defaults to strategy_random_uniform.
        size_params (_type_, optional): _description_. Defaults to {'radius': 5.0, 'height': 20.0, 'width': 5.0, 'length': 5.0}.

    Returns:
        ObstaclesData: object with information about obstacles \n
            data (np.ndarray): Matrix (N, 6)  conatining coordinates of obstacles and their dimensions. \n
            shape_type (str): 'CYLINDER' lub 'BOX'
    """

    # Creating copy of world boundaries to avoid modifying the original data
    spawn_min = world.min_bounds.copy()
    spawn_max = world.max_bounds.copy()
    
    # Ensure all obstacles are located on the ground (z-axis)
    spawn_min[2] = 0.0
    spawn_max[2] = 0.0 
    
    # Counting obstacle positions with given strategy
    positions = placement_strategy(spawn_min, spawn_max, n_obstacles)
    
    # Prepare data matrix for obstacles
    dimensions = np.zeros((n_obstacles, 3))
    
    # Fill data matrix with obstacle coordinates and dimensions
    # Different shapes can be used depending on the requirements
    if shape_type == 'CYLINDER':
        # [radius, height, 0]
        r = size_params.get('radius', 5.0)
        h = size_params.get('height', 10.0)
        dimensions[:, 0] = r
        dimensions[:, 1] = h
        
    elif shape_type == 'BOX':
        # [length, width, height]
        length = size_params.get('length', 5.0)
        width = size_params.get('width', 5.0)
        height = size_params.get('height', 10.0)
        dimensions[:, 0] = length
        dimensions[:, 1] = width
        dimensions[:, 2] = height
    
    # Compound everything into single matrix (N, 6)
    # [pos_x, pos_y, pos_z, dim1, dim2, dim3]
    full_data = np.hstack([positions, dimensions])
    
    return ObstaclesData(
        data=full_data,
        shape_type=shape_type
    )
