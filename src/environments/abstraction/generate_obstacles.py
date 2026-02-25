import numpy as np
from typing import Callable, NamedTuple, Literal, Optional, Protocol
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

class PlacementStrategy(Protocol):
    """
    Protocol for obstacle generation strategies.
    Defines the signature that any grid generation function must follow.
    """
    def __call__(
        self, 
        min_b: NDArray[np.float64], 
        max_b: NDArray[np.float64], 
        count: int,
        start_positions: Optional[NDArray[np.float64]] = None,
        target_positions: Optional[NDArray[np.float64]] = None,
        safe_radius: float = 15.0
    ) -> NDArray[np.float64]:
        """
        Generates obstacle positions within bounds.
        
        Args:
            min_b: [min_x, min_y, min_z]
            max_b: [max_x, max_y, max_z]
            count: Number of obstacles to generate
            start_positions: (Optional) Protected start zones (N, 3)
            target_positions: (Optional) Protected target zones (N, 3)
            safe_radius: (Optional) Radius around protected zones
            
        Returns:
            (N, 3) Array of obstacle center positions.
        """
        ...

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

def strategy_grid_jitter(
    min_b: np.ndarray, 
    max_b: np.ndarray, 
    count: int,
    start_positions: Optional[np.ndarray] = None,
    target_positions: Optional[np.ndarray] = None,
    safe_radius: float = 15.0
) -> np.ndarray:
    """
    Generates obstacles in a grid pattern preserving aspect ratio,
    while excluding areas around start and target positions.
    
    Args:
        min_b (np.ndarray): [min_x, min_y, min_z]
        max_b (np.ndarray): [max_x, max_y, max_z]
        count (int): exact number of obstacles required
        start_positions (np.ndarray, optional): Array of start points (N, 3) to protect.
        target_positions (np.ndarray, optional): Array of target points (N, 3) to protect.
        safe_radius (float): Radius around start/target where no obstacles can exist.
    """
    # 1. Obliczenie wymiarów obszaru
    lx = max_b[0] - min_b[0]
    ly = max_b[1] - min_b[1]
    
    if lx <= 0 or ly <= 0:
        return np.zeros((count, 3))

    # 2. Szacowanie rozmiaru siatki z NADMIAREM (Oversampling)
    # Generujemy więcej punktów (np. 1.5x lub 2x), żeby po odfiltrowaniu stref bezpiecznych
    # wciąż mieć z czego wybrać wymagane 'count' przeszkód.
    oversample_factor = 2.0  # Generujemy 2x więcej potencjalnych miejsc
    target_count = int(count * oversample_factor)
    
    ratio = lx / ly
    nx = np.sqrt(target_count * ratio)
    ny = np.sqrt(target_count / ratio)
    
    nx = max(1, int(np.round(nx)))
    ny = max(1, int(np.round(ny)))
    
    # 3. Korekta rozmiaru siatki
    while nx * ny < target_count:
        if nx / ny < ratio:
            nx += 1
        else:
            ny += 1
    
    # 4. Generowanie pełnej siatki
    x = np.linspace(min_b[0], max_b[0], nx)
    y = np.linspace(min_b[1], max_b[1], ny)
    xv, yv = np.meshgrid(x, y)
    
    # Wszystkie potencjalne pozycje
    candidates = np.column_stack([xv.ravel(), yv.ravel(), np.zeros(nx*ny)])
    
    # 5. Dodanie szumu (Jitter) PRZED filtrowaniem
    # Dzięki temu sprawdzamy rzeczywistą pozycję przeszkody, a nie idealną kratkę
    avg_block_size = (lx/nx + ly/ny) / 2
    noise_scale = avg_block_size * 0.15
    noise = np.random.normal(0, noise_scale, candidates.shape)
    noise[:, 2] = 0
    candidates += noise
    
    # 6. Filtrowanie (Maskowanie stref bezpiecznych)
    mask = np.ones(candidates.shape[0], dtype=bool)
    
    # Sprawdzenie kolizji ze Startem
    if start_positions is not None:
        # candidates: (M, 3), starts: (N, 3) -> dystans każdy z każdym
        # Dla dużych siatek robimy to w pętli lub przez broadcasting
        # Tutaj prosty broadcasting: (M, 1, 2) - (1, N, 2) -> (M, N, 2)
        # Bierzemy tylko XY dla bezpieczeństwa (ignorujemy Z)
        cand_xy = candidates[:, :2][:, None, :]
        starts_xy = start_positions[:, :2][None, :, :]
        
        dists = np.sqrt(np.sum((cand_xy - starts_xy)**2, axis=2))
        # Jeśli jakikolwiek dystans < safe_radius, to False (odrzucamy)
        in_danger_zone = np.any(dists < safe_radius, axis=1)
        mask = mask & (~in_danger_zone)

    # Sprawdzenie kolizji z Celem
    if target_positions is not None:
        cand_xy = candidates[:, :2][:, None, :]
        targets_xy = target_positions[:, :2][None, :, :]
        dists = np.sqrt(np.sum((cand_xy - targets_xy)**2, axis=2))
        in_danger_zone = np.any(dists < safe_radius, axis=1)
        mask = mask & (~in_danger_zone)
        
    # Zastosowanie maski
    valid_candidates = candidates[mask]
    
    # 7. Wybór ostatecznych punktów
    available = valid_candidates.shape[0]
    
    if available < count:
        print(f"[WARN] Nie udało się wygenerować {count} przeszkód z zachowaniem bezpiecznej strefy.")
        print(f"       Zwiększ obszar lub zmniejsz liczbę przeszkód/promień. Dostępne: {available}")
        # Zwracamy tyle ile mamy, resztę dopełniamy zerami (lub po prostu mniej przeszkód)
        # Tutaj zwracam po prostu mniej przeszkód, żeby nie tworzyć "duchów" w (0,0,0)
        return valid_candidates
        
    indices = np.random.choice(available, count, replace=False)
    final_positions = valid_candidates[indices]
    
    return final_positions

def generate_obstacles(
    world: WorldData,
    n_obstacles: int,
    shape_type: Literal['CYLINDER', 'BOX'] = 'CYLINDER',
    placement_strategy: PlacementStrategy = strategy_random_uniform,
    size_params: dict = {'radius': 5.0, 'height': 20.0, 'width': 5.0, 'length': 5.0},
    start_positions: np.ndarray = None,
    target_positions: np.ndarray = None,
    safe_radius: float = 15.0
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
    positions = placement_strategy(spawn_min, spawn_max, n_obstacles, start_positions, target_positions, safe_radius)
    
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
