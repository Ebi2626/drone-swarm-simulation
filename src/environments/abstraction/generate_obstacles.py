import numpy as np
from typing import NamedTuple, Optional, Protocol, TypedDict
from typing_extensions import NotRequired
from numpy.typing import NDArray
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.obstacles.ObstacleShape import ObstacleShape

class SizeParams(TypedDict):
    """Wymiary przeszkody — `height` i `width` wymagane, `length` opcjonalny (BOX)."""
    height: float
    width: float
    length: NotRequired[float]


class ObstaclesData(NamedTuple):
    """Macierzowa reprezentacja przeszkód `(N, 6)` z typem kształtu.

    Attributes:
        data: `(N, 6)` z wierszami `[x, y, z, dim1, dim2, dim3]`:
              - CYLINDER: `[x, y, z, radius, height, 0.0]`,
              - BOX:      `[x, y, z, length_x, width_y, height_z]`.
        shape_type: `ObstacleShape.CYLINDER` lub `ObstacleShape.BOX`.
    """
    data: NDArray[np.float64]
    shape_type: ObstacleShape

    @property
    def count(self) -> int:
        """Liczba przeszkód (`N`)."""
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
        safe_radius: float = 1.5
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

def strategy_random_uniform(
    min_b: np.ndarray,
    max_b: np.ndarray,
    count: int,
    start_positions: Optional[np.ndarray] = None,
    target_positions: Optional[np.ndarray] = None,
    safe_radius: float = 30.0,
    rng: np.random.Generator | int | None = None,
    *args
) -> np.ndarray:
    """Wygeneruj `count` losowych pozycji `(N, 3)` z odrzucaniem stref bezpiecznych start/target.

    Args:
        min_b, max_b: `(3,)` granice świata `[min/max_x, _y, _z]`.
        count: Wymagana liczba pozycji.
        start_positions, target_positions: `(N, 3)` punkty chronione (lub `None`).
        safe_radius: Minimalna odległość od dowolnego punktu chronionego.
        rng: Ziarno deterministyczne dla `np.random.default_rng`.

    Returns:
        `(count, 3)` macierz pozycji `[x, y, z]`.
    """

    # 0. Usawienie seeda
    rng = np.random.default_rng(rng)

    # 1. Agregacja i unifikacja wszystkich chronionych punktów do formatu 2D
    protected_points = []
    if start_positions is not None:
        protected_points.append(np.atleast_2d(start_positions))
    if target_positions is not None:
        protected_points.append(np.atleast_2d(target_positions))
        
    if protected_points:
        protected_array = np.vstack(protected_points)
    else:
        protected_array = np.empty((0, 3))
        
    positions = np.zeros((0, 3))
    
    # 2. Metoda akceptacji-odrzucenia (Rejection Sampling)
    while len(positions) < count:
        needed = count - len(positions)
        
        # Generowanie paczki kandydatów
        candidates = rng.uniform(low=min_b, high=max_b, size=(needed, 3))
        
        if len(protected_array) > 0:
            # Wektoryzowane obliczanie odległości euklidesowych (Broadcasting)
            # candidates kształt: (needed, 1, 3), protected_array kształt: (1, K, 3)
            diff = candidates[:, np.newaxis, :] - protected_array[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2) # kształt: (needed, K)
            
            # Pobranie najmniejszej odległości do JAKIEGOKOLWIEK punktu chronionego
            min_distances = np.min(distances, axis=1)
            
            # Akceptacja tylko tych kandydatów, którzy są bezpiecznie oddaleni
            valid_candidates = candidates[min_distances >= safe_radius]
        else:
            valid_candidates = candidates
            
        # Dodanie zaakceptowanych kandydatów do głównej puli
        positions = np.vstack((positions, valid_candidates))
        
    return positions

def strategy_grid_jitter(
    min_b: np.ndarray,
    max_b: np.ndarray,
    count: int,
    start_positions: Optional[np.ndarray] = None,
    target_positions: Optional[np.ndarray] = None,
    safe_radius: float = 15.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Wygeneruj pozycje na siatce zachowującej proporcje + jitter; wyklucza strefy chronione.

    Args:
        min_b, max_b: `(3,)` granice świata.
        count: Wymagana liczba pozycji.
        start_positions, target_positions: `(N, 3)` punkty chronione (lub `None`).
        safe_radius: Promień strefy bezpieczeństwa.
        rng: Ziarno deterministyczne.

    Returns:
        `(≤count, 3)` pozycje. Gdy filtr odrzuci za dużo punktów, zwraca
        mniej niż `count` z `WARN` w stdout.
    """
    # 0. Ustawienie seeda
    rng = np.random.default_rng(rng)

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
    noise = rng.normal(0, noise_scale, candidates.shape)
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
    available = int(valid_candidates.shape[0])
    if available < count:
        print(f"[WARN] Nie udało się wygenerować {count} przeszkód z zachowaniem bezpiecznej strefy.")
        print(f"       Zwiększ obszar lub zmniejsz liczbę przeszkód/promień. Dostępne: {available}")
        # Zwracamy tyle ile mamy, resztę dopełniamy zerami (lub po prostu mniej przeszkód)
        # Tutaj zwracam po prostu mniej przeszkód, żeby nie tworzyć "duchów" w (0,0,0)
        return valid_candidates
        
    indices = rng.choice(available, count, replace=False)
    final_positions = valid_candidates[indices]
    
    return final_positions

def strategy_empty(
    min_b: NDArray[np.float64],
    max_b: NDArray[np.float64],
    count: int,
    start_positions: Optional[NDArray[np.float64]] = None,
    target_positions: Optional[NDArray[np.float64]] = None,
    safe_radius: float = 15.0,
    rng: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    """No-op strategia dla pustego środowiska — zwraca `(0, 3)` macierz."""
    # Zwraca pustą macierz o poprawnym kształcie (0 wierszy, 3 kolumny: X, Y, Z)
    return np.empty((0, 3), dtype=np.float64)

def generate_obstacles(
    world: WorldData,
    n_obstacles: int,
    shape_type: ObstacleShape = ObstacleShape.BOX,
    placement_strategy: PlacementStrategy = strategy_random_uniform,
    size_params: SizeParams = {'height': 20.0, 'width': 5.0, 'length': 5.0},
    start_positions: np.ndarray = None,
    target_positions: np.ndarray = None,
    safe_radius: float = 15.0,
    rng: np.random.Generator | int | None = None,
) -> ObstaclesData:
    """Wygeneruj `ObstaclesData` zadanego kształtu i ilości wybraną strategią rozmieszczenia.

    Args:
        world: `WorldData` z granicami świata.
        n_obstacles: Liczba przeszkód do wygenerowania.
        shape_type: `ObstacleShape.CYLINDER` lub `ObstacleShape.BOX`.
        placement_strategy: Strategia rozmieszczania (sygnatura `PlacementStrategy`).
        size_params: Wymiary przeszkód (`width`/`height`/opcjonalnie `length`).
        start_positions, target_positions: Punkty chronione przekazywane do strategii.
        safe_radius: Promień strefy bezpieczeństwa.
        rng: Ziarno deterministyczne.

    Returns:
        `ObstaclesData(data=(N, 6), shape_type=…)`; faktyczne `N` może być
        mniejsze niż `n_obstacles`, gdy strategia nie zmieściła wszystkich punktów.
    """
    # Seeding
    rng = np.random.default_rng(rng)

    # Creating copy of world boundaries to avoid modifying the original data
    spawn_min = world.min_bounds.copy()
    spawn_max = world.max_bounds.copy()
    
    # Ensure all obstacles are located on the ground (z-axis)
    spawn_min[2] = 0.0
    spawn_max[2] = 0.0 
    
    # Counting obstacle positions with given strategy
    positions = placement_strategy(spawn_min, spawn_max, n_obstacles, start_positions, target_positions, safe_radius, rng=rng)
    
    actual_n = positions.shape[0]
    dimensions = np.zeros((actual_n, 3), dtype=np.float64)

    if shape_type == ObstacleShape.CYLINDER:
        r = size_params.get('width', 5.0)
        h = size_params.get('height', 10.0)
        dimensions[:, 0] = r
        dimensions[:, 1] = h

    elif shape_type == ObstacleShape.BOX:
        length = size_params.get('length', 5.0)
        width = size_params.get('width', 5.0)
        height = size_params.get('height', 10.0)
        dimensions[:, 0] = length
        dimensions[:, 1] = width
        dimensions[:, 2] = height

    full_data = np.hstack([positions, dimensions])
    return ObstaclesData(data=full_data, shape_type=shape_type)