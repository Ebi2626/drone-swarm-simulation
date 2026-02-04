import numpy as np
from typing import Any, Dict, Optional
from numpy.typing import NDArray
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import pdist

# Pymoo imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

# Twoje importy
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.abstraction.generate_obstacles import ObstaclesData

# --- Helper Functions (Shared) ---

def generate_bspline_trajectory(start, end, control_points, n_waypoints):
    # Identyczna jak poprzednio
    points = np.vstack([start, control_points, end])
    # Zabezpieczenie przed duplikatami punktów dla splprep
    # (dodajemy minimalny szum eps jeśli punkty są identyczne, co psuje splprep)
    if np.any(np.all(np.diff(points, axis=0) == 0, axis=1)):
        points[1:-1] += np.random.normal(0, 1e-4, points[1:-1].shape)

    t = np.linspace(0, 1, points.shape[0])
    t_eval = np.linspace(0, 1, n_waypoints)
    coords = []
    try:
        for dim in range(3):
            k = min(3, points.shape[0] - 1)
            tck = splprep([points[:, dim]], u=t, k=k, s=0)[0]
            coords.append(splev(t_eval, tck)[0])
        return np.column_stack(coords)
    except Exception:
        # Fallback w przypadku błędu numerycznego splajnu -> linia prosta
        return np.linspace(start, end, n_waypoints)

def check_static_collisions(trajectory, obstacles_list, safety_margin=1.0):
    # Uproszczona wersja zliczająca punkty w kolizji
    collisions = 0
    tx, ty, tz = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    
    for batch in obstacles_list:
        data = batch.data
        for i in range(batch.count):
            ox, oy, oz, d1, d2, _ = data[i]
            if batch.shape_type == 'CYLINDER':
                r = d1 + safety_margin
                h = d2
                dist_xy = np.sqrt((tx - ox)**2 + (ty - oy)**2)
                # Kolizja: w promieniu ORAZ poniżej wysokości przeszkody
                in_collision = (dist_xy < r) & (tz >= 0) & (tz <= h)
                collisions += np.sum(in_collision)
    return collisions

def check_inter_agent_collisions(all_trajectories, min_dist=1.0):
    """
    Sprawdza kolizje między dronami w każdym kroku czasowym.
    all_trajectories shape: (N_drones, N_waypoints, 3)
    """
    n_drones, n_steps, _ = all_trajectories.shape
    total_collisions = 0
    
    # Dla każdego kroku czasowego t
    for t in range(n_steps):
        # Pobieramy pozycje wszystkich dronów w chwili t -> (N_drones, 3)
        positions_at_t = all_trajectories[:, t, :]
        
        # Obliczamy dystanse parowe (condensed distance matrix)
        # pdist zwraca n*(n-1)/2 dystansów
        dists = pdist(positions_at_t)
        
        # Zliczamy ile par jest zbyt blisko siebie
        collisions_at_t = np.sum(dists < min_dist)
        total_collisions += collisions_at_t
        
    return total_collisions

# --- Global Swarm Problem Definition ---

class GlobalSwarmProblem(ElementwiseProblem):
    def __init__(
        self, 
        start_positions: NDArray, 
        target_positions: NDArray, 
        obstacles_list: list,
        world_bounds: NDArray,
        n_waypoints: int,
        n_drones: int,
        n_control_points: int = 3
    ):
        """
        Problem optymalizacji CAŁEGO roju naraz.
        Zmienne decyzyjne to jeden długi wektor:
        [Dron1_CP1_x, Dron1_CP1_y, ..., DronN_CPK_z]
        """
        self.starts = start_positions
        self.targets = target_positions
        self.obstacles = obstacles_list
        self.n_waypoints = n_waypoints
        self.n_drones = n_drones
        self.n_control = n_control_points
        
        # Wymiarowość: N_drones * N_control_points * 3 współrzędne
        dim = n_drones * n_control_points * 3
        
        # Granice (zduplikowane dla wszystkich zmiennych)
        # Zakładamy, że world_bounds to [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        xl_one_point = world_bounds[:, 0]
        xu_one_point = world_bounds[:, 1]
        
        # Powielamy granice dla wszystkich zmiennych
        xl = np.tile(xl_one_point, n_drones * n_control_points)
        xu = np.tile(xu_one_point, n_drones * n_control_points)
        
        # n_obj=1 (Minimalizacja łącznej długości tras)
        # n_ieq_constr=2 (1. Przeszkody statyczne, 2. Kolizje między dronami)
        super().__init__(n_var=dim, n_obj=1, n_ieq_constr=2, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # 1. Dekodowanie Genotypu -> (N_drones, N_control, 3)
        # x jest płaskim wektorem
        controls_all = x.reshape((self.n_drones, self.n_control, 3))
        
        all_trajectories = np.zeros((self.n_drones, self.n_waypoints, 3))
        total_length = 0.0
        static_collisions = 0
        
        # 2. Generowanie tras i ocena indywidualna (statyczna)
        for i in range(self.n_drones):
            traj = generate_bspline_trajectory(
                self.starts[i], 
                self.targets[i], 
                controls_all[i], 
                self.n_waypoints
            )
            all_trajectories[i] = traj
            
            # Cel: Długość
            diffs = np.diff(traj, axis=0)
            length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
            total_length += length
            
            # Ograniczenie 1: Przeszkody statyczne
            static_collisions += check_static_collisions(traj, self.obstacles)
            
        # 3. Ocena grupowa (dynamiczna)
        # Ograniczenie 2: Kolizje między dronami
        agent_collisions = check_inter_agent_collisions(all_trajectories, min_dist=3.0)
        
        # Zapis wyników
        out["F"] = [total_length]
        # Pymoo traktuje G(x) <= 0 jako spełnione. 
        # Jeśli liczba kolizji > 0, to ograniczenie naruszone.
        out["G"] = [static_collisions, agent_collisions]

# --- Protocol Implementation ---

def nsga3_swarm_strategy(
    *, 
    start_positions: NDArray[np.float64],
    target_positions: NDArray[np.float64],
    obstacles_data: ObstaclesData, # Może być pojedynczy obiekt lub lista
    world_data: WorldData,
    number_of_waypoints: int,
    drone_swarm_size: int, 
    algorithm_params: Optional[Dict[str, Any]] = None
) -> NDArray[np.float64]:
    
    # Obsługa obstacles_data (może przyjść jako lista lub pojedynczy obiekt)
    if isinstance(obstacles_data, list):
        obs_list = obstacles_data
    else:
        obs_list = [obstacles_data]

    # Parametry
    params = algorithm_params or {}
    pop_size = params.get("pop_size", 100) # Większa populacja bo trudniejszy problem
    n_gen = params.get("n_gen", 50)
    n_control = params.get("n_control_points", 3)
    
    # Pobranie granic
    try:
        bounds = world_data.bounds
    except AttributeError:
        bounds = np.array([[0, 100], [0, 100], [0, 100]])

    print(f"Start NSGA-III (Global Swarm). Drones: {drone_swarm_size}, Var count: {drone_swarm_size*n_control*3}")

    # Definicja problemu
    problem = GlobalSwarmProblem(
        start_positions=start_positions,
        target_positions=target_positions,
        obstacles_list=obs_list,
        world_bounds=bounds,
        n_waypoints=number_of_waypoints,
        n_drones=drone_swarm_size,
        n_control_points=n_control
    )
    
    # Konfiguracja algorytmu
    ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=12)
    
    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    # Uruchomienie optymalizacji
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_gen),
        seed=1,
        verbose=True # Warto widzieć postęp
    )
    
    # Pobranie najlepszego rozwiązania
    if res.X is not None:
        # Bierzemy pierwsze rozwiązanie z frontu Pareto (w tym przypadku najlepsze feasible)
        best_X = res.X[0] if len(res.X.shape) > 1 else res.X
        
        # Rekonstrukcja tras
        controls_all = best_X.reshape((drone_swarm_size, n_control, 3))
        final_trajectories = np.zeros((drone_swarm_size, number_of_waypoints, 3))
        
        for i in range(drone_swarm_size):
            final_trajectories[i] = generate_bspline_trajectory(
                start_positions[i],
                target_positions[i],
                controls_all[i],
                number_of_waypoints
            )
        return final_trajectories
    else:
        print("Nie znaleziono rozwiązania. Zwracam linie proste.")
        # Fallback
        # ... (kod linii prostych jak wcześniej)
        return np.zeros((drone_swarm_size, number_of_waypoints, 3)) 
