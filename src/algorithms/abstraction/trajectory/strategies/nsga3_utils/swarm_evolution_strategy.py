""""
Swarm Evolution Strategy Orchestrator.
Główny punkt wejścia dla strategii NSGA-III (Wersja Polyline).
Zamiast B-Spline używa interpolowanej łamanej z ograniczeniami geometrycznymi.
"""

from typing import Any, Dict, Optional, List, Union
import numpy as np
from numpy.typing import NDArray

# Pymoo imports
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# Komponenty wewnętrzne
from .objective_constrains import VectorizedEvaluator
from .decision_maker import (
    DecisionStrategyProtocol, 
    EqualWeightsDecision, 
    SafetyPriorityDecision, 
    KneePointDecision
)

# Typy z projektu
try:
    from src.environments.abstraction.generate_world_boundaries import WorldData
    from src.environments.abstraction.generate_obstacles import ObstaclesData
except ImportError:
    class WorldData: bounds: Any
    class ObstaclesData: pass


# --- Helper: Resampling Polyline ---


def resample_polyline_batch(
    waypoints: NDArray[np.float64], 
    num_samples: int = 100
) -> NDArray[np.float64]:
    """
    Interpoluje liniowo punkty trasy (waypoints) do zadanej liczby gęstych punktów.
    Działa dla wsadu (Batch): (Pop, Drones, Waypoints, 3) -> (Pop, Drones, Num_Samples, 3).
    """
    pop_size, n_drones, n_in, dims = waypoints.shape
    
    # Przygotowanie wyjścia
    flat_waypoints = waypoints.reshape(-1, n_in, dims) # (Batch, N_In, 3)
    flat_out = np.empty((pop_size * n_drones, num_samples, dims), dtype=waypoints.dtype)
    
    t_in = np.linspace(0, 1, n_in)
    t_out = np.linspace(0, 1, num_samples)
    
    for i in range(pop_size * n_drones):
        w = flat_waypoints[i]
        for d in range(dims):
            flat_out[i, :, d] = np.interp(t_out, t_in, w[:, d])
            
    return flat_out.reshape(pop_size, n_drones, num_samples, dims)


# --- Klasa Problemu Pymoo ---
class SwarmOptimizationProblem(Problem):
    """
    Definicja problemu optymalizacyjnego (Wersja Polyline).
    Zmienne decyzyjne to bezpośrednio punkty pośrednie trasy.
    """
    def __init__(
        self, 
        n_drones: int, 
        n_inner_points: int,  # Zmieniona nazwa z n_control
        n_output_samples: int, # Docelowa liczba punktów (np. 100)
        bounds: NDArray,
        evaluator: VectorizedEvaluator,
        start_pos: NDArray,
        target_pos: NDArray
    ):
        self.n_drones = n_drones
        self.n_inner_points = n_inner_points
        self.n_output_samples = n_output_samples
        self.evaluator = evaluator
        
        # Przechowujemy Start i Cel do doklejenia w _evaluate
        self.starts = start_pos
        self.targets = target_pos
                
        # Liczba zmiennych decyzyjnych: N_Drones * Inner_Points * 3 (XYZ)
        n_var = n_drones * n_inner_points * 3

        # Rozszerzamy granice o margines
        margin = 50.0 
        xl_one_point = bounds[:, 0] - margin
        xu_one_point = bounds[:, 1] + margin
        
        print("[Problem] Liczba zmiennych:", n_var)
        print("[Problem] Granice X:", xl_one_point[0], xu_one_point[0])
        
        # Powielamy granice
        xl = np.tile(xl_one_point, n_drones * n_inner_points)
        xu = np.tile(xu_one_point, n_drones * n_inner_points)
        
        # n_constr=5: Battery, Separation, ObstacleSafety, Uniformity, Smoothness
        super().__init__(n_var=n_var, n_obj=3, n_ieq_constr=5, xl=xl, xu=xu)

    def _evaluate(self, x: NDArray, out: Dict[str, Any], *args, **kwargs) -> None:
        # x shape: (Pop_Size, N_Var)
        pop_size = x.shape[0]
        
        # 1. Dekodowanie zmiennych (Inner Waypoints)
        # Kształt: (Pop, Drones, Inner, 3)
        inner_waypoints = x.reshape(pop_size, self.n_drones, self.n_inner_points, 3)
        
        # 2. Doklejanie Startu i Celu (Tworzenie pełnej łamanej)
        # starts: (N_Drones, 3) -> (Pop, N_Drones, 1, 3)
        starts_bc = np.tile(self.starts[None, :, None, :], (pop_size, 1, 1, 1))
        targets_bc = np.tile(self.targets[None, :, None, :], (pop_size, 1, 1, 1))
        
        # Sparse Polyline: Start -> Inner -> Target
        sparse_trajectory = np.concatenate([starts_bc, inner_waypoints, targets_bc], axis=2)
        
        # 3. Resampling do gęstej trajektorii (wymagane przez Evaluator i fizykę)
        # Zamienia rzadkie punkty (np. 7) na gęste (np. 100)
        trajectories = resample_polyline_batch(sparse_trajectory, num_samples=self.n_output_samples)
            
        # 4. Ewaluacja (Cele i Ograniczenia)
        # Ewaluator policzy 3 cele (F) i 5 ograniczeń (G)
        self.evaluator.evaluate(trajectories, out)


# --- Sampling Heurystyczny ---
class HeuristicSampling(Sampling):
    """
    Inicjalizuje populację punktami leżącymi na prostej Start->Cel.
    Generuje tylko punkty wewnętrzne (inner points).
    """
    def __init__(self, start_pos: NDArray, target_pos: NDArray, n_inner_points: int, n_drones: int):
        super().__init__()
        self.start = start_pos
        self.target = target_pos
        self.n_inner_points = n_inner_points
        self.n_drones = n_drones

    def _do(self, problem, n_samples, **kwargs):
        # Generujemy punkty wewnętrzne w zakresie (0, 1) wykluczając 0 i 1 (bo to Start i Cel)
        # np. dla 3 pkt wew: 0.25, 0.50, 0.75
        t_vals = np.linspace(0, 1, self.n_inner_points + 2)[1:-1]
        t = t_vals.reshape(1, 1, self.n_inner_points, 1)
        
        s = self.start[None, :, None, :]
        e = self.target[None, :, None, :]
        
        # Punkty na prostej
        points = s + t * (e - s)
        
        # Powielamy dla populacji
        X = np.tile(points, (n_samples, 1, 1, 1))
        
        # Szum (mniejszy niż wcześniej, by nie psuć separacji)
        noise = np.random.normal(0, 0.2, X.shape)
        X += noise
        
        # Spłaszczamy do zmiennych decyzyjnych
        X_flat = X.reshape(n_samples, -1)
        
        # Clipping
        if problem.xl is not None and problem.xu is not None:
            X_flat = np.clip(X_flat, problem.xl, problem.xu)
            
        return X_flat


# --- Helper: Dynamiczne partycje ---
def calculate_n_partitions(pop_size: int, n_obj: int = 3) -> int:
    if n_obj != 3: return 12 
    p = int(np.sqrt(2 * pop_size))
    return max(2, p - 1)


# --- Główna Funkcja Strategii ---
def nsga3_swarm_strategy(
    *, 
    start_positions: NDArray[np.float64],
    target_positions: NDArray[np.float64],
    obstacles_data: Union[Any, List[Any]], 
    world_data: WorldData,
    number_of_waypoints: int, # Docelowa liczba punktów wyjściowych (np. 100)
    drone_swarm_size: int, 
    algorithm_params: Optional[Dict[str, Any]] = None
) -> NDArray[np.float64]:
    
    # 1. Konfiguracja Parametrów
    params = algorithm_params or {}
    pop_size = params.get("pop_size", 100)
    n_gen = params.get("n_gen", 100)
    
    # Liczba punktów optymalizowanych (wewnętrznych)
    # Domyślnie bierzemy 10% liczby waypointów wyjściowych, lub min 5
    n_inner = params.get("n_inner_waypoints", max(5, int(number_of_waypoints * 0.1)))
    
    decision_mode = params.get("decision_mode", "knee_point")
    
    if isinstance(obstacles_data, list):
        obs_list = obstacles_data
    else:
        obs_list = [obstacles_data]

    bounds = world_data.bounds
    
    # 2. NSGA-III Setup
    n_partitions = calculate_n_partitions(pop_size, n_obj=3)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
    actual_pop_size = ref_dirs.shape[0]
    
    print(f"[NSGA-III Polyline] Start. Pop: {pop_size} (Ref: {actual_pop_size}), Gen: {n_gen}, Inner Pts: {n_inner}")

    # 3. Inicjalizacja
    evaluator = VectorizedEvaluator(
        obstacles=obs_list,
        start_pos=start_positions,
        target_pos=target_positions,
        params=params
    )
    
    problem = SwarmOptimizationProblem(
        n_drones=drone_swarm_size,
        n_inner_points=n_inner,
        n_output_samples=number_of_waypoints, # To jest np. 100
        bounds=bounds,
        evaluator=evaluator,
        start_pos=start_positions,
        target_pos=target_positions
    )
    
    sampling = HeuristicSampling(
        start_pos=start_positions,
        target_pos=target_positions,
        n_inner_points=n_inner,
        n_drones=drone_swarm_size
    )
    
    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=sampling,
        crossover=SBX(prob=0.9, eta=10),
        mutation=PM(eta=10, prob=0.3),
        eliminate_duplicates=True
    )
    
    # 4. Optymalizacja
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_gen),
        seed=1,
        verbose=True,
        save_history=True
    )

    if res.history:
        last_pop = res.history[-1].pop
        print(f"Ostatnia populacja: {len(last_pop)} osobników")
        # Logowanie dla debugowania
        if len(last_pop) > 0:
            print("Przykładowe G:", last_pop.get("G")[:3])
    
    # 5. Wybór rozwiązania
    if res.X is not None and len(res.X) > 0:
        dm: DecisionStrategyProtocol
        if decision_mode == "safety":
            dm = SafetyPriorityDecision()
        elif decision_mode == "equal":
            dm = EqualWeightsDecision()
        else:
            dm = KneePointDecision()
            
        best_idx = dm.select_best(res.F, res.G)
        best_genotype_flat = res.X[best_idx]
        
        # Rekonstrukcja trasy dla zwycięzcy
        # Musimy powtórzyć logikę z _evaluate dla pojedynczego osobnika
        inner = best_genotype_flat.reshape(1, drone_swarm_size, n_inner, 3)
        s = start_positions[None, :, None, :]
        t = target_positions[None, :, None, :]
        sparse = np.concatenate([s, inner, t], axis=2)
        final_traj = resample_polyline_batch(sparse, num_samples=number_of_waypoints)
        
        print(f"[NSGA-III] Wybrano rozwiązanie indeks: {best_idx}")
        return final_traj[0] # Zwracamy (Drones, Waypoints, 3)
        
    else:
        print("[NSGA-III] Błąd krytyczny: Brak rozwiązań. Zwracam linię prostą.")
        # Fallback
        t = np.linspace(0, 1, number_of_waypoints)
        out = np.empty((drone_swarm_size, number_of_waypoints, 3))
        for d in range(drone_swarm_size):
            for i in range(3):
                out[d, :, i] = np.interp(t, [0, 1], [start_positions[d, i], target_positions[d, i]])
        return out
