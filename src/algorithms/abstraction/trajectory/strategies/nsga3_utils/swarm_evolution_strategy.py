"""
Swarm Evolution Strategy Orchestrator.
Główny punkt wejścia dla strategii NSGA-III. Łączy komponenty matematyczne, 
ocenę (fitness) i logikę decyzyjną w gotowy do użycia algorytm.

Implementuje protokół: TrajectoryStrategyProtocol.
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
from .core_math import precompute_bspline_matrix, trajectory_from_genotype
from .objective_constrains import VectorizedEvaluator
from .decision_maker import (
    DecisionStrategyProtocol, 
    EqualWeightsDecision, 
    SafetyPriorityDecision, 
    KneePointDecision
)

# Typy z projektu (zakładamy, że są dostępne w PYTHONPATH)
try:
    from src.environments.abstraction.generate_world_boundaries import WorldData
    from src.environments.abstraction.generate_obstacles import ObstaclesData
except ImportError:
    # Fallback dla celów definicji typów (jeśli uruchamiane izolowanie)
    class WorldData: bounds: Any
    class ObstaclesData: pass


# --- Klasa Problemu Pymoo ---

class SwarmOptimizationProblem(Problem):
    """
    Definicja problemu optymalizacyjnego dla Pymoo.
    Łączy genotyp (punkty kontrolne) z funkcjami oceny (Evaluator).
    """
    def __init__(
        self, 
        n_drones: int, 
        n_control: int, 
        n_waypoints: int,
        bounds: NDArray,
        evaluator: VectorizedEvaluator
    ):
        self.n_drones = n_drones
        self.n_control = n_control
        self.n_waypoints = n_waypoints
        self.evaluator = evaluator
        
        # Prekomputacja macierzy B-Spline (raz na cały proces)
        self.basis_matrix = precompute_bspline_matrix(n_control, n_waypoints)
        
        # Liczba zmiennych decyzyjnych: N_Drones * N_Control * 3 (XYZ)
        n_var = n_drones * n_control * 3

        # Rozszerzamy granice o 20% poza świat, żeby dać algorytmowi "rozbieg"
        margin = 50.0 
        
        # Granice zmiennych (Bounding Box świata)
        xl_one_point = bounds[:, 0] - margin
        xu_one_point = bounds[:, 1] + margin
        
        # Powielamy granice dla wszystkich punktów kontrolnych wszystkich dronów
        xl = np.tile(xl_one_point, n_drones * n_control)
        xu = np.tile(xu_one_point, n_drones * n_control)
        
        super().__init__(n_var=n_var, n_obj=3, n_ieq_constr=3, xl=xl, xu=xu)

    def _evaluate(self, x: NDArray, out: Dict[str, Any], *args, **kwargs) -> None:
        # x shape: (Pop_Size, N_Var)
        pop_size = x.shape[0]
        
        # 1. Dekodowanie genotypu -> (Pop, Drones, Controls, 3)
        try:
            import cupy as cp
            has_cupy = True
        except ImportError:
            has_cupy = False
            
        if has_cupy:
            x_gpu = cp.asarray(x)
            genotype = x_gpu.reshape(pop_size, self.n_drones, self.n_control, 3)
            # Basis matrix też na GPU
            basis_gpu = cp.asarray(self.basis_matrix)
            trajectories = trajectory_from_genotype(genotype, basis_gpu)
        else:
            genotype = x.reshape(pop_size, self.n_drones, self.n_control, 3)
            trajectories = trajectory_from_genotype(genotype, self.basis_matrix)
            
        # 2. Ewaluacja (Cele i Ograniczenia)
        self.evaluator.evaluate(trajectories, out)
        
        # 3. Powrót do CPU dla Pymoo
        if has_cupy:
            out["F"] = cp.asnumpy(out["F"])
            out["G"] = cp.asnumpy(out["G"])


# --- Sampling Heurystyczny ---

class HeuristicSampling(Sampling):
    """
    Inicjalizuje populację liniami prostymi z małym szumem.
    Zapewnia "dobry start" spełniający ograniczenia budżetowe.
    """
    def __init__(self, start_pos: NDArray, target_pos: NDArray, n_control: int, n_drones: int):
        super().__init__()
        self.start = start_pos
        self.target = target_pos
        self.n_control = n_control
        self.n_drones = n_drones

    def _do(self, problem, n_samples, **kwargs):
        # t: wektor postępu [0, ..., 1]
        t = np.linspace(0, 1, self.n_control).reshape(1, 1, self.n_control, 1)
        
        s = self.start[None, :, None, :]
        e = self.target[None, :, None, :]
        
        # Linia prosta
        lines = s + t * (e - s)
        
        # Powielamy dla całej populacji
        X = np.tile(lines, (n_samples, 1, 1, 1))
        
        # --- ZMIANA: Zmniejszony szum (sigma 1.0 -> 0.2) ---
        # Przy rozstawie dronów co 2.0m, szum 1.0 powodował, że 
        # drony często "wskakiwały" na siebie (kolizja < 1.5m).
        noise = np.random.normal(0, 0.2, X.shape)
        X += noise
        
        # Spłaszczamy
        X_flat = X.reshape(n_samples, -1)
        
        # --- FIX: Przycinanie do granic (z poprzedniego kroku) ---
        if problem.xl is not None and problem.xu is not None:
            X_flat = np.clip(X_flat, problem.xl, problem.xu)
            
        return X_flat


# --- Helper: Dynamiczne partycje dla NSGA-III ---

def calculate_n_partitions(pop_size: int, n_obj: int = 3) -> int:
    """
    Oblicza liczbę partycji dla punktów referencyjnych Das-Dennis,
    aby liczba punktów była zbliżona do zadanej wielkości populacji.
    
    Wzór na liczbę punktów ref: (M + p - 1)! / (p! * (M-1)!)
    Dla M=3: H = (p+2)*(p+1)/2
    Chcemy H ~= pop_size => p ~= sqrt(2 * pop_size)
    """
    if n_obj != 3:
        # Fallback dla innej liczby celów (uproszczony)
        return 12 
        
    # Dla M=3, p ~= sqrt(2 * pop_size)
    p = int(np.sqrt(2 * pop_size))
    # Korekta w dół, żeby nie przekroczyć drastycznie
    return max(2, p - 1)


# --- Główna Funkcja Strategii ---

def nsga3_swarm_strategy(
    *, 
    start_positions: NDArray[np.float64],
    target_positions: NDArray[np.float64],
    obstacles_data: Union[Any, List[Any]], 
    world_data: WorldData,
    number_of_waypoints: int,
    drone_swarm_size: int, 
    algorithm_params: Optional[Dict[str, Any]] = None
) -> NDArray[np.float64]:
    
    # 1. Konfiguracja Parametrów
    params = algorithm_params or {}
    pop_size = params.get("pop_size", 100)
    n_gen = params.get("n_gen", 100)
    n_control = params.get("n_control_points", 5)
    decision_mode = params.get("decision_mode", "knee_point")
    
    if isinstance(obstacles_data, list):
        obs_list = obstacles_data
    else:
        obs_list = [obstacles_data]

    try:
        bounds = world_data.bounds
    except AttributeError:
        bounds = np.array([[-100, 100], [-100, 100], [0, 100]])
    
    # 2. Obliczanie parametrów algorytmu
    # Dynamicznie dobieramy n_partitions do wielkości populacji
    n_partitions = calculate_n_partitions(pop_size, n_obj=3)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
    
    actual_pop_size = ref_dirs.shape[0]
    # Jeśli wyliczona populacja różni się drastycznie, Pymoo i tak dobierze pop_size,
    # ale staramy się trzymać je blisko.
    
    print(f"[NSGA-III] Start. Pop: {pop_size} (RefDirs: {actual_pop_size}), Gen: {n_gen}, Ctrl: {n_control}")

    # 3. Inicjalizacja komponentów
    evaluator = VectorizedEvaluator(
        obstacles=obs_list,
        start_pos=start_positions,
        target_pos=target_positions,
        params=params
    )
    
    problem = SwarmOptimizationProblem(
        n_drones=drone_swarm_size,
        n_control=n_control,
        n_waypoints=number_of_waypoints,
        bounds=bounds,
        evaluator=evaluator
    )
    
    sampling = HeuristicSampling(
        start_pos=start_positions,
        target_pos=target_positions,
        n_control=n_control,
        n_drones=drone_swarm_size
    )
    
    algorithm = NSGA3(
        pop_size=pop_size, # Pymoo użyje tego jako wymuszonej wielkości populacji
        ref_dirs=ref_dirs,
        sampling=sampling,
        crossover=SBX(prob=0.9, eta=10),
        mutation=PM(eta=10, prob=0.3),
        eliminate_duplicates=True
    )
    
    # 4. Uruchomienie Optymalizacji
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_gen),
        seed=1,
        verbose=True
    )
    
    # 5. Decyzja
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
        
        genotype = best_genotype_flat.reshape(1, drone_swarm_size, n_control, 3)
        final_traj = trajectory_from_genotype(genotype, problem.basis_matrix)
        
        print(f"[NSGA-III] Znaleziono rozwiązanie. Wybrano indeks: {best_idx}")
        return final_traj[0]
        
    else:
        print("[NSGA-III] Błąd: Nie znaleziono rozwiązań dopuszczalnych! Zwracam prostą linię.")
        # Fallback z zerowym szumem, aby zagwarantować linię prostą
        fallback_sampling = HeuristicSampling(start_positions, target_positions, n_control, drone_swarm_size)
        flat = fallback_sampling._do(problem, 1)
        # Zerujemy szum (nadpisujemy logikę próbkowania dla fallbacku)
        # Ręczna rekonstrukcja dla pewności
        t = np.linspace(0, 1, n_control).reshape(1, 1, n_control, 1)
        s = start_positions[None, :, None, :]
        e = target_positions[None, :, None, :]
        lines = s + t * (e - s)
        gen = lines # Bez szumu!
        
        traj = trajectory_from_genotype(gen, problem.basis_matrix)
        return traj[0]
