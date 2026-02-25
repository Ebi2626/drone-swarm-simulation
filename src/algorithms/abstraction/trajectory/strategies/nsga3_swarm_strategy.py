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

from src.algorithms.abstraction.trajectory.strategies.nsga3_utils.core_math import get_xp

# Komponenty wewnętrzne
# UWAGA: Upewnij się, że importujesz poprawnie swoje moduły
# Jeśli masz problem z importem VectorizedEvaluator, sprawdź ścieżkę
try:
    from src.algorithms.abstraction.trajectory.strategies.nsga3_utils.objective_constrains import VectorizedEvaluator
    from src.algorithms.abstraction.trajectory.strategies.nsga3_utils.decision_maker import (
        DecisionStrategyProtocol, 
        EqualWeightsDecision, 
        SafetyPriorityDecision, 
        KneePointDecision
    )
except ImportError:
    # Fallback na starą strukturę, jeśli powyższa nie zadziała
    print("Importowanie plików nie powiodło się")

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
    Wersja w pełni wektoryzowana (bez pętli for).
    
    Args:
        waypoints: (Pop, Drones, N_In, 3)
        num_samples: N_Out
        
    Returns:
        (Pop, Drones, N_Out, 3)
    """
    xp = get_xp(waypoints)
    pop_size, n_drones, n_in, dims = waypoints.shape
    
    # 1. Spłaszczenie do (Batch, N_In, 3) dla wygody operacji
    batch_size = pop_size * n_drones
    flat_waypoints = waypoints.reshape(batch_size, n_in, dims)
    
    # 2. Obliczenie indeksów i wag (alfa) dla interpolacji liniowej
    # Tworzymy wirtualną oś czasu dla punktów wyjściowych: 0 .. N_In-1
    # Np. dla 100 punktów wyjściowych z 5 wejściowych, wartości będą: 0.0, 0.04, ..., 4.0
    u = xp.linspace(0, n_in - 1, num_samples)
    
    # Indeks punktu "po lewej" (floor)
    idx = xp.floor(u).astype(int)
    
    # Zabezpieczenie krawędzi: ostatni punkt (wartość N_In-1)
    # musi korzystać z segmentu [N_In-2, N_In-1], a nie szukać N_In.
    idx = xp.clip(idx, 0, n_in - 2)
    
    # Waga interpolacji (część ułamkowa): alpha = (t - t_left)
    # Dla ostatniego punktu wyjdzie 1.0 (czyli czysty punkt końcowy)
    alpha = u - idx
    
    # Rozszerzenie wymiarów alpha do (1, Num_Samples, 1) dla broadcastingu
    # (działa na wszystkie batche i wszystkie wymiary XYZ tak samo)
    alpha = alpha.reshape(1, num_samples, 1)
    
    # 3. Gather - Pobranie punktów P0 i P1 dla każdego kroku próbkowania
    # Fancy indexing: flat_waypoints[:, idx, :] automatycznie wybiera odpowiednie punkty
    # dla każdego elementu w batchu.
    # Wynik: (Batch, Num_Samples, 3)
    p0 = flat_waypoints[:, idx, :]     # Punkty początkowe segmentów
    p1 = flat_waypoints[:, idx + 1, :] # Punkty końcowe segmentów
    
    # 4. Interpolacja: P(t) = P0 + alpha * (P1 - P0)
    # (Batch, Num, 3) + (1, Num, 1) * (Batch, Num, 3) -> Broadcasting zadziała
    flat_out = p0 + alpha * (p1 - p0)
    
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
        n_inner_points: int,  
        n_output_samples: int, 
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
        
        print(f"[Problem Polyline] Zmienne: {n_var}, Granice X: {xl_one_point[0]:.1f} - {xu_one_point[0]:.1f}")
        
        # Powielamy granice
        xl = np.tile(xl_one_point, n_drones * n_inner_points)
        xu = np.tile(xu_one_point, n_drones * n_inner_points)
        
        # n_constr=5: Battery, Separation, ObstacleSafety, Uniformity, Smoothness
        super().__init__(n_var=n_var, n_obj=3, n_ieq_constr=5, xl=xl, xu=xu)


    def _evaluate(self, x: NDArray, out: Dict[str, Any], *args, **kwargs) -> None:
        # x shape: (Pop_Size, N_Var)
        pop_size = x.shape[0]
        
        # 1. Dekodowanie zmiennych (Inner Waypoints)
        inner_waypoints = x.reshape(pop_size, self.n_drones, self.n_inner_points, 3)
        
        # 2. Doklejanie Startu i Celu (Tworzenie pełnej łamanej)
        starts_bc = np.tile(self.starts[None, :, None, :], (pop_size, 1, 1, 1))
        targets_bc = np.tile(self.targets[None, :, None, :], (pop_size, 1, 1, 1))
        
        # Sparse Polyline: Start -> Inner -> Target
        sparse_trajectory = np.concatenate([starts_bc, inner_waypoints, targets_bc], axis=2)
        
        # 3. Resampling do gęstej trajektorii
        trajectories = resample_polyline_batch(sparse_trajectory, num_samples=self.n_output_samples)
            
        # 4. Ewaluacja
        self.evaluator.evaluate(trajectories, out)


# --- Sampling Heurystyczny ---

class HeuristicSampling(Sampling):
    """
    Inicjalizuje populację punktami leżącymi wokół prostej Start->Cel.
    Generuje punkty z dużym rozrzutem, aby zwiększyć szansę na ominięcie przeszkód.
    """
    def __init__(self, start_pos: NDArray, target_pos: NDArray, n_inner_points: int, n_drones: int):
        super().__init__()
        self.start = start_pos
        self.target = target_pos
        self.n_inner_points = n_inner_points
        self.n_drones = n_drones

    def _do(self, problem, n_samples, **kwargs):
        # 1. Baza: Punkty na prostej
        t_vals = np.linspace(0, 1, self.n_inner_points + 2)[1:-1]
        t = t_vals.reshape(1, 1, self.n_inner_points, 1)
        
        s = self.start[None, :, None, :]
        e = self.target[None, :, None, :]
        
        points = s + t * (e - s) # (1, Drones, Inner, 3)
        X = np.tile(points, (n_samples, 1, 1, 1))
        
        # 2. Szum (Agresywny)
        # Zamiast 0.2m, dajemy np. 20-50m rozrzutu w poziomie (X, Y)
        # W pionie (Z) mniej, żeby nie latały pod ziemią lub w kosmosie
        
        # Szum XY: sigma = 30m
        noise_xy = np.random.normal(0, 30.0, (n_samples, self.n_drones, self.n_inner_points, 2))
        
        # Szum Z: sigma = 5m (lekkie wahania wysokości)
        noise_z = np.random.normal(0, 5.0, (n_samples, self.n_drones, self.n_inner_points, 1))
        
        X[..., :2] += noise_xy
        X[..., 2] += noise_z
        
        # 3. Spłaszczanie i przycinanie do granic świata
        X_flat = X.reshape(n_samples, -1)
        
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
    number_of_waypoints: int, 
    drone_swarm_size: int, 
    algorithm_params: Optional[Dict[str, Any]] = None
) -> NDArray[np.float64]:
    
    # 1. Konfiguracja Parametrów
    params = algorithm_params or {}
    pop_size = params.get("pop_size", 100)
    n_gen = params.get("n_gen", 100)
    
    # Domyślna liczba punktów wewnętrznych (kontrolnych)
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
        n_output_samples=number_of_waypoints, 
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

    # 5. Wybór rozwiązania
    if res.X is not None and len(res.X) > 0:
        dm: DecisionStrategyProtocol
        try:
            if decision_mode == "safety": dm = SafetyPriorityDecision()
            elif decision_mode == "equal": dm = EqualWeightsDecision()
            else: dm = KneePointDecision()
            
            best_idx = dm.select_best(res.F, res.G)
            best_genotype_flat = res.X[best_idx]
            
            # Rekonstrukcja
            inner = best_genotype_flat.reshape(1, drone_swarm_size, n_inner, 3)
            s = start_positions[None, :, None, :]
            t = target_positions[None, :, None, :]
            sparse = np.concatenate([s, inner, t], axis=2)
            final_traj = resample_polyline_batch(sparse, num_samples=number_of_waypoints)
            
            print(f"[NSGA-III] Wybrano rozwiązanie indeks: {best_idx}")
            return final_traj[0]
        except Exception as e:
            print(f"[NSGA-III] Błąd podczas wyboru rozwiązania: {e}")
            # Fallback w razie błędu decydenta
            t = np.linspace(0, 1, number_of_waypoints)
            out = np.empty((drone_swarm_size, number_of_waypoints, 3))
            for d in range(drone_swarm_size):
                for i in range(3):
                    out[d, :, i] = np.interp(t, [0, 1], [start_positions[d, i], target_positions[d, i]])
            return out
        
    else:
        print("[NSGA-III] Brak rozwiązań. Zwracam linię prostą.")
        t = np.linspace(0, 1, number_of_waypoints)
        out = np.empty((drone_swarm_size, number_of_waypoints, 3))
        for d in range(drone_swarm_size):
            for i in range(3):
                out[d, :, i] = np.interp(t, [0, 1], [start_positions[d, i], target_positions[d, i]])
        return out
