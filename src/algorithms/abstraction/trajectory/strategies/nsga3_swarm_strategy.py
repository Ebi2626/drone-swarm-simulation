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
from pymoo.core.termination import Termination
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from src.algorithms.abstraction.trajectory.strategies.nsga3_utils.core_math import get_xp
from src.environments.obstacles.ObstacleShape import ObstacleShape

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


class MultiConditionTermination(Termination):
    def __init__(self, n_max_gen, min_feasible_needed):
        super().__init__()
        self.n_max_gen = n_max_gen
        self.min_feasible_needed = min_feasible_needed

    def _update(self, algorithm):
        # Pobieramy aktualną generację
        n_gen = algorithm.n_iter
        
        # Pobieramy populację i sprawdzamy, ile rozwiązań jest dopuszczalnych (feasible)
        # algorithm.pop.get("feasible") zwraca wektor boolowski (True tam, gdzie G <= 0)
        feasible_mask = algorithm.pop.get("feasible")
        n_feasible = np.sum(feasible_mask) if feasible_mask is not None else 0

        # Obliczamy stopień postępu (0.0 do 1.0)
        # Terminacja następuje, gdy progress >= 1.0
        gen_progress = n_gen / self.n_max_gen
        feasible_progress = n_feasible / self.min_feasible_needed if self.min_feasible_needed > 0 else 0
        
        # Zwracamy maksymalny postęp - algorytm zakończy się, gdy dowolny warunek zostanie spełniony
        return max(gen_progress, feasible_progress)


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
        world_data: WorldData,
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
        # Kopiujemy i rzutujemy na float, aby nie nadpisać oryginalnych danych
        xl_one_point = np.array(world_data.min_bounds, dtype=float) - margin
        xu_one_point = np.array(world_data.max_bounds, dtype=float) + margin
        
        # [POPRAWKA] Twarda podłoga dla osi Z. Bezpieczna minimalna wysokość (np. 0.5m)
        MIN_Z_ALTITUDE = 0.5 
        xl_one_point[2] = max(MIN_Z_ALTITUDE, world_data.min_bounds[2])
        
        print(f"[Problem Polyline] Zmienne: {n_var}, Granice X: {xl_one_point[0]:.1f} - {xu_one_point[0]:.1f}")
        print(f"[Problem Polyline] Granice Z: {xl_one_point[2]:.1f} - {xu_one_point[2]:.1f}")
        
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
    def __init__(
            self, 
            start_pos: NDArray, 
            target_pos: NDArray, 
            n_inner_points: int, 
            n_drones: int,
            world_data: WorldData,
            obstacles_data: ObstaclesData
            ):
        super().__init__()
        self.start = start_pos
        self.target = target_pos
        self.n_inner_points = n_inner_points
        self.n_drones = n_drones
        self.world_data = world_data
        self.obstacles_data = obstacles_data

    def _do(self, problem, n_samples, **kwargs):
        # 1. Baza: Punkty na prostej
        t_vals = np.linspace(0, 1, self.n_inner_points + 2)[1:-1]
        t = t_vals.reshape(1, 1, self.n_inner_points, 1)
        
        s = self.start[None, :, None, :]
        e = self.target[None, :, None, :]
        
        points = s + t * (e - s) # (1, Drones, Inner, 3)
        X = np.tile(points, (n_samples, 1, 1, 1))
        
        # 2. Szum (Agresywny w poziomie (wymiar przeszkody), bezpieczny w pionie)
        if self.obstacles_data is not None and self.obstacles_data.data is not None and len(self.obstacles_data.data) > 0 and self.obstacles_data.data[0] is not None:
            if self.obstacles_data.shape_type == ObstacleShape.BOX:
                horizontal_dims = self.obstacles_data.data[:, 3:5]
            elif self.obstacles_data.shape_type == ObstacleShape.CYLINDER:
                horizontal_dims = self.obstacles_data.data[:, 3:4]
            max_horizontal_size = np.max(horizontal_dims)
            noise_xy = np.random.normal(0, max_horizontal_size, (n_samples, self.n_drones, self.n_inner_points, 2))
            X[..., :2] += noise_xy
        
        # Przestrzeń warstwowa zamiast Gaussa z obcinaniem.
        # Drony dostają losową wysokość w bezpiecznym korytarzu powietrznym.
        min_safe_z = 0.5    # Bezpieczna "podłoga"
        max_flight_z = self.world_data.max_bounds[2] # Maksymalny pułap operacyjny dla optymalizacji
        
        # Zabezpieczenie przed przekroczeniem górnych limitów mapy/świata
        if problem.xu is not None:
            # Zakładamy, że indeks 2 to oś Z w spłaszczonej tablicy xu
            max_flight_z = min(max_flight_z, problem.xu[2] - 3.0)
            
        print("Lecimy poniżej: ", max_flight_z)
        # Losujemy wartości równomiernie, co w 100% zachowuje różnorodność genetyczną
        random_z = np.random.uniform(min_safe_z, max_flight_z, (n_samples, self.n_drones, self.n_inner_points))
        X[..., 2] = random_z
        
        # 3. Spłaszczanie i przycinanie do granic świata (XY)
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
    eta_c = params.get("eta_c")
    eta_m = params.get("eta_m")
    crossover_prob = params.get("crossover_prob", 0.9)
    mutation_prob = params.get("mutation_prob", 0.1)
    min_ideal_solutions = params.get("min_ideal_solutions", 30)
    
    # Domyślna liczba punktów wewnętrznych (kontrolnych)
    n_inner = params.get("n_inner_waypoints", max(5, int(number_of_waypoints * 0.1)))
    
    decision_mode = params.get("decision_mode", "knee_point")

    termination = MultiConditionTermination(
        n_max_gen=n_gen, 
        min_feasible_needed=min_ideal_solutions
    )
    
    if isinstance(obstacles_data, list):
        obs_list = obstacles_data
    else:
        obs_list = [obstacles_data]
    
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
        world_data=world_data,
        evaluator=evaluator,
        start_pos=start_positions,
        target_pos=target_positions
    )
    
    sampling = HeuristicSampling(
        start_pos=start_positions,
        target_pos=target_positions,
        n_inner_points=n_inner,
        n_drones=drone_swarm_size,
        world_data=world_data,
        obstacles_data=obstacles_data
    )
    
    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=sampling,
        crossover=SBX(prob=crossover_prob, eta=eta_c),
        mutation=PM(eta=eta_m, prob=mutation_prob),
        eliminate_duplicates=True
    )
    
    # 4. Optymalizacja
    res = minimize(
        problem,
        algorithm,
        termination=termination,
        seed=1,
        verbose=True,
        save_history=True
    )

    # 5. Wybór rozwiązania
    if res.X is not None and len(res.X) > 0:
        dm: DecisionStrategyProtocol
        try:
            if decision_mode == "safety": 
                dm = SafetyPriorityDecision()
            elif decision_mode == "equal": 
                dm = EqualWeightsDecision()
            else: 
                dm = KneePointDecision()
            
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
            raise LookupError("Results not found")
        
    else:
        print("[NSGA-III] Brak rozwiązań. Zwracam linię prostą.")
        t = np.linspace(0, 1, number_of_waypoints)
        out = np.empty((drone_swarm_size, number_of_waypoints, 3))

        MIN_SAFE_ALTITUDE = params.get("min_safe_altitude", 1.0)  # metry nad ziemią
        print("Target positions: ")

        for d in range(drone_swarm_size):
            # Osie X i Y: zwykła interpolacja liniowa
            for i in range(2):
                out[d, :, i] = np.interp(t, [0, 1], [start_positions[d, i], target_positions[d, i]])
                print(target_positions[d, i])
            
            # Oś Z: interpolacja z gwarantowaną minimalną wysokością
            z_start  = max(start_positions[d, 2],  MIN_SAFE_ALTITUDE)
            z_target = max(target_positions[d, 2], MIN_SAFE_ALTITUDE)
            out[d, :, 2] = np.interp(t, [0, 1], [z_start, z_target])

        return out
