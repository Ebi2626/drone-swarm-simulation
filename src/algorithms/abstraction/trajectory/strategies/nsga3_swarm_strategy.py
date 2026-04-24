"""
Swarm Evolution Strategy Orchestrator.
Główny punkt wejścia dla strategii NSGA-III.
Wersja B-Spline Control Polygon - optymalizator ewaluuje węzły kontrolne krzywej,
co gwarantuje 100% bezpieczeństwo wygładzonej trajektorii.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

# Pymoo imports
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.termination import Termination
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.algorithms.abstraction.trajectory.strategies.shared.StraightLineNoiseSampling import StraightLineNoiseSampling
from src.algorithms.abstraction.trajectory.strategies.shared.bspline_utils import generate_bspline_batch
from src.algorithms.abstraction.trajectory.strategies.timing_utils import TimingCollector
from src.utils.optimization_history_writer import OptimizationHistoryWriter

# Komponenty wewnętrzne
from src.algorithms.abstraction.trajectory.objective_constrains import VectorizedEvaluator
from src.algorithms.abstraction.trajectory.strategies.nsga3_utils.decision_maker import (
    DecisionStrategyProtocol, 
    EqualWeightsDecision, 
    SafetyPriorityDecision, 
    KneePointDecision
)

from src.environments.abstraction.generate_world_boundaries import WorldData


class MultiConditionTermination(Termination):
    def __init__(self, n_max_gen, min_feasible_needed):
        super().__init__()
        self.n_max_gen = n_max_gen
        self.min_feasible_needed = min_feasible_needed

    def _update(self, algorithm):
        n_gen = algorithm.n_iter
        feasible_mask = algorithm.pop.get("feasible")
        n_feasible = np.sum(feasible_mask) if feasible_mask is not None else 0

        gen_progress = n_gen / self.n_max_gen
        feasible_progress = n_feasible / self.min_feasible_needed if self.min_feasible_needed > 0 else 0
        
        return max(gen_progress, feasible_progress)


# --- Klasa Problemu Pymoo ---

class SwarmOptimizationProblem(Problem):
    """
    Definicja problemu optymalizacyjnego (Wersja B-Spline Control Polygon).
    Zmienne decyzyjne to bezpośrednio węzły kontrolne krzywej B-Spline.
    """
    def __init__(
        self,
        n_drones: int,
        n_inner_points: int,
        world_data: WorldData,
        evaluator: VectorizedEvaluator,
        start_pos: NDArray,
        target_pos: NDArray,
        min_altitude: float | None = None,
        max_altitude: float | None = None,
    ):
        self.n_drones = n_drones
        self.n_inner_points = n_inner_points
        self.evaluator = evaluator

        # Przechowujemy Start i Cel do doklejenia w _evaluate
        self.starts = start_pos
        self.targets = target_pos

        # Liczba zmiennych decyzyjnych: N_Drones * Inner_Points * 3 (XYZ)
        n_var = n_drones * n_inner_points * 3

        # Rozszerzamy granice o margines
        margin = 10.0  # Zmniejszony margines, by punkty kontrolne nie uciekały w nieskończoność
        xl_one_point = np.array(world_data.min_bounds, dtype=float) - margin
        xu_one_point = np.array(world_data.max_bounds, dtype=float) + margin

        # Dolny/górny limit Z spójny z endpointami:
        #   - nie chcemy wymusić Inner_Z > Start_Z (to tworzy niepotrzebny bump),
        #   - ale nie pozwalamy też schodzić poniżej ziemi ± margines.
        min_endpoint_z = float(min(start_pos[:, 2].min(), target_pos[:, 2].min()))
        max_endpoint_z = float(max(start_pos[:, 2].max(), target_pos[:, 2].max()))
        ground_z = float(world_data.min_bounds[2])
        ceiling_z = float(world_data.max_bounds[2])

        default_min_z = max(ground_z + 0.5, min_endpoint_z - 0.2)
        default_max_z = min(ceiling_z - 0.5, max_endpoint_z + 20.0)
        eff_min_z = float(min_altitude) if min_altitude is not None else default_min_z
        eff_max_z = float(max_altitude) if max_altitude is not None else default_max_z
        xl_one_point[2] = eff_min_z
        xu_one_point[2] = eff_max_z

        print(f"[Problem B-Spline] Zmienne: {n_var}, Granice X: {xl_one_point[0]:.1f} - {xu_one_point[0]:.1f}")
        print(f"[Problem B-Spline] Granice Z: {xl_one_point[2]:.2f} - {xu_one_point[2]:.2f} "
              f"(endpoint Z range: [{min_endpoint_z:.2f}, {max_endpoint_z:.2f}])")

        # Powielamy granice
        xl = np.tile(xl_one_point, n_drones * n_inner_points)
        xu = np.tile(xu_one_point, n_drones * n_inner_points)

        super().__init__(n_var=n_var, n_obj=3, n_ieq_constr=3, xl=xl, xu=xu)

    def _evaluate(self, x: NDArray, out: Dict[str, Any], *args, **kwargs) -> None:
        pop_size = x.shape[0]
        
        # 1. Dekodowanie zmiennych (Inner Control Points)
        inner_waypoints = x.reshape(pop_size, self.n_drones, self.n_inner_points, 3)
        
        # 2. Doklejanie Startu i Celu (Tworzenie pełnego wieloboku kontrolnego)
        starts_bc = np.tile(self.starts[None, :, None, :], (pop_size, 1, 1, 1))
        targets_bc = np.tile(self.targets[None, :, None, :], (pop_size, 1, 1, 1))
        
        # Control Polygon: Start -> Inner Nodes -> Target
        control_polygon = np.concatenate([starts_bc, inner_waypoints, targets_bc], axis=2)

        # 3. Ewaluacja bezpośrednio na węzłach kontrolnych (Broad-Phase Collision)
        self.evaluator.evaluate(control_polygon, out)


# --- Helper: Dynamiczne partycje ---

class _HistoryCallback(Callback):
    """Pymoo callback — loguje stan populacji po każdej generacji."""

    def __init__(self, writer: OptimizationHistoryWriter):
        super().__init__()
        self._writer = writer

    def notify(self, algorithm):
        pop = algorithm.pop
        objectives = pop.get("F")
        decisions = pop.get("X")
        if objectives is not None and decisions is not None:
            self._writer.put_generation_data({
                "objectives_matrix": objectives.copy(),
                "decisions_matrix": decisions.copy(),
            })

def calculate_n_partitions(pop_size: int, n_obj: int = 3) -> int:
    if n_obj != 3: 
        return 12 
    p = int(np.sqrt(2 * pop_size))
    return max(2, p - 1)


# --- Główna Funkcja Strategii ---

def nsga3_swarm_strategy(
    *, 
    start_positions: NDArray[np.float64],
    target_positions: NDArray[np.float64],
    obstacles_data: ObstaclesData, 
    world_data: WorldData, 
    number_of_waypoints: int, 
    drone_swarm_size: int, 
    algorithm_params: Optional[Dict[str, Any]] = None,
    timing: Optional["TimingCollector"] = None 
) -> NDArray[np.float64]:
    
    # --- Konfiguracja pomiaru czasu ---
    local_timing = False
    if timing is None:
        try:
            from src.algorithms.abstraction.trajectory.strategies.timing_utils import TimingCollector
            timing = TimingCollector("NSGA3_Swarm")
            local_timing = True
        except ImportError:
            timing = None

    from contextlib import nullcontext
    _measure = timing.measure if timing is not None else lambda *a, **kw: nullcontext()

    try:
        with _measure("total_optimization"):
            
            # 1. Konfiguracja Parametrów
            params = algorithm_params or {}
            pop_size = params.get("pop_size", 100)
            n_gen = params.get("n_gen", 100)
            eta_c = params.get("eta_c", 15)
            eta_m = params.get("eta_m", 20)
            crossover_prob = params.get("crossover_prob", 0.9)
            mutation_prob = params.get("mutation_prob", 0.1)
            min_ideal_solutions = params.get("min_ideal_solutions", 30)
            
            n_inner = params.get("n_inner_waypoints", max(5, int(number_of_waypoints * 0.1)))
            decision_mode = params.get("decision_mode", "knee_point")

            termination = MultiConditionTermination(
                n_max_gen=n_gen, 
                min_feasible_needed=min_ideal_solutions
            )
                        
            # 2. NSGA-III Setup
            n_partitions = calculate_n_partitions(pop_size, n_obj=3)
            ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
            actual_pop_size = ref_dirs.shape[0]
            
            from hydra.core.hydra_config import HydraConfig
            output_dir = HydraConfig.get().runtime.output_dir
            writer = OptimizationHistoryWriter(
                output_dir=os.path.join(output_dir, "optimization_history")
            )

            print(f"[NSGA-III B-Spline] Start. Pop: {pop_size} (Ref: {actual_pop_size}), Gen: {n_gen}, Control Pts: {n_inner}")

            # 3. Inicjalizacja
            with _measure("initialization"):
                evaluator = VectorizedEvaluator(
                    obstacles=obstacles_data,
                    start_pos=start_positions,
                    target_pos=target_positions,
                    n_inner_points=n_inner,
                    params=params
                )

                problem = SwarmOptimizationProblem(
                    n_drones=drone_swarm_size,
                    n_inner_points=n_inner,
                    world_data=world_data,
                    evaluator=evaluator,
                    start_pos=start_positions,
                    target_pos=target_positions,
                    min_altitude=params.get("min_altitude"),
                    max_altitude=params.get("max_altitude"),
                )

                # Szum anizotropowy — mniej w Z, bo zakres Z (start↔cel) jest zwykle
                # rzędu pojedynczych metrów i izotropowy σ=2m zepchnąłby całą
                # populację początkową na dolny limit Z po klipowaniu.
                noise_std_xy = float(params.get("noise_std_xy", 2.0))
                noise_std_z = float(params.get("noise_std_z", 0.3))

                sampling = StraightLineNoiseSampling(
                    start_pos=start_positions,
                    target_pos=target_positions,
                    n_inner_points=n_inner,
                    n_drones=drone_swarm_size,
                    noise_std=noise_std_xy,
                    noise_std_z=noise_std_z,
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
            with _measure("optimization"):
                callback = _HistoryCallback(writer)
                try:
                    res = minimize(
                        problem,
                        algorithm,
                        termination=termination,
                        seed=1,
                        verbose=True,
                        save_history=True,
                        callback=callback,
                    )
                except Exception:
                    raise
                finally:
                    writer.close()

            # 5. Wybór rozwiązania i wygładzanie
            if res.X is not None and len(res.X) > 0:
                with _measure("decision_and_reconstruction"):
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
                        
                        # Rekonstrukcja wieloboku kontrolnego
                        inner = best_genotype_flat.reshape(1, drone_swarm_size, n_inner, 3)
                        s = start_positions[None, :, None, :]
                        t = target_positions[None, :, None, :]
                        best_control_points = np.concatenate([s, inner, t], axis=2)
                        # --- B-SPLINE RECONSTRUCTION ---
                        # Przekształcenie węzłów kontrolnych w ciągłą i gładką trajektorię
                        final_traj = generate_bspline_batch(best_control_points, num_samples=number_of_waypoints)
                        
                        print(f"[NSGA-III] Wybrano rozwiązanie indeks: {best_idx}")
                        return final_traj[0]
                    except Exception as e:
                        print(f"[NSGA-III] Błąd podczas wyboru rozwiązania: {e}")
                        raise LookupError("Results not found")
                
            else:
                with _measure("fallback"):
                    print("[NSGA-III] Brak rozwiązań. Zwracam linię prostą.")
                    t_vals = np.linspace(0, 1, number_of_waypoints)
                    out = np.empty((drone_swarm_size, number_of_waypoints, 3))

                    MIN_SAFE_ALTITUDE = params.get("min_safe_altitude", 1.0)
                    for d in range(drone_swarm_size):
                        for i in range(2):
                            out[d, :, i] = np.interp(t_vals, [0, 1], [start_positions[d, i], target_positions[d, i]])
                        
                        z_start  = max(start_positions[d, 2],  MIN_SAFE_ALTITUDE)
                        z_target = max(target_positions[d, 2], MIN_SAFE_ALTITUDE)
                        out[d, :, 2] = np.interp(t_vals, [0, 1], [z_start, z_target])

                    return out

    finally:
        if local_timing and timing is not None:
            try:
                from hydra.core.hydra_config import HydraConfig
                out_dir = HydraConfig.get().runtime.output_dir
                timing.save_csv(os.path.join(out_dir, "optimization_timings.csv"))
            except Exception as e:
                print(f"[NSGA-III] Nie udało się zapisać logów czasowych: {e}")