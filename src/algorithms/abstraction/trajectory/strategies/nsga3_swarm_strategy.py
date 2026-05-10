"""
NSGA-III strategy entry point. Wersja B-Spline Control Polygon — optymalizator
operuje na węzłach kontrolnych krzywej (nie na próbkach po wygładzeniu), więc
ograniczenia kolizji liczone na wieloboku kontrolnym są zachowawcze: krzywa
B-Spline leży w convex hull swojego control polygon (de Boor 1978), zatem
feasibility punktów kontrolnych implikuje feasibility krzywej.
"""

import logging
import math
import os
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.algorithms.abstraction.trajectory.strategies.shared.StraightLineNoiseSampling import StraightLineNoiseSampling
from src.algorithms.abstraction.trajectory.strategies.shared.bspline_utils import generate_bspline_batch
from src.algorithms.abstraction.trajectory.strategies.timing_utils import TimingCollector
from src.utils.optimization_history_writer import OptimizationHistoryWriter
from src.algorithms.abstraction.trajectory.objective_constrains import VectorizedEvaluator
from src.algorithms.abstraction.trajectory.strategies.nsga3_utils.decision_maker import (
    DecisionStrategyProtocol, 
    EqualWeightsDecision, 
    SafetyPriorityDecision, 
    KneePointDecision
)

from src.environments.abstraction.generate_world_boundaries import WorldData
from src.utils.SeedRegistry import SeedRegistry

logger = logging.getLogger(__name__)


class SwarmOptimizationProblem(Problem):
    """Pymoo problem: zmienne decyzyjne = węzły kontrolne B-Spline (XYZ per drone
    per waypoint). 5 obiektywów, 3 ieq constraints — definicje w
    :class:`VectorizedEvaluator`.
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

        self.starts = start_pos
        self.targets = target_pos

        n_var = n_drones * n_inner_points * 3

        margin = 10.0
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

        logger.info(f"[Problem B-Spline] Zmienne: {n_var}, Granice X: {xl_one_point[0]:.1f} - {xu_one_point[0]:.1f}")
        logger.info(f"[Problem B-Spline] Granice Z: {xl_one_point[2]:.2f} - {xu_one_point[2]:.2f} "
              f"(endpoint Z range: [{min_endpoint_z:.2f}, {max_endpoint_z:.2f}])")

        xl = np.tile(xl_one_point, n_drones * n_inner_points)
        xu = np.tile(xu_one_point, n_drones * n_inner_points)

        super().__init__(n_var=n_var, n_obj=5, n_ieq_constr=3, xl=xl, xu=xu)

    def _evaluate(self, x: NDArray, out: Dict[str, Any], *args, **kwargs) -> None:
        pop_size = x.shape[0]

        inner_waypoints = x.reshape(pop_size, self.n_drones, self.n_inner_points, 3)

        starts_bc = np.tile(self.starts[None, :, None, :], (pop_size, 1, 1, 1))
        targets_bc = np.tile(self.targets[None, :, None, :], (pop_size, 1, 1, 1))

        control_polygon = np.concatenate([starts_bc, inner_waypoints, targets_bc], axis=2)

        self.evaluator.evaluate(control_polygon, out)


class _HistoryCallback(Callback):
    """Pymoo callback — loguje stan populacji po każdej generacji.

    Wysyła do `OptimizationHistoryWriter` pełny zestaw per-gen metryk
    (objectives, decisions, feasibility, CV, elapsed_s, eval_count) zgodny
    z konwencją używaną przez SOO strategie (SSA/OOA/MSFFOA) — jednolite
    dane wejściowe dla ETL `_load_optimization_history`.
    """

    def __init__(self, writer: OptimizationHistoryWriter,
                 problem: Optional["SwarmOptimizationProblem"] = None) -> None:
        super().__init__()
        import time
        self._writer = writer
        self._problem = problem
        # Initialize at construction so first `notify` zwraca prawdziwy
        # czas pierwszej generacji (od zbudowania callbacka do końca gen 1)
        # zamiast 0. Inaczej `total_elapsed = sum(elapsed_s)` zaniża czas
        # o jeden gen — krytyczne dla małych n_gen.
        self._gen_t0: float = time.monotonic()

    def notify(self, algorithm):
        import time

        pop = algorithm.pop
        objectives = pop.get("F")
        decisions = pop.get("X")
        if objectives is None or decisions is None:
            return

        # G z populacji pymoo. Nasz `SwarmOptimizationProblem(n_ieq_constr=3)`
        # zawsze ma "G" w Population — `pop.get("G")` zwraca array (n_pop, 3).
        # Gdyby problem był bez ograniczeń, pymoo nie ma "G" w Population
        # i `.get` zwraca array Noneów; obsługa fallbackiem przez try/except.
        try:
            constraints = pop.get("G")
            if constraints is None or (
                hasattr(constraints, "dtype") and constraints.dtype == object
            ):
                constraints = None
        except Exception:
            constraints = None

        # Wallclock per gen — `_gen_t0` zainicjalizowane w `__init__`
        # gwarantuje że pierwsza generacja ma niezerowy elapsed_s.
        now = time.monotonic()
        elapsed = now - self._gen_t0
        self._gen_t0 = now

        # NFE: jeśli mamy referencję do problemu z evaluatorem,
        # użyj `evaluator.individuals_evaluated` (true NFE).
        # Fallback: `algorithm.evaluator.n_eval` (pymoo native).
        n_eval = 0
        if self._problem is not None and hasattr(self._problem, "evaluator"):
            ev = self._problem.evaluator
            if hasattr(ev, "individuals_evaluated"):
                n_eval = int(ev.individuals_evaluated)
        if n_eval == 0:
            n_eval = int(getattr(getattr(algorithm, "evaluator", None), "n_eval", 0))

        from src.utils.per_gen_metrics import per_gen_metrics_from_FG

        self._writer.put_generation_data(
            per_gen_metrics_from_FG(
                objectives=objectives.copy(),
                constraints=constraints.copy() if constraints is not None else None,
                decisions=decisions.copy(),
                elapsed_s=elapsed,
                eval_count_cumulative=n_eval,
            )
        )
def calculate_n_partitions(pop_size: int, n_obj: int) -> int:
    """
    Wylicza optymalną liczbę partycji (p) dla metody Das-Dennis,
    aby liczba punktów referencyjnych (H) była jak najbliższa pop_size.
    """
    best_p = 1
    min_diff = float('inf')
    
    for p in range(1, 50):
        # Wzór na liczbę punktów referencyjnych
        H = math.comb(n_obj + p - 1, p)
        
        diff = abs(H - pop_size)
        if diff < min_diff:
            min_diff = diff
            best_p = p
        elif H > pop_size:
            # Ponieważ H rośnie monotonicznie wraz z p, 
            # jeśli przekroczyliśmy pop_size i różnica rośnie, to mamy optimum.
            break
            
    return max(1, best_p)


def nsga3_swarm_strategy(
    *, 
    start_positions: NDArray[np.float64],
    target_positions: NDArray[np.float64],
    obstacles_data: ObstaclesData, 
    world_data: WorldData, 
    number_of_waypoints: int, 
    drone_swarm_size: int, 
    algorithm_params: Optional[Dict[str, Any]] = None,
    timing: Optional["TimingCollector"] = None,
    seeds: SeedRegistry = None,
) -> NDArray[np.float64]:


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
            params = algorithm_params or {}
            pop_size = params.get("pop_size", 100)
            n_gen = params.get("n_gen", 100)
            eta_c = params.get("eta_c", 15)
            eta_m = params.get("eta_m", 20)
            crossover_prob = params.get("crossover_prob", 0.9)
            mutation_prob = params.get("mutation_prob", 0.1)

            n_inner = params.get("n_inner_waypoints", max(5, int(number_of_waypoints * 0.1)))
            decision_mode = params.get("decision_mode", "knee_point")

            # Jednolita pula obliczeniowa: NSGA-III wykonuje dokładnie `n_gen`
            # generacji niezależnie od liczby feasible solutions, tak jak mealpy
            # SSA/OOA i MSFFOA wykonują dokładnie `epoch`/`max_generations`.
            # Dawniej `MultiConditionTermination` przerywała wcześniej, gdy
            # znaleziono `min_ideal_solutions` feasible — to powodowało
            # niesprawiedliwe porównanie metaheurystyk i `res.X is None`
            # w pymoo (patrz plan.md, Krok 3).
            termination = get_termination("n_gen", n_gen)

            n_partitions = calculate_n_partitions(pop_size, n_obj=5)
            ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=n_partitions)
            actual_pop_size = ref_dirs.shape[0]
            
            from hydra.core.hydra_config import HydraConfig
            output_dir = HydraConfig.get().runtime.output_dir
            writer = OptimizationHistoryWriter(
                output_dir=os.path.join(output_dir, "optimization_history")
            )

            logger.info(f"[NSGA-III B-Spline] Start. Pop: {pop_size} (Ref: {actual_pop_size}), Gen: {n_gen}, Control Pts: {n_inner}")

            with _measure("initialization"):
                evaluator = VectorizedEvaluator(
                    obstacles=obstacles_data,
                    start_pos=start_positions,
                    target_pos=target_positions,
                    n_inner_points=n_inner,
                    params=params,
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
                    rng=seeds.rng("sampling")
                )
                
                algorithm = NSGA3(
                    pop_size=pop_size,
                    ref_dirs=ref_dirs,
                    sampling=sampling,
                    crossover=SBX(prob=crossover_prob, eta=eta_c),
                    mutation=PM(eta=eta_m, prob=mutation_prob),
                    eliminate_duplicates=True
                )

            with _measure("optimization"):
                callback = _HistoryCallback(writer, problem=problem)
                try:
                    res = minimize(
                        problem,
                        algorithm,
                        termination=termination,
                        seed=seeds.seed("optimizer"),
                        verbose=True,
                        save_history=True,
                        callback=callback,
                    )
                except Exception:
                    raise
                finally:
                    writer.close()

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

                        inner = best_genotype_flat.reshape(1, drone_swarm_size, n_inner, 3)
                        s = start_positions[None, :, None, :]
                        t = target_positions[None, :, None, :]
                        best_control_points = np.concatenate([s, inner, t], axis=2)
                        final_traj = generate_bspline_batch(best_control_points, num_samples=number_of_waypoints)
                        
                        logger.info(f"[NSGA-III] Wybrano rozwiązanie indeks: {best_idx}")
                        return final_traj[0]
                    except Exception as e:
                        logger.warning(f"[NSGA-III] Błąd podczas wyboru rozwiązania: {e}")
                        raise LookupError("Results not found")
                
            else:
                with _measure("fallback", success=False):
                    logger.warning("[NSGA-III] Brak rozwiązań. Zwracam linię prostą.")
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
                logger.warning(f"[NSGA-III] Nie udało się zapisać logów czasowych: {e}")