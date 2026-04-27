"""
MSFOA Swarm Strategy.

Single-objective trajectory optimization using the Multiple Swarm Fruit Fly
Optimization Algorithm based strictly on the scientific paper:
Shi, K., Zhang, X., & Xia, S. (2020).

Pipeline:
1. The MSFFOAOptimizer explores the space of sparse inner waypoints (Polylines).
2. The TrajectorySOOAdapter aggregates multi-objective collisions linearly (O(N) complexity).
3. The final best polyline is smoothed using B-Spline post-processing via
   generate_bspline_batch.

REFACTORING NOTE (Rygor Naukowy i Porównywalność z NSGA-III):
- Inicjalizacja populacji: Wykorzystuje StraightLineNoiseSampling oraz dokładnie
  tę samą klasę SwarmOptimizationProblem co NSGA-III. Gwarantuje to identyczne
  wymiary, granice (clipping) oraz rozkład szumu początkowego.
- Pomiary czasu: Używa TimingCollector z identycznymi nazwami faz co NSGA-III.
- Historia optymalizacji: Zapisuje macierze decyzji/celów per generacja w formacie 
  identycznym do NSGA-III (OptimizationHistoryWriter).
- Wygładzanie: Zastąpiono dedykowany kod B-Spline wspólną funkcją generate_bspline_batch.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from hydra.core.hydra_config import HydraConfig

# Strategie MSFOA i Adapter SOO
from src.algorithms.abstraction.trajectory.strategies.nsga3_swarm_strategy import SwarmOptimizationProblem
from src.algorithms.abstraction.trajectory.strategies.core_msffoa import (
    MSFFOAOptimizer,
)
from src.algorithms.abstraction.trajectory.strategies.soo_adapter import (
    TrajectorySOOAdapter,
)

# Ewaluator i dane świata
from src.algorithms.abstraction.trajectory.objective_constrains import (
    VectorizedEvaluator,
)
from src.environments.abstraction.generate_world_boundaries import WorldData

# Współdzielone komponenty (wspólne z NSGA-III)
from src.algorithms.abstraction.trajectory.strategies.shared.bspline_utils import (
    generate_bspline_batch,
)
from src.algorithms.abstraction.trajectory.strategies.timing_utils import (
    TimingCollector,
)
from src.algorithms.abstraction.trajectory.strategies.shared.StraightLineNoiseSampling import (
    StraightLineNoiseSampling,
)
from src.utils.optimization_history_writer import OptimizationHistoryWriter

logger = logging.getLogger(__name__)


def msffoa_strategy(
    *,
    start_positions: NDArray[np.float64],
    target_positions: NDArray[np.float64],
    obstacles_data: Union[Any, List[Any]],
    world_data: WorldData,
    number_of_waypoints: int,
    drone_swarm_size: int,
    algorithm_params: Optional[Dict[str, Any]] = None,
    timing: Optional["TimingCollector"] = None,
) -> NDArray[np.float64]:
    """
    Trajectory generation via Multiple Swarm Fruit Fly Optimization Algorithm.
    (Shi, Zhang & Xia, 2020)
    """
    params = algorithm_params or {}

    # --- Konfiguracja pomiaru czasu (wzorzec z NSGA-III) ---
    local_timing = False
    if timing is None:
        try:
            timing = TimingCollector("MSFOA_Swarm")
            local_timing = True
        except Exception:
            timing = None

    from contextlib import nullcontext
    _measure = timing.measure if timing is not None else lambda *a, **kw: nullcontext()

    try:
        with _measure("total_optimization"):

            # 1. Konfiguracja parametrów
            pop_size: int          = params.get("pop_size", 200)
            max_generations: int   = params.get("epochs", 500)
            n_inner: int           = params.get("n_inner_waypoints", max(5, int(number_of_waypoints * 0.1)))
            seed: int              = params.get("seed", 42)

            w_list   = params.get("objective_weights", [0.05, 100.0, 0.1])
            weights  = np.array(w_list, dtype=np.float64)
            penalty_weight: float  = params.get("penalty_weight", 1.0)

            n_swarms:  int   = params.get("n_swarms", 5)
            coe1:      float = params.get("coe1", 0.8)
            coe2:      float = params.get("coe2", 0.2)

            if pop_size % n_swarms != 0:
                pop_size = (pop_size // n_swarms) * n_swarms
                logger.info(f"[MSFOA] Adjusted pop_size to {pop_size} to be divisible by n_swarms ({n_swarms}).")

            logger.info(
                f"[MSFOA B-Spline] Start. Pop: {pop_size}, Epochs: {max_generations}, "
                f"Control Pts: {n_inner}, Weights: {weights}, Penalty: {penalty_weight}"
            )

            try:
                output_dir = HydraConfig.get().runtime.output_dir
            except ValueError:
                output_dir = os.getcwd()

            writer = OptimizationHistoryWriter(
                output_dir=os.path.join(output_dir, "optimization_history")
            )

            # 2. Inicjalizacja środowiska i populacji
            with _measure("initialization"):
                evaluator = VectorizedEvaluator(
                    obstacles=obstacles_data,
                    start_pos=start_positions,
                    target_pos=target_positions,
                    n_inner_points=n_inner,
                    params=params,
                )

                adapter = TrajectorySOOAdapter(
                    evaluator=evaluator,
                    start_positions=start_positions,
                    target_positions=target_positions,
                    n_drones=drone_swarm_size,
                    n_inner=n_inner,
                    weights=weights,
                    penalty_weight=penalty_weight,
                )

                adapter._f_ref = np.maximum(adapter._f_ref, 1.0)

                logger.info(f"[MSFOA] F_ref (normalization scales): {adapter._f_ref}")

                # Dynamiczny próg (Shi et al., 2020)
                t_vals = np.linspace(0, 1, n_inner + 2)[1:-1]
                t_reshaped   = t_vals.reshape(1, 1, n_inner, 1)
                starts_bc    = start_positions[np.newaxis, :, np.newaxis, :]
                targets_bc   = target_positions[np.newaxis, :, np.newaxis, :]
                base_inner   = starts_bc + t_reshaped * (targets_bc - starts_bc)

                initial_fitness = float(adapter(base_inner)[0])
                threshold_ratio = float(params.get("threshold_ratio", 0.1))

                # Paper Sec. 1 podaje threshold jako parametr stały. Adaptujemy
                # do dynamicznej kalibracji `initial_fitness · threshold_ratio`,
                # bo bezwzględna wartość fitness zależy od F_ref normalization
                # i jest trudna do ustawienia z góry. Floor 0.01 chroni przed
                # zerowym progiem, gdy initial_fitness jest patologicznie niski
                # (np. perfect feasible straight-line z penalty=0).
                THRESHOLD_FLOOR = 0.01
                dynamic_threshold = max(initial_fitness * threshold_ratio, THRESHOLD_FLOOR)

                logger.info(f"[MSFOA] Initial Straight-Line Fitness: {initial_fitness:.4f}")
                logger.info(f"[MSFOA] Dynamic Threshold: {dynamic_threshold:.4f} (Ratio: {threshold_ratio})")

                # --- INSTANCJONOWANIE TEGO SAMEGO PROBLEMU CO NSGA-III ---
                # Gwarantuje to identyczne limity clippingu (xl, xu)
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

                # Generowanie i formatowanie zaszumionej linii z clippingiem Pymoo
                initial_population_flat = sampling._do(problem, pop_size)
                initial_population = initial_population_flat.reshape(
                    pop_size, drone_swarm_size, n_inner, 3
                )

                # Granice clippingu zgodne z NSGA-III i OOA: bierzemy je
                # bezpośrednio z SwarmOptimizationProblem, który buduje je z
                # marginesem świata (XY) oraz Z-range wyznaczonym przez
                # min/max endpoint_z. Dzięki temu wszystkie trzy optymalizatory
                # pracują na dokładnie tej samej przestrzeni poszukiwań.
                # problem.xl / problem.xu mają kształt (N_drones * N_inner * 3,),
                # a każda trójka jest identyczna — wystarczy pierwsza.
                xl_point = np.asarray(problem.xl[:3], dtype=np.float64)
                xu_point = np.asarray(problem.xu[:3], dtype=np.float64)

                # Konfigurowalne kroki perturbacji (paper Sec. 1 — R jako parametr).
                # YAML pozwala podać listę 3-elementową; przekazujemy None gdy
                # nie ma override'u, żeby konstruktor użył paper-tuned defaults.
                sg_frac = params.get("step_global_frac")
                sl_frac = params.get("step_local_frac")
                step_global_frac = (
                    np.asarray(sg_frac, dtype=np.float64) if sg_frac is not None else None
                )
                step_local_frac = (
                    np.asarray(sl_frac, dtype=np.float64) if sl_frac is not None else None
                )

                # Przekazanie wspólnej populacji inicjalnej
                optimizer = MSFFOAOptimizer(
                    pop_size=pop_size,
                    n_drones=drone_swarm_size,
                    n_inner=n_inner,
                    world_min_bounds=xl_point,
                    world_max_bounds=xu_point,
                    start_positions=start_positions,
                    target_positions=target_positions,
                    fitness_function=adapter,
                    max_generations=max_generations,
                    seed=seed,
                    n_swarms=n_swarms,
                    coe1=coe1,
                    coe2=coe2,
                    threshold=dynamic_threshold,
                    step_global_frac=step_global_frac,
                    step_local_frac=step_local_frac,
                    history_writer=writer,
                    initial_population=initial_population,
                )

            # 3. Optymalizacja
            with _measure("optimization"):
                try:
                    # MSFFOAOptimizer.optimize() zwraca parę (best_pos, best_fitness).
                    _, best_fitness = optimizer.optimize()
                finally:
                    writer.close()

            logger.info(f"[MSFOA] Optimization Finished. Best Polyline Fitness: {best_fitness:.4f}")

            # 4. Rekonstrukcja trajektorii i wygładzanie
            with _measure("decision_and_reconstruction"):
                sparse_trajectory = optimizer.get_best_dense_trajectory()
                control_points_batch = sparse_trajectory[np.newaxis, ...]
                
                logger.info("[MSFOA] Applying B-Spline Post-Processing...")
                final_dense_traj = generate_bspline_batch(
                    control_points_batch, num_samples=number_of_waypoints
                )
                
                return final_dense_traj[0]

    except Exception as e:
        logger.warning(f"[MSFOA] Optimization error: {e}. Returning straight-line fallback.")

    finally:
        if local_timing and timing is not None:
            try:
                out_dir = HydraConfig.get().runtime.output_dir
                timing.save_csv(os.path.join(out_dir, "optimization_timings.csv"))
            except Exception as e:
                logger.warning(f"[MSFOA] Nie udało się zapisać logów czasowych: {e}")

    # 5. Fallback w przypadku błędu
    with _measure("fallback", success=False):
        logger.warning("[MSFOA] Fallback: generating straight-line trajectory.")
        t_line = np.linspace(0, 1, number_of_waypoints)
        out    = np.empty((drone_swarm_size, number_of_waypoints, 3))
        min_safe_alt = params.get("min_safe_altitude", 1.0)

        for d in range(drone_swarm_size):
            for axis in range(2):
                out[d, :, axis] = np.interp(
                    t_line, [0, 1], [start_positions[d, axis], target_positions[d, axis]],
                )
            z_start = max(float(start_positions[d, 2]), min_safe_alt)
            z_target = max(float(target_positions[d, 2]), min_safe_alt)
            out[d, :, 2] = np.interp(t_line, [0, 1], [z_start, z_target])

        return out