"""
MSFOA Swarm Strategy.

Single-objective trajectory optimization using the Multiple Swarm Fruit Fly
Optimization Algorithm (Shi, Zhang & Xia, 2020).

Ceteris paribus z NSGA-III: ta sama ``SwarmOptimizationProblem`` (clipping
bounds), ten sam ``StraightLineNoiseSampling`` (initial population), ten sam
``OptimizationHistoryWriter`` (per-gen logging), ta sama
``generate_bspline_batch`` (post-processing). Co się różni: scalar fitness
przez ``TrajectorySOOAdapter`` zamiast Pareto frontu.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from hydra.core.hydra_config import HydraConfig

from src.algorithms.abstraction.trajectory.strategies.nsga3_swarm_strategy import SwarmOptimizationProblem
from src.algorithms.abstraction.trajectory.strategies.core_msffoa import (
    MSFFOAOptimizer,
)
from src.algorithms.abstraction.trajectory.strategies.soo_adapter import (
    HARD_INFEASIBLE_BASE,
    TrajectorySOOAdapter,
)
from src.algorithms.abstraction.trajectory.objective_constrains import (
    VectorizedEvaluator,
)
from src.environments.abstraction.generate_world_boundaries import WorldData
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
from src.utils.SeedRegistry import SeedRegistry

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
    seeds: SeedRegistry = None,

) -> NDArray[np.float64]:
    """Wygeneruj trajektorię roju algorytmem MSFOA (Shi, Zhang & Xia 2020).

    Implementacja `TrajectoryStrategyProtocol` — patrz `count_trajectories.py`.
    Skalaryzuje 5 obiektywów `VectorizedEvaluator` przez `TrajectorySOOAdapter`
    z domyślnymi wagami z `algorithm_params['objective_weights']`.

    Args:
        start_positions: `(N, 3)` pozycje startowe dronów [m].
        target_positions: `(N, 3)` pozycje docelowe dronów [m].
        obstacles_data: Geometria przeszkód statycznych.
        world_data: Granice świata symulacji.
        number_of_waypoints: Docelowa liczba punktów `W` w trajektorii
            (po post-processingu B-spline).
        drone_swarm_size: Rozmiar roju `N`.
        algorithm_params: Hiperparametry MSFOA — `pop_size`, `epochs`,
            `n_inner_waypoints`, `objective_weights`, `penalty_weight`,
            `n_swarms`, `coe1`, `coe2`, `noise_std_xy`, `noise_std_z`,
            `threshold_ratio`. `None` ⇒ wartości domyślne.
        timing: Opcjonalny `TimingCollector` dla pomiaru faz; `None` ⇒
            tworzony lokalnie i zapisywany na końcu do CSV.
        seeds: `SeedRegistry` z subseedami `sampling` i `optimizer`.

    Returns:
        `(N, W, 3)` trajektoria po wygładzeniu B-spline. W razie błędu
        zwraca trajektorię awaryjną — linię prostą z minimalną wysokością.
    """
    params = algorithm_params or {}

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
            pop_size: int          = params.get("pop_size", 200)
            max_generations: int   = params.get("epochs", 500)
            n_inner: int           = params.get("n_inner_waypoints", max(5, int(number_of_waypoints * 0.1)))

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

            with _measure("initialization"):
                evaluator = VectorizedEvaluator(
                    obstacles=obstacles_data,
                    start_pos=start_positions,
                    target_pos=target_positions,
                    n_inner_points=n_inner,
                    params=params
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

                # Anchor wstępny dla threshold (Shi et al., 2020 Eq. 5-8).
                # Faktyczna kalibracja jest odroczona do momentu, gdy znamy
                # initial_population — pozwala to wykryć Big-M domination i użyć
                # mediany feasible jako anchora zamiast magnitudy violation.
                t_vals = np.linspace(0, 1, n_inner + 2)[1:-1]
                t_reshaped   = t_vals.reshape(1, 1, n_inner, 1)
                starts_bc    = start_positions[np.newaxis, :, np.newaxis, :]
                targets_bc   = target_positions[np.newaxis, :, np.newaxis, :]
                base_inner   = starts_bc + t_reshaped * (targets_bc - starts_bc)

                initial_fitness = float(adapter(base_inner)[0])
                threshold_ratio = float(params.get("threshold_ratio", 0.1))
                logger.info(f"[MSFOA] Initial Straight-Line Fitness: {initial_fitness:.4f}")

                # Identyczna definicja problemu co NSGA-III gwarantuje te same
                # limity clippingu (xl, xu) — ceteris paribus.
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
                    rng=seeds.rng("sampling")
                )

                initial_population_flat = sampling._do(problem, pop_size)
                initial_population = initial_population_flat.reshape(
                    pop_size, drone_swarm_size, n_inner, 3
                )

                # Wstępny anchor threshold (Shi et al. 2020 Eq. 5-8). Jeśli
                # `threshold_ratio is None` (paperowy tryb statyczny) ta wartość
                # zostaje na stałe; w trybie adaptacyjnym jest nadpisywana po
                # _initialize_swarms na podstawie liderów.
                THRESHOLD_FLOOR = 0.01
                if initial_fitness < HARD_INFEASIBLE_BASE:
                    # Straight-line feasible — Big-M nie wpływa, oryginalna kalibracja
                    threshold_anchor = initial_fitness
                    anchor_source = "straight-line (feasible)"
                else:
                    # Big-M dominacja — sample skalę feasible z init pop
                    init_pop_fits = np.asarray(adapter(initial_population), dtype=np.float64)
                    feasible_in_init = init_pop_fits < HARD_INFEASIBLE_BASE
                    if feasible_in_init.any():
                        threshold_anchor = float(np.median(init_pop_fits[feasible_in_init]))
                        anchor_source = (
                            f"median feasible ({int(feasible_in_init.sum())}/"
                            f"{init_pop_fits.size} feasible w init pop)"
                        )
                    else:
                        # 100% init pop infeasible — analytical fallback. Suma |w|
                        # to górne oszacowanie skali feasible przy F_norm ~ 1
                        # (każdy objective znormalizowany do straight-line ≤ 1).
                        threshold_anchor = float(np.sum(np.abs(weights)))
                        anchor_source = "sum(|weights|) (no feasible in init pop)"
                dynamic_threshold = max(threshold_anchor * threshold_ratio, THRESHOLD_FLOOR)
                logger.info(
                    f"[MSFOA] Initial threshold anchor={threshold_anchor:.4f} "
                    f"({anchor_source}) → initial_threshold={dynamic_threshold:.4f} "
                    f"(ratio={threshold_ratio}) — będzie przeliczone adaptacyjnie "
                    f"po init populacji w optimizerze"
                )

                # problem.xl / problem.xu są (N_drones * N_inner * 3,) z trójkami
                # identycznymi per-waypoint — wystarczy pierwsza. Identyczne
                # bounds u NSGA-III i OOA.
                xl_point = np.asarray(problem.xl[:3], dtype=np.float64)
                xu_point = np.asarray(problem.xu[:3], dtype=np.float64)

                # paper Sec. 1 — R jako parametr; None → paper-tuned defaults
                # w konstruktorze MSFFOAOptimizer.
                sg_frac = params.get("step_global_frac")
                sl_frac = params.get("step_local_frac")
                step_global_frac = (
                    np.asarray(sg_frac, dtype=np.float64) if sg_frac is not None else None
                )
                step_local_frac = (
                    np.asarray(sl_frac, dtype=np.float64) if sl_frac is not None else None
                )

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
                    rng=seeds.rng("optimizer"),
                    n_swarms=n_swarms,
                    coe1=coe1,
                    coe2=coe2,
                    threshold=dynamic_threshold,
                    threshold_ratio=threshold_ratio,
                    step_global_frac=step_global_frac,
                    step_local_frac=step_local_frac,
                    history_writer=writer,
                    initial_population=initial_population,
                )

            with _measure("optimization"):
                try:
                    _, best_fitness = optimizer.optimize()
                finally:
                    writer.close()

            logger.info(f"[MSFOA] Optimization Finished. Best Polyline Fitness: {best_fitness:.4f}")

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