"""
MSFFOA Swarm Strategy.
Single-objective trajectory optimization using the Multi-Strategy Fruit Fly
Optimization Algorithm with pure-numpy vectorized math.

Aggregates the multi-objective output of VectorizedEvaluator (F: 3 objectives,
G: 5 constraints) into a scalar fitness via TrajectorySOOAdapter, then drives
MSFFOAOptimizer through smell-based and vision-based search phases.
"""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from hydra.core.hydra_config import HydraConfig

from src.algorithms.abstraction.trajectory.strategies.core_msffoa import (
    MSFFOAOptimizer,
)
from src.algorithms.abstraction.trajectory.strategies.soo_adapter import (
    TrajectorySOOAdapter,
)
from src.algorithms.abstraction.trajectory.objective_constrains import (
    VectorizedEvaluator,
)
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.utils.optimization_history_writer import OptimizationHistoryWriter


def msffoa_strategy(
    *,
    start_positions: NDArray[np.float64],
    target_positions: NDArray[np.float64],
    obstacles_data: Union[Any, List[Any]],
    world_data: WorldData,
    number_of_waypoints: int,
    drone_swarm_size: int,
    algorithm_params: Optional[Dict[str, Any]] = None,
) -> NDArray[np.float64]:
    """Trajectory generation via Multi-Strategy Fruit Fly Optimization Algorithm.

    Drop-in replacement for nsga3_swarm_strategy / osprey_swarm_strategy.
    Uses pure numpy — no external optimization libraries.

    Args:
        start_positions: (N, 3) drone start positions.
        target_positions: (N, 3) drone target positions.
        obstacles_data: Environment obstacles (single object or list).
        world_data: World boundary data.
        number_of_waypoints: Dense output trajectory length.
        drone_swarm_size: Number of drones in the swarm.
        algorithm_params: Algorithm hyperparameters (from Hydra config).

    Returns:
        (N_drones, N_waypoints, 3) optimized trajectory tensor.
    """
    params = algorithm_params or {}

    # --- Algorithm parameters ---
    pop_size: int = params.get("pop_size", 200)
    max_generations: int = params.get("epochs", 500)
    n_inner: int = params.get(
        "n_inner_waypoints", max(5, int(number_of_waypoints * 0.1)),
    )
    seed: int = params.get("seed", 42)

    # Objective weights
    w_list = params.get("objective_weights", [0.05, 100.0, 0.1])
    weights = np.array(w_list, dtype=np.float64)
    penalty_weight: float = params.get("penalty_weight", 1.0)

    # MSFFOA-specific hyperparameters
    levy_beta: float = params.get("levy_beta", 1.5)
    sigma_min_fraction: float = params.get("sigma_min_fraction", 0.01)

    # --- Normalize obstacles list ---
    obs_list: List[Any] = (
        obstacles_data if isinstance(obstacles_data, list) else [obstacles_data]
    )

    print(
        f"[MSFFOA] Start. Pop: {pop_size}, Epochs: {max_generations}, "
        f"Inner Pts: {n_inner}, Weights: {weights}, "
        f"Penalty: {penalty_weight}, Levy beta: {levy_beta}"
    )

    output_dir = HydraConfig.get().runtime.output_dir
    writer = OptimizationHistoryWriter(
        output_dir=os.path.join(output_dir, "optimization_history")
    )

    try:
        # --- VectorizedEvaluator ---
        evaluator = VectorizedEvaluator(
            obstacles=obs_list,
            start_pos=start_positions,
            target_pos=target_positions,
            params=params,
        )

        # --- SOO Adapter (handles normalization + weakest-link penalty) ---
        adapter = TrajectorySOOAdapter(
            evaluator=evaluator,
            start_positions=start_positions,
            target_positions=target_positions,
            n_drones=drone_swarm_size,
            n_inner=n_inner,
            n_output_samples=number_of_waypoints,
            weights=weights,
            penalty_weight=penalty_weight,
            history_writer=writer,
        )

        print(f"[MSFFOA] F_ref (normalization scales): {adapter._f_ref}")

        # --- MSFFOAOptimizer ---
        # The adapter accepts inner waypoints (Pop, D, Inner, 3) and handles
        # trajectory reconstruction internally, so we wire it as _evaluate
        # to bypass the optimizer's default _build_dense -> fitness_fn path.
        optimizer = MSFFOAOptimizer(
            pop_size=pop_size,
            n_drones=drone_swarm_size,
            n_inner=n_inner,
            n_output_samples=number_of_waypoints,
            world_min_bounds=np.asarray(world_data.min_bounds, dtype=np.float64),
            world_max_bounds=np.asarray(world_data.max_bounds, dtype=np.float64),
            start_positions=start_positions,
            target_positions=target_positions,
            fitness_function=adapter,  # unused directly, kept for API completeness
            max_generations=max_generations,
            seed=seed,
            levy_beta=levy_beta,
            sigma_min_fraction=sigma_min_fraction,
        )
        optimizer._evaluate = adapter

        # --- Run optimization ---
        best_inner, best_fitness = optimizer.optimize()

        print(f"[MSFFOA] Finished. Best fitness: {best_fitness:.4f}")

        # --- Reconstruct dense trajectory from best solution ---
        final_traj = optimizer.get_best_dense_trajectory()
        return final_traj  # (N_drones, N_waypoints, 3)

    except Exception as e:
        print(f"[MSFFOA] Optimization error: {e}. Returning straight-line fallback.")
    finally:
        writer.close()

    # --- Fallback: straight-line trajectory ---
    print("[MSFFOA] Fallback: generating straight-line trajectory.")
    t_line = np.linspace(0, 1, number_of_waypoints)
    out = np.empty((drone_swarm_size, number_of_waypoints, 3))
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
