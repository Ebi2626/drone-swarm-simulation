"""
Single-Objective Adapter for VectorizedEvaluator.

Bridges the multi-objective VectorizedEvaluator (F: 3 objectives, G: 5 constraints)
to a scalar fitness value suitable for single-objective optimizers like MSFFOAOptimizer.

Golden Rules enforced:
  1. Objective Normalization — F is divided by F_ref (straight-line reference)
     before weighting, so objectives operate on a dimensionless ~1.0 scale.
  2. Weakest-Link Penalty — the worst single constraint violation (np.max)
     defines the penalty, not the sum. One catastrophic violation cannot be
     diluted by four clean constraints.
"""

from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from src.algorithms.abstraction.trajectory.objective_constrains import (
    VectorizedEvaluator,
)
from src.algorithms.abstraction.trajectory.strategies.core_msffoa import (
    _generate_bezier_curve
)

class TrajectorySOOAdapter:
    """Converts VectorizedEvaluator outputs into a single scalar fitness.

    Designed as a callable: an instance can be passed directly as the
    ``fitness_function`` argument to ``MSFFOAOptimizer``.

    Pipeline (per call):
        1. Receive inner waypoints ``(Pop, D, Inner, 3)``.
        2. Prepend start positions, append target positions.
        3. Resample sparse polyline to dense trajectory.
        4. Query ``VectorizedEvaluator`` -> ``F (Pop, 3)``, ``G (Pop, 5)``.
        5. Normalize ``F`` by ``F_ref`` (Golden Rule #1).
        6. Aggregate ``G`` via weakest-link max (Golden Rule #2).
        7. Return ``(Pop,)`` scalar fitness.

    Args:
        evaluator: Pre-configured VectorizedEvaluator instance.
        start_positions: (N_drones, 3) fixed start positions.
        target_positions: (N_drones, 3) fixed target positions.
        n_drones: Number of drones.
        n_inner: Number of inner waypoints per drone.
        n_output_samples: Dense trajectory length for evaluation.
        weights: (3,) objective weights [w_path_length, w_collision_risk, w_elevation].
        penalty_weight: Multiplier for the weakest-link constraint penalty.
    """

    def __init__(
        self,
        evaluator: VectorizedEvaluator,
        start_positions: NDArray[np.float64],
        target_positions: NDArray[np.float64],
        n_drones: int,
        n_inner: int,
        n_output_samples: int,
        weights: NDArray[np.float64],
        penalty_weight: float = 100.0,
    ) -> None:
        self.evaluator = evaluator
        self.n_drones = n_drones
        self.n_inner = n_inner
        self.n_output_samples = n_output_samples
        self.weights = np.asarray(weights, dtype=np.float64)  # (3,)
        self.penalty_weight = penalty_weight

        # Broadcast-ready endpoint shapes: (1, D, 1, 3)
        self._starts_bc = start_positions[np.newaxis, :, np.newaxis, :]
        self._targets_bc = target_positions[np.newaxis, :, np.newaxis, :]

        # Golden Rule #1: compute F_ref from a straight-line trajectory
        self._f_ref = self._compute_reference_scales()

    # ------------------------------------------------------------------
    # Reference scale computation (Golden Rule #1)
    # ------------------------------------------------------------------

    def _compute_reference_scales(self) -> NDArray[np.float64]:
        """Evaluate a straight-line trajectory to obtain F_ref for normalization.

        F2 (collision risk) is often zero on the straight line, so a floor of
        1.0 is applied to prevent division by zero while keeping the raw scale
        for objectives that lack a meaningful reference.

        Returns:
            (3,) reference objective values, each >= 1.0.
        """
        t_vals = np.linspace(0, 1, self.n_inner + 2)[1:-1]
        t = t_vals.reshape(1, 1, self.n_inner, 1)

        inner_ref = self._starts_bc + t * (self._targets_bc - self._starts_bc)
        sparse_ref = np.concatenate(
            [self._starts_bc, inner_ref, self._targets_bc], axis=2,
        )
        traj_ref = _generate_bezier_curve(sparse_ref, self.n_output_samples)

        out_ref: Dict[str, Any] = {}
        self.evaluator.evaluate(traj_ref, out_ref)
        f_ref = out_ref["F"][0]  # (3,)

        return np.maximum(f_ref, 1.0)

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(self, inner_waypoints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate a batch of inner-waypoint populations.

        Args:
            inner_waypoints: (Pop_size, N_drones, N_inner, 3).

        Returns:
            (Pop_size,) scalar fitness values (lower is better).
        """
        pop_size = inner_waypoints.shape[0]

        starts = np.broadcast_to(self._starts_bc, (pop_size, self.n_drones, 1, 3)).copy()
        targets = np.broadcast_to(self._targets_bc, (pop_size, self.n_drones, 1, 3)).copy()
        
        # Traktujemy wygenerowane przez MSFFOA punkty jako PUNKTY KONTROLNE
        control_points = np.concatenate([starts, inner_waypoints, targets], axis=2)

        # GENERUJEMY GŁADKĄ KRZYWĄ BEZIERA zamiast liniowej interpolacji
        # Ewaluator dostanie n_output_samples punktów, które fizycznie nie mają ostrych kątów
        trajectories = _generate_bezier_curve(control_points, self.n_output_samples)

        # 3. Query VectorizedEvaluator
        out: Dict[str, Any] = {}
        self.evaluator.evaluate(trajectories, out)

        F = out["F"]  # (Pop_size, 3)
        G = out["G"]  # (Pop_size, 5)

        # 4. Golden Rule #1: Normalize objectives by reference scales
        F_norm = F / self._f_ref[np.newaxis, :]  # (Pop_size, 3)

        # 5. Weighted objective sum (vectorized dot product over Pop_size)
        obj_values = F_norm @ self.weights  # (Pop_size,)

        # 6. Golden Rule #2: Weakest-link penalty (max violation, not sum)
        penalties = self.penalty_weight * np.max(
            np.maximum(0.0, G), axis=1,
        )  # (Pop_size,)

        return obj_values + penalties
