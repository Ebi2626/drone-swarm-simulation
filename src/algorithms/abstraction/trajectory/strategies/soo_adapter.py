"""
Generic Single-Objective Adapter for VectorizedEvaluator.

Bridges the multi-objective VectorizedEvaluator (F: objectives, G: constraints)
to a scalar fitness value suitable for single-objective metaheuristics 
(e.g., PSO, DE, FOA variants).

Golden Rules enforced:
1. Objective Normalization — F is divided by F_ref (straight-line reference)
   before weighting, so objectives operate on a dimensionless ~1.0 scale.
2. Weakest-Link Penalty — the worst single constraint violation (np.max)
   defines the penalty, not the sum. One catastrophic violation cannot be
   diluted by clean constraints.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

# Zmiana nazwy importu zgodnie z fizyczną nazwą w projekcie
from src.algorithms.abstraction.trajectory.objective_constrains import VectorizedEvaluator


class TrajectorySOOAdapter:
    """Converts VectorizedEvaluator outputs into a single scalar fitness.

    Designed as a callable: an instance can be passed directly as the
    ``fitness_function`` argument to any SOO optimizer that operates on
    inner waypoints tensors.

    Pipeline (per call):
    1. Receive inner waypoints ``(Pop, N_drones, Inner, 3)``.
    2. Prepend start positions, append target positions (Polyline construction).
    3. Query ``VectorizedEvaluator`` -> ``F (Pop, M)``, ``G (Pop, K)``.
    4. Normalize ``F`` by ``F_ref`` (Golden Rule #1).
    5. Aggregate ``G`` via weakest-link max (Golden Rule #2).
    6. Return ``(Pop,)`` scalar fitness.

    Args:
        evaluator: Pre-configured VectorizedEvaluator instance.
        start_positions: (N_drones, 3) fixed start positions.
        target_positions: (N_drones, 3) fixed target positions.
        n_drones: Number of drones.
        n_inner: Number of inner waypoints per drone.
        weights: (M,) objective weights [w_path_length, w_smoothness, w_collision_risk].
        penalty_weight: Multiplier for the weakest-link constraint penalty.
        history_writer: Optional logger for saving generational data.
    """

    def __init__(
        self,
        evaluator: VectorizedEvaluator,
        start_positions: NDArray[np.float64],
        target_positions: NDArray[np.float64],
        n_drones: int,
        n_inner: int,
        weights: NDArray[np.float64],
        penalty_weight: float = 100.0,
        history_writer: Optional[Any] = None,
    ) -> None:
        self.evaluator = evaluator
        self.n_drones = n_drones
        self.n_inner = n_inner
        self.weights = np.asarray(weights, dtype=np.float64)
        self.penalty_weight = penalty_weight
        self.history_writer = history_writer

        # Broadcast-ready endpoint shapes: (1, N_drones, 1, 3)
        self._starts_bc = np.asarray(start_positions, dtype=np.float64)[np.newaxis, :, np.newaxis, :]
        self._targets_bc = np.asarray(target_positions, dtype=np.float64)[np.newaxis, :, np.newaxis, :]

        # Golden Rule #1: compute F_ref from a straight-line trajectory
        self._f_ref = self._compute_reference_scales()
        
        # State tracking for logger/optimizer access
        self.last_objectives: NDArray[np.float64] | None = None
        self.last_constraints: NDArray[np.float64] | None = None
        self._debug_printed: bool = False

    # ------------------------------------------------------------------
    # Reference scale computation (Golden Rule #1)
    # ------------------------------------------------------------------

    def _compute_reference_scales(self) -> NDArray[np.float64]:
        """Evaluate a straight-line trajectory to obtain F_ref for normalization.

        F2 (collision risk) is often zero on the straight line. A tiny epsilon
        is applied to prevent DivisionByZero, preserving the original scale 
        for structurally small objectives (like smoothness).

        Returns:
            (M,) reference objective values, guaranteed > 0.
        """
        # Generate evenly spaced intermediate points along the straight line
        t_vals = np.linspace(0, 1, self.n_inner + 2)[1:-1]
        t = t_vals.reshape(1, 1, self.n_inner, 1)

        # Interpolation via broadcasting
        inner_ref = self._starts_bc + t * (self._targets_bc - self._starts_bc)
        
        # Assemble sparse points
        sparse_ref = np.concatenate(
            [self._starts_bc, inner_ref, self._targets_bc], axis=2
        )

        out_ref: Dict[str, Any] = {}
        self.evaluator.evaluate(sparse_ref, out_ref)
        
        f_ref = out_ref["F"][0]  # Take the first (and only) population element's objectives

        # Use 1e-6 instead of 1.0 to avoid flattening naturally small values (e.g., smoothness)
        return np.maximum(f_ref, 1e-6)

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

        # Use np.broadcast_to to create memory-efficient views without copying
        starts = np.broadcast_to(self._starts_bc, (pop_size, self.n_drones, 1, 3))
        targets = np.broadcast_to(self._targets_bc, (pop_size, self.n_drones, 1, 3))

        # Reconstruct the full polyline (discrete waypoints)
        trajectories = np.concatenate([starts, inner_waypoints, targets], axis=2)

        # Query VectorizedEvaluator
        out: Dict[str, Any] = {}
        self.evaluator.evaluate(trajectories, out)

        F = out["F"]  # Objectives: (Pop_size, M)
        G = out["G"]  # Constraints: (Pop_size, K)

        self.last_objectives = F
        self.last_constraints = G

        # Golden Rule #1: Normalize objectives by reference scales
        # F shapes: (Pop_size, M) / (M,) -> Broadcasts naturally
        F_norm = F / self._f_ref

        # Weighted objective sum (vectorized dot product)
        # F_norm @ weights mathematically executes \sum_{i=1}^{M} w_i * F_{norm, i}
        obj_values = F_norm @ self.weights  # Result shape: (Pop_size,)

        # Golden Rule #2: Weakest-link penalty (max violation, not sum)
        # pymoo standard: G <= 0 is feasible, G > 0 is a violation.
        # np.maximum(0.0, G) clips all feasible scores to zero.
        # np.max(..., axis=1) finds the most severe violation for each population member.
        penalties = self.penalty_weight * np.max(np.maximum(0.0, G), axis=1)

        # Optional: Print debug info only on the first pass
        if not self._debug_printed:
            print(f"\n[DEBUG SOO] Obj Norm (First Mem): {F_norm[0]} | Raw G: {G[0]}")
            print(f"[DEBUG SOO] Fitness: {obj_values[0]:.4f} | Penalty: {penalties[0]:.4f}")
            self._debug_printed = True

        return obj_values + penalties
