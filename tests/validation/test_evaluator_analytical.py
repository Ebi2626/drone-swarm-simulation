"""
End-to-end analytical validation of VectorizedEvaluator (§6.2).

These tests run the REAL Numba pipeline (evaluate_bspline_trajectory_sync)
against trajectories with analytically known costs. No mocks are used —
this validates the complete evaluation chain from control points through
B-spline sampling to objective/constraint computation.

Reference environment: empty world (no obstacles), 2 drones flying
straight lines from [15/20, 1, 5] to [15/20, 99, 5].

Expected analytical values for straight-line trajectory:
  f1 ≈ 196.0   (2 drones × 98m Euclidean distance)
  f2 = 0.0     (z = preferred_height = 5.0, dz = 0)
  f3 = 0.0     (no obstacles)
  f4 ≈ 0.0     (collinear segments, arccos(1) = 0)
  f5 = 0.0     (5m separation > 2m min_drone_distance)
  G  ≤ 0       (all constraints feasible)
"""
import numpy as np
import pytest

from src.algorithms.abstraction.trajectory.objective_constrains import VectorizedEvaluator
from tests.validation.conftest import _build_straight_line_control_points


# ============================================================
# HELPER
# ============================================================

def _evaluate_straight_line(
    start_positions, target_positions, n_inner_waypoints, empty_world_params,
    straight_line_control_points,
):
    """Create evaluator + evaluate straight-line trajectory, return (F, G)."""
    evaluator = VectorizedEvaluator(
        obstacles=None,
        start_pos=start_positions,
        target_pos=target_positions,
        n_inner_points=n_inner_waypoints,
        params=empty_world_params,
    )
    out: dict = {}
    evaluator.evaluate(straight_line_control_points, out)
    return out["F"], out["G"]


# ============================================================
# TEST 1: f1 — trajectory cost ≈ total Euclidean path length
# ============================================================

class TestStraightLineF1:
    """
    f1 = f_length + f_shape.

    For a perfect straight line:
      - f_length = sum of B-spline arc lengths ≈ 2 × 98m = 196m
        (B-spline interpolation introduces slight deviation from
         the control polygon length, hence atol=5.0)
      - f_shape = 0 (all control points lie on the start→target line,
        so cross-product distance to ideal vector = 0)
    """

    def test_f1_equals_path_length(
        self, start_positions, target_positions, n_inner_waypoints,
        empty_world_params, straight_line_control_points,
    ):
        F, _ = _evaluate_straight_line(
            start_positions, target_positions, n_inner_waypoints,
            empty_world_params, straight_line_control_points,
        )
        f1 = float(F[0, 0])
        # 2 drones × 98m = 196m; B-spline arc may differ slightly
        assert 180.0 < f1 < 220.0, (
            f"f1={f1:.2f}, expected ≈196.0 (2×98m straight line)"
        )


# ============================================================
# TEST 2: f2 — height + angle cost = 0 at preferred height
# ============================================================

class TestStraightLineF2:
    """
    f2 = f_height + f_angle.

    For z = preferred_height = 5.0 everywhere:
      - f_height = sum(|z - 5|) = 0
      - f_angle = sum(arctan(|dz|/||dxy||)) = 0  (dz = 0 everywhere)
    """

    def test_f2_zero_at_preferred_height(
        self, start_positions, target_positions, n_inner_waypoints,
        empty_world_params, straight_line_control_points,
    ):
        F, _ = _evaluate_straight_line(
            start_positions, target_positions, n_inner_waypoints,
            empty_world_params, straight_line_control_points,
        )
        f2 = float(F[0, 1])
        assert f2 == pytest.approx(0.0, abs=1e-6), (
            f"f2={f2:.6f}, expected 0.0 (z=preferred_height, no vertical change)"
        )


# ============================================================
# TEST 3: f3 — threat cost = 0 (no obstacles)
# ============================================================

class TestStraightLineF3:

    def test_f3_zero_no_obstacles(
        self, start_positions, target_positions, n_inner_waypoints,
        empty_world_params, straight_line_control_points,
    ):
        F, _ = _evaluate_straight_line(
            start_positions, target_positions, n_inner_waypoints,
            empty_world_params, straight_line_control_points,
        )
        f3 = float(F[0, 2])
        assert f3 == 0.0, (
            f"f3={f3}, expected exactly 0.0 (empty world, no obstacles)"
        )


# ============================================================
# TEST 4: f4 — turn cost ≈ 0 (collinear segments)
# ============================================================

class TestStraightLineF4:
    """
    f4 = sum(arccos(dot(u_i, u_{i+1}))²).

    For collinear control points all unit vectors u_i are identical,
    so dot = 1.0, arccos(1) = 0, f4 = 0.
    """

    def test_f4_zero_collinear(
        self, start_positions, target_positions, n_inner_waypoints,
        empty_world_params, straight_line_control_points,
    ):
        F, _ = _evaluate_straight_line(
            start_positions, target_positions, n_inner_waypoints,
            empty_world_params, straight_line_control_points,
        )
        f4 = float(F[0, 3])
        assert f4 == pytest.approx(0.0, abs=1e-6), (
            f"f4={f4:.6f}, expected 0.0 (collinear segments)"
        )


# ============================================================
# TEST 5: f5 — coordination cost = 0 (well-separated drones)
# ============================================================

class TestStraightLineF5:
    """
    f5 uses exponential penalty for drones closer than min_drone_distance.

    Drone separation = |20 - 15| = 5m at every control point.
    min_drone_distance = 2m. Since 5m > 2m, mask a_ij = 0 everywhere → f5 = 0.
    """

    def test_f5_zero_separated_drones(
        self, start_positions, target_positions, n_inner_waypoints,
        empty_world_params, straight_line_control_points,
    ):
        F, _ = _evaluate_straight_line(
            start_positions, target_positions, n_inner_waypoints,
            empty_world_params, straight_line_control_points,
        )
        f5 = float(F[0, 4])
        assert f5 == pytest.approx(0.0, abs=1e-6), (
            f"f5={f5:.6f}, expected 0.0 (5m separation > 2m threshold)"
        )


# ============================================================
# TEST 6: All constraints feasible (G ≤ 0)
# ============================================================

class TestStraightLineConstraints:
    """
    G = [obs_collision_sum, swarm_hard - 0.01, kinematic_penalty].

    For straight line in empty world:
      - G[0] = 0 (no obstacles)
      - G[1] = swarm_collisions_hard - 0.01 ≤ 0 (5m > 2m min distance)
      - G[2] = kinematic_penalty = 0 (uniform spacing, no acceleration)
    """

    def test_constraints_feasible(
        self, start_positions, target_positions, n_inner_waypoints,
        empty_world_params, straight_line_control_points,
    ):
        _, G = _evaluate_straight_line(
            start_positions, target_positions, n_inner_waypoints,
            empty_world_params, straight_line_control_points,
        )
        for j in range(G.shape[1]):
            assert float(G[0, j]) <= 0.0, (
                f"G[0, {j}]={float(G[0, j]):.4f} > 0 — constraint violated "
                f"for analytical straight-line trajectory in empty world"
            )


# ============================================================
# TEST 7: Perturbed trajectory increases costs
# ============================================================

class TestPerturbedTrajectoryIncreasesCosts:
    """
    Adding random noise to control points must increase f1, f2, f4.
    This validates monotonicity: worse trajectories → higher costs.

    We perturb ONLY the inner control points (not start/target endpoints)
    to ensure the trajectory still connects start → target.
    """

    def test_perturbed_trajectory_increases_costs(
        self, start_positions, target_positions, n_inner_waypoints,
        empty_world_params, straight_line_control_points,
    ):
        evaluator = VectorizedEvaluator(
            obstacles=None,
            start_pos=start_positions,
            target_pos=target_positions,
            n_inner_points=n_inner_waypoints,
            params=empty_world_params,
        )

        # Evaluate baseline (straight line)
        out_base: dict = {}
        evaluator.evaluate(straight_line_control_points, out_base)
        F_base = out_base["F"][0]

        # Perturb inner control points (indices 1:-1)
        rng = np.random.default_rng(42)
        perturbed = straight_line_control_points.copy()
        n_control = perturbed.shape[2]
        # XY noise: ±5m, Z noise: ±2m (keeps drones within world bounds)
        perturbed[0, :, 1:n_control - 1, 0] += rng.uniform(-5, 5, size=(2, n_control - 2))
        perturbed[0, :, 1:n_control - 1, 1] += rng.uniform(-5, 5, size=(2, n_control - 2))
        perturbed[0, :, 1:n_control - 1, 2] += rng.uniform(-2, 2, size=(2, n_control - 2))

        out_pert: dict = {}
        evaluator.evaluate(perturbed, out_pert)
        F_pert = out_pert["F"][0]

        # f1 increases (longer path + shape deviation)
        assert F_pert[0] > F_base[0], (
            f"f1: perturbed={F_pert[0]:.2f} should be > baseline={F_base[0]:.2f}"
        )
        # f2 increases (height deviations from preferred_height)
        assert F_pert[1] > F_base[1], (
            f"f2: perturbed={F_pert[1]:.2f} should be > baseline={F_base[1]:.2f}"
        )
        # f4 increases (non-collinear segments create turn costs)
        assert F_pert[3] > F_base[3], (
            f"f4: perturbed={F_pert[3]:.2f} should be > baseline={F_base[3]:.2f}"
        )


# ============================================================
# TEST 8: Minimal optimization in empty world produces finite
#          feasible trajectories (integration test)
# ============================================================

class TestMinimalOptimizationEmpty:
    """
    Integration test: run SSA strategy with minimal budget (pop=10, epochs=3)
    in empty world and verify:
      1. Returned trajectories are finite (no NaN/Inf)
      2. f3 = 0 (no obstacles in empty world)
      3. No "stuck" drones (trajectory covers > 50% of start→target distance)

    This validates the full pipeline: strategy → evaluator → output,
    without mocks.
    """

    def test_minimal_ssa_optimization_empty(
        self, start_positions, target_positions, empty_world_params,
    ):
        from src.algorithms.abstraction.trajectory.strategies.ssa_strategy import (
            ssa_swarm_strategy,
        )
        from src.environments.abstraction.generate_world_boundaries import (
            generate_world_boundaries,
        )
        from src.utils.SeedRegistry import SeedRegistry

        seeds = SeedRegistry(master_seed=12345)

        # Empty world matching configs/environment/empty.yaml
        world_data = generate_world_boundaries(
            width=30.0, length=100.0, height=12.0, ground_height=0.1,
        )

        algorithm_params = {
            "pop_size": 10,
            "epochs": 3,
            "n_inner_waypoints": 7,
            "ST": 0.6,
            "PD": 0.4,
            "SD": 0.1,
            "objective_weights": [0.05, 0.5, 0.8, 1.0, 0.25],
            "penalty_weight": 2.0,
            "obstacle_safety_margin": 1.0,
            "min_drone_distance": 2.0,
            "k_factor": 2.0,
            "absolute_min_node_dist": 2.0,
            "max_accel_limit": 2.0,
            "preferred_height": 5.0,
            "coordination_penalty_factor": 1.2,
            "noise_std_xy": 7.0,
            "noise_std_z": 1.5,
            "cruise_speed": 6.0,
        }

        # Minimal budget — we only care about pipeline correctness, not quality
        trajectories = ssa_swarm_strategy(
            start_positions=start_positions,
            target_positions=target_positions,
            obstacles_data=None,
            world_data=world_data,
            number_of_waypoints=100,
            drone_swarm_size=2,
            algorithm_params=algorithm_params,
            seeds=seeds,
        )

        # Shape: (num_drones, num_waypoints, 3)
        assert trajectories.ndim == 3, (
            f"Expected 3D array, got shape {trajectories.shape}"
        )
        assert trajectories.shape[0] == 2, "Expected 2 drones"
        assert trajectories.shape[2] == 3, "Expected 3D coordinates"

        # 1. Finite values (no NaN/Inf from Numba pipeline)
        assert np.all(np.isfinite(trajectories)), (
            "Trajectories contain NaN or Inf values"
        )

        # 2. f3 = 0 for empty world — evaluate the output trajectory
        # Convert waypoints back to control points shape for evaluator
        n_wp = trajectories.shape[1]
        cp = trajectories[np.newaxis, :, :, :]  # (1, 2, n_wp, 3)

        evaluator = VectorizedEvaluator(
            obstacles=None,
            start_pos=start_positions,
            target_pos=target_positions,
            n_inner_points=n_wp - 2,
            params=empty_world_params,
        )
        out: dict = {}
        evaluator.evaluate(cp, out)
        f3 = float(out["F"][0, 2])
        assert f3 == 0.0, (
            f"f3={f3} in empty world, expected 0.0 (no obstacles)"
        )

        # 3. No stuck drones — each drone covers significant distance
        for d in range(2):
            total_dist = np.linalg.norm(
                target_positions[d] - start_positions[d]
            )
            covered = np.linalg.norm(
                trajectories[d, -1] - trajectories[d, 0]
            )
            ratio = covered / total_dist
            assert ratio > 0.5, (
                f"Drone {d} stuck: covered {covered:.1f}m of {total_dist:.1f}m "
                f"({ratio:.1%})"
            )
