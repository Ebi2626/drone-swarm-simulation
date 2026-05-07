"""
Fixtures for end-to-end analytical validation of VectorizedEvaluator.

These tests use the REAL Numba pipeline (no mocks) to verify that
known-cost trajectories produce expected objective values. This directly
addresses experiment_restrictions.md §6.2: "test walidacyjny z analitycznie
znanym kosztem."

Environment: empty world (configs/environment/empty.yaml)
  - 2 drones, 30×100×12m, ground=0.1m
  - start: [[15,1,5],[20,1,5]]  target: [[15,99,5],[20,99,5]]
  - No obstacles
"""
import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def empty_world_params() -> dict:
    """
    Evaluator params matching configs/optimizer/ssa.yaml kinematic section,
    but with preferred_height=5.0 to match the empty env start/target z=5.
    """
    return {
        "k_factor": 2.0,
        "absolute_min_node_dist": 2.0,
        "obstacle_safety_margin": 1.0,
        "min_drone_distance": 2.0,
        "max_accel_limit": 2.0,
        "preferred_height": 5.0,
        "coordination_penalty_factor": 1.2,
    }


@pytest.fixture
def start_positions() -> NDArray[np.float64]:
    """Start positions from configs/environment/empty.yaml."""
    return np.array([[15.0, 1.0, 5.0], [20.0, 1.0, 5.0]], dtype=np.float64)


@pytest.fixture
def target_positions() -> NDArray[np.float64]:
    """Target positions from configs/environment/empty.yaml."""
    return np.array([[15.0, 99.0, 5.0], [20.0, 99.0, 5.0]], dtype=np.float64)


@pytest.fixture
def n_inner_waypoints() -> int:
    """Inner waypoint count — 11 matches MSFFOA/OOA/NSGA-III configs."""
    return 11


def _build_straight_line_control_points(
    start_positions: NDArray[np.float64],
    target_positions: NDArray[np.float64],
    n_inner: int,
) -> NDArray[np.float64]:
    """
    Build control points forming a perfect straight line between start and
    target for each drone.

    Returns shape (1, NDrones, n_inner+2, 3) — 1 individual in the population.

    The n_inner+2 control points are evenly spaced along the line segment
    [start, target], including start and target as the first and last points.
    This is the analytically optimal trajectory in an empty environment:
    minimal path length, zero turns, constant height, zero threat.
    """
    n_drones = start_positions.shape[0]
    n_control = n_inner + 2
    cp = np.zeros((1, n_drones, n_control, 3), dtype=np.float64)

    for d in range(n_drones):
        for dim in range(3):
            cp[0, d, :, dim] = np.linspace(
                start_positions[d, dim],
                target_positions[d, dim],
                n_control,
            )
    return cp


@pytest.fixture
def straight_line_control_points(
    start_positions, target_positions, n_inner_waypoints
) -> NDArray[np.float64]:
    """
    Perfect straight-line control points for 2 drones in empty world.
    Shape: (1, 2, 13, 3).

    Analytical properties:
      - Path length per drone: ||target - start|| = 98.0m (Y: 1→99)
      - Total f1 ≈ 196.0 (2 × 98m), plus negligible shape cost
      - z = 5.0 everywhere = preferred_height → f2 = 0
      - No obstacles → f3 = 0
      - All segments collinear → f4 = 0
      - Inter-drone distance = 5m > min_drone_distance=2m → f5 = 0
      - Constraints G all ≤ 0 (feasible)
    """
    return _build_straight_line_control_points(
        start_positions, target_positions, n_inner_waypoints
    )
