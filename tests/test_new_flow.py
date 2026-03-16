"""
Validation tests for the refactored flow:
- BaseAlgorithm no longer requires parent
- TrajectoryFollowingAlgorithm accepts trajectories directly
- count_trajectories integrates with nsga3_swarm_strategy
- EmptyWorld has _generate_world_def()
"""

import numpy as np

def test_base_algorithm_no_parent():
    """BaseAlgorithm constructor should not require parent."""
    from src.algorithms.BaseAlgorithm import BaseAlgorithm
    import inspect
    sig = inspect.signature(BaseAlgorithm.__init__)
    params = list(sig.parameters.keys())
    assert "parent" not in params, f"BaseAlgorithm still has 'parent' in __init__: {params}"
    assert "num_drones" in params
    assert "params" in params
    print("[OK] BaseAlgorithm no longer requires 'parent'")


def test_trajectory_follower_accepts_trajectories():
    """TrajectoryFollowingAlgorithm should accept trajectories in constructor."""
    from src.algorithms.TrajectoryFollowingAlgorithm import TrajectoryFollowingAlgorithm
    import inspect
    sig = inspect.signature(TrajectoryFollowingAlgorithm.__init__)
    params = list(sig.parameters.keys())
    assert "trajectories" in params, f"Missing 'trajectories' param: {params}"
    assert "parent" not in params, f"Still has 'parent' param: {params}"

    # Create with dummy trajectories
    n_drones = 2
    n_waypoints = 10
    dummy_traj = np.random.rand(n_drones, n_waypoints, 3)
    follower = TrajectoryFollowingAlgorithm(
        num_drones=n_drones,
        trajectories=dummy_traj,
        params={"acceptance_radius": 0.5, "simulation_freq_hz": 240}
    )
    assert follower._cached_trajectories.shape == (n_drones, n_waypoints, 3)
    assert follower.acceptance_radius == 0.5
    print("[OK] TrajectoryFollowingAlgorithm accepts trajectories directly")


def test_trajectory_follower_rejects_none():
    """TrajectoryFollowingAlgorithm should raise if trajectories is None."""
    from src.algorithms.TrajectoryFollowingAlgorithm import TrajectoryFollowingAlgorithm
    try:
        TrajectoryFollowingAlgorithm(num_drones=1, trajectories=None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "requires pre-computed trajectories" in str(e)
    print("[OK] TrajectoryFollowingAlgorithm rejects None trajectories")


def test_linear_algorithm_still_works():
    """LinearTestAlgorithm should work with the new BaseAlgorithm."""
    from src.algorithms.LinearTestAlgorithm import LinearTestAlgorithm
    alg = LinearTestAlgorithm(num_drones=1, params={"target_pos": [10, 0, 1]})
    assert alg.num_drones == 1
    print("[OK] LinearTestAlgorithm still works with updated BaseAlgorithm")


def test_empty_world_has_generate_world_def():
    """EmptyWorld should have _generate_world_def method."""
    from src.environments.EmptyWorld import EmptyWorld
    assert hasattr(EmptyWorld, '_generate_world_def'), "EmptyWorld missing _generate_world_def"
    print("[OK] EmptyWorld has _generate_world_def")


def test_count_trajectories_integration():
    """count_trajectories with nsga3_swarm_strategy should produce valid trajectories."""
    from src.algorithms.abstraction.count_trajectories import count_trajectories
    from src.algorithms.abstraction.trajectory.strategies.nsga3_swarm_strategy import nsga3_swarm_strategy
    from src.environments.obstacles.ObstacleShape import ObstacleShape

    class MockWorldData:
        def __init__(self):
            self.bounds = np.array([[0, 200], [0, 1000], [0, 120]])

    class MockObstaclesData:
        def __init__(self):
            self.count = 0
            self.data = np.zeros((0, 6))
            self.shape_type = ObstacleShape.CYLINDER

    n_drones = 2
    n_waypoints = 20
    start_pos = np.array([[95.0, 1.0, 0.5], [105.0, 1.0, 0.5]])
    target_pos = np.array([[95.0, 999.0, 0.5], [105.0, 999.0, 0.5]])

    params = {
        "pop_size": 20,
        "n_gen": 3,
        "n_inner_waypoints": 3,
        "uniformity_std": 50.0,
        "max_jerk": 500.0,
    }

    traj = count_trajectories(
        world_data=MockWorldData(),
        obstacles_data=MockObstaclesData(),
        counting_protocol=nsga3_swarm_strategy,
        drone_swarm_size=n_drones,
        number_of_waypoints=n_waypoints,
        start_positions=start_pos,
        target_positions=target_pos,
        algorithm_params=params
    )

    assert traj.shape == (n_drones, n_waypoints, 3), f"Wrong shape: {traj.shape}"

    # Verify anchoring: first and last waypoints should be near start/target
    for d in range(n_drones):
        err_start = np.linalg.norm(traj[d, 0] - start_pos[d])
        err_end = np.linalg.norm(traj[d, -1] - target_pos[d])
        assert err_start < 1.0, f"Drone {d} start error: {err_start}"
        assert err_end < 1.0, f"Drone {d} end error: {err_end}"

    print(f"[OK] count_trajectories returned shape {traj.shape} with correct anchoring")


if __name__ == "__main__":
    test_base_algorithm_no_parent()
    test_trajectory_follower_accepts_trajectories()
    test_trajectory_follower_rejects_none()
    test_linear_algorithm_still_works()
    test_empty_world_has_generate_world_def()
    print("\n--- Running integration test (NSGA-III optimization, may take a moment) ---\n")
    test_count_trajectories_integration()
    print("\n=== All tests passed ===")
