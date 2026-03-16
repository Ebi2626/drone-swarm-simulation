import numpy as np
from src.algorithms.BaseAlgorithm import BaseAlgorithm
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel


class TrajectoryFollowingAlgorithm(BaseAlgorithm):
    """PID-based waypoint follower with simple lookahead and altitude floor."""

    def __init__(self, num_drones, trajectories, params=None):
        super().__init__(num_drones, params)

        if trajectories is None:
            raise ValueError("TrajectoryFollowingAlgorithm requires pre-computed trajectories.")

        self._cached_trajectories = np.asarray(trajectories, dtype=np.float64)
        if self._cached_trajectories.ndim != 3 or self._cached_trajectories.shape[0] != num_drones:
            raise ValueError(
                "Trajectories must have shape (n_drones, n_waypoints, 3). "
                f"Got {self._cached_trajectories.shape}."
            )

        drone_model_name = self.params.get("drone_model", "CF2X")
        drone_model = getattr(DroneModel, drone_model_name, DroneModel.CF2X)
        self.controllers = [DSLPIDControl(drone_model=drone_model) for _ in range(num_drones)]

        self.current_waypoint_indices = np.zeros(num_drones, dtype=int)
        self.acceptance_radius = float(self.params.get("acceptance_radius", 1.0))
        self.lookahead_steps = int(self.params.get("lookahead_steps", 2))
        self.min_target_z = float(self.params.get("min_target_z", 1.0))
        self.control_timestep = 1.0 / float(self.params.get("control_freq_hz", 48))

        print(
            f"[TrajectoryFollower] Loaded trajectory with {self._cached_trajectories.shape[1]} waypoints. "
            f"dt={self.control_timestep:.4f}s, acceptance_radius={self.acceptance_radius}, "
            f"lookahead={self.lookahead_steps}, min_target_z={self.min_target_z}"
        )

    def _select_target(self, drone_path: np.ndarray, current_pos: np.ndarray, drone_idx: int) -> np.ndarray:
        current_idx = self.current_waypoint_indices[drone_idx]

        while current_idx < len(drone_path) - 1:
            candidate = drone_path[current_idx]
            if np.linalg.norm(candidate - current_pos) >= self.acceptance_radius:
                break
            current_idx += 1

        self.current_waypoint_indices[drone_idx] = current_idx
        lookahead_idx = min(current_idx + self.lookahead_steps, len(drone_path) - 1)
        target_pos = drone_path[lookahead_idx].copy()
        target_pos[2] = max(target_pos[2], self.min_target_z)
        return target_pos

    def compute_actions(self, current_states, current_time):
        actions = []

        for i in range(self.num_drones):
            state = current_states[i]
            pos = state[0:3]
            drone_path = self._cached_trajectories[i]
            target_pos = self._select_target(drone_path, pos, i)

            action, _, _ = self.controllers[i].computeControlFromState(
                control_timestep=self.control_timestep,
                state=state,
                target_pos=target_pos,
                target_rpy=np.array([0.0, 0.0, 0.0]),
            )
            actions.append(action)

        return np.array(actions)
