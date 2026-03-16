import numpy as np
from src.algorithms.BaseAlgorithm import BaseAlgorithm
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

class TrajectoryFollowingAlgorithm(BaseAlgorithm):
    """
    PID-based waypoint follower. Receives pre-computed trajectories
    and follows them using DSLPIDControl.

    Args:
        num_drones: Number of drones in the swarm.
        trajectories: Pre-computed trajectories, shape (n_drones, n_waypoints, 3).
        params: Optional dict with follower parameters (acceptance_radius, simulation_freq_hz, drone_model).
    """
    def __init__(self, num_drones, trajectories, params=None):
        super().__init__(num_drones, params)

        if trajectories is None:
            raise ValueError("TrajectoryFollowingAlgorithm requires pre-computed trajectories.")
        self._cached_trajectories = np.asarray(trajectories)

        drone_model_name = self.params.get("drone_model", "CF2X")
        drone_model = getattr(DroneModel, drone_model_name, DroneModel.CF2X)
        self.controllers = [DSLPIDControl(drone_model=drone_model) for _ in range(num_drones)]

        self.current_waypoint_indices = np.zeros(num_drones, dtype=int)
        self.acceptance_radius = self.params.get("acceptance_radius", 0.2)

        print(f"[TrajectoryFollower] Loaded trajectory with {self._cached_trajectories.shape[1]} waypoints.")

    def compute_actions(self, current_states, current_time):
        actions = []

        for i in range(self.num_drones):
            state = current_states[i]
            pos = state[0:3]

            drone_path = self._cached_trajectories[i]

            current_idx = self.current_waypoint_indices[i]
            target_pos = drone_path[current_idx]

            dist = np.linalg.norm(target_pos - pos)

            if dist < self.acceptance_radius:
                if current_idx < len(drone_path) - 1:
                    self.current_waypoint_indices[i] += 1
                    target_pos = drone_path[self.current_waypoint_indices[i]]

            action, _, _ = self.controllers[i].computeControlFromState(
                control_timestep=self.params.get("simulation_freq_hz", 240) ** -1,
                state=state,
                target_pos=target_pos,
                target_rpy=np.array([0, 0, 0])
            )
            actions.append(action)

        return np.array(actions)
