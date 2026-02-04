from typing import Any, Dict, Optional, Protocol
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.abstraction.generate_obstacles import ObstaclesData
from numpy.typing import NDArray
import numpy as np

class TrajectoryStrategyProtocol(Protocol):
    """
    Strategy for trajectory counting. It gets all parameters needed for mulitcriteria optimization methods
    and allow to pass additional configuration arguments for specific implementations.

    Args:
        start_positions (NDArray[np.float64], (N,3)): Array with starting positions for each drone. \n
        target_positions (NDArray[np.float64], (N,3)): Array with target positions for each drone.
        obstacles_data (ObstaclesData): Object with positions of obstacles. \n
        world_data (WorldData): Object with world dimensions, min and max bounds, center of the world. \n
        number_of_waypoints (int): number of waypoints to be generated. \n
        drone_swarm_size (int): number of drones in the swarm. \n
        algorithm_params (Optional[Dict[str, Any]], optional): _description_. Defaults to None.

    Returns:
        NDArray[np.float64]: Tensor with following shape: (N, W, 3)\n
                        N - amount of drones\n
                        W - amount of waypoints (n_waypoints)\n
                        3 - coordinates x, y, z
    """
    def __call__(
        self,
        *, 
        start_positions: NDArray[np.float64],
        target_positions: NDArray[np.float64],
        obstacles_data: ObstaclesData,
        world_data: WorldData,
        number_of_waypoints: int,
        drone_swarm_size: int, 
        algorithm_params: Optional[Dict[str, Any]] = None
    ) -> NDArray[np.float64]:
        pass

def count_trajectories(
        world_data: WorldData,
        obstacles_data: ObstaclesData, 
        counting_protocol: TrajectoryStrategyProtocol, 
        drone_swarm_size: int, 
        number_of_waypoints: int,
        start_positions: NDArray[np.float64],
        target_positions: NDArray[np.float64],
        algorithm_params: Optional[Dict[str, Any]] = None
        ) -> NDArray[np.float64]:
    """
    Trajectory generation based on given world_data, obstacles_data and strategy. \n
    Strategy should represent mathematical implementation of specific bio-inspired algorithm.

    Args:
        world_data (WorldData): Object with world dimensions, min and max bounds, center of the world. \n
        obstacles_data (ObstaclesData): Object with positions of obstacles. \n
        counting_protocol (TrajectoryStrategyProtocol): Object representing strategy for trajectory counting. \n
        drone_swarm_size (int): number of drones in the swarm. \n
        number_of_waypoints (int): number of waypoints to be generated. \n
        start_positions (NDArray[np.float64], (N,3)): Array with starting positions for each drone. \n
        target_positions (NDArray[np.float64], (N,3)): Array with target positions for each drone. \n
        algorithm_params (Optional[Dict[str, Any]]): Parameters for the optimization algorithm. \n

    Returns:
        NDArray[np.float64]: Tensor with following shape: (N, W, 3)\n
                        N - amount of drones\n
                        W - amount of waypoints (n_waypoints)\n
                        3 - coordinates x, y, z
    """

    # TODO: implement methods to count time of optimization
    # TODO: implement methods to validate input

    trajectories = counting_protocol(
        start_positions=start_positions,
        target_positions=target_positions,
        obstacles_data=obstacles_data,
        world_data=world_data,
        number_of_waypoints=number_of_waypoints,
        drone_swarm_size=drone_swarm_size,
        algorithm_params=algorithm_params
    )

    return trajectories
