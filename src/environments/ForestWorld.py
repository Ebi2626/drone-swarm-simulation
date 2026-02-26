import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from src.environments.SwarmBaseWorld import SwarmBaseWorld
from src.environments.abstraction.generate_obstacles import ObstaclesData, generate_obstacles, strategy_random_uniform
from src.environments.abstraction.generate_world_boundaries import generate_world_boundaries
from src.environments.obstacles.ObstacleShape import ObstacleShape
from src.utils.ConfigValidator import ConfigValidator
from src.utils.config_parser import sanitize_init_params
from src.utils.postions_to_tensor import positions_to_tensor

class ForestWorld(SwarmBaseWorld):
    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        physics: Physics = Physics.PYB,
        initial_xyzs=None,
        initial_rpys=None,
        end_xyzs=None,
        drone_number: int = None,
        ground_position: float = 0.1,
        track_length: float = 1000.0,
        track_width: float = 50.0,
        track_height: float = 12.0,
        obstacles_number: int = 100,
        obstacle_width: float = 1.0,
        obstacle_length: float = None,
        obstacle_height: float = 10.0,
        **kwargs,
    ):
        drone_model, physics, initial_xyzs, end_xyzs, initial_rpys = sanitize_init_params(
            drone_model, physics, initial_xyzs, end_xyzs, initial_rpys
        )
        self.config_validator = ConfigValidator(expected_obstacle_shape=ObstacleShape.CYLINDER)
        self.config_validator.validate(
            initial_xyzs,
            end_xyzs,
            drone_number,
            obstacles_number,
            obstacle_width,
            obstacle_length,
            obstacle_height,
            track_length,
            track_width,
            track_height,
            ground_position,
            drone_model
        )

        self.initial_xyzs = initial_xyzs
        self.end_xyzs = end_xyzs
        self.ground_position = ground_position
        self.track_length = track_length
        self.track_width = track_width
        self.track_height = track_height
        self.obstacles_number = obstacles_number
        self.obstacle_width = obstacle_width
        self.obstacle_height = obstacle_height
    
        super().__init__(
            drone_model=drone_model,
            physics=physics,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            ground_position=ground_position,
            obstacles=True,
            **kwargs,
        )

    def _generate_world_def(self):
        return generate_world_boundaries(self.track_width, self.track_length, self.track_height, self.ground_position)

    def _create_forrest(self, obstacles: np.ndarray):
        for i in range(obstacles.shape[0]):
            obstacle = obstacles[i, :]
            self._create_tree(obstacle)

    def _create_tree(self, obstacle: np.ndarray):
        x, y, z = obstacle[0], obstacle[1], obstacle[2]
        radius, height = obstacle[3], obstacle[4]
        color = [0.5, 0.8, 0.3, 1.0]  # Green color for the tree
                
        base_z = z + height/2

        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height,
        )

        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color
        )

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, base_z],
        )

    def generate_obstacles(self) -> ObstaclesData:
        return generate_obstacles(
            self._generate_world_def(),
            n_obstacles=self.obstacles_number,
            shape_type=ObstacleShape.Cylinder,
            placement_strategy=strategy_random_uniform,
            size_params={'radius': self.obstacle_width, 'height': self.obstacle_height},
            start_positions=positions_to_tensor(self.initial_xyzs),
            target_positions=positions_to_tensor(self.end_xyzs)
        )
    
    def draw_obstacles(self, obstacles: ObstaclesData) -> None:
        print("[DEBUG] Generating forrest environment...") 
        self._init_render_silently()
        self._setup_environment(self.track_length, self.track_width, self.track_height, ground_color=[0.4, 0.4, 0.45, 1.0])
        self._create_forrest(obstacles.data)
        print("[DEBUG] Forrest environment generated.")
