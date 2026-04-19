import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from src.environments.SwarmBaseWorld import SwarmBaseWorld
from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.utils.config_parser import sanitize_init_params

class ForestWorld(SwarmBaseWorld):
    def __init__(
        self,
        world_data: WorldData,
        obstacles_data: ObstaclesData,
        num_drones: int | None = None,  # Złapanie zmiennej z Hydry chroni kwargs przed duplikatem
        drone_model: DroneModel = DroneModel.CF2X,
        physics: Physics = Physics.PYB,
        initial_xyzs=None,
        end_xyzs=None,
        initial_rpys=None,
        primary_num_drones: int | None = None,
        dynamic_obstacles_enabled: bool = False,
        num_dynamic_obstacles: int = 0,
        **kwargs
    ):
        drone_model, physics, initial_xyzs, end_xyzs, initial_rpys = sanitize_init_params(
            drone_model, physics, initial_xyzs, end_xyzs, initial_rpys
        )

        super().__init__(
            world_data=world_data,
            obstacles_data=obstacles_data,
            primary_num_drones=primary_num_drones,
            dynamic_obstacles_enabled=dynamic_obstacles_enabled,
            num_dynamic_obstacles=num_dynamic_obstacles,
            drone_model=drone_model,
            physics=physics,
            num_drones=len(initial_xyzs),
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            obstacles=True,
            **kwargs
        )

    # ------------------------------------------------------------------ #
    # Implementacja rysowania przeszkód (drzew)                          #
    # ------------------------------------------------------------------ #
    def draw_obstacles(self):
        print("[DEBUG]: Drawing obstacles")
        self._create_forrest(self.obstacles.data) 

    # ------------------------------------------------------------------ #
    # Geometria przeszkód (bez zmian)                                    #
    # ------------------------------------------------------------------ #

    def _create_forrest(self, obstacles: np.ndarray):
        for i in range(obstacles.shape[0]):
            self._create_tree(obstacles[i, :])

    def _create_tree(self, obstacle: np.ndarray):
        x, y, z = obstacle[0], obstacle[1], obstacle[2]
        radius, height = obstacle[3], obstacle[4]
        color = [0.5, 0.8, 0.3, 1.0]  # Zielony kolor dla drzewa
                
        base_z = z + height / 2

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