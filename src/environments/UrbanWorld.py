import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from src.environments.SwarmBaseWorld import SwarmBaseWorld
from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.utils.config_parser import sanitize_init_params


class UrbanWorld(SwarmBaseWorld):
    def __init__(
        self,
        world_data: WorldData,        # ← wymiary świata
        obstacles_data: ObstaclesData, # ← dane o przeszkodach
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

        self.end_xyzs = end_xyzs

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
    # Implementacja rysowania przeszkód                                  #
    # ------------------------------------------------------------------ #
    def draw_obstacles(self):
        print("[DEBUG]: Drawing obstacles")
        self._create_city(self.obstacles.data) 
      
    # ------------------------------------------------------------------ #
    # Geometria przeszkód                                                #
    # ------------------------------------------------------------------ #

    def _create_building(self, obstacle: np.ndarray) -> None:
        shade = np.random.uniform(0.6, 0.9) # Zmienna losowa niekontorlowana seedem - to tylko kolor
        x, y, z = obstacle[0], obstacle[1], obstacle[2]
        length, width, height = obstacle[3], obstacle[4], obstacle[5]
        base_z = z + height / 2
        print("[DEBUG]: x,y,z", x, y, z)
        print("[DEBUG]: length, width, height", length, width, height)


        col_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[length/2, width/2, height/2]
        )
        vis_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[length/2, width/2, height/2],
            rgbaColor=[shade, shade, shade, 1]
        )
        print("[DEBUG]: Tworzenie przeszkody: ", vis_shape, col_shape)
        p.createMultiBody(0, col_shape, vis_shape, [x, y, base_z])


    def _create_city(self, obstacles: np.ndarray) -> None:
        for i in range(obstacles.shape[0]):
            self._create_building(obstacles[i, :])