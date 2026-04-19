from gym_pybullet_drones.utils.enums import DroneModel, Physics
from src.environments.SwarmBaseWorld import SwarmBaseWorld
from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.utils.config_parser import sanitize_init_params

class EmptyWorld(SwarmBaseWorld):
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

        
    def draw_obstacles(self) -> None:
        print("[DEBUG] Generating empty environment...") 