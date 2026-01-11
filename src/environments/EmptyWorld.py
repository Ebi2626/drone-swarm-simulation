import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from src.environments.SwarmBaseWorld import SwarmBaseWorld
from src.utils.config_parser import sanitize_init_params

class EmptyWorld(SwarmBaseWorld):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 48,
                 gui: bool = True,
                 record: bool = False,
                 obstacles: bool = True,
                 user_debug_gui: bool = True
                 ):
        
        drone_model, physics, initial_xyzs, initial_rpys = sanitize_init_params(
            drone_model, physics, initial_xyzs, initial_rpys
        )
        
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui
                         )
        
    def _addObstacles(self):
        print("[DEBUG] Setting up empty world...")
        self._init_render_silently()

        self._setup_environment(length=500.0, width=500.0, ceiling_height=0.0, ground_color=[0.5, 0.5, 0.55, 1.0])

        print("[DEBUG] Empty world setup complete.")