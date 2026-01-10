import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from src.simulation.worlds.SwarmBaseWorld import SwarmBaseWorld


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
                 obstacles: bool = False,
                 user_debug_gui: bool = True
                 ):
        
        # --- FIX: Konwersja String -> Enum ---
        # Jeśli Hydra przysłała string (np. "CF2X"), zamień go na DroneModel.CF2X
        if isinstance(drone_model, str):
            # Używamy getattr, aby dynamicznie pobrać wartość z klasy Enum
            # np. DroneModel['CF2X'] lub getattr(DroneModel, 'CF2X')
            try:
                drone_model = DroneModel[drone_model]
            except KeyError:
                # Fallback, jeśli nazwa jest niepoprawna (opcjonalnie)
                print(f"Warning: Unknown drone model '{drone_model}', defaulting to CF2X")
                drone_model = DroneModel.CF2X

        if isinstance(physics, str):
            try:
                physics = Physics[physics] 
            except KeyError:
                print(f"Warning: Unknown physics engine '{physics}', defaulting to PYB")
                physics = Physics.PYB
        # -------------------------------------

        self.flight_duration_sec = 120
        
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
