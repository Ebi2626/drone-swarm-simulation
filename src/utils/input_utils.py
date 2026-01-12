import pybullet as p
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

class CommandType(Enum):
    TOGGLE_SIMULATION = auto()
    SWITCH_DRONE_CAMERA = auto()
    EXIT = auto()

@dataclass
class SimulationCommand:
    type: CommandType
    payload: Optional[int] = None

class InputHandler:
    def __init__(self, num_drones: int):
        self.num_drones = num_drones

    def get_command(self) -> Optional[SimulationCommand]:
        keys = p.getKeyboardEvents()
        
        if ord(' ') in keys and (keys[ord(' ')] & p.KEY_WAS_TRIGGERED):            
            return SimulationCommand(CommandType.TOGGLE_SIMULATION)

        for k, v in keys.items():
            if (v & p.KEY_WAS_TRIGGERED):
                if ord('0') < k < ord('9'):
                    selected_id = k - ord('1')
                    if selected_id < self.num_drones:
                        return SimulationCommand(
                            CommandType.SWITCH_DRONE_CAMERA, 
                            payload=selected_id
                        )
                if k == ord('0'):
                    if self.num_drones >= 10:
                        return SimulationCommand(
                            CommandType.SWITCH_DRONE_CAMERA,
                            payload=9
                        )
        return None
