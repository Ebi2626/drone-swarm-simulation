import pybullet as p
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

class CommandType(Enum):
    """Komendy interaktywne z klawiatury PyBullet GUI."""
    TOGGLE_SIMULATION = auto()
    SWITCH_DRONE_CAMERA = auto()
    TOGGLE_LIDAR_RAYS = auto()
    EXIT = auto()


@dataclass
class SimulationCommand:
    """Komenda symulacji: typ + opcjonalny payload (np. indeks drona dla SWITCH)."""
    type: CommandType
    payload: Optional[int] = None


class InputHandler:
    """Mapuje klawisze PyBullet (`getKeyboardEvents`) na `SimulationCommand`."""

    def __init__(self, num_drones: int):
        """Skonfiguruj handler dla `num_drones` (limit klawiszy `1..9`/`0`)."""
        self.num_drones = num_drones

    def get_command(self) -> Optional[SimulationCommand]:
        """Sprawdź klawisze; zwróć `SimulationCommand` lub `None`, gdy nic nie naciśnięto."""
        keys = p.getKeyboardEvents()
        
        if ord(' ') in keys and (keys[ord(' ')] & p.KEY_WAS_TRIGGERED):            
            return SimulationCommand(CommandType.TOGGLE_SIMULATION)
        
        if (ord('l') in keys and (keys[ord('l')] & p.KEY_WAS_TRIGGERED)) or \
           (ord('L') in keys and (keys[ord('L')] & p.KEY_WAS_TRIGGERED)):
            return SimulationCommand(CommandType.TOGGLE_LIDAR_RAYS)

        for k, v in keys.items():
            if (v & p.KEY_WAS_TRIGGERED):
                if ord('0') < k <= ord('9'):
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
