from enum import Enum

class ObstacleShape(str, Enum):
    """
    Enum representing different shapes of obstacles.
    """
    BOX = "BOX"           # Required: length, width, height
    CYLINDER = "CYLINDER" # Required: width (as a radius), height, (length is ignored)