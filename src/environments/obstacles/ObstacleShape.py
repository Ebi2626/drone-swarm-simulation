from enum import Enum

class ObstacleShape(str, Enum):
    """Typ kształtu przeszkody — `BOX` (length+width+height) lub `CYLINDER` (radius+height)."""
    BOX = "BOX"           # Required: length, width, height
    CYLINDER = "CYLINDER" # Required: width (as a radius), height, (length is ignored)