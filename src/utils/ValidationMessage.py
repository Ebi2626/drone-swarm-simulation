from enum import Enum

class ValidationMessage(str, Enum):
    # Obstacles errors
    INVALID_INITIAL_POINTS = "Number of initial positions ({}) does not match the number of drones ({})."
    INVALID_END_POINTS = "Number of end positions ({}) does not match the number of drones ({})."
    MISSING_SHAPE_DIMENSION = "Lack of dimension '{}' for obstacle shape '{}'."
    MISSING_HEIHGT = "Lack of obstacle height"
    INVALID_OBSTACLE_AMOUNT = "Number of obstacles ({}) must be an positive integer."
    TOO_MUCH_DIMENSIONS_FOR_CYLINDER = "Cylinder obstacle expects only two dimension (width/radius) and height. Given additional length ({})"

    # Track errors
    LACK_OF_TRACK_DIMENSION = "Lack of ({}) dimension for track."
    MISSING_GROUND_POSITION = "Lack of ground position"
    WRONG_GROUND_POSITION = "Incorrect ground position ({}). Expected value is a float number between 0 and 1"

    # Drone errors
    INVALID_DRONE_TYPE = "Invalid drone type ({}). Expected one of the following: 'cf2x', 'cf2p', 'racer'."
    
    def format(self, *args, **kwargs) -> str:
        """
        Nadpisujemy metodę format, aby można było łatwo wstrzykiwać parametry
        bezpośrednio na obiekcie Enum, zachowując jego wartość bazową.
        """
        return self.value.format(*args, **kwargs)