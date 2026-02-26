from src.environments.obstacles.ObstacleShape import ObstacleShape
from src.utils.ValidationMessage import ValidationMessage
from gym_pybullet_drones.utils.enums import DroneModel

class ConfigValidator():
    def __init__(
            self,
            expected_obstacle_shape: ObstacleShape
    ):
        self.expected_obstacle_shape = expected_obstacle_shape

    def _validate_initial_positions(self, initial_xyzs, drone_number):
        number_of_initial_positions = len(initial_xyzs)
        if  number_of_initial_positions != drone_number:
            raise ValueError(
                ValidationMessage.INVALID_INITIAL_POINTS.format(number_of_initial_positions, drone_number)
            )

    def _validate_end_positions(self, end_xyzs, drone_number):
        number_of_end_positions = len(end_xyzs)
        if  number_of_end_positions != drone_number:
            raise ValueError(
                ValidationMessage.INVALID_END_POINTS.format(number_of_end_positions, drone_number)
            )

    def _validate_obstacles_parameters(self, obstacles_number, obstacle_width, obstacle_length, obstacle_height):
        if obstacles_number == 0:
            raise ValueError(
                ValidationMessage.INVALID_OBSTACLE_AMOUNT.format(obstacles_number)
            )
        if (obstacle_width is None or obstacle_length is None) and self.expected_obstacle_shape is ObstacleShape.BOX:
            raise ValueError(
                ValidationMessage.MISSING_SHAPE_DIMENSION.format("width/length", self.expected_obstacle_shape)
            )
        if self.expected_obstacle_shape is ObstacleShape.CYLINDER and obstacle_length is not None:
            raise ValueError(
                ValidationMessage.TOO_MUCH_DIMENSIONS_FOR_CYLINDER.format(obstacle_length)
            )
        if obstacle_height is None:
            raise ValueError(
                ValidationMessage.MISSING_HEIHGT
            )

    def _validate_world_boundaries_parameters(self, track_length, track_width, track_height, ground_position):
        if track_width is None:
            raise ValueError(
                ValidationMessage.LACK_OF_TRACK_DIMENSION.format("width")
            )
        if track_height is None:
            raise ValueError(
                ValidationMessage.LACK_OF_TRACK_DIMENSION.format("height")
            )
        if track_length is None:
            raise ValueError(
                ValidationMessage.LACK_OF_TRACK_DIMENSION.format("length")
            )
        if ground_position is None:
            raise ValueError(
                ValidationMessage.MISSING_GROUND_POSITION
            )
        if ground_position < 0 or ground_position > 1:
            raise ValueError(
                ValidationMessage.INVALID_GROUND_POSITION
                .format(ground_position)
            )

    def _validate_drone_parameters(self, drone_model):
        if not isinstance(drone_model, DroneModel):
            raise ValueError(
                ValidationMessage.INVALID_DRONE_TYPE.format(drone_model)
            )

    def validate(
            self,
            initial_xyzs,
            end_xyzs,
            drone_number,
            obstacles_number,
            obstacle_width,
            obstacle_length,
            obstacle_height,
            track_length,
            track_width,
            track_height,
            ground_position,
            drone_model,
    ):
        self._validate_initial_positions(initial_xyzs, drone_number)
        self._validate_end_positions(end_xyzs, drone_number)
        self._validate_obstacles_parameters(obstacles_number, obstacle_width, obstacle_length, obstacle_height)
        self._validate_world_boundaries_parameters(obstacles_number, obstacle_width, obstacle_length, obstacle_height)
        self._validate_world_boundaries_parameters(track_length, track_width, track_height, ground_position)
        self._validate_drone_parameters(drone_model)
        

