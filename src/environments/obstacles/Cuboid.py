from src.environments.obstacles.Obstacle import Obstacle

class Cuboid(Obstacle):
    def __init__(self, position, width, length, height, color=(0.5, 0.5, 0.5)):
        super().__init__("CUBOID", position, color)
        self.width = width
        self.length = length
        self.height = height
        self.dimensions = (width, length, height)
