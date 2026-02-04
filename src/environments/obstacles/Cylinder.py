from src.environments.obstacles.Obstacle import Obstacle

class Cylinder(Obstacle):
    def __init__(self, position, radius, height, color=(0.5, 0.5, 0.5)):
        super().__init__("CYLINDER", position, color)
        self.radius = radius
        self.height = height
        self.dimensions = (radius, height)