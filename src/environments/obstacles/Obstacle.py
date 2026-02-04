class Obstacle:
    def __init__(self, type, position, color=(0.5, 0.5, 0.5), dimensions=None):
        self.position = position
        self.color = color
        self.type = type
        if dimensions is not None:
            self.dimensions = dimensions