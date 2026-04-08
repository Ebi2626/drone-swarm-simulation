from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    def __init__(self, parent, num_drones, params=None):
        self.parent = parent
        self.num_drones = num_drones
        self.params = params or {}

    @abstractmethod
    def compute_actions(self, current_states, current_time):
        pass
