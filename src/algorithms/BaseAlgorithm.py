from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    def __init__(self, num_drones, params=None):
        self.num_drones = num_drones
        self.params = params or {}

    @abstractmethod
    def compute_actions(self, current_states, current_time):
        pass
