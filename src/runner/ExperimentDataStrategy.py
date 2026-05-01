from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

from src.utils.SeedRegistry import SeedRegistry

if TYPE_CHECKING:
    from main import ExperimentRunner

class ExperimentDataStrategy(ABC):
    @abstractmethod
    def prepare_data(self, runner: "ExperimentRunner", seeds: SeedRegistry):
        pass