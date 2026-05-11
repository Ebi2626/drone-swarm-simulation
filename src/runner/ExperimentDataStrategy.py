from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

from src.utils.SeedRegistry import SeedRegistry

if TYPE_CHECKING:
    from main import ExperimentRunner

class ExperimentDataStrategy(ABC):
    """Strategia przygotowania danych eksperymentu (Strategy Pattern).

    Konkretne implementacje (`GenerationDataStrategy`, `ReplayDataStrategy`)
    inicjalizują `runner.world_data`, `runner.obstacles_data` i
    `runner.drones_trajectories` z odpowiedniego źródła.
    """

    @abstractmethod
    def prepare_data(self, runner: "ExperimentRunner", seeds: SeedRegistry):
        """Wypełnij stan `runner` danymi świata/przeszkód/trajektorii dronów.

        Args:
            runner: Główny `ExperimentRunner` — implementacja modyfikuje
                jego pola in-place.
            seeds: Rejestr ziaren dla reprodukowalności.
        """
        pass