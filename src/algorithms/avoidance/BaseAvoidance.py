from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import EvasionContext
from src.trajectory.BSplineTrajectory import BSplineTrajectory

@dataclass
class EvasionPlan:
    """Wynik planowania uniku: lokalny spline + informacja gdzie wracamy na bazową trasę."""
    evasion_spline: BSplineTrajectory
    rejoin_point: np.ndarray          
    rejoin_base_arc: float            
    preferred_axis: str               


class BaseAvoidance(ABC):
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs

    @abstractmethod
    def compute_evasion_plan(self, context: EvasionContext) -> EvasionPlan | None:
        """
        Rozwiązuje zunifikowany problem optymalizacji trajektorii uniku.

        Dzięki hermetyzacji w obiekcie EvasionContext, algorytm nie musi samodzielnie
        wyliczać punktu powrotu (rejoin) ani granic przeszukiwania. Jego jedynym celem
        jest odnalezienie optymalnej, bezkolizyjnej ścieżki w podanej przestrzeni (search_space_min/max),
        która minimalizuje odchylenie od trasy bazowej i zużycie energii.

        :param context: Zunifikowany obiekt EvasionContext.
        :return: EvasionPlan lub None, jeżeli optymalizator nie znalazł bezpiecznego rozwiązania.
        """
        pass