from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import EvasionContext
from src.trajectory.BSplineTrajectory import BSplineTrajectory
from src.utils.optimization_metrics import OnlineOptimizationRecord

@dataclass
class EvasionPlan:
    """Wynik planowania uniku: lokalny spline + informacja gdzie wracamy na bazową trasę."""
    evasion_spline: BSplineTrajectory
    rejoin_point: np.ndarray
    rejoin_base_arc: float
    preferred_axis: str
    astar_success: bool = True
    fallback_used: bool = False
    planning_wall_time_s: float = 0.0


class BaseAvoidance(ABC):
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
        # Common-contract: convergence trace ostatniego trigger'a (per-generation
        # best_fitness). Pusta gdy optimizer nie wystawił trace'u (np. AStar).
        # SwarmFlightController czyta to po `compute_evasion_plan` i przekazuje
        # do `SimulationLogger.log_convergence_trace`.
        self.last_convergence_trace: List[float] = []

    @abstractmethod
    def compute_evasion_plan(
        self, context: EvasionContext
    ) -> Tuple[EvasionPlan | None, OnlineOptimizationRecord]:
        """
        Rozwiązuje zunifikowany problem optymalizacji trajektorii uniku.

        Common-contract (Krok 3.3 plan.md): KAŻDY plug-in MUSI zwrócić tuple
        `(EvasionPlan | None, OnlineOptimizationRecord)`. Rekord wypełnia
        identyfikację (run_id MOŻE być pusty — uzupełnia integrator), grupy A
        (optimizer summary) i B (decision; sentinele NaN/"" gdy plan=None).
        Grupę D (outcome) wypełnia `update_online_optimization_outcome` po
        BLEND_END / collision.

        :param context: Zunifikowany obiekt EvasionContext.
        :return: krotka `(plan, record)`. `plan is None` gdy optimizer nie
            znalazł bezpiecznego rozwiązania — `record.status` opisuje powód.
        """
        pass
