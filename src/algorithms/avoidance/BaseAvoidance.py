from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import EvasionContext
from src.trajectory.BSplineTrajectory import BSplineTrajectory
from src.utils.optimization_metrics import OnlineOptimizationRecord

@dataclass
class EvasionPlan:
    """Wynik planowania uniku: lokalny spline + punkt rejoin na bazową trasę."""
    evasion_spline: BSplineTrajectory
    rejoin_point: np.ndarray
    rejoin_base_arc: float
    preferred_axis: str
    fallback_used: bool = False
    planning_wall_time_s: float = 0.0


class BaseAvoidance(ABC):
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
        # Common-contract: convergence trace ostatniego trigger'a (per-generation
        # best_fitness). SwarmFlightController czyta to po `compute_evasion_plan`
        # i przekazuje do `SimulationLogger.log_convergence_trace`.
        self.last_convergence_trace: List[float] = []

    @abstractmethod
    def compute_evasion_plan(
        self, context: EvasionContext
    ) -> Tuple[EvasionPlan | None, OnlineOptimizationRecord]:
        """Rozwiązuje zunifikowany problem optymalizacji trajektorii uniku.

        Common-contract: KAŻDY plug-in MUSI zwrócić tuple `(EvasionPlan | None,
        OnlineOptimizationRecord)`. Rekord wypełnia identyfikację (run_id MOŻE
        być pusty — uzupełnia integrator), grupy A (optimizer summary) i B
        (decision; sentinele NaN/"" gdy plan=None). Grupę D (outcome) wypełnia
        `update_online_optimization_outcome` po BLEND_END / collision.

        Returns: `(plan, record)` gdzie `plan is None` ⇒ optimizer nie znalazł
        bezpiecznego rozwiązania (powód w `record.status`).
        """
        pass
