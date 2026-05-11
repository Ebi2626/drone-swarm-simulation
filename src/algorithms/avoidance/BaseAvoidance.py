"""Bazowy interfejs strategii uniku online dla `SwarmFlightController`.

`EvasionPlan` jest wynikiem planowania, `BaseAvoidance` definiuje abstrakcyjny
kontrakt `compute_evasion_plan(context) → (plan, record)`. Konkretne
implementacje (np. `GenericOptimizingAvoidance`) używają sub-strategii
`IObstaclePredictor`, `IPathRepresentation`, `IFitnessEvaluator`,
`IPathOptimizer` (zob. `interfaces.py`).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import EvasionContext
from src.trajectory.BSplineTrajectory import BSplineTrajectory
from src.utils.optimization_metrics import OnlineOptimizationRecord


@dataclass
class EvasionPlan:
    """Wynik planowania uniku zwracany przez `BaseAvoidance.compute_evasion_plan`.

    Pola:
        evasion_spline: Lokalny B-spline manewru uniku.
        rejoin_point: `(3,)` punkt powrotu na trasę bazową [m].
        rejoin_base_arc: Długość łuku bazowej trasy w punkcie powrotu [m].
        preferred_axis: Oś manewru — `'right' | 'left' | 'up' | 'down'`.
        fallback_used: `True`, gdy planer użył ścieżki awaryjnej (jakość
            niższa, ale dostępna w czasie).
        planning_wall_time_s: Czas wallclock planowania [s].
    """
    evasion_spline: BSplineTrajectory
    rejoin_point: np.ndarray
    rejoin_base_arc: float
    preferred_axis: str
    fallback_used: bool = False
    planning_wall_time_s: float = 0.0


class BaseAvoidance(ABC):
    """Abstrakcyjny kontrakt strategii uniku online.

    Atrybuty publiczne:
        name: Krótki identyfikator strategii (do logów).
        params: Dowolny słownik konfiguracji wstrzykiwany przez Hydra.
        last_convergence_trace: Per-generation `best_fitness` ostatniego
            wywołania `compute_evasion_plan` (czytany przez
            `SwarmFlightController` i przekazywany do `SimulationLogger`).
    """

    def __init__(self, name: str, **kwargs):
        """Zapamiętaj nazwę i parametry strategii.

        Args:
            name: Krótka etykieta (np. `"NSGA3Avoidance"`).
            **kwargs: Parametry konfiguracji — trafiają do `self.params`,
                czytane m.in. przez `SwarmFlightController` (`trigger_ttc`,
                `evasion_time_min`, `rejoin_arc_distance_m`, …).
        """
        self.name = name
        self.params = kwargs
        self.last_convergence_trace: List[float] = []

    @abstractmethod
    def compute_evasion_plan(
        self, context: EvasionContext
    ) -> Tuple[EvasionPlan | None, OnlineOptimizationRecord]:
        """Rozwiąż problem planowania trajektorii uniku dla pojedynczego trigger'a.

        Każda implementacja MUSI zwrócić krotkę `(plan, record)`. Rekord
        wypełnia identyfikację (run_id można zostawić pusty — uzupełni
        integrator), grupę A (optimizer summary) i B (decision; sentinele
        `NaN`/`""` gdy `plan is None`). Grupę D (outcome) wypełnia później
        `update_online_optimization_outcome` po BLEND_END lub kolizji.

        Args:
            context: `EvasionContext` z aktualnym stanem drona, zagrożeniem
                i geometrią świata (zob. `ThreatAnalyzer.py`).

        Returns:
            Krotka `(plan, record)`:
            - `plan is None` ⇒ optymalizator nie znalazł bezpiecznego
              rozwiązania (powód w `record.status`).
            - `record` zawsze niepusty — używany do logu nawet gdy plan brak.
        """
        pass
