from __future__ import annotations

import logging
import time

from src.algorithms.avoidance.BaseAvoidance import BaseAvoidance, EvasionPlan
from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import EvasionContext
from src.algorithms.avoidance.budget import (
    HardDeadlineExceeded,
    TimeBudget,
    hard_deadline,
)
from src.algorithms.avoidance.interfaces import (
    IFitnessEvaluator,
    IObstaclePredictor,
    IPathOptimizer,
    IPathRepresentation,
    PathProblem,
)

logger = logging.getLogger(__name__)


class GenericOptimizingAvoidance(BaseAvoidance):
    """Strategy implementujący `BaseAvoidance` przez kompozycję 4 sub-strategii.

    Architektura zgodna ze specyfikacją z `plan.md` (Faza 1):
      - `IObstaclePredictor` — model przyszłej pozycji przeszkody.
      - `IPathRepresentation` — konwersja waypointów (lub genów) na BSpline.
      - `IFitnessEvaluator` — koszt / ocena trajektorii (dla AStara: tylko axis_score).
      - `IPathOptimizer` — silnik wybierający waypointy (AStar w Fazie 1;
        NSGA-III/MSFFOA/SSA/OOA w Fazie 2).

    Wszystkie sub-strategy są instancjowane przez Hydrę (`_target_` w yaml),
    więc podmiana algorytmu nie wymaga ingerencji w żaden plik .py.

    Mechanizm budżetu (zob. `plan.md` → Engineering Notes):
      - Cooperative `TimeBudget` przekazywany do `optimize()` — primary defense.
      - SIGALRM `hard_deadline(time_budget_s × hard_kill_factor)` jako outer
        circuit breaker — odpala się gdy cooperative zawiedzie (bug w optimizerze,
        deadlock w native code). Po stripowaniu zwracamy fallback bez planu
        (TRACKING kontynuuje), eksperyment się nie wiesza.
    """

    def __init__(
        self,
        predictor: IObstaclePredictor,
        path_representation: IPathRepresentation,
        fitness: IFitnessEvaluator,
        optimizer: IPathOptimizer,
        time_budget_s: float = 1.0,
        hard_kill_factor: float = 1.5,
        name: str = "Generic Optimizing Avoidance",
        **kwargs,
    ) -> None:
        # `**kwargs` trafia do `self.params` przez `BaseAvoidance.__init__` —
        # to tam `TrajectoryFollowingAlgorithm` szuka progów wyzwolenia
        # (`trigger_ttc`, `trigger_distance_base`, `evasion_time_min`,
        # `margin_velocity_gain`, `rejoin_arc_distance_m`, …). Patrz wywołania
        # `avoidance_algorithm.params.get(...)` w TrajectoryFollowingAlgorithm.py.
        super().__init__(name=name, **kwargs)
        self.predictor = predictor
        self.path_representation = path_representation
        self.fitness = fitness
        self.optimizer = optimizer
        self.time_budget_s = float(time_budget_s)
        self.hard_kill_factor = float(hard_kill_factor)

    def compute_evasion_plan(self, context: EvasionContext) -> EvasionPlan | None:
        t_plan_start = time.perf_counter()

        problem = PathProblem(
            context=context,
            predictor=self.predictor,
            fitness=self.fitness,
            path_repr=self.path_representation,
        )
        budget = TimeBudget.start_now(self.time_budget_s)
        hard_seconds = self.time_budget_s * self.hard_kill_factor

        # Outer circuit breaker — patrz `budget.hard_deadline` docstring.
        try:
            with hard_deadline(hard_seconds):
                result = self.optimizer.optimize(problem, budget)
        except HardDeadlineExceeded as e:
            wall = time.perf_counter() - t_plan_start
            logger.error(
                f"GenericOptimizingAvoidance: d{context.drone_id} — HARD DEADLINE "
                f"({hard_seconds:.3f}s) — circuit breaker zadziałał po {wall:.3f}s. "
                f"Optimizer ZIGNOROWAŁ cooperative budget (bug?). Plan: None. "
                f"Detal: {e}"
            )
            return None

        if result.status != "ok" or result.waypoints is None:
            wall = time.perf_counter() - t_plan_start
            logger.warning(
                f"GenericOptimizingAvoidance: d{context.drone_id} — optimizer "
                f"status={result.status} (elapsed={result.elapsed_s:.3f}s, "
                f"wall_total={wall:.3f}s). Plan: None. Extra: {result.extra}"
            )
            return None

        axis_chosen = result.extra.get("axis_chosen")
        spline = self.path_representation.waypoints_to_spline(
            result.waypoints, context, axis_name=axis_chosen
        )
        if spline is None:
            wall = time.perf_counter() - t_plan_start
            logger.warning(
                f"GenericOptimizingAvoidance: d{context.drone_id} — "
                f"path_representation zwrócił None (BSpline build / tangent reject). "
                f"Plan: None. Wall total: {wall:.3f}s."
            )
            return None

        wall_total = time.perf_counter() - t_plan_start
        return EvasionPlan(
            evasion_spline=spline,
            rejoin_point=context.rejoin_point,
            rejoin_base_arc=context.rejoin_base_arc,
            preferred_axis=axis_chosen if axis_chosen is not None else "unknown",
            astar_success=not bool(result.extra.get("fallback_used", False)),
            fallback_used=bool(result.extra.get("fallback_used", False)),
            planning_wall_time_s=float(wall_total),
        )
