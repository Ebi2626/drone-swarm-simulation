from __future__ import annotations

import json
import logging
import time
from typing import Any, Tuple

import numpy as np

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
    OptimizationResult,
    PathProblem,
)
from src.utils.optimization_metrics import (
    OnlineOptimizationRecord,
)

logger = logging.getLogger(__name__)

_NAN = float("nan")


class GenericOptimizingAvoidance(BaseAvoidance):
    """Strategy implementujący `BaseAvoidance` przez kompozycję 4 sub-strategii:
      - `IObstaclePredictor` — model przyszłej pozycji przeszkody.
      - `IPathRepresentation` — konwersja waypointów (lub genów) na BSpline.
      - `IFitnessEvaluator` — koszt / ocena trajektorii.
      - `IPathOptimizer` — silnik wybierający waypointy (NSGA3/MSFFOA/SSA/OOA
        Online).

    Wszystkie sub-strategy są instancjowane przez Hydrę (`_target_` w yaml),
    więc podmiana algorytmu nie wymaga ingerencji w żaden plik .py.

    Mechanizm budżetu czasu:
      - Cooperative `TimeBudget` przekazywany do `optimize()` — primary defense.
      - SIGALRM `hard_deadline(time_budget_s × hard_kill_factor)` jako outer
        circuit breaker — odpala się gdy cooperative zawiedzie (bug w optimizerze,
        deadlock w native code). Po przerwaniu zwracamy fallback bez planu
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
        sampling_seed: int | None = None,
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
        # RNG dla pre-generacji populacji ceteris paribus. Wspólny Generator
        # gwarantuje identyczną sekwencję U(lb, ub) niezależnie od backendu
        # PRNG w mealpy (PCG64) / pymoo (MT19937) / custom MSFOA (PCG64).
        # Stan RNG awansuje między kolejnymi trigger'ami, więc każdy trigger
        # dostaje inną (ale deterministyczną) populację.
        self._sampling_rng = np.random.default_rng(sampling_seed)
        # Run-id wstrzykiwane przez integrator (SwarmFlightController). Default
        # pusty — common-contract dopuszcza, integrator MOŻE zostawić "" jeśli
        # nie ma centralnego rejestru runów. `analyze_online_optimization.py`
        # po prostu pominie filtrowanie po run_id.
        self.run_id: str = ""

    def compute_evasion_plan(
        self, context: EvasionContext
    ) -> Tuple[EvasionPlan | None, OnlineOptimizationRecord]:
        t_plan_start = time.perf_counter()
        # Trace zerowany na każdy trigger — optimizer.optimize podstawi swój
        # przez `result.extra["convergence_trace"]`.
        self.last_convergence_trace = []

        # Pre-generacja populacji ceteris paribus: jeśli optimizer deklaruje
        # population_size > 0, generujemy U(lb, ub) ze wspólnego RNG ZANIM
        # budżet czasu ruszy — koszt ~1 µs dla (20, 2).
        initial_pop = None
        pop_size = getattr(self.optimizer, "population_size", 0)
        if pop_size > 0:
            try:
                lb, ub = self.path_representation.gene_bounds(context)
                K = self.path_representation.gene_dim(context)
                initial_pop = (
                    lb[None, :]
                    + self._sampling_rng.uniform(0.0, 1.0, size=(pop_size, K))
                    * (ub - lb)[None, :]
                )
            except NotImplementedError:
                pass  # path_repr nie wspiera genów — pomijamy populację

        problem = PathProblem(
            context=context,
            predictor=self.predictor,
            fitness=self.fitness,
            path_repr=self.path_representation,
            initial_population=initial_pop,
        )
        budget = TimeBudget.start_now(self.time_budget_s)
        hard_seconds = self.time_budget_s * self.hard_kill_factor

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
            record = self._build_record(
                context=context,
                result=None,
                plan=None,
                wallclock_s=wall,
                fallback_status="failed",
                fallback_reason=f"hard_deadline_exceeded: {e}",
            )
            return None, record

        # Common-contract trace: optimizer może wystawić `convergence_trace`
        # w `extra` (lista per-gen best_fitness); brak ⇒ pusta lista.
        trace = list(result.extra.get("convergence_trace", []) or [])
        self.last_convergence_trace = [float(x) for x in trace]

        if result.status != "ok" or result.waypoints is None:
            wall = time.perf_counter() - t_plan_start
            logger.warning(
                f"GenericOptimizingAvoidance: d{context.drone_id} — optimizer "
                f"status={result.status} (elapsed={result.elapsed_s:.3f}s, "
                f"wall_total={wall:.3f}s). Plan: None. Extra: {result.extra}"
            )
            record = self._build_record(
                context=context,
                result=result,
                plan=None,
                wallclock_s=wall,
            )
            return None, record

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
            record = self._build_record(
                context=context,
                result=result,
                plan=None,
                wallclock_s=wall,
                fallback_status="failed",
                fallback_reason="waypoints_to_spline_returned_none",
            )
            return None, record

        wall_total = time.perf_counter() - t_plan_start
        plan = EvasionPlan(
            evasion_spline=spline,
            rejoin_point=context.rejoin_point,
            rejoin_base_arc=context.rejoin_base_arc,
            preferred_axis=axis_chosen if axis_chosen is not None else "unknown",
            fallback_used=bool(result.extra.get("fallback_used", False)),
            planning_wall_time_s=float(wall_total),
        )
        record = self._build_record(
            context=context,
            result=result,
            plan=plan,
            wallclock_s=wall_total,
        )
        return plan, record

    def _build_record(
        self,
        *,
        context: EvasionContext,
        result: OptimizationResult | None,
        plan: EvasionPlan | None,
        wallclock_s: float,
        fallback_status: str | None = None,
        fallback_reason: str | None = None,
    ) -> OnlineOptimizationRecord:
        """Konstruuje `OnlineOptimizationRecord` dla 1 trigger'a.

        Optimizer wystawia w `result.extra` common-contract pola:
        `algorithm`, `best_fitness`, `evaluations_completed`,
        `generations_completed`, `wallclock_s`, `reason`. Brakujące są wypełniane
        sentinelami (NaN / ""), co pozwala śledzić triggery które wyleciały
        z hard-deadline'em zanim optimizer zdążył zwrócić wynik.

        Grupa B (decision) wypełniana z `plan` lub sentinelami gdy plan=None.
        Grupa D (outcome) zostaje na `pending` — uzupełnia integrator.
        """
        extra: dict[str, Any] = result.extra if result is not None else {}

        algorithm = str(extra.get("algorithm", type(self.optimizer).__name__))
        status = fallback_status if fallback_status is not None else (
            result.status if result is not None else "failed"
        )
        reason = fallback_reason if fallback_reason is not None else str(
            extra.get("reason", "")
        )

        best_fitness = float(extra.get("best_fitness", _NAN))
        evaluations_completed = int(extra.get("evaluations_completed", 0))
        generations_completed = int(extra.get("generations_completed", 0))

        if plan is not None:
            spline = plan.evasion_spline
            # `plan.preferred_axis` ∈ {right, left, up, down}; magic "unknown"
            # (gdy optimizer nie wstawił `axis_chosen` w extra) normalizujemy
            # do "" → NULL po stronie ETL/CSV.
            chosen_axis = (
                str(plan.preferred_axis)
                if plan.preferred_axis in ("right", "left", "up", "down")
                else ""
            )
            try:
                wpts = np.asarray(spline.waypoints, dtype=float).tolist()
                plan_waypoints_json = json.dumps(wpts)
            except Exception:
                plan_waypoints_json = ""
            plan_total_duration_s = float(getattr(spline, "total_duration", _NAN))
            plan_arc_length_m = float(getattr(spline, "arc_length", _NAN))
        else:
            chosen_axis = ""
            plan_waypoints_json = ""
            plan_total_duration_s = _NAN
            plan_arc_length_m = _NAN

        return OnlineOptimizationRecord(
            run_id=self.run_id,
            drone_id=int(context.drone_id),
            trigger_time=float(context.current_time),
            algorithm=algorithm,
            status=status,
            reason=reason,
            best_fitness=best_fitness,
            evaluations_completed=evaluations_completed,
            generations_completed=generations_completed,
            wallclock_s=float(wallclock_s),
            time_budget_s=float(self.time_budget_s),
            chosen_axis=chosen_axis,
            plan_waypoints_json=plan_waypoints_json,
            plan_total_duration_s=plan_total_duration_s,
            plan_arc_length_m=plan_arc_length_m,
        )
