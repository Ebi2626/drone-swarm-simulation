"""Testy common-contract `OnlineOptimizationRecord` (Krok 3.3 plan.md).

Każde wywołanie `BaseAvoidance.compute_evasion_plan` MUSI zwrócić tuple
`(EvasionPlan | None, OnlineOptimizationRecord)`. Rekord ma być wypełniony
identyfikacją + grupami A i B niezależnie czy plan się powiódł.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.algorithms.avoidance.GenericOptimizingAvoidance import (
    GenericOptimizingAvoidance,
)
from src.algorithms.avoidance.budget import TimeBudget
from src.algorithms.avoidance.fitness.WeightedSumFitness import WeightedSumFitness
from src.algorithms.avoidance.interfaces import OptimizationResult
from src.algorithms.avoidance.path.AxisChooser import AxisChooser
from src.algorithms.avoidance.path.SingleArcDeflection import SingleArcDeflection
from src.algorithms.avoidance.predictors.ConstantVelocityPredictor import (
    ConstantVelocityPredictor,
)
from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import (
    EvasionContext,
    KinematicState,
    ThreatAlert,
)
from src.utils.optimization_metrics import (
    OUTCOME_PENDING,
    OnlineOptimizationRecord,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def head_on_context() -> EvasionContext:
    drone_state = KinematicState(
        position=np.array([0.0, 0.0, 5.0]),
        velocity=np.array([5.0, 0.0, 0.0]),
        radius=0.4,
    )
    obs_state = KinematicState(
        position=np.array([10.0, 0.0, 5.0]),
        velocity=np.array([-5.0, 0.0, 0.0]),
        radius=0.5,
    )
    threat = ThreatAlert(
        obstacle_state=obs_state,
        distance=10.0,
        time_to_collision=1.0,
        relative_velocity=np.array([10.0, 0.0, 0.0]),
    )
    base_spline = MagicMock()
    base_spline.profile.cruise_speed = 5.0
    base_spline.profile.max_accel = 2.0
    base_spline.cruise_speed = 5.0
    base_spline.max_accel = 2.0
    base_spline.arc_length = 100.0
    return EvasionContext(
        drone_id=7,
        current_time=12.5,
        drone_state=drone_state,
        threat=threat,
        base_spline=base_spline,
        rejoin_point=np.array([20.0, 0.0, 5.0]),
        rejoin_base_arc=20.0,
        world_bounds=(np.array([0.0, -10.0, 0.0]), np.array([30.0, 10.0, 10.0])),
        search_space_min=np.array([0.0, -8.0, 1.0]),
        search_space_max=np.array([30.0, 8.0, 9.0]),
    )


class _StubOptimizer:
    """Optimizer-stub o sztywnym wyniku — pozwala kontrolować rekord."""

    def __init__(self, result: OptimizationResult) -> None:
        self._result = result

    def optimize(self, problem, budget):
        return self._result


def _make_avoidance(optimizer: _StubOptimizer) -> GenericOptimizingAvoidance:
    return GenericOptimizingAvoidance(
        name="test",
        predictor=ConstantVelocityPredictor(),
        path_representation=SingleArcDeflection(
            axis_chooser=AxisChooser(), min_applied_cruise_ratio=0.0,
        ),
        fitness=WeightedSumFitness(),
        optimizer=optimizer,
        time_budget_s=0.5,
    )


# --------------------------------------------------------------------------- #
# Contract tests                                                              #
# --------------------------------------------------------------------------- #


class TestComputeEvasionPlanReturnsTuple:
    """Niezależnie czy plan jest, zawsze zwraca tuple `(plan, record)`."""

    def test_returns_tuple_of_plan_and_record(self, head_on_context) -> None:
        # Optimizer stub: status=failed → plan będzie None, ale tuple zwracana.
        opt = _StubOptimizer(OptimizationResult(
            waypoints=None, elapsed_s=0.01, status="failed",
            extra={"reason": "stubbed_failure", "algorithm": "STUB"},
        ))
        avoidance = _make_avoidance(opt)
        result = avoidance.compute_evasion_plan(head_on_context)
        assert isinstance(result, tuple)
        assert len(result) == 2
        plan, record = result
        assert plan is None
        assert isinstance(record, OnlineOptimizationRecord)

    def test_record_has_identification_and_group_a(self, head_on_context) -> None:
        opt = _StubOptimizer(OptimizationResult(
            waypoints=None, elapsed_s=0.02, status="failed",
            extra={
                "algorithm": "FAKE",
                "reason": "stub_no_feasible",
                "best_fitness": 99.5,
                "evaluations_completed": 42,
                "generations_completed": 7,
            },
        ))
        avoidance = _make_avoidance(opt)
        _, rec = avoidance.compute_evasion_plan(head_on_context)
        assert rec.drone_id == 7
        assert rec.trigger_time == 12.5
        assert rec.algorithm == "FAKE"
        assert rec.status == "failed"
        assert rec.reason == "stub_no_feasible"
        assert rec.best_fitness == 99.5
        assert rec.evaluations_completed == 42
        assert rec.generations_completed == 7
        assert rec.time_budget_s == 0.5

    def test_failed_plan_has_sentinel_group_b(self, head_on_context) -> None:
        opt = _StubOptimizer(OptimizationResult(
            waypoints=None, elapsed_s=0.0, status="failed",
            extra={"algorithm": "X", "reason": "fail"},
        ))
        avoidance = _make_avoidance(opt)
        _, rec = avoidance.compute_evasion_plan(head_on_context)
        assert rec.chosen_axis == ""
        assert rec.plan_waypoints_json == ""
        assert np.isnan(rec.plan_total_duration_s)
        assert np.isnan(rec.plan_arc_length_m)

    def test_pending_outcome_until_finalized(self, head_on_context) -> None:
        opt = _StubOptimizer(OptimizationResult(
            waypoints=None, elapsed_s=0.0, status="failed",
            extra={"algorithm": "X", "reason": "fail"},
        ))
        avoidance = _make_avoidance(opt)
        _, rec = avoidance.compute_evasion_plan(head_on_context)
        assert rec.outcome == OUTCOME_PENDING


class TestConvergenceTraceCapture:
    def test_trace_populated_from_optimizer_extra(self, head_on_context) -> None:
        opt = _StubOptimizer(OptimizationResult(
            waypoints=None, elapsed_s=0.0, status="failed",
            extra={
                "algorithm": "MOCK", "reason": "fail",
                "convergence_trace": [10.0, 8.0, 5.0, 3.0],
            },
        ))
        avoidance = _make_avoidance(opt)
        avoidance.compute_evasion_plan(head_on_context)
        assert avoidance.last_convergence_trace == [10.0, 8.0, 5.0, 3.0]

    def test_trace_empty_when_optimizer_omits_it(self, head_on_context) -> None:
        opt = _StubOptimizer(OptimizationResult(
            waypoints=None, elapsed_s=0.0, status="failed",
            extra={"algorithm": "X", "reason": "fail"},
        ))
        avoidance = _make_avoidance(opt)
        avoidance.compute_evasion_plan(head_on_context)
        assert avoidance.last_convergence_trace == []

    def test_trace_reset_between_triggers(self, head_on_context) -> None:
        # First trigger has trace.
        opt = _StubOptimizer(OptimizationResult(
            waypoints=None, elapsed_s=0.0, status="failed",
            extra={"algorithm": "X", "reason": "fail",
                   "convergence_trace": [1.0, 2.0]},
        ))
        avoidance = _make_avoidance(opt)
        avoidance.compute_evasion_plan(head_on_context)
        assert avoidance.last_convergence_trace == [1.0, 2.0]

        # Swap to optimizer without trace; trace should be reset.
        opt2 = _StubOptimizer(OptimizationResult(
            waypoints=None, elapsed_s=0.0, status="failed",
            extra={"algorithm": "X", "reason": "fail"},
        ))
        avoidance.optimizer = opt2
        avoidance.compute_evasion_plan(head_on_context)
        assert avoidance.last_convergence_trace == []


class TestRunIdInjection:
    """`run_id` jest wstrzykiwane przez integrator (SwarmFlightController),
    avoidance trzyma pole settable. Pusty default jest OK."""

    def test_default_run_id_empty(self, head_on_context) -> None:
        opt = _StubOptimizer(OptimizationResult(
            waypoints=None, elapsed_s=0.0, status="failed",
            extra={"algorithm": "X", "reason": "fail"},
        ))
        avoidance = _make_avoidance(opt)
        _, rec = avoidance.compute_evasion_plan(head_on_context)
        assert rec.run_id == ""

    def test_settable_run_id(self, head_on_context) -> None:
        opt = _StubOptimizer(OptimizationResult(
            waypoints=None, elapsed_s=0.0, status="failed",
            extra={"algorithm": "X", "reason": "fail"},
        ))
        avoidance = _make_avoidance(opt)
        avoidance.run_id = "run_42"
        _, rec = avoidance.compute_evasion_plan(head_on_context)
        assert rec.run_id == "run_42"
