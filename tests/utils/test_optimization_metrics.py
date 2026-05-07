"""Testy common-contract dataclass dla metryki optymalizacji online."""
from __future__ import annotations

import json

import pytest

from src.utils.optimization_metrics import (
    OUTCOME_PENDING,
    OUTCOME_REJOINED_OK,
    ConvergenceSample,
    OnlineOptimizationRecord,
    convergence_sample_headers,
    online_record_headers,
    record_to_dict,
)


class TestOnlineOptimizationRecord:
    def test_minimal_record_construction_with_defaults(self) -> None:
        """Wymagane pola (identyfikacja + grupy A) wystarczają — grupa B/D ma defaults."""
        rec = OnlineOptimizationRecord(
            run_id="r1",
            drone_id=0,
            trigger_time=10.5,
            algorithm="SSA",
            status="ok",
            reason="ok",
            best_fitness=42.0,
            evaluations_completed=200,
            generations_completed=20,
            wallclock_s=0.5,
            time_budget_s=0.5,
        )
        assert rec.outcome == OUTCOME_PENDING
        assert rec.chosen_axis == ""

    def test_full_record_construction_all_fields(self) -> None:
        rec = OnlineOptimizationRecord(
            run_id="r1",
            drone_id=2,
            trigger_time=12.34,
            algorithm="NSGA3",
            status="ok",
            reason="ok",
            best_fitness=15.0,
            evaluations_completed=500,
            generations_completed=50,
            wallclock_s=0.49,
            time_budget_s=0.5,
            chosen_axis="right",
            plan_waypoints_json=json.dumps([[0, 0, 5], [5, 1, 5], [10, 0, 5]]),
            plan_total_duration_s=2.0,
            plan_arc_length_m=10.5,
            outcome=OUTCOME_REJOINED_OK,
            pos_err_at_rejoin_m=0.15,
            vel_err_at_rejoin_mps=0.5,
            time_to_rejoin_s=2.05,
        )
        assert rec.outcome == OUTCOME_REJOINED_OK
        assert rec.algorithm == "NSGA3"


class TestCommonContract:
    """Każdy z 4 algorytmów avoidance MUSI produkować rekordy z tym samym
    zestawem pól. Test waliduje że dataclass NIE pozwala na pominięcie pól
    wymaganych (Identyfikacja + grupa A) — kompilacyjny błąd przy missing arg.
    """

    @pytest.mark.parametrize("algorithm", ["SSA", "OOA", "MSFOA", "NSGA3"])
    def test_record_for_each_algorithm_has_identical_fields(
        self, algorithm: str
    ) -> None:
        rec = OnlineOptimizationRecord(
            run_id="r",
            drone_id=0,
            trigger_time=0.0,
            algorithm=algorithm,
            status="ok",
            reason="ok",
            best_fitness=0.0,
            evaluations_completed=0,
            generations_completed=0,
            wallclock_s=0.0,
            time_budget_s=0.5,
        )
        d = record_to_dict(rec)
        # Common contract: WSZYSTKIE klasy avoidance mają TE SAME pola.
        expected_keys = set(online_record_headers())
        assert set(d.keys()) == expected_keys

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(TypeError):
            OnlineOptimizationRecord(  # type: ignore[call-arg]
                run_id="r",
                drone_id=0,
                # missing trigger_time, algorithm, ...
            )


class TestConvergenceSample:
    def test_construction(self) -> None:
        s = ConvergenceSample(
            run_id="r",
            drone_id=0,
            trigger_time=10.0,
            algorithm="SSA",
            generation=5,
            best_fitness=33.3,
        )
        d = record_to_dict(s)
        assert set(d.keys()) == set(convergence_sample_headers())
        assert d["generation"] == 5
        assert d["best_fitness"] == 33.3


class TestHeaders:
    def test_online_record_headers_are_stable_order(self) -> None:
        # Pierwsze pola = identyfikacja (run_id, drone_id, trigger_time, algorithm).
        h = online_record_headers()
        assert h[:4] == ["run_id", "drone_id", "trigger_time", "algorithm"]
        # Outcome jest na końcu (wypełniany później).
        assert "outcome" in h
        assert "pos_err_at_rejoin_m" in h

    def test_convergence_sample_headers(self) -> None:
        h = convergence_sample_headers()
        assert h == [
            "run_id", "drone_id", "trigger_time", "algorithm",
            "generation", "best_fitness",
        ]
