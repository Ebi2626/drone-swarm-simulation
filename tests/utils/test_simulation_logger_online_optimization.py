"""Testy rozszerzeń SimulationLogger dla metryki optymalizacji online."""
from __future__ import annotations

import csv
import os
import tempfile

import pytest

from src.utils.optimization_metrics import (
    OUTCOME_COLLIDED_DRONE,
    OUTCOME_PENDING,
    OUTCOME_REJOINED_OK,
    OnlineOptimizationRecord,
)
from src.utils.SimulationLogger import SimulationLogger


@pytest.fixture
def logger():
    with tempfile.TemporaryDirectory() as tmp:
        yield SimulationLogger(
            output_dir=tmp, log_freq=10, ctrl_freq=100, num_drones=5
        )


def _make_record(drone_id: int = 0, trigger_time: float = 10.0,
                 algorithm: str = "SSA") -> OnlineOptimizationRecord:
    return OnlineOptimizationRecord(
        run_id="test",
        drone_id=drone_id,
        trigger_time=trigger_time,
        algorithm=algorithm,
        status="ok",
        reason="ok",
        best_fitness=42.0,
        evaluations_completed=100,
        generations_completed=10,
        wallclock_s=0.4,
        time_budget_s=0.5,
        chosen_axis="right",
        plan_waypoints_json="[[0,0,5],[10,0,5]]",
        plan_total_duration_s=2.0,
        plan_arc_length_m=10.0,
    )


class TestLogOnlineOptimizationTrigger:
    def test_appends_to_buffer(self, logger: SimulationLogger) -> None:
        rec = _make_record()
        logger.log_online_optimization_trigger(rec)
        assert len(logger.online_optimization_buffer) == 1
        assert logger.online_optimization_buffer[0]["drone_id"] == 0
        assert logger.online_optimization_buffer[0]["outcome"] == OUTCOME_PENDING

    def test_multiple_triggers_accumulate(self, logger: SimulationLogger) -> None:
        for i in range(3):
            logger.log_online_optimization_trigger(
                _make_record(drone_id=i, trigger_time=float(i))
            )
        assert len(logger.online_optimization_buffer) == 3


class TestUpdateOnlineOptimizationOutcome:
    def test_finds_pk_and_modifies_record(self, logger: SimulationLogger) -> None:
        logger.log_online_optimization_trigger(_make_record(drone_id=2, trigger_time=12.34))
        logger.update_online_optimization_outcome(
            drone_id=2, trigger_time=12.34,
            outcome=OUTCOME_REJOINED_OK,
            pos_err_at_rejoin_m=0.15,
            vel_err_at_rejoin_mps=0.3,
            time_to_rejoin_s=1.8,
        )
        rec = logger.online_optimization_buffer[0]
        assert rec["outcome"] == OUTCOME_REJOINED_OK
        assert rec["pos_err_at_rejoin_m"] == 0.15
        assert rec["time_to_rejoin_s"] == 1.8

    def test_pk_match_handles_floating_point_tolerance(
        self, logger: SimulationLogger
    ) -> None:
        logger.log_online_optimization_trigger(_make_record(trigger_time=10.0))
        # 10.0 + 1e-9 ≈ 10.0 (powinno znaleźć).
        logger.update_online_optimization_outcome(
            drone_id=0, trigger_time=10.0 + 1e-9,
            outcome=OUTCOME_REJOINED_OK,
        )
        assert logger.online_optimization_buffer[0]["outcome"] == OUTCOME_REJOINED_OK

    def test_no_match_logs_warning_does_not_crash(
        self, logger: SimulationLogger, capsys
    ) -> None:
        logger.log_online_optimization_trigger(_make_record(drone_id=0))
        logger.update_online_optimization_outcome(
            drone_id=99, trigger_time=99.9, outcome=OUTCOME_REJOINED_OK,
        )
        captured = capsys.readouterr()
        assert "brak match" in captured.out

    def test_collision_outcome(self, logger: SimulationLogger) -> None:
        logger.log_online_optimization_trigger(_make_record(drone_id=3, trigger_time=20.0))
        logger.update_online_optimization_outcome(
            drone_id=3, trigger_time=20.0, outcome=OUTCOME_COLLIDED_DRONE,
        )
        assert logger.online_optimization_buffer[0]["outcome"] == OUTCOME_COLLIDED_DRONE


class TestLogConvergenceTrace:
    def test_appends_n_rows(self, logger: SimulationLogger) -> None:
        trace = [100.0, 50.0, 25.0, 10.0, 5.0]
        logger.log_convergence_trace(
            run_id="r1", drone_id=0, trigger_time=10.0,
            algorithm="SSA", trace=trace,
        )
        assert len(logger.convergence_traces_buffer) == 5

    def test_generation_index_starts_at_zero(self, logger: SimulationLogger) -> None:
        logger.log_convergence_trace(
            run_id="r1", drone_id=0, trigger_time=10.0,
            algorithm="SSA", trace=[1.0, 2.0, 3.0],
        )
        gens = [r["generation"] for r in logger.convergence_traces_buffer]
        assert gens == [0, 1, 2]

    def test_empty_trace_appends_nothing(self, logger: SimulationLogger) -> None:
        logger.log_convergence_trace(
            run_id="r1", drone_id=0, trigger_time=10.0,
            algorithm="SSA", trace=[],
        )
        assert len(logger.convergence_traces_buffer) == 0


class TestSaveWritesCSVs:
    def test_save_writes_online_optimization_csv(
        self, logger: SimulationLogger
    ) -> None:
        logger.log_online_optimization_trigger(_make_record())
        logger.save()
        path = os.path.join(logger.output_dir, "online_optimization.csv")
        assert os.path.exists(path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["algorithm"] == "SSA"
        assert rows[0]["outcome"] == OUTCOME_PENDING

    def test_save_writes_convergence_traces_csv(
        self, logger: SimulationLogger
    ) -> None:
        logger.log_convergence_trace(
            run_id="r", drone_id=0, trigger_time=10.0,
            algorithm="SSA", trace=[5.0, 2.5, 1.0],
        )
        logger.save()
        path = os.path.join(logger.output_dir, "convergence_traces.csv")
        assert os.path.exists(path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert rows[2]["best_fitness"] == "1.0"

    def test_save_skips_csv_when_buffer_empty(
        self, logger: SimulationLogger
    ) -> None:
        logger.save()
        assert not os.path.exists(
            os.path.join(logger.output_dir, "online_optimization.csv")
        )
        assert not os.path.exists(
            os.path.join(logger.output_dir, "convergence_traces.csv")
        )

    def test_save_clears_buffers(self, logger: SimulationLogger) -> None:
        logger.log_online_optimization_trigger(_make_record())
        logger.log_convergence_trace(
            run_id="r", drone_id=0, trigger_time=10.0,
            algorithm="SSA", trace=[1.0],
        )
        logger.save()
        assert logger.online_optimization_buffer == []
        assert logger.convergence_traces_buffer == []
