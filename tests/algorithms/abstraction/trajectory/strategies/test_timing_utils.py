"""Tests for the shared timing utility module."""

import time

import pytest

from src.algorithms.abstraction.trajectory.strategies.timing_utils import (
    TimingCollector,
    TimingRecord,
)


class TestTimingRecord:
    def test_is_frozen(self):
        rec = TimingRecord(
            algorithm_name="A",
            stage_name="s",
            wall_time_s=0.1,
            cpu_time_s=0.05,
            timestamp_utc="2026-01-01T00:00:00.000+00:00",
            success=True,
        )
        with pytest.raises(AttributeError):
            rec.wall_time_s = 999  # type: ignore[misc]


class TestTimingCollector:
    def test_measure_records_wall_and_cpu_time(self):
        tc = TimingCollector("TestAlgo")

        with tc.measure("sleep_stage"):
            time.sleep(0.05)

        assert len(tc.records) == 1
        rec = tc.records[0]
        assert rec.algorithm_name == "TestAlgo"
        assert rec.stage_name == "sleep_stage"
        assert rec.wall_time_s >= 0.04  # generous lower bound
        assert rec.cpu_time_s >= 0.0
        assert rec.success is True
        assert rec.timestamp_utc  # non-empty ISO string

    def test_measure_on_exception_records_failure(self):
        tc = TimingCollector("Fail")

        with pytest.raises(ValueError, match="boom"):
            with tc.measure("bad_stage"):
                raise ValueError("boom")

        assert len(tc.records) == 1
        assert tc.records[0].success is False
        assert tc.records[0].wall_time_s >= 0.0

    def test_multiple_stages(self):
        tc = TimingCollector("Multi")

        with tc.measure("a"):
            pass
        with tc.measure("b"):
            pass

        assert len(tc.records) == 2
        assert tc.records[0].stage_name == "a"
        assert tc.records[1].stage_name == "b"

    def test_total_wall_and_cpu_time(self):
        tc = TimingCollector("Sum")

        with tc.measure("x"):
            time.sleep(0.02)
        with tc.measure("y"):
            time.sleep(0.02)

        assert tc.total_wall_time_s >= 0.03
        assert tc.total_cpu_time_s >= 0.0

    def test_to_dicts(self):
        tc = TimingCollector("Dict")

        with tc.measure("stage1"):
            pass

        rows = tc.to_dicts()
        assert len(rows) == 1
        assert isinstance(rows[0], dict)
        assert rows[0]["algorithm_name"] == "Dict"
        assert rows[0]["stage_name"] == "stage1"
        assert "wall_time_s" in rows[0]
        assert "cpu_time_s" in rows[0]

    def test_summary_contains_algorithm_name(self):
        tc = TimingCollector("MySolver")

        with tc.measure("a"):
            pass

        s = tc.summary()
        assert "MySolver" in s
        assert "1 stage(s)" in s

    def test_empty_collector(self):
        tc = TimingCollector("Empty")
        assert tc.records == []
        assert tc.total_wall_time_s == 0.0
        assert tc.total_cpu_time_s == 0.0
        assert tc.to_dicts() == []

    def test_cpu_time_grows_under_cpu_work(self):
        tc = TimingCollector("CPU")

        with tc.measure("compute"):
            total = 0
            for i in range(200_000):
                total += i

        assert tc.records[0].cpu_time_s > 0.0
