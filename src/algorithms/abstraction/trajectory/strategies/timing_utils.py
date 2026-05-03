"""
Lightweight, shared timing utilities for trajectory optimization strategies.

Provides a context manager and a collector for measuring wall-clock and CPU
time of algorithm stages. Zero external dependencies (stdlib only), no global
state, no coupling to Hydra or simulation loggers.

Usage::

    timer = TimingCollector("MSFFOA")

    with timer.measure("optimization"):
        best, fitness = optimizer.optimize()

    with timer.measure("trajectory_reconstruction"):
        traj = optimizer.get_best_dense_trajectory()

    # Access results
    for rec in timer.records:
        print(f"{rec.stage_name}: {rec.wall_time_s:.3f}s wall, {rec.cpu_time_s:.3f}s cpu")

    # Export to list of dicts (e.g. for CSV serialization)
    rows = timer.to_dicts()
"""

from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List


@dataclass(frozen=True, slots=True)
class TimingRecord:
    """Single timing measurement."""

    algorithm_name: str
    stage_name: str
    wall_time_s: float
    cpu_time_s: float
    timestamp_utc: str
    success: bool


class TimingCollector:
    """Collects :class:`TimingRecord` entries for one algorithm run.

    Args:
        algorithm_name: Human-readable algorithm identifier
            (e.g. ``"NSGA-III"``, ``"MSFFOA"``).
    """

    def __init__(self, algorithm_name: str) -> None:
        self.algorithm_name = algorithm_name
        self.records: List[TimingRecord] = []

    @contextmanager
    def measure(
        self,
        stage_name: str,
        success: bool = True,
    ) -> Generator[None, None, None]:
        """Context manager that times the enclosed block.

        On normal exit ``success`` is taken from the explicit argument
        (default ``True``); if an exception propagates, ``success=False``
        regardless of the argument and the exception is **not** suppressed.

        Use ``success=False`` to mark *expected* unhappy-path stages such
        as fallback branches — those run cleanly (no exception) yet must
        not be treated as a successful optimization in downstream ETL.

        Args:
            stage_name: Label for the measured stage (e.g. ``"optimization"``).
            success: Outcome flag for normal exit. Pass ``False`` for
                fallback / known-failure stages.
        """
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        ok = success
        try:
            yield
        except BaseException:
            ok = False
            raise
        finally:
            wall_elapsed = time.perf_counter() - wall_start
            cpu_elapsed = time.process_time() - cpu_start
            self.records.append(
                TimingRecord(
                    algorithm_name=self.algorithm_name,
                    stage_name=stage_name,
                    wall_time_s=round(wall_elapsed, 6),
                    cpu_time_s=round(cpu_elapsed, 6),
                    timestamp_utc=ts,
                    success=ok,
                )
            )

    @property
    def total_wall_time_s(self) -> float:
        return sum(r.wall_time_s for r in self.records)

    @property
    def total_cpu_time_s(self) -> float:
        return sum(r.cpu_time_s for r in self.records)

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Return all records as plain dicts (CSV-friendly)."""
        return [asdict(r) for r in self.records]

    def save_csv(self, filepath: str) -> None:
        """Write all collected records to a CSV file.

        Does nothing when there are no records. The parent directory
        must already exist.

        Args:
            filepath: Absolute path to the output CSV file.
        """
        if not self.records:
            return

        fieldnames = [
            "algorithm_name",
            "stage_name",
            "wall_time_s",
            "cpu_time_s",
            "timestamp_utc",
            "success",
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.records:
                writer.writerow({
                    "algorithm_name": rec.algorithm_name,
                    "stage_name": rec.stage_name,
                    "wall_time_s": rec.wall_time_s,
                    "cpu_time_s": rec.cpu_time_s,
                    "timestamp_utc": rec.timestamp_utc,
                    "success": rec.success,
                })

    def summary(self) -> str:
        """One-line human-readable summary."""
        n = len(self.records)
        w = self.total_wall_time_s
        c = self.total_cpu_time_s
        return (
            f"[{self.algorithm_name}] {n} stage(s), "
            f"wall={w:.3f}s, cpu={c:.3f}s"
        )