"""Testy ETL CSV → DB dla `populate_online_metrics` (Faza 1 plan.md).

Sprawdza:
- `online_optimization.csv` zostaje zassany do `online_optimization_tasks`.
- `convergence_traces.csv` (po fix literówki) → `online_convergence_traces`.
- Brak CSV nie wywala pipeline'u — tabele puste, brak wyjątku.
"""
from __future__ import annotations

import csv
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.analysis.db.initialize_database import initialize_database
from src.analysis.db.populate_online_metrics import populate_online_metrics


@pytest.fixture
def tmp_run() -> tuple[Path, sqlite3.Connection]:
    """Tymczasowy katalog runa + zainicjalizowana baza."""
    with tempfile.TemporaryDirectory() as tmp:
        exp_dir = Path(tmp) / "exp"
        run_dir = exp_dir / "msffoa_forest_msffoa_seed1"
        run_dir.mkdir(parents=True)
        db_path = initialize_database(exp_dir)
        with sqlite3.connect(db_path) as conn:
            # `runs` row jest wymagany przez FK.
            conn.execute(
                """
                INSERT INTO runs (run_id, run_dir_name, source_path,
                    optimizer_algo, avoidance_algo, environment, seed,
                    algorithm_pair)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_dir.name, run_dir.name, str(run_dir),
                 "msffoa", "msffoa", "forest", 1, "msffoa + msffoa"),
            )
            conn.commit()
            yield run_dir, conn


def _write_online_csv(run_dir: Path, n_rows: int = 2) -> None:
    with (run_dir / "online_optimization.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id", "drone_id", "trigger_time", "algorithm", "status", "reason",
            "best_fitness", "evaluations_completed", "generations_completed",
            "wallclock_s", "time_budget_s", "chosen_axis", "plan_waypoints_json",
            "plan_total_duration_s", "plan_arc_length_m", "outcome",
            "pos_err_at_rejoin_m", "vel_err_at_rejoin_mps", "time_to_rejoin_s",
        ])
        for i in range(n_rows):
            w.writerow([
                run_dir.name, i, 1.0 + i, "MSFOA", "ok", "ok",
                50.0, 100, 10, 0.4, 0.5, "right", "[]",
                1.5, 5.0, "rejoined_ok", 0.1, 0.05, 1.0,
            ])


def _write_convergence_csv(run_dir: Path, n_gens: int = 5) -> None:
    with (run_dir / "convergence_traces.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "drone_id", "trigger_time", "algorithm", "generation", "best_fitness"])
        for g in range(n_gens):
            w.writerow([run_dir.name, 0, 1.0, "MSFOA", g, 100.0 - g * 10])


class TestPopulateOnlineMetricsCSVPresent:
    def test_loads_online_optimization_tasks(self, tmp_run) -> None:
        run_dir, conn = tmp_run
        _write_online_csv(run_dir, n_rows=3)
        populate_online_metrics(conn, run_dir.name, run_dir)
        n = conn.execute(
            "SELECT COUNT(*) FROM online_optimization_tasks WHERE run_id = ?",
            (run_dir.name,),
        ).fetchone()[0]
        assert n == 3

    def test_loads_convergence_traces(self, tmp_run) -> None:
        run_dir, conn = tmp_run
        _write_online_csv(run_dir, n_rows=1)
        _write_convergence_csv(run_dir, n_gens=5)
        populate_online_metrics(conn, run_dir.name, run_dir)
        n = conn.execute(
            "SELECT COUNT(*) FROM online_convergence_traces WHERE run_id = ?",
            (run_dir.name,),
        ).fetchone()[0]
        assert n == 5

    def test_convergence_csv_correct_filename(self, tmp_run) -> None:
        """Regression: literówka `convergance_traces.csv` została naprawiona
        do `convergence_traces.csv` (zgodnie z `SimulationLogger.save()`)."""
        run_dir, conn = tmp_run
        _write_online_csv(run_dir, n_rows=1)
        # Plik z literówką nie powinien być akceptowany.
        with (run_dir / "convergance_traces.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_id", "drone_id", "trigger_time", "algorithm", "generation", "best_fitness"])
            w.writerow([run_dir.name, 0, 1.0, "MSFOA", 0, 99.0])
        populate_online_metrics(conn, run_dir.name, run_dir)
        n = conn.execute(
            "SELECT COUNT(*) FROM online_convergence_traces WHERE run_id = ?",
            (run_dir.name,),
        ).fetchone()[0]
        assert n == 0  # plik z literówką ignorowany


class TestPopulateOnlineMetricsCSVAbsent:
    def test_no_csv_no_crash(self, tmp_run) -> None:
        run_dir, conn = tmp_run
        # Bez CSV — tabela powinna pozostać pusta, brak wyjątku.
        populate_online_metrics(conn, run_dir.name, run_dir)
        n = conn.execute("SELECT COUNT(*) FROM online_optimization_tasks").fetchone()[0]
        assert n == 0
        n = conn.execute("SELECT COUNT(*) FROM online_convergence_traces").fetchone()[0]
        assert n == 0


class TestSchemaIntegrity:
    def test_online_optimization_tasks_pk(self, tmp_run) -> None:
        run_dir, conn = tmp_run
        _write_online_csv(run_dir, n_rows=2)
        populate_online_metrics(conn, run_dir.name, run_dir)
        # Re-run idempotentnie nie duplikuje (INSERT OR REPLACE).
        populate_online_metrics(conn, run_dir.name, run_dir)
        n = conn.execute("SELECT COUNT(*) FROM online_optimization_tasks").fetchone()[0]
        assert n == 2

    def test_convergence_fk_links_to_task(self, tmp_run) -> None:
        run_dir, conn = tmp_run
        _write_online_csv(run_dir, n_rows=1)
        _write_convergence_csv(run_dir, n_gens=3)
        populate_online_metrics(conn, run_dir.name, run_dir)
        # JOIN po PK kompozytowym.
        n = conn.execute(
            """
            SELECT COUNT(*) FROM online_convergence_traces c
            JOIN online_optimization_tasks o
              ON o.run_id = c.run_id
             AND o.drone_id = c.drone_id
             AND o.trigger_time = c.trigger_time
            WHERE c.run_id = ?
            """,
            (run_dir.name,),
        ).fetchone()[0]
        assert n == 3
