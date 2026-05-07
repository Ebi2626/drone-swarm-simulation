"""Regression tests dla `populate_online_metrics` po refaktorze 2026-05-07.

Weryfikuje:
- CHECK constraint na `chosen_axis` ∈ ('right','left','up','down') OR NULL
- normalizacja legacy magic string "unknown" → NULL
- "" (empty string) → NULL
- prawidłowy axis 'right'/'left'/'up'/'down' przechodzi
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest


def _setup_db(tmp_path: Path):
    from src.analysis.db.initialize_database import initialize_database

    exp_dir = tmp_path / "exp"
    run_dir = exp_dir / "msffoa_forest_msffoa_seed1"
    run_dir.mkdir(parents=True)
    db_path = initialize_database(exp_dir)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """INSERT INTO runs (run_id, run_dir_name, source_path, optimizer_algo,
        avoidance_algo, environment, seed, algorithm_pair)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("run1", run_dir.name, str(run_dir),
         "msffoa", "msffoa", "forest", 1, "msffoa_msffoa"),
    )
    conn.commit()
    return conn, run_dir


def _write_online_csv(path: Path, rows: list[dict]) -> None:
    headers = [
        "run_id", "drone_id", "trigger_time", "algorithm", "status", "reason",
        "best_fitness", "evaluations_completed", "generations_completed",
        "wallclock_s", "time_budget_s", "chosen_axis", "plan_waypoints_json",
        "plan_total_duration_s", "plan_arc_length_m", "outcome",
        "pos_err_at_rejoin_m", "vel_err_at_rejoin_mps", "time_to_rejoin_s",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _base_row(**overrides) -> dict:
    base = {
        "run_id": "run1", "drone_id": 0, "trigger_time": 1.0,
        "algorithm": "MSFOA", "status": "ok", "reason": "ok",
        "best_fitness": 50.0, "evaluations_completed": 100,
        "generations_completed": 10, "wallclock_s": 0.4,
        "time_budget_s": 0.5, "chosen_axis": "",
        "plan_waypoints_json": "[]", "plan_total_duration_s": 1.5,
        "plan_arc_length_m": 5.0, "outcome": "rejoined_ok",
        "pos_err_at_rejoin_m": 0.1, "vel_err_at_rejoin_mps": 0.05,
        "time_to_rejoin_s": 1.0,
    }
    base.update(overrides)
    return base


class TestChosenAxisNormalization:
    def test_unknown_normalized_to_null(self, tmp_path: Path) -> None:
        from src.analysis.db.populate_online_metrics import populate_online_metrics
        conn, run_dir = _setup_db(tmp_path)

        _write_online_csv(
            run_dir / "online_optimization.csv",
            [_base_row(chosen_axis="unknown", trigger_time=1.0, drone_id=0)],
        )
        populate_online_metrics(conn, "run1", run_dir)
        conn.commit()

        result = conn.execute(
            "SELECT chosen_axis FROM online_optimization_tasks"
        ).fetchone()
        assert result["chosen_axis"] is None

    def test_empty_string_normalized_to_null(self, tmp_path: Path) -> None:
        from src.analysis.db.populate_online_metrics import populate_online_metrics
        conn, run_dir = _setup_db(tmp_path)

        _write_online_csv(
            run_dir / "online_optimization.csv",
            [_base_row(chosen_axis="", trigger_time=1.0, drone_id=0)],
        )
        populate_online_metrics(conn, "run1", run_dir)
        conn.commit()

        result = conn.execute(
            "SELECT chosen_axis FROM online_optimization_tasks"
        ).fetchone()
        assert result["chosen_axis"] is None

    def test_valid_axis_passes(self, tmp_path: Path) -> None:
        from src.analysis.db.populate_online_metrics import populate_online_metrics
        conn, run_dir = _setup_db(tmp_path)

        _write_online_csv(
            run_dir / "online_optimization.csv",
            [
                _base_row(chosen_axis="right", trigger_time=1.0, drone_id=0),
                _base_row(chosen_axis="left", trigger_time=2.0, drone_id=0),
                _base_row(chosen_axis="up", trigger_time=3.0, drone_id=0),
                _base_row(chosen_axis="down", trigger_time=4.0, drone_id=0),
            ],
        )
        populate_online_metrics(conn, "run1", run_dir)
        conn.commit()

        result = sorted(
            r["chosen_axis"]
            for r in conn.execute(
                "SELECT chosen_axis FROM online_optimization_tasks"
            ).fetchall()
        )
        assert result == ["down", "left", "right", "up"]

    def test_invalid_axis_normalized_to_null(self, tmp_path: Path) -> None:
        """Każda wartość poza right/left/up/down (np. 'X', 'W') → NULL.
        Test broni przed silent INSERT-ami z niespodziewanymi enumami."""
        from src.analysis.db.populate_online_metrics import populate_online_metrics
        conn, run_dir = _setup_db(tmp_path)

        _write_online_csv(
            run_dir / "online_optimization.csv",
            [
                _base_row(chosen_axis="X", trigger_time=1.0, drone_id=0),
                _base_row(chosen_axis="W", trigger_time=2.0, drone_id=0),
            ],
        )
        populate_online_metrics(conn, "run1", run_dir)
        conn.commit()

        result = conn.execute(
            "SELECT chosen_axis FROM online_optimization_tasks"
        ).fetchall()
        assert all(r["chosen_axis"] is None for r in result)


class TestSchemaCheck:
    def test_direct_insert_with_unknown_rejected(self, tmp_path: Path) -> None:
        """Ręczny INSERT z chosen_axis='unknown' MUSI paść na CHECK."""
        conn, _ = _setup_db(tmp_path)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO online_optimization_tasks (
                    run_id, drone_id, trigger_time, algorithm, status,
                    wallclock_s, time_budget_s, chosen_axis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("run1", 0, 1.0, "MSFOA", "ok", 0.4, 0.5, "unknown"),
            )
