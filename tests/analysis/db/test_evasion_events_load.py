"""Regression tests dla `_load_evasion_events` po refaktorze 2026-05-07.

Weryfikuje:
- usunięcie kolumny `astar_success` ze schemy + populator akceptuje ją
  w starych CSV (silent ignore)
- nowa kolumna `ttc_source` ('oracle_discrete' | 'continuous' | NULL)
- nowa kolumna `preferred_axis` ('right'|'left'|'up'|'down'|NULL) z
  parsowaniem `notes="axis=..."` jako fallback dla starych CSV
- "axis=unknown" → preferred_axis=NULL (a nie magic string)
"""
from __future__ import annotations

import csv
import sqlite3
import tempfile
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
        """
        INSERT INTO runs (
            run_id, run_dir_name, source_path, optimizer_algo, avoidance_algo,
            environment, seed, algorithm_pair
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("run1", run_dir.name, str(run_dir), "MSFFOA", "MSFFOA",
         "forest", 1, "MSFFOA_MSFFOA"),
    )
    conn.commit()
    return conn, run_dir


def _write_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow(row)


class TestSchemaShape:
    def test_astar_success_column_removed(self, tmp_path: Path) -> None:
        conn, _ = _setup_db(tmp_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(evasion_events)").fetchall()}
        assert "astar_success" not in cols, "astar_success powinno być usunięte"

    def test_ttc_source_column_present(self, tmp_path: Path) -> None:
        conn, _ = _setup_db(tmp_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(evasion_events)").fetchall()}
        assert "ttc_source" in cols
        assert "preferred_axis" in cols


class TestLegacyCSVCompat:
    """Stare CSV (sprzed 2026-05-07) miały `astar_success` + `notes='axis=...'`."""

    def test_legacy_csv_with_astar_success_loads(self, tmp_path: Path) -> None:
        from src.analysis.db.populate_database import _load_evasion_events
        conn, run_dir = _setup_db(tmp_path)

        legacy_headers = [
            "time", "drone_id", "event_type", "mode",
            "ttc", "dist_to_threat",
            "threat_x", "threat_y", "threat_z",
            "threat_vx", "threat_vy", "threat_vz",
            "rejoin_x", "rejoin_y", "rejoin_z", "rejoin_arc",
            "astar_success", "fallback_used",
            "pos_error_at_rejoin", "vel_error_at_rejoin",
            "planning_wall_time_s", "notes",
        ]
        rows = [
            {
                "time": 1.5, "drone_id": 0, "event_type": "plan_built", "mode": 1,
                "ttc": 3.4, "dist_to_threat": 42.0,
                "threat_x": 10, "threat_y": 5, "threat_z": 3,
                "threat_vx": 0, "threat_vy": 0, "threat_vz": 0,
                "rejoin_x": 50, "rejoin_y": 5, "rejoin_z": 3, "rejoin_arc": 1.2,
                "astar_success": "true", "fallback_used": "false",
                "pos_error_at_rejoin": 0.5, "vel_error_at_rejoin": 0.1,
                "planning_wall_time_s": 0.05, "notes": "axis=right",
            },
            {
                "time": 2.0, "drone_id": 1, "event_type": "plan_built", "mode": 1,
                "ttc": 2.9, "dist_to_threat": 478.0,
                "threat_x": 100, "threat_y": 0, "threat_z": 5,
                "threat_vx": -1, "threat_vy": 0, "threat_vz": 0,
                "rejoin_x": 60, "rejoin_y": 0, "rejoin_z": 5, "rejoin_arc": 0.8,
                "astar_success": "true", "fallback_used": "false",
                "pos_error_at_rejoin": 0.3, "vel_error_at_rejoin": 0.05,
                "planning_wall_time_s": 0.04, "notes": "axis=unknown",
            },
        ]
        csv_path = run_dir / "evasion_events.csv"
        _write_csv(csv_path, legacy_headers, rows)

        _load_evasion_events(conn, "run1", csv_path)
        conn.commit()

        result = conn.execute(
            "SELECT preferred_axis, ttc_source, notes, fallback_used "
            "FROM evasion_events ORDER BY event_index"
        ).fetchall()
        assert len(result) == 2
        # Pierwszy wiersz: notes='axis=right' → preferred_axis='right', notes wyczyszczone
        assert result[0]["preferred_axis"] == "right"
        assert result[0]["notes"] is None
        # Drugi wiersz: notes='axis=unknown' → preferred_axis=NULL (a nie magic string)
        assert result[1]["preferred_axis"] is None
        assert result[1]["notes"] is None
        # ttc_source z legacy CSV jest NULL (nie miał tej kolumny)
        assert all(r["ttc_source"] is None for r in result)


class TestNewCSVFormat:
    """Nowe CSV (od 2026-05-07) mają `ttc_source` i `preferred_axis` jako kolumny."""

    def test_new_csv_with_explicit_columns(self, tmp_path: Path) -> None:
        from src.analysis.db.populate_database import _load_evasion_events
        conn, run_dir = _setup_db(tmp_path)

        new_headers = [
            "time", "drone_id", "event_type", "mode",
            "ttc", "ttc_source", "dist_to_threat",
            "threat_x", "threat_y", "threat_z",
            "threat_vx", "threat_vy", "threat_vz",
            "rejoin_x", "rejoin_y", "rejoin_z", "rejoin_arc",
            "preferred_axis", "fallback_used",
            "pos_error_at_rejoin", "vel_error_at_rejoin",
            "planning_wall_time_s", "notes",
        ]
        rows = [
            {
                "time": 1.5, "drone_id": 0, "event_type": "plan_built", "mode": 1,
                "ttc": 3.4, "ttc_source": "oracle_discrete", "dist_to_threat": 42.0,
                "threat_x": 10, "threat_y": 5, "threat_z": 3,
                "threat_vx": 0, "threat_vy": 0, "threat_vz": 0,
                "rejoin_x": 50, "rejoin_y": 5, "rejoin_z": 3, "rejoin_arc": 1.2,
                "preferred_axis": "right", "fallback_used": "false",
                "pos_error_at_rejoin": 0.5, "vel_error_at_rejoin": 0.1,
                "planning_wall_time_s": 0.05, "notes": "",
            },
            {
                "time": 2.0, "drone_id": 1, "event_type": "trigger", "mode": 0,
                "ttc": 0.8, "ttc_source": "continuous", "dist_to_threat": 5.5,
                "threat_x": 5.5, "threat_y": 0, "threat_z": 5,
                "threat_vx": -2, "threat_vy": 0, "threat_vz": 0,
                "rejoin_x": "", "rejoin_y": "", "rejoin_z": "", "rejoin_arc": "",
                "preferred_axis": "", "fallback_used": "",
                "pos_error_at_rejoin": "", "vel_error_at_rejoin": "",
                "planning_wall_time_s": "", "notes": "",
            },
        ]
        csv_path = run_dir / "evasion_events.csv"
        _write_csv(csv_path, new_headers, rows)

        _load_evasion_events(conn, "run1", csv_path)
        conn.commit()

        result = conn.execute(
            "SELECT preferred_axis, ttc_source, ttc, dist_to_threat "
            "FROM evasion_events ORDER BY event_index"
        ).fetchall()
        assert result[0]["preferred_axis"] == "right"
        assert result[0]["ttc_source"] == "oracle_discrete"
        # Drugi wiersz: trigger bez axis (nie ma planu jeszcze)
        assert result[1]["preferred_axis"] is None
        assert result[1]["ttc_source"] == "continuous"

    def test_invalid_axis_yields_null(self, tmp_path: Path) -> None:
        """Nielegalna wartość preferred_axis (np. 'W') → NULL (CHECK constraint)."""
        from src.analysis.db.populate_database import _load_evasion_events
        conn, run_dir = _setup_db(tmp_path)

        new_headers = [
            "time", "drone_id", "event_type", "mode",
            "ttc", "ttc_source", "dist_to_threat",
            "threat_x", "threat_y", "threat_z",
            "threat_vx", "threat_vy", "threat_vz",
            "rejoin_x", "rejoin_y", "rejoin_z", "rejoin_arc",
            "preferred_axis", "fallback_used",
            "pos_error_at_rejoin", "vel_error_at_rejoin",
            "planning_wall_time_s", "notes",
        ]
        rows = [{
            "time": 1.0, "drone_id": 0, "event_type": "plan_built", "mode": 1,
            "ttc": 1.0, "ttc_source": "continuous", "dist_to_threat": 5.0,
            "threat_x": 0, "threat_y": 0, "threat_z": 0,
            "threat_vx": 0, "threat_vy": 0, "threat_vz": 0,
            "rejoin_x": 0, "rejoin_y": 0, "rejoin_z": 0, "rejoin_arc": 0,
            "preferred_axis": "W",  # nielegalne — nie right/left/up/down
            "fallback_used": "false",
            "pos_error_at_rejoin": 0, "vel_error_at_rejoin": 0,
            "planning_wall_time_s": 0, "notes": "",
        }]
        csv_path = run_dir / "evasion_events.csv"
        _write_csv(csv_path, new_headers, rows)

        _load_evasion_events(conn, "run1", csv_path)
        conn.commit()

        result = conn.execute("SELECT preferred_axis FROM evasion_events").fetchone()
        assert result["preferred_axis"] is None
