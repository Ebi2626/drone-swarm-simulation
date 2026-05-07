"""Regression tests dla `_register_run_files` po refaktorze 2026-05-07.

Weryfikuje:
- usunięcie kolumn `checksum` i `extra_json` (YAGNI)
- wypełnianie `modified_at` (ISO 8601 z `stat().st_mtime`) dla istniejących plików
- wypełnianie `row_count` (n_lines - 1) dla plików .csv
- `row_count` NULL dla h5 / log / nieistniejących
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest


def _setup(tmp_path: Path):
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


def _write_csv(path: Path, n_data_rows: int, n_cols: int = 3) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"col{i}" for i in range(n_cols)])
        for i in range(n_data_rows):
            w.writerow([i] * n_cols)


class TestSchemaShape:
    def test_checksum_column_removed(self, tmp_path: Path) -> None:
        conn, _ = _setup(tmp_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(run_files)").fetchall()}
        assert "checksum" not in cols

    def test_extra_json_column_removed(self, tmp_path: Path) -> None:
        conn, _ = _setup(tmp_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(run_files)").fetchall()}
        assert "extra_json" not in cols

    def test_active_columns_present(self, tmp_path: Path) -> None:
        conn, _ = _setup(tmp_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(run_files)").fetchall()}
        assert {"row_count", "modified_at", "size_bytes", "file_format"}.issubset(cols)


class TestPopulation:
    def test_csv_row_count_filled(self, tmp_path: Path) -> None:
        from src.analysis.db.populate_database import _register_run_files
        conn, run_dir = _setup(tmp_path)

        # Stwórz 3 sample CSV z różnymi rozmiarami
        _write_csv(run_dir / "trajectories.csv", n_data_rows=42)
        _write_csv(run_dir / "collisions.csv", n_data_rows=0)  # tylko header
        _write_csv(run_dir / "evasion_events.csv", n_data_rows=999)

        _register_run_files(conn, "run1", run_dir)
        conn.commit()

        rows = {
            r["file_role"]: dict(r)
            for r in conn.execute(
                "SELECT file_role, exists_flag, row_count, modified_at, size_bytes "
                "FROM run_files"
            ).fetchall()
        }
        assert rows["trajectories_csv"]["row_count"] == 42
        assert rows["collisions_csv"]["row_count"] == 0
        assert rows["evasion_events_csv"]["row_count"] == 999

    def test_modified_at_is_iso8601(self, tmp_path: Path) -> None:
        from src.analysis.db.populate_database import _register_run_files
        conn, run_dir = _setup(tmp_path)

        _write_csv(run_dir / "trajectories.csv", n_data_rows=5)
        _register_run_files(conn, "run1", run_dir)
        conn.commit()

        result = conn.execute(
            "SELECT modified_at FROM run_files WHERE file_role='trajectories_csv'"
        ).fetchone()
        modified = result["modified_at"]
        assert modified is not None
        # ISO 8601 with timezone: 2026-05-07T15:30:42+00:00
        assert "T" in modified
        assert modified.endswith("+00:00")

    def test_nonexistent_file_has_null_row_count_and_mtime(self, tmp_path: Path) -> None:
        from src.analysis.db.populate_database import _register_run_files
        conn, run_dir = _setup(tmp_path)

        # NIE tworzymy żadnego pliku
        _register_run_files(conn, "run1", run_dir)
        conn.commit()

        for r in conn.execute("SELECT row_count, modified_at, exists_flag FROM run_files").fetchall():
            assert r["exists_flag"] == 0
            assert r["row_count"] is None
            assert r["modified_at"] is None

    def test_h5_has_null_row_count(self, tmp_path: Path) -> None:
        """h5 i log mają NULL row_count (różne semantyki granulacji)."""
        from src.analysis.db.populate_database import _register_run_files
        conn, run_dir = _setup(tmp_path)

        h5_dir = run_dir / "optimization_history"
        h5_dir.mkdir()
        (h5_dir / "optimization_history.h5").write_text("dummy")  # dummy h5 file
        (run_dir / "main.log").write_text("log line\n" * 50)

        _register_run_files(conn, "run1", run_dir)
        conn.commit()

        h5_row = conn.execute(
            "SELECT row_count, exists_flag FROM run_files WHERE file_role='optimization_history_h5'"
        ).fetchone()
        log_row = conn.execute(
            "SELECT row_count, exists_flag FROM run_files WHERE file_role='main_log'"
        ).fetchone()
        assert h5_row["exists_flag"] == 1
        assert h5_row["row_count"] is None  # h5: NULL
        assert log_row["exists_flag"] == 1
        assert log_row["row_count"] is None  # log: NULL
