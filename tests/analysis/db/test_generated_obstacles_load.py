"""Regression tests dla `_load_generated_obstacles` po refaktorze 2026-05-07.

Weryfikuje:
- nowa kolumna `shape_type` ('cylinder' | 'box')
- usunięcie `unused_dim`, dodanie `length` i `width`
- CHECK constraint: cylinder MUSI mieć `radius`, box MUSI mieć `length+width`
- backward-compat dla starych CSV cylindra (z `unused_dim` jako 6. kolumną)
- BOX nie pisze już length do kolumny `radius` (regresja na bug 549-551)
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest


def _setup_db(tmp_path: Path):
    from src.analysis.db.initialize_database import initialize_database

    exp_dir = tmp_path / "exp"
    run_dir_forest = exp_dir / "msffoa_forest_msffoa_seed1"
    run_dir_urban = exp_dir / "msffoa_urban_msffoa_seed1"
    run_dir_forest.mkdir(parents=True)
    run_dir_urban.mkdir(parents=True)

    db_path = initialize_database(exp_dir)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """INSERT INTO runs (run_id, run_dir_name, source_path, optimizer_algo,
        avoidance_algo, environment, seed, algorithm_pair)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("forest_run", run_dir_forest.name, str(run_dir_forest),
         "msffoa", "msffoa", "forest", 1, "msffoa_msffoa"),
    )
    conn.execute(
        """INSERT INTO runs (run_id, run_dir_name, source_path, optimizer_algo,
        avoidance_algo, environment, seed, algorithm_pair)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("urban_run", run_dir_urban.name, str(run_dir_urban),
         "msffoa", "msffoa", "urban", 1, "msffoa_msffoa"),
    )
    conn.commit()
    return conn, run_dir_forest, run_dir_urban


def _write_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow(row)


class TestSchema:
    def test_unused_dim_column_removed(self, tmp_path: Path) -> None:
        conn, _, _ = _setup_db(tmp_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(generated_obstacles)").fetchall()}
        assert "unused_dim" not in cols

    def test_new_columns_present(self, tmp_path: Path) -> None:
        conn, _, _ = _setup_db(tmp_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(generated_obstacles)").fetchall()}
        assert {"shape_type", "radius", "length", "width", "height"}.issubset(cols)


class TestCylinderLoad:
    def test_cylinder_csv_5_columns(self, tmp_path: Path) -> None:
        from src.analysis.db.populate_database import _load_generated_obstacles
        conn, fdir, _ = _setup_db(tmp_path)

        headers = ["x", "y", "z", "radius", "height"]
        rows = [
            {"x": 10.0, "y": 20.0, "z": 0.0, "radius": 1.5, "height": 10.0},
            {"x": 30.0, "y": 40.0, "z": 0.0, "radius": 2.0, "height": 8.0},
        ]
        csv_path = fdir / "generated_obstacles.csv"
        _write_csv(csv_path, headers, rows)

        _load_generated_obstacles(conn, "forest_run", csv_path)
        conn.commit()

        result = conn.execute(
            """SELECT shape_type, radius, length, width, height
            FROM generated_obstacles WHERE run_id='forest_run' ORDER BY obstacle_index"""
        ).fetchall()
        assert len(result) == 2
        for r in result:
            assert r["shape_type"] == "cylinder"
            assert r["radius"] is not None
            assert r["length"] is None
            assert r["width"] is None
            assert r["height"] is not None
        assert result[0]["radius"] == 1.5
        assert result[1]["radius"] == 2.0

    def test_legacy_cylinder_csv_with_unused_dim_loads(self, tmp_path: Path) -> None:
        """Stary CSV cylindra (6 kolumn z `unused_dim`) wciąż się ładuje."""
        from src.analysis.db.populate_database import _load_generated_obstacles
        conn, fdir, _ = _setup_db(tmp_path)

        legacy_headers = ["x", "y", "z", "radius", "height", "unused_dim"]
        rows = [{"x": 10.0, "y": 20.0, "z": 0.0, "radius": 1.0, "height": 10.0, "unused_dim": 0.0}]
        csv_path = fdir / "generated_obstacles.csv"
        _write_csv(csv_path, legacy_headers, rows)

        _load_generated_obstacles(conn, "forest_run", csv_path)
        conn.commit()

        result = conn.execute(
            "SELECT shape_type, radius, height FROM generated_obstacles WHERE run_id='forest_run'"
        ).fetchone()
        assert result["shape_type"] == "cylinder"
        assert result["radius"] == 1.0
        assert result["height"] == 10.0


class TestBoxLoad:
    def test_box_csv_writes_length_width_separately(self, tmp_path: Path) -> None:
        """REGRESJA: BOX poprzednio (przed 2026-05-07) zapisywał length→radius
        i width→unused_dim. Teraz: każdy wymiar w swojej kolumnie."""
        from src.analysis.db.populate_database import _load_generated_obstacles
        conn, _, udir = _setup_db(tmp_path)

        headers = ["x", "y", "z", "length", "width", "height"]
        rows = [
            {"x": 100.0, "y": 200.0, "z": 0.0, "length": 15.0, "width": 12.0, "height": 18.0},
        ]
        csv_path = udir / "generated_obstacles.csv"
        _write_csv(csv_path, headers, rows)

        _load_generated_obstacles(conn, "urban_run", csv_path)
        conn.commit()

        result = conn.execute(
            """SELECT shape_type, radius, length, width, height
            FROM generated_obstacles WHERE run_id='urban_run'"""
        ).fetchone()
        assert result["shape_type"] == "box"
        assert result["radius"] is None, "radius dla box MUSI być NULL (regresja na bug 549)"
        assert result["length"] == 15.0
        assert result["width"] == 12.0
        assert result["height"] == 18.0


class TestCheckConstraints:
    def test_cylinder_with_length_rejected(self, tmp_path: Path) -> None:
        """Zabezpieczenie: ręczny INSERT z mieszaną semantyką musi paść
        na CHECK constraint."""
        conn, _, _ = _setup_db(tmp_path)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO generated_obstacles (run_id, obstacle_index, x, y, z,
                shape_type, radius, length, width, height) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("forest_run", 0, 0.0, 0.0, 0.0, "cylinder", 1.0, 5.0, None, 10.0),
            )

    def test_box_without_length_rejected(self, tmp_path: Path) -> None:
        conn, _, _ = _setup_db(tmp_path)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO generated_obstacles (run_id, obstacle_index, x, y, z,
                shape_type, radius, length, width, height) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("urban_run", 0, 0.0, 0.0, 0.0, "box", None, None, 10.0, 10.0),
            )
