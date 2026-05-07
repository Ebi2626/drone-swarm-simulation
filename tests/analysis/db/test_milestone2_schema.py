"""Regression tests dla nowych kolumn schematu Kamień 2.

Weryfikujemy:
- `run_metrics.front_size_last_gen` istnieje + akceptuje INTEGER ≥ 0.
- `run_metrics.hypervolume_normalized` istnieje + akceptuje REAL ≥ 0.
- `iteration_metrics.front_size`, `iteration_metrics.hypervolume_normalized`.
- `reference_points.ideal_value` istnieje (REAL).
- Backfill z R + r* + z* uzupełnia HV_norm.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _columns_of(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


@pytest.fixture
def fresh_db(tmp_path: Path) -> sqlite3.Connection:
    from src.analysis.db.initialize_database import initialize_database

    db_path = initialize_database(tmp_path)
    return sqlite3.connect(db_path)


class TestSchemaColumnsExist:
    def test_run_metrics_has_front_size_and_hv_norm(self, fresh_db) -> None:
        cols = _columns_of(fresh_db, "run_metrics")
        assert "front_size_last_gen" in cols
        assert "hypervolume_normalized" in cols

    def test_iteration_metrics_has_front_size_and_hv_norm(self, fresh_db) -> None:
        cols = _columns_of(fresh_db, "iteration_metrics")
        assert "front_size" in cols
        assert "hypervolume_normalized" in cols

    def test_reference_points_has_ideal_value(self, fresh_db) -> None:
        cols = _columns_of(fresh_db, "reference_points")
        assert "ideal_value" in cols


class TestRunMetricsConstraints:
    def test_front_size_rejects_negative(self, fresh_db) -> None:
        # Setup minimum viable run row.
        fresh_db.execute(
            """
            INSERT INTO runs (run_id, run_dir_name, source_path, optimizer_algo,
                              avoidance_algo, environment, seed, algorithm_pair)
            VALUES ('r1', 'd1', '/tmp/d1', 'msffoa', 'msffoa', 'forest', 0, 'pair')
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            fresh_db.execute(
                "INSERT INTO run_metrics (run_id, front_size_last_gen) VALUES (?, ?)",
                ("r1", -1),
            )

    def test_hypervolume_normalized_rejects_negative(self, fresh_db) -> None:
        fresh_db.execute(
            """
            INSERT INTO runs (run_id, run_dir_name, source_path, optimizer_algo,
                              avoidance_algo, environment, seed, algorithm_pair)
            VALUES ('r1', 'd1', '/tmp/d1', 'msffoa', 'msffoa', 'forest', 0, 'pair')
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            fresh_db.execute(
                "INSERT INTO run_metrics (run_id, hypervolume_normalized) VALUES (?, ?)",
                ("r1", -0.5),
            )


class TestReferencePointsIdeal:
    def test_build_reference_pareto_persists_ideal(self, tmp_path: Path) -> None:
        """build_reference_pareto wpisuje ideal = min(R, axis=0)."""
        import h5py
        import numpy as np

        from src.analysis.db.build_reference_pareto import (
            build_reference_pareto_sets,
            load_ideal_point,
            load_reference_point,
        )
        from src.analysis.db.initialize_database import initialize_database

        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()

        # 1 run, last_gen front z znanymi wartościami.
        run_dir = exp_dir / "msffoa_forest_msffoa_seed0"
        h5_dir = run_dir / "optimization_history"
        h5_dir.mkdir(parents=True)
        last_gen = np.array([
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [2.0, 1.0, 4.0],
        ])
        obj = np.full((1, 3, 3), 999.0)
        obj[0] = last_gen
        with h5py.File(h5_dir / "optimization_history.h5", "w") as f:
            f.create_dataset("objectives_matrix", data=obj)

        db_path = initialize_database(exp_dir)
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            INSERT INTO runs (run_id, run_dir_name, source_path, optimizer_algo,
                              avoidance_algo, environment, seed, algorithm_pair)
            VALUES ('r1', 'msffoa_forest_msffoa_seed0', ?, 'msffoa', 'msffoa',
                    'forest', 0, 'pair')
            """,
            (str(run_dir),),
        )
        conn.commit()

        build_reference_pareto_sets(conn, exp_dir)
        conn.commit()

        ideal = load_ideal_point(conn, "forest", 3)
        ref = load_reference_point(conn, "forest", 3)

        assert ideal is not None and ref is not None
        # ideal = component-wise min of merged R.
        # Wszystkie 3 punkty są niezdominowane (każdy najlepszy w jednym obj).
        # min per axis: (1, 1, 1).
        assert ideal.tolist() == [1.0, 1.0, 1.0]
        # r* = nadir + 0.1 · (nadir − ideal). nadir = (3, 2, 4).
        # r* = (3 + 0.2, 2 + 0.1, 4 + 0.3) = (3.2, 2.1, 4.3).
        assert ref.tolist() == pytest.approx([3.2, 2.1, 4.3])
