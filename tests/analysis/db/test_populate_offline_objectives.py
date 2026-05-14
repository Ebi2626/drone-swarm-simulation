"""Testy ekstrakcji best-feasible F-vector z optimization_history.h5."""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from src.analysis.db.initialize_database import initialize_database
from src.analysis.db.populate_offline_objectives import populate_offline_objectives


@pytest.fixture
def fresh_db():
    """Pusta baza + 1 wpis runa + pusty `run_metrics`."""
    with tempfile.TemporaryDirectory() as tmp:
        exp_dir = Path(tmp) / "exp"
        exp_dir.mkdir()
        db_path = initialize_database(exp_dir)
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            run_id = "msffoa_forest_msffoa_seed1"
            conn.execute(
                """
                INSERT INTO runs (run_id, run_dir_name, source_path,
                    optimizer_algo, avoidance_algo, environment, seed, algorithm_pair)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, run_id, str(exp_dir / run_id),
                 "msffoa", "msffoa", "forest", 1, "msffoa + msffoa"),
            )
            conn.execute(
                "INSERT INTO run_metrics (run_id) VALUES (?)",
                (run_id,),
            )
            conn.commit()
            yield conn, run_id, exp_dir


def _write_h5(
    h5_path: Path,
    objectives_matrix: np.ndarray,
    feasible_mask: np.ndarray | None = None,
) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("objectives_matrix", data=objectives_matrix)
        if feasible_mask is not None:
            f.create_dataset("feasible_mask", data=feasible_mask.astype(bool))


def _write_hydra_config(
    run_dir: Path,
    objective_weights: list[float] | None = None,
) -> None:
    """Write minimal `.hydra/config.yaml` z `objective_weights` (opcjonalnie)."""
    import yaml
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"optimizer": {"algorithm_params": {}}}
    if objective_weights is not None:
        cfg["optimizer"]["algorithm_params"]["objective_weights"] = objective_weights
    with open(hydra_dir / "config.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)


class TestExtract5Obj:
    def test_picks_best_f1_in_last_gen(self, fresh_db) -> None:
        conn, run_id, exp_dir = fresh_db
        run_dir = exp_dir / run_id
        run_dir.mkdir(exist_ok=True)
        _write_hydra_config(run_dir, [0.05, 0.5, 0.8, 1.0, 0.25])

        # 3 generacje × 4 individuals × 5 objectives.
        obj = np.full((3, 4, 5), 999.0, dtype=np.float64)
        obj[2, 2] = [10.0, 20.0, 30.0, 40.0, 50.0]
        obj[2, 0] = [100.0, 200.0, 300.0, 400.0, 500.0]

        h5_path = run_dir / "optimization_history" / "optimization_history.h5"
        _write_h5(h5_path, obj)
        populate_offline_objectives(conn, run_id, h5_path, run_dir=run_dir)

        row = conn.execute(
            """
            SELECT final_objective, final_objective_f1_trajectory,
                   final_objective_f2_height_angle, total_threat_cost,
                   total_turn_penalty, total_coordination_cost,
                   final_objectives_json, final_objective_weights_json
            FROM run_metrics WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        # final_objective NIE jest ustawiany przez populate_offline_objectives
        # po 2026-05-13 — czeka na post-pass populate_final_objective_aggregated.
        assert row[0] is None
        assert row[1] == pytest.approx(10.0)  # f1
        assert row[2] == pytest.approx(20.0)  # f2
        assert row[3] == pytest.approx(30.0)  # threat
        assert row[4] == pytest.approx(40.0)  # turn
        assert row[5] == pytest.approx(50.0)  # coordination
        assert json.loads(row[6]) == [10.0, 20.0, 30.0, 40.0, 50.0]
        assert json.loads(row[7]) == [0.05, 0.5, 0.8, 1.0, 0.25]

    def test_uses_only_last_generation(self, fresh_db) -> None:
        """Best w gen=0 nie powinien wygrać — bierzemy LAST gen."""
        conn, run_id, exp_dir = fresh_db
        run_dir = exp_dir / run_id
        run_dir.mkdir(exist_ok=True)
        _write_hydra_config(run_dir, [0.05, 0.5, 0.8, 1.0, 0.25])

        obj = np.full((3, 2, 5), 999.0, dtype=np.float64)
        obj[0, 0] = [1.0, 2.0, 3.0, 4.0, 5.0]   # super w gen 0 — IGNOROWANY
        obj[2, 0] = [50.0, 51.0, 52.0, 53.0, 54.0]
        obj[2, 1] = [99.0, 99.0, 99.0, 99.0, 99.0]
        h5_path = exp_dir / "h.h5"
        _write_h5(h5_path, obj)
        populate_offline_objectives(conn, run_id, h5_path, run_dir=run_dir)

        row = conn.execute(
            "SELECT final_objective_f1_trajectory FROM run_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert row[0] == pytest.approx(50.0)

    def test_feasible_mask_filters_infeasible(self, fresh_db) -> None:
        conn, run_id, exp_dir = fresh_db
        run_dir = exp_dir / run_id
        run_dir.mkdir(exist_ok=True)
        _write_hydra_config(run_dir, [0.05, 0.5, 0.8, 1.0, 0.25])

        obj = np.array([[
            [10.0, 1.0, 1.0, 1.0, 1.0],   # infeasible
            [50.0, 5.0, 5.0, 5.0, 5.0],   # feasible — wybieramy mimo wyższego f1
        ]], dtype=np.float64)
        feasible = np.array([[False, True]])
        h5_path = exp_dir / "h.h5"
        _write_h5(h5_path, obj, feasible_mask=feasible)

        populate_offline_objectives(conn, run_id, h5_path, run_dir=run_dir)
        row = conn.execute(
            "SELECT final_objective_f1_trajectory FROM run_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert row[0] == pytest.approx(50.0)


class TestWeightsLoading:
    """Test ładowania wag z .hydra/config.yaml + canonical fallback."""

    def test_canonical_fallback_when_config_missing(self, fresh_db) -> None:
        """Brak .hydra/config.yaml → canonical weights."""
        conn, run_id, exp_dir = fresh_db
        run_dir = exp_dir / run_id
        run_dir.mkdir(exist_ok=True)
        # NIE tworzymy .hydra/config.yaml

        obj = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float64)
        h5_path = exp_dir / "h.h5"
        _write_h5(h5_path, obj)
        populate_offline_objectives(conn, run_id, h5_path, run_dir=run_dir)

        weights = json.loads(conn.execute(
            "SELECT final_objective_weights_json FROM run_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0])
        assert weights == [0.05, 0.5, 0.8, 1.0, 0.25]

    def test_canonical_fallback_when_weights_missing_from_config(self, fresh_db) -> None:
        """NSGA-III: config bez objective_weights → canonical fallback."""
        conn, run_id, exp_dir = fresh_db
        run_dir = exp_dir / run_id
        run_dir.mkdir(exist_ok=True)
        _write_hydra_config(run_dir, objective_weights=None)  # brak wag

        obj = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float64)
        h5_path = exp_dir / "h.h5"
        _write_h5(h5_path, obj)
        populate_offline_objectives(conn, run_id, h5_path, run_dir=run_dir)

        weights = json.loads(conn.execute(
            "SELECT final_objective_weights_json FROM run_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0])
        assert weights == [0.05, 0.5, 0.8, 1.0, 0.25]

    def test_custom_weights_loaded(self, fresh_db) -> None:
        """Custom wagi w configu są poprawnie wczytane."""
        conn, run_id, exp_dir = fresh_db
        run_dir = exp_dir / run_id
        run_dir.mkdir(exist_ok=True)
        custom = [0.1, 0.2, 0.3, 0.4, 0.5]
        _write_hydra_config(run_dir, custom)

        obj = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float64)
        h5_path = exp_dir / "h.h5"
        _write_h5(h5_path, obj)
        populate_offline_objectives(conn, run_id, h5_path, run_dir=run_dir)

        weights = json.loads(conn.execute(
            "SELECT final_objective_weights_json FROM run_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0])
        assert weights == custom


class TestNoH5:
    def test_missing_h5_writes_weights_only(self, fresh_db) -> None:
        """Brak h5 — F-vec NULL, ale wagi zapisane (z canonical fallback)."""
        conn, run_id, exp_dir = fresh_db
        run_dir = exp_dir / run_id
        run_dir.mkdir(exist_ok=True)
        populate_offline_objectives(
            conn, run_id, exp_dir / "nonexistent.h5", run_dir=run_dir,
        )
        row = conn.execute(
            "SELECT final_objective, final_objective_f1_trajectory, "
            "final_objective_weights_json FROM run_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert row[0] is None
        assert row[1] is None
        # weights zapisane mimo braku h5 (canonical fallback).
        assert json.loads(row[2]) == [0.05, 0.5, 0.8, 1.0, 0.25]

    def test_empty_h5_writes_weights_only(self, fresh_db) -> None:
        conn, run_id, exp_dir = fresh_db
        run_dir = exp_dir / run_id
        run_dir.mkdir(exist_ok=True)
        h5_path = exp_dir / "empty.h5"
        with h5py.File(h5_path, "w") as f:
            pass  # no datasets
        populate_offline_objectives(conn, run_id, h5_path, run_dir=run_dir)
        row = conn.execute(
            "SELECT final_objective, final_objective_weights_json "
            "FROM run_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert row[0] is None
        assert json.loads(row[1]) == [0.05, 0.5, 0.8, 1.0, 0.25]


class TestLegacy3Obj:
    def test_legacy_3obj_falls_back(self, fresh_db) -> None:
        """Stary VectorizedEvaluator zapisywał 3 objectives — graceful fallback."""
        conn, run_id, exp_dir = fresh_db
        run_dir = exp_dir / run_id
        run_dir.mkdir(exist_ok=True)
        _write_hydra_config(run_dir, [0.05, 0.5, 0.8, 1.0, 0.25])

        obj = np.array([[
            [100.0, 200.0, 300.0],
            [400.0, 500.0, 600.0],
        ]], dtype=np.float64)
        h5_path = exp_dir / "h.h5"
        _write_h5(h5_path, obj)
        populate_offline_objectives(conn, run_id, h5_path, run_dir=run_dir)
        row = conn.execute(
            """
            SELECT final_objective_f1_trajectory,
                   total_coordination_cost, final_objectives_json
            FROM run_metrics WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        assert row[0] == pytest.approx(100.0)
        # Coordination nie istnieje w legacy → NULL.
        assert row[1] is None
        assert json.loads(row[2]) == [100.0, 200.0, 300.0]


class TestIdempotency:
    def test_rerun_overwrites_not_duplicates(self, fresh_db) -> None:
        conn, run_id, exp_dir = fresh_db
        run_dir = exp_dir / run_id
        run_dir.mkdir(exist_ok=True)
        _write_hydra_config(run_dir, [0.05, 0.5, 0.8, 1.0, 0.25])

        obj = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float64)
        h5_path = exp_dir / "h.h5"
        _write_h5(h5_path, obj)
        populate_offline_objectives(conn, run_id, h5_path, run_dir=run_dir)
        populate_offline_objectives(conn, run_id, h5_path, run_dir=run_dir)
        n = conn.execute("SELECT COUNT(*) FROM run_metrics").fetchone()[0]
        assert n == 1
