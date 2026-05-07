"""Unit tests dla `build_reference_pareto` — merged ND-front cross-run.

Reference: Riquelme et al. (2015) §4.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from src.analysis.db.build_reference_pareto import (
    DEFAULT_REF_POINT_MARGIN,
    backfill_moo_quality_with_reference,
    build_reference_pareto_sets,
    load_reference_point,
    load_reference_set,
)
from src.analysis.db.initialize_database import initialize_database


def _make_run(
    exp_dir: Path,
    run_id: str,
    optimizer_algo: str,
    seed: int,
    last_gen_objs: np.ndarray,
) -> Path:
    """Tworzy katalog runa z h5 zawierającym `last_gen_objs` w ostatniej gen."""
    run_dir = exp_dir / run_id
    h5_dir = run_dir / "optimization_history"
    h5_dir.mkdir(parents=True, exist_ok=True)
    # 2 generacje × pop × n_obj. Pierwsza dummy, druga = last_gen_objs.
    pop, n_obj = last_gen_objs.shape
    obj = np.full((2, pop, n_obj), 999.0, dtype=np.float64)
    obj[1] = last_gen_objs
    with h5py.File(h5_dir / "optimization_history.h5", "w") as f:
        f.create_dataset("objectives_matrix", data=obj)
    return run_dir


def _insert_run(conn: sqlite3.Connection, run_id: str, run_dir: Path,
                optimizer_algo: str, seed: int, environment: str = "forest") -> None:
    conn.execute(
        """
        INSERT INTO runs (
            run_id, run_dir_name, source_path, optimizer_algo, avoidance_algo,
            environment, seed, algorithm_pair
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, run_id, str(run_dir), optimizer_algo, optimizer_algo,
         environment, seed, f"{optimizer_algo}_{optimizer_algo}"),
    )


@pytest.fixture
def two_runs_db(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    # Run A: front zawiera punkt zdominowany przez B.
    last_gen_a = np.array([[1.0, 4.0], [2.0, 2.5], [5.0, 5.0]])
    # Run B: front lepszy w jednej części.
    last_gen_b = np.array([[1.5, 3.0], [3.0, 1.0], [2.0, 2.0]])
    run_a = _make_run(exp_dir, "msffoa_forest_a_seed0", "MSFOA", 0, last_gen_a)
    run_b = _make_run(exp_dir, "msffoa_forest_b_seed1", "MSFOA", 1, last_gen_b)

    db_path = initialize_database(exp_dir)
    conn = sqlite3.connect(db_path)
    _insert_run(conn, "run_a", run_a, "MSFOA", 0)
    _insert_run(conn, "run_b", run_b, "MSFOA", 1)
    conn.commit()
    return conn, exp_dir


class TestBuildReferenceParetoSets:
    def test_merged_front_dominates_inputs(self, two_runs_db) -> None:
        conn, exp_dir = two_runs_db
        refs = build_reference_pareto_sets(conn, exp_dir)
        assert ("forest", 2) in refs
        R = refs[("forest", 2)]
        assert R.shape[1] == 2
        # Punkty dominowane (np. [5,5]) NIE mogą być w R.
        for pt in R:
            assert not (pt[0] == 5.0 and pt[1] == 5.0)
        # Najlepszy w f1 (1.0) i najlepszy w f2 (1.0) muszą się znaleźć.
        coords = {tuple(r) for r in R}
        assert (1.0, 4.0) in coords or (1.5, 3.0) in coords
        assert (3.0, 1.0) in coords

    def test_persists_to_db(self, two_runs_db) -> None:
        conn, exp_dir = two_runs_db
        build_reference_pareto_sets(conn, exp_dir)
        rows = conn.execute(
            "SELECT environment, n_obj, point_idx, objective_j, value FROM reference_pareto_sets"
        ).fetchall()
        assert len(rows) > 0
        envs = {r[0] for r in rows}
        assert envs == {"forest"}

    def test_load_reference_set_roundtrip(self, two_runs_db) -> None:
        conn, exp_dir = two_runs_db
        refs = build_reference_pareto_sets(conn, exp_dir)
        loaded = load_reference_set(conn, "forest", 2)
        expected = refs[("forest", 2)]
        # Te same punkty (po sortowaniu).
        loaded_sorted = loaded[np.lexsort(loaded.T)]
        expected_sorted = expected[np.lexsort(expected.T)]
        np.testing.assert_allclose(loaded_sorted, expected_sorted)

    def test_idempotent(self, two_runs_db) -> None:
        conn, exp_dir = two_runs_db
        build_reference_pareto_sets(conn, exp_dir)
        n1 = conn.execute("SELECT COUNT(*) FROM reference_pareto_sets").fetchone()[0]
        build_reference_pareto_sets(conn, exp_dir)
        n2 = conn.execute("SELECT COUNT(*) FROM reference_pareto_sets").fetchone()[0]
        assert n1 == n2  # DELETE+INSERT, nie duplikuje


class TestReferencePoint:
    """Reference point r* dla HV — Ishibuchi 2018 §4: nadir + ε·(nadir−ideal)."""

    def test_persists_per_env_nobj(self, two_runs_db) -> None:
        conn, exp_dir = two_runs_db
        build_reference_pareto_sets(conn, exp_dir)
        rows = conn.execute(
            "SELECT environment, n_obj, objective_j, value, margin, method "
            "FROM reference_points ORDER BY objective_j"
        ).fetchall()
        assert len(rows) == 2  # 2-obj problem, 1 (env, n_obj) grupa
        for r in rows:
            assert r[0] == "forest"
            assert r[1] == 2
            assert r[4] == DEFAULT_REF_POINT_MARGIN
            assert r[5] == "nadir_plus_eps_range"

    def test_r_star_dominates_nadir_strictly(self, two_runs_db) -> None:
        # r* musi dominować nadir komponentowo (warunek konieczny by HV > 0).
        conn, exp_dir = two_runs_db
        refs = build_reference_pareto_sets(conn, exp_dir)
        R = refs[("forest", 2)]
        nadir = np.max(R, axis=0)
        r_star = load_reference_point(conn, "forest", 2)
        assert r_star is not None
        assert np.all(r_star > nadir), f"r*={r_star} musi być > nadir={nadir}"

    def test_load_returns_none_for_unknown(self, two_runs_db) -> None:
        conn, exp_dir = two_runs_db
        build_reference_pareto_sets(conn, exp_dir)
        assert load_reference_point(conn, "unknown_env", 2) is None
        assert load_reference_point(conn, "forest", 99) is None

    def test_idempotent_replace(self, two_runs_db) -> None:
        conn, exp_dir = two_runs_db
        build_reference_pareto_sets(conn, exp_dir)
        n1 = conn.execute("SELECT COUNT(*) FROM reference_points").fetchone()[0]
        build_reference_pareto_sets(conn, exp_dir)
        n2 = conn.execute("SELECT COUNT(*) FROM reference_points").fetchone()[0]
        assert n1 == n2 == 2  # 1 (env, n_obj) × 2 obj


class TestBackfillHV:
    """Backfill HV: po `backfill_moo_quality_with_reference` w
    `optimization_generation_stats` powinno być `hypervolume`."""

    def test_hv_inserted_after_backfill(self, two_runs_db) -> None:
        conn, exp_dir = two_runs_db
        refs = build_reference_pareto_sets(conn, exp_dir)
        backfill_moo_quality_with_reference(conn, refs)
        rows = conn.execute(
            "SELECT COUNT(*) FROM optimization_generation_stats "
            "WHERE source_name='moo_quality' AND metric_name='hypervolume'"
        ).fetchone()
        assert rows[0] > 0, "HV nie został wpisany do optimization_generation_stats"
