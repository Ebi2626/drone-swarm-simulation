"""Testy post-pass aggregatora `final_objective` = Σ w_i · F[i] / F_ref_env[i].

Setup: pusta baza + N runów z fixture'ami F_best + weights → asercja
że `final_objective` == oczekiwany weighted-normalized-sum.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.analysis.db.initialize_database import initialize_database
from src.analysis.db.populate_final_objective_aggregated import (
    populate_final_objective_aggregated,
    load_f_ref_per_environment,
)


def _insert_run(
    conn: sqlite3.Connection,
    run_id: str,
    environment: str,
    optimizer_algo: str,
    seed: int,
    f_best: list[float] | None,
    weights: list[float] | None,
) -> None:
    """Wstaw row do `runs` + `run_metrics` z opcjonalnym F_best i wagami."""
    conn.execute(
        """
        INSERT INTO runs (run_id, run_dir_name, source_path,
            optimizer_algo, avoidance_algo, environment, seed, algorithm_pair)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, run_id, "/tmp/" + run_id,
         optimizer_algo, optimizer_algo, environment, seed,
         f"{optimizer_algo} + {optimizer_algo}"),
    )
    cols = ["run_id"]
    vals: list = [run_id]
    if f_best is not None:
        cols += [
            "final_objective_f1_trajectory",
            "final_objective_f2_height_angle",
            "total_threat_cost",
            "total_turn_penalty",
            "total_coordination_cost",
        ]
        vals += [float(v) for v in f_best]
    if weights is not None:
        cols.append("final_objective_weights_json")
        vals.append(json.dumps(weights))
    placeholders = ", ".join("?" for _ in cols)
    conn.execute(
        f"INSERT INTO run_metrics ({', '.join(cols)}) VALUES ({placeholders})",
        vals,
    )


@pytest.fixture
def fresh_db():
    with tempfile.TemporaryDirectory() as tmp:
        exp_dir = Path(tmp) / "exp"
        exp_dir.mkdir()
        db_path = initialize_database(exp_dir)
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
            conn.commit()


class TestPerEnvironmentMedianFRef:
    def test_two_envs_isolated_medians(self, fresh_db) -> None:
        """Median F_ref liczone niezależnie per environment."""
        conn = fresh_db
        # forest: 3 runów z f1 ∈ {10, 20, 30}, urban: 3 runów z f1 ∈ {100, 200, 300}.
        # Median(forest)=[20,2,3,4,5], median(urban)=[200,20,30,40,50].
        weights = [0.05, 0.5, 0.8, 1.0, 0.25]
        _insert_run(conn, "r1", "forest", "msffoa", 1, [10, 1, 1, 1, 1], weights)
        _insert_run(conn, "r2", "forest", "ooa", 1, [20, 2, 3, 4, 5], weights)
        _insert_run(conn, "r3", "forest", "ssa", 1, [30, 5, 5, 5, 5], weights)
        _insert_run(conn, "r4", "urban", "msffoa", 1, [100, 10, 10, 10, 10], weights)
        _insert_run(conn, "r5", "urban", "ooa", 1, [200, 20, 30, 40, 50], weights)
        _insert_run(conn, "r6", "urban", "ssa", 1, [300, 50, 50, 50, 50], weights)

        populate_final_objective_aggregated(conn)

        f_ref = load_f_ref_per_environment(conn)
        assert set(f_ref.keys()) == {"forest", "urban"}
        # Median (3 values): drugi w sorted.
        np.testing.assert_allclose(
            f_ref["forest"], [20.0, 2.0, 3.0, 4.0, 5.0],
        )
        np.testing.assert_allclose(
            f_ref["urban"], [200.0, 20.0, 30.0, 40.0, 50.0],
        )

    def test_weighted_sum_formula(self, fresh_db) -> None:
        """final_objective = Σ w_i · F[i] / F_ref_env[i]."""
        conn = fresh_db
        weights = [0.05, 0.5, 0.8, 1.0, 0.25]
        # 3 runy w forest → median = [20, 2, 3, 4, 5].
        _insert_run(conn, "r1", "forest", "msffoa", 1, [10, 1, 1, 1, 1], weights)
        _insert_run(conn, "r2", "forest", "ooa", 1, [20, 2, 3, 4, 5], weights)
        _insert_run(conn, "r3", "forest", "ssa", 1, [30, 5, 5, 5, 5], weights)

        populate_final_objective_aggregated(conn)

        # Expected dla r2 (=median): Σ w_i · 1.0 = sum(w) = 2.6.
        # Expected dla r1: 0.05·(10/20) + 0.5·(1/2) + 0.8·(1/3) + 1.0·(1/4) + 0.25·(1/5)
        #                = 0.025 + 0.25 + 0.2667 + 0.25 + 0.05 = 0.8417
        # Expected dla r3: 0.05·(30/20) + 0.5·(5/2) + 0.8·(5/3) + 1.0·(5/4) + 0.25·(5/5)
        #                = 0.075 + 1.25 + 1.333 + 1.25 + 0.25 = 4.1583
        rows = dict(conn.execute(
            "SELECT run_id, final_objective FROM run_metrics"
        ).fetchall())
        assert rows["r1"] == pytest.approx(
            0.05 * 0.5 + 0.5 * 0.5 + 0.8 * (1/3) + 1.0 * 0.25 + 0.25 * 0.2,
            rel=1e-6,
        )
        assert rows["r2"] == pytest.approx(sum(weights), rel=1e-6)
        assert rows["r3"] == pytest.approx(
            0.05 * 1.5 + 0.5 * 2.5 + 0.8 * (5/3) + 1.0 * 1.25 + 0.25 * 1.0,
            rel=1e-6,
        )

    def test_zero_component_uses_floor(self, fresh_db) -> None:
        """F_ref_env ≤ 1e-9 → mianownik 1.0 (neutralny), nie inf."""
        conn = fresh_db
        weights = [0.05, 0.5, 0.8, 1.0, 0.25]
        # f3 (threat) = 0 we wszystkich runach → median(f3) = 0 → użyje 1.0.
        _insert_run(conn, "r1", "forest", "msffoa", 1, [10, 1, 0, 1, 1], weights)
        _insert_run(conn, "r2", "forest", "ooa", 1, [20, 2, 0, 4, 5], weights)
        _insert_run(conn, "r3", "forest", "ssa", 1, [30, 5, 0, 5, 5], weights)

        populate_final_objective_aggregated(conn)

        f_ref = load_f_ref_per_environment(conn)
        # f3 powinien być 1.0 (neutralny), nie 0.
        assert f_ref["forest"][2] == pytest.approx(1.0)

        # r2 (=median dla pozostałych obj): 0.05 + 0.5 + 0.8·0/1 + 1.0 + 0.25 = 1.8
        rows = dict(conn.execute(
            "SELECT run_id, final_objective FROM run_metrics"
        ).fetchall())
        assert rows["r2"] == pytest.approx(
            0.05 * 1 + 0.5 * 1 + 0.8 * 0 + 1.0 * 1 + 0.25 * 1, rel=1e-6,
        )


class TestRobustness:
    def test_idempotent_overwrite(self, fresh_db) -> None:
        """Re-run nadpisuje, nie duplikuje wpisów normalizacji."""
        conn = fresh_db
        weights = [0.05, 0.5, 0.8, 1.0, 0.25]
        _insert_run(conn, "r1", "forest", "msffoa", 1, [10, 1, 1, 1, 1], weights)
        _insert_run(conn, "r2", "forest", "ooa", 1, [20, 2, 3, 4, 5], weights)

        populate_final_objective_aggregated(conn)
        populate_final_objective_aggregated(conn)

        n = conn.execute(
            "SELECT COUNT(*) FROM offline_objective_normalization"
        ).fetchone()[0]
        assert n == 5  # 5 f_idx × 1 env

    def test_no_runs_no_op_with_warning(self, fresh_db, caplog) -> None:
        conn = fresh_db
        import logging
        with caplog.at_level(logging.WARNING):
            populate_final_objective_aggregated(conn)
        assert "brak runów z f_best" in caplog.text.lower()

    def test_missing_f_components_skipped(self, fresh_db) -> None:
        """Run z którąkolwiek per-obj kolumną NULL jest pomijany w median."""
        conn = fresh_db
        weights = [0.05, 0.5, 0.8, 1.0, 0.25]
        # r1 ma kompletny F, r2 ma NULL w jednej kolumnie → pomijany.
        _insert_run(conn, "r1", "forest", "msffoa", 1, [10, 1, 1, 1, 1], weights)
        conn.execute(
            """
            INSERT INTO runs (run_id, run_dir_name, source_path,
                optimizer_algo, avoidance_algo, environment, seed, algorithm_pair)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("r2", "r2", "/tmp/r2", "ooa", "ooa", "forest", 1, "ooa + ooa"),
        )
        # F2 NULL → niekompletny.
        conn.execute(
            """
            INSERT INTO run_metrics (run_id,
                final_objective_f1_trajectory, total_threat_cost,
                total_turn_penalty, total_coordination_cost,
                final_objective_weights_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("r2", 50.0, 5.0, 5.0, 5.0, json.dumps(weights)),
        )

        populate_final_objective_aggregated(conn)

        f_ref = load_f_ref_per_environment(conn)
        # Median = same r1 (r2 pominięty).
        np.testing.assert_allclose(f_ref["forest"], [10, 1, 1, 1, 1])

        # r2 nie dostaje final_objective (brak F2).
        fo_r2 = conn.execute(
            "SELECT final_objective FROM run_metrics WHERE run_id = 'r2'"
        ).fetchone()[0]
        assert fo_r2 is None

    def test_missing_weights_skipped(self, fresh_db) -> None:
        """Run bez weights_json nie dostaje final_objective."""
        conn = fresh_db
        weights = [0.05, 0.5, 0.8, 1.0, 0.25]
        _insert_run(conn, "r1", "forest", "msffoa", 1, [10, 1, 1, 1, 1], weights)
        _insert_run(conn, "r2", "forest", "ooa", 1, [20, 2, 3, 4, 5], weights=None)

        populate_final_objective_aggregated(conn)
        rows = dict(conn.execute(
            "SELECT run_id, final_objective FROM run_metrics"
        ).fetchall())
        assert rows["r1"] is not None
        assert rows["r2"] is None


class TestRankingPreservation:
    """Kluczowy invariant: ranking algos w jednym env-seed-bloku jest
    niezmienniczy względem F_ref median (vs faithful straight-line F_ref).
    """

    def test_ranking_invariant_within_env(self, fresh_db) -> None:
        conn = fresh_db
        weights = [0.05, 0.5, 0.8, 1.0, 0.25]
        # 3 algos w forest seed=1, każdy z różnym F.
        _insert_run(conn, "r_best", "forest", "nsga-3", 1,
                    [10, 1, 1, 1, 1], weights)
        _insert_run(conn, "r_med", "forest", "msffoa", 1,
                    [50, 5, 5, 5, 5], weights)
        _insert_run(conn, "r_worst", "forest", "ooa", 1,
                    [100, 10, 10, 10, 10], weights)

        populate_final_objective_aggregated(conn)
        rows = dict(conn.execute(
            "SELECT run_id, final_objective FROM run_metrics"
        ).fetchall())

        # Ranking: r_best < r_med < r_worst (lower=better).
        assert rows["r_best"] < rows["r_med"] < rows["r_worst"]
