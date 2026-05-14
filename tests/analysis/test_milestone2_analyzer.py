"""Regression tests dla Kamień 2 — split offline/online + failure_rate.

Weryfikujemy:
1. `_dedup_offline` zwija duplikaty per (env, opt, seed) — kluczowe by Friedman
   nie traktował 4 avoidance variant-ów jako niezależnych datasetów (Demšar
   2006 §3.1 false-positive risk).
2. `_compute_failure_flag` poprawnie identyfikuje failure runy (HV=0,
   front_size=0, collision_count>0).
3. ExperimentAnalyzer eksportuje `failure_rate.csv` i bar plot.
4. Friedman z block_cols=("environment", "seed") na zdedupowanym DF zwraca
   `n_datasets` równe rzeczywistej liczbie (env × seed) — nie inflowane.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest


os.environ.setdefault("MPLBACKEND", "Agg")


class TestDedupOffline:
    def test_collapses_avoidance_variants(self) -> None:
        from src.analysis.analyzer.ExperimentAnalyzer import _dedup_offline

        # 1 (env,opt,seed) z 4 avoidance variants — wszystkie powinny się zwinąć.
        df = pd.DataFrame({
            "environment": ["forest"] * 4,
            "optimizer": ["msffoa"] * 4,
            "seed": [0] * 4,
            "avoidance": ["msffoa", "ooa", "ssa", "nsga-3"],
            "hypervolume": [10.0, 10.0, 10.0, 10.0],
        })
        out = _dedup_offline(df)
        assert len(out) == 1

    def test_preserves_distinct_seeds(self) -> None:
        from src.analysis.analyzer.ExperimentAnalyzer import _dedup_offline

        # 2 seedy × 4 avoidance = 8 wierszy; po dedup → 2.
        df = pd.DataFrame({
            "environment": ["forest"] * 8,
            "optimizer": ["msffoa"] * 8,
            "seed": [0, 0, 0, 0, 1, 1, 1, 1],
            "avoidance": ["msffoa", "ooa", "ssa", "nsga-3"] * 2,
            "hypervolume": [10.0] * 8,
        })
        out = _dedup_offline(df)
        assert len(out) == 2
        assert set(out["seed"]) == {0, 1}

    def test_empty_df_passes_through(self) -> None:
        from src.analysis.analyzer.ExperimentAnalyzer import _dedup_offline

        out = _dedup_offline(pd.DataFrame())
        assert out.empty


class TestComputeFailureFlag:
    """Nowa semantyka (zob. reports/failure_success_methodology.md):
    - is_offline_failure = tracking_phase_collisions > 0 (plan offline kolizyjny)
    - is_online_failure = evasion_phase_collisions > 0 (algorytm unikania zawiódł)
    - is_hv_degenerate = HV=0 OR front_size=0 (osobna diagnostyka, NIE failure)
    """

    def test_evasion_phase_collision_marks_online_failure(self) -> None:
        from src.analysis.analyzer.ExperimentAnalyzer import _compute_failure_flag

        df = pd.DataFrame({
            "tracking_phase_collisions": [0, 0],
            "evasion_phase_collisions": [0, 3],
            "front_size_last_gen": [10, 10],
            "hypervolume": [5.0, 5.0],
        })
        out = _compute_failure_flag(df)
        assert out["is_online_failure"].tolist() == [0, 1]
        assert out["is_offline_failure"].tolist() == [0, 0]

    def test_tracking_phase_collision_marks_offline_failure(self) -> None:
        from src.analysis.analyzer.ExperimentAnalyzer import _compute_failure_flag

        df = pd.DataFrame({
            "tracking_phase_collisions": [0, 2],
            "evasion_phase_collisions": [0, 0],
            "front_size_last_gen": [10, 10],
            "hypervolume": [5.0, 5.0],
        })
        out = _compute_failure_flag(df)
        assert out["is_offline_failure"].tolist() == [0, 1]
        assert out["is_online_failure"].tolist() == [0, 0]

    def test_zero_hv_marks_hv_degenerate_not_failure(self) -> None:
        from src.analysis.analyzer.ExperimentAnalyzer import _compute_failure_flag

        df = pd.DataFrame({
            "tracking_phase_collisions": [0, 0],
            "evasion_phase_collisions": [0, 0],
            "front_size_last_gen": [10, 10],
            "hypervolume": [5.0, 0.0],
        })
        out = _compute_failure_flag(df)
        # HV=0 to NIE jest failure — to osobna diagnostyka.
        assert out["is_offline_failure"].tolist() == [0, 0]
        assert out["is_online_failure"].tolist() == [0, 0]
        assert out["is_hv_degenerate"].tolist() == [0, 1]

    def test_zero_front_size_marks_hv_degenerate(self) -> None:
        from src.analysis.analyzer.ExperimentAnalyzer import _compute_failure_flag

        df = pd.DataFrame({
            "tracking_phase_collisions": [0, 0],
            "evasion_phase_collisions": [0, 0],
            "front_size_last_gen": [10, 0],
            "hypervolume": [5.0, 5.0],
        })
        out = _compute_failure_flag(df)
        assert out["is_hv_degenerate"].tolist() == [0, 1]
        assert out["is_offline_failure"].tolist() == [0, 0]

    def test_null_hv_marks_hv_degenerate(self) -> None:
        from src.analysis.analyzer.ExperimentAnalyzer import _compute_failure_flag

        df = pd.DataFrame({
            "tracking_phase_collisions": [0, 0],
            "evasion_phase_collisions": [0, 0],
            "front_size_last_gen": [10, 10],
            "hypervolume": [5.0, None],
        })
        out = _compute_failure_flag(df)
        assert out["is_hv_degenerate"].tolist() == [0, 1]
        assert out["is_offline_failure"].tolist() == [0, 0]

    def test_orthogonality_offline_and_online_failures(self) -> None:
        """Sprawdza wszystkie 4 kombinacje (offline, online) ∈ {0,1}²."""
        from src.analysis.analyzer.ExperimentAnalyzer import _compute_failure_flag

        df = pd.DataFrame({
            "tracking_phase_collisions": [0, 0, 2, 1],
            "evasion_phase_collisions":  [0, 1, 0, 3],
        })
        out = _compute_failure_flag(df)
        assert out["is_offline_failure"].tolist() == [0, 0, 1, 1]
        assert out["is_online_failure"].tolist()  == [0, 1, 0, 1]


class TestFriedmanBlockCols:
    def test_dedup_offline_avoids_inflated_n(self) -> None:
        """Bez dedup'u Friedman widziałby 4× duplikatów per (env, seed) i
        nadinflowałby N. Z dedup'em N = n_seeds (1 env × 4 seedy = 4)."""
        from src.analysis.analyzer.ExperimentAnalyzer import _dedup_offline
        from src.analysis.analyzer.statistical_tests import friedman_with_nemenyi

        rows = []
        for opt in ("a", "b", "c"):
            base = {"a": 1.0, "b": 2.0, "c": 3.0}[opt]
            for seed in range(4):
                for av in ("av1", "av2", "av3", "av4"):
                    rows.append({
                        "environment": "forest",
                        "optimizer": opt,
                        "seed": seed,
                        "avoidance": av,
                        "metric": base + 0.01 * seed,
                    })
        df = pd.DataFrame(rows)

        # WITHOUT dedup: N = 16 (4 seedy × 4 avoidance). Pseudo-replication.
        fr_inflated = friedman_with_nemenyi(
            df, metric="metric", block_cols=("environment", "seed", "avoidance"),
        )
        assert fr_inflated.n_datasets == 16

        # WITH dedup + block_cols=(env, seed): N = 4 (prawdziwe).
        df_dedup = _dedup_offline(df)
        fr_correct = friedman_with_nemenyi(
            df_dedup, metric="metric", block_cols=("environment", "seed"),
        )
        assert fr_correct.n_datasets == 4
