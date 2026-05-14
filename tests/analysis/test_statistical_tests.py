"""Unit tests dla `statistical_tests` — Friedman + Nemenyi, A12, Wilson 95% CI.

Trzy testy statystyczne używane w eksperymencie (zob.
`reports/statistical_tests_methodology.md`):

- Friedman + Nemenyi (Demšar 2006) — global test + post-hoc parami
- Vargha-Delaney A12 (Vargha & Delaney 2000) — miara wielkości efektu
- Wilson 95% CI (Wilson 1927; Newcombe 1998) — przedział ufności dla proporcji
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.analyzer.statistical_tests import (
    friedman_with_nemenyi,
    summary_stats,
    vargha_delaney_a12,
    wilson_proportion_ci,
)


def _toy_df(perfect_winner: bool = True) -> pd.DataFrame:
    """3 algorytmy × 5 datasetów (env=fixed, seed varies). Algorytm 'A'
    zawsze ma niższe wartości niż 'B' i 'C' gdy perfect_winner=True."""
    records = []
    for seed in range(5):
        if perfect_winner:
            records.append({"optimizer": "A", "environment": "e1", "seed": seed, "metric": 1.0 + 0.05 * seed})
            records.append({"optimizer": "B", "environment": "e1", "seed": seed, "metric": 5.0 + 0.05 * seed})
            records.append({"optimizer": "C", "environment": "e1", "seed": seed, "metric": 9.0 + 0.05 * seed})
        else:
            records.append({"optimizer": "A", "environment": "e1", "seed": seed, "metric": 5.0})
            records.append({"optimizer": "B", "environment": "e1", "seed": seed, "metric": 5.0})
            records.append({"optimizer": "C", "environment": "e1", "seed": seed, "metric": 5.0})
    return pd.DataFrame(records)


class TestFriedman:
    def test_lower_is_better_ranks_winner_first(self) -> None:
        df = _toy_df(perfect_winner=True)
        result = friedman_with_nemenyi(df, metric="metric", higher_is_better=False)
        assert result.n_algorithms == 3
        assert result.n_datasets == 5
        assert result.average_ranks["A"] == pytest.approx(1.0, abs=1e-9)
        assert result.average_ranks["B"] == pytest.approx(2.0, abs=1e-9)
        assert result.average_ranks["C"] == pytest.approx(3.0, abs=1e-9)
        # CD dla k=3, N=5, alpha=0.05: q=2.343, sqrt(3*4/(6*5))=sqrt(0.4)
        expected_cd = 2.343 * np.sqrt(3 * 4 / (6 * 5))
        assert result.cd_nemenyi == pytest.approx(expected_cd, abs=1e-3)

    def test_higher_is_better_inverts_ranks(self) -> None:
        df = _toy_df(perfect_winner=True)
        result = friedman_with_nemenyi(df, metric="metric", higher_is_better=True)
        # Teraz C (najwyższe) ma rank 1.
        assert result.average_ranks["C"] == pytest.approx(1.0, abs=1e-9)
        assert result.average_ranks["A"] == pytest.approx(3.0, abs=1e-9)


class TestA12:
    def test_perfect_winner_has_a12_one(self) -> None:
        # Lower is better; A < B w 100% par ⇒ A12(A,B) = 1.0 (A "wygrywa").
        df = _toy_df(perfect_winner=True)
        results = vargha_delaney_a12(df, metric="metric", higher_is_better=False)
        for r in results:
            if r.alg_a == "A" and r.alg_b == "B":
                assert r.a12 == pytest.approx(1.0)
                assert r.magnitude == "large"
            if r.alg_a == "B" and r.alg_b == "A":
                pytest.fail("Nie powinno być (B, A)")

    def test_no_difference_a12_half(self) -> None:
        df = _toy_df(perfect_winner=False)
        results = vargha_delaney_a12(df, metric="metric", higher_is_better=False)
        for r in results:
            assert r.a12 == pytest.approx(0.5)
            assert r.magnitude == "negligible"


class TestSummaryStats:
    def test_summary_stats_columns(self) -> None:
        df = _toy_df(perfect_winner=True)
        summary = summary_stats(df, metric="metric")
        assert {
            "n", "mean", "std", "min", "max", "median", "q25", "q75",
        }.issubset(summary.columns)
        # Bootstrap CI columns no longer emitted.
        assert "ci95_low" not in summary.columns
        assert "ci95_high" not in summary.columns
        assert len(summary) == 3  # 3 algorytmy × 1 env
        # min ≤ q25 ≤ median ≤ q75 ≤ max — order statistics invariant.
        assert (summary["min"] <= summary["q25"]).all()
        assert (summary["q25"] <= summary["median"]).all()
        assert (summary["median"] <= summary["q75"]).all()
        assert (summary["q75"] <= summary["max"]).all()


class TestWilsonProportionCi:
    def test_zero_proportion_lower_bound_zero(self) -> None:
        """p̂=0 → ci_low = 0 (Wald CI dałby ujemne)."""
        lo, hi = wilson_proportion_ci(0, 30)
        assert lo == 0.0
        assert hi > 0  # górna granica > 0 (Wilson nie zerowy)

    def test_full_proportion_upper_bound_one(self) -> None:
        """p̂=1 → ci_high ≈ 1 (Wald dałby >1)."""
        lo, hi = wilson_proportion_ci(30, 30)
        assert hi == pytest.approx(1.0, abs=1e-9)
        assert lo < 1.0

    def test_ci_brackets_point_estimate(self) -> None:
        """CI musi zawierać p̂ wewnątrz."""
        lo, hi = wilson_proportion_ci(7, 30)
        p_hat = 7 / 30
        assert lo < p_hat < hi

    def test_zero_trials_returns_nan(self) -> None:
        lo, hi = wilson_proportion_ci(0, 0)
        assert np.isnan(lo) and np.isnan(hi)

    def test_typical_safety_critical_case(self) -> None:
        """NSGA-III: 2 failures of 30 — sanity check known good interval."""
        lo, hi = wilson_proportion_ci(2, 30)
        # Wilson CI dla 2/30 ≈ [0.019, 0.213]
        assert 0.01 < lo < 0.03
        assert 0.19 < hi < 0.23
