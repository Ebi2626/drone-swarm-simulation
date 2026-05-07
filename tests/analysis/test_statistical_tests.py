"""Unit tests dla `statistical_tests` — Friedman, Wilcoxon, A12, bootstrap.

Reference: Demšar (2006); Vargha & Delaney (2000); Arcuri & Briand (2014).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.analyzer.statistical_tests import (
    bootstrap_ci,
    friedman_with_nemenyi,
    summary_with_ci,
    vargha_delaney_a12,
    wilcoxon_pairwise,
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


class TestWilcoxon:
    def test_pairwise_with_holm_correction(self) -> None:
        df = _toy_df(perfect_winner=True)
        pairs = wilcoxon_pairwise(df, metric="metric")
        assert len(pairs) == 3  # C(3,2)=3 pary
        for p in pairs:
            assert p.p_value <= p.p_value_holm + 1e-12
            assert p.n == 5
        # Wszystkie różnice istotne kierunkowo.
        for p in pairs:
            if p.alg_a == "A":
                assert p.median_diff < 0  # A < B i A < C
            if p.alg_a == "B":
                assert p.median_diff < 0  # B < C


class TestA12:
    def test_perfect_winner_has_a12_zero(self) -> None:
        # Lower is better; A < B w 100% par ⇒ A12(A,B) = 1.0 (A "wygrywa").
        df = _toy_df(perfect_winner=True)
        results = vargha_delaney_a12(df, metric="metric", higher_is_better=False)
        for r in results:
            if r.alg_a == "A" and r.alg_b == "B":
                assert r.a12 == pytest.approx(1.0)
                assert r.magnitude == "large"
            if r.alg_a == "B" and r.alg_b == "A":
                # nieistnieje — pętla generuje a < b w sortowaniu alfa
                pytest.fail("Nie powinno być (B, A)")

    def test_no_difference_a12_half(self) -> None:
        df = _toy_df(perfect_winner=False)
        results = vargha_delaney_a12(df, metric="metric", higher_is_better=False)
        for r in results:
            assert r.a12 == pytest.approx(0.5)
            assert r.magnitude == "negligible"


class TestBootstrap:
    def test_constant_vector_zero_width(self) -> None:
        point, lo, hi = bootstrap_ci([3.0, 3.0, 3.0], n_resamples=500, rng_seed=0)
        assert point == 3.0 and lo == 3.0 and hi == 3.0

    def test_summary_with_ci_shape(self) -> None:
        df = _toy_df(perfect_winner=True)
        summary = summary_with_ci(df, metric="metric", n_resamples=200)
        assert {"n", "mean", "std", "median", "q25", "q75", "ci95_low", "ci95_high"}.issubset(
            summary.columns
        )
        assert len(summary) == 3  # 3 algorytmy × 1 env

    def test_empty_input_returns_nan(self) -> None:
        point, lo, hi = bootstrap_ci([], n_resamples=10)
        assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)
