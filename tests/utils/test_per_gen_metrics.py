"""Unit tests dla `src.utils.per_gen_metrics`.

Helper liczy per-generation feasibility/CV/elapsed_s/eval_count_cumulative
z F+G; używany przez wszystkie strategie (SSA, OOA, MSFFOA, NSGA-III) jako
ujednolicony wkład do `OptimizationHistoryWriter.put_generation_data(...)`.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.per_gen_metrics import (
    FEASIBILITY_EPS,
    per_gen_metrics_from_FG,
)


class TestPerGenMetricsFromFG:
    def test_no_constraints_all_feasible(self) -> None:
        F = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        d = np.zeros((3, 5))
        out = per_gen_metrics_from_FG(F, None, d, elapsed_s=0.5, eval_count_cumulative=42)
        assert np.array_equal(out["objectives_matrix"], F)
        assert np.array_equal(out["decisions_matrix"], d)
        assert np.all(out["feasible_mask"])
        assert np.allclose(out["constraint_violation"], 0.0)
        assert out["elapsed_s"][0] == pytest.approx(0.5)
        assert out["eval_count_cumulative"][0] == 42

    def test_2d_constraints_summed(self) -> None:
        # G = (pop, n_g); CV[i] = Σ max(0, G[i, j])
        F = np.array([[1.0], [1.0], [1.0]])
        G = np.array([
            [-1.0, 0.0,  2.0],   # CV = 0 + 0 + 2 = 2 (infeasible)
            [-2.0, -3.0, -1.0],  # CV = 0 (feasible)
            [0.5, 0.5, -1.0],    # CV = 0.5 + 0.5 + 0 = 1.0 (infeasible)
        ])
        d = np.zeros((3, 4))
        out = per_gen_metrics_from_FG(F, G, d, elapsed_s=0.0, eval_count_cumulative=0)
        np.testing.assert_allclose(out["constraint_violation"], [2.0, 0.0, 1.0])
        assert list(out["feasible_mask"]) == [False, True, False]

    def test_1d_constraints_treated_as_total_cv(self) -> None:
        # 1D G traktowane jako już-zsumowany CV per individual
        # (np. legacy formaty z mealpy). max(0, ·) zachowane.
        F = np.array([[1.0], [2.0]])
        G = np.array([-0.5, 1.5])
        d = np.zeros((2, 3))
        out = per_gen_metrics_from_FG(F, G, d, elapsed_s=0.1, eval_count_cumulative=10)
        np.testing.assert_allclose(out["constraint_violation"], [0.0, 1.5])
        assert list(out["feasible_mask"]) == [True, False]

    def test_feasibility_epsilon_boundary(self) -> None:
        # CV ≤ FEASIBILITY_EPS ⇒ feasible. Boundary: CV == eps → feasible.
        F = np.array([[1.0], [1.0]])
        G = np.array([
            [FEASIBILITY_EPS / 2, 0.0],   # CV ≈ 5e-7 < eps → feasible
            [FEASIBILITY_EPS * 2, 0.0],   # CV ≈ 2e-6 > eps → infeasible
        ])
        d = np.zeros((2, 1))
        out = per_gen_metrics_from_FG(F, G, d, 0.0, 0)
        assert list(out["feasible_mask"]) == [True, False]

    def test_dtypes_explicit(self) -> None:
        F = np.array([[1.0]])
        d = np.zeros((1, 1))
        out = per_gen_metrics_from_FG(F, None, d, 0.5, 100)
        assert out["objectives_matrix"].dtype == np.float64
        assert out["decisions_matrix"].dtype == np.float64
        assert out["constraint_violation"].dtype == np.float64
        assert out["feasible_mask"].dtype == np.bool_
        assert out["elapsed_s"].dtype == np.float64
        assert out["eval_count_cumulative"].dtype == np.int64

    def test_elapsed_eval_count_shape_is_1d(self) -> None:
        # `_load_optimization_history` używa `_safe_array(ds, gen)` które
        # czyta `ds[gen]` — zakłada że zapisane datasety mają shape (n_gens, ...).
        # Stąd pojedyncza generacja musi dostarczać shape (1,).
        F = np.array([[1.0]])
        d = np.zeros((1, 1))
        out = per_gen_metrics_from_FG(F, None, d, 1.5, 99)
        assert out["elapsed_s"].shape == (1,)
        assert out["eval_count_cumulative"].shape == (1,)
