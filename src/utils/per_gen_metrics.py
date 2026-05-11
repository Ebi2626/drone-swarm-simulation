"""Wspólne helpery do obliczania per-generation metryk dla
`OptimizationHistoryWriter.put_generation_data(...)`.

Cel: ujednolicić między strategiami (SSA, MSFFOA, OOA, NSGA-III) dane
zapisywane do `optimization_history.h5`. ETL `_load_optimization_history`
oczekuje datasetów: `objectives_matrix`, `decisions_matrix`, `feasible_mask`,
`constraint_violation`, `elapsed_s`, `eval_count_cumulative`. Bez tych
ostatnich czterech kolumny `iteration_metrics.{feasible_*, elapsed_s,
eval_count_cumulative, constraint_violation_*}` byłyby 100% NULL.

Konwencje:
- `feasible_mask: np.ndarray (pop,) bool` — True gdy CV[i] ≤ ε.
- `constraint_violation: np.ndarray (pop,) float` — total CV per individual,
  Σ max(0, G_j) (pymoo convention G ≤ 0 = feasible).
- `elapsed_s: np.array([dt], float)` — wallclock danej generacji.
- `eval_count_cumulative: np.array([n_eval], int64)` — kumulatywna NFE
  (`evaluator.individuals_evaluated`).

Reference: Hansen, Auger, Ros, Finck & Pošík (2009) "Comparing Results of 31
Algorithms from the BBOB-2009", GECCO Companion §3.3 — NFE jako standard
porównywania algorytmów meta-heurystycznych.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


# Tolerancja constraint-violation: Σ max(0, G) ≤ FEASIBILITY_EPS ⇒ feasible.
FEASIBILITY_EPS = 1e-6


def per_gen_metrics_from_FG(
    objectives: np.ndarray,
    constraints: np.ndarray | None,
    decisions: np.ndarray,
    elapsed_s: float,
    eval_count_cumulative: int,
) -> Dict[str, np.ndarray]:
    """Złóż per-gen payload do `put_generation_data` z gotowych F + G.

    Args:
        objectives: `(pop, n_obj)` F-vector z `VectorizedEvaluator.evaluate`.
        constraints: `(pop, n_g)` G; `None` ⇒ feasibility = True dla wszystkich.
        decisions: `(pop, n_var)` macierz decyzyjna do zapisu.
        elapsed_s: Wallclock generacji [s].
        eval_count_cumulative: Kumulatywna NFE.

    Returns:
        Słownik `{objectives_matrix, decisions_matrix, constraint_violation,
        feasible_mask, elapsed_s, eval_count_cumulative}` gotowy do
        `OptimizationHistoryWriter.put_generation_data`.
    """
    F = np.asarray(objectives, dtype=np.float64)
    pop = F.shape[0]

    if constraints is None:
        cv = np.zeros(pop, dtype=np.float64)
        feasible = np.ones(pop, dtype=bool)
    else:
        G = np.asarray(constraints, dtype=np.float64)
        if G.ndim == 1:
            cv = np.maximum(0.0, G)
        else:
            cv = np.sum(np.maximum(0.0, G), axis=1)
        feasible = cv <= FEASIBILITY_EPS

    return {
        "objectives_matrix": F,
        "decisions_matrix": np.asarray(decisions, dtype=np.float64),
        "constraint_violation": cv.astype(np.float64),
        "feasible_mask": feasible.astype(np.bool_),
        "elapsed_s": np.array([float(elapsed_s)], dtype=np.float64),
        "eval_count_cumulative": np.array([int(eval_count_cumulative)], dtype=np.int64),
    }


def per_gen_metrics_re_evaluate(
    evaluator: Any,
    decisions_for_eval: np.ndarray,
    decisions_for_log: np.ndarray,
    elapsed_s: float,
    eval_count_cumulative: int,
) -> Dict[str, np.ndarray]:
    """Re-ewaluuj populację `evaluator.evaluate(...)` i zwróć payload per-gen.

    Re-eval zwiększa `evaluator.individuals_evaluated` — przekazuj
    `eval_count_cumulative` SPRZED re-evalu, by NFE odzwierciedlała tylko
    optymalizację, a nie instrumentację.

    Args:
        evaluator: `VectorizedEvaluator` z metodą `evaluate(X, out)`.
        decisions_for_eval: Dane wejściowe dla evaluatora (zwykle
            `(pop, n_drones, n_inner+2, 3)`).
        decisions_for_log: `(pop, n_var)` decyzje do zapisu.
        elapsed_s: Wallclock generacji [s].
        eval_count_cumulative: NFE pre-eval.

    Returns:
        Słownik per-gen jak w `per_gen_metrics_from_FG`.
    """
    out: Dict[str, Any] = {}
    evaluator.evaluate(decisions_for_eval, out)
    return per_gen_metrics_from_FG(
        objectives=out["F"],
        constraints=out.get("G"),
        decisions=decisions_for_log,
        elapsed_s=elapsed_s,
        eval_count_cumulative=eval_count_cumulative,
    )
