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
    """Składa dict-słownik gotowy do `put_generation_data` z F+G (już policzone).

    Args:
        objectives: F z `VectorizedEvaluator.evaluate(...)`, shape (pop, n_obj).
        constraints: G z `VectorizedEvaluator.evaluate(...)`, shape (pop, n_g)
            lub None (brak ograniczeń → feasibility=True dla wszystkich).
        decisions: macierz decyzyjna w postaci do zapisu, shape (pop, n_var).
        elapsed_s: czas wallclock tej generacji (sekundy).
        eval_count_cumulative: NFE kumulatywna (zwykle
            `evaluator.individuals_evaluated`).
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
    """Wariant gdy F/G nie są zacache'owane — re-ewaluuje populację via
    `evaluator.evaluate(...)`.

    Uwaga: re-eval zwiększa `evaluator.individuals_evaluated` — przekazuj
    `eval_count_cumulative` ZE SNAPSHOTU PRZED re-evalem, by liczba
    odzwierciedlała tylko optymalizację, nie instrumentację.

    Args:
        evaluator: VectorizedEvaluator instance.
        decisions_for_eval: dane wejściowe dla `evaluator.evaluate(...)`,
            shape (pop, n_drones, n_inner+2, 3) zwykle.
        decisions_for_log: macierz do zapisu w `decisions_matrix`,
            zwykle reshape do (pop, n_var).
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
