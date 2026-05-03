"""IPathOptimizer dla NSGA-III (Non-dominated Sorting Genetic Algorithm III).

NSGA-III to wielokryterialny algorytm ewolucyjny zaprojektowany przez
Deba & Jaina (2014) dla many-objective optimization. W odróżnieniu od
mealpy SSA/OOA i custom MSFOA (które są single-objective), NSGA-III pracuje
na wektorze `[c_safety, c_energy, c_jerk, c_symmetry]` zwracanym przez
`WeightedSumFitness.evaluate_components`. Zachowuje to bezpośrednią
porównywalność z offline NSGA-3 strategią (`nsga3_swarm_strategy.py`),
która też używa multi-obj NSGA-III.

Pętla:
  1. Pymoo `minimize(problem, NSGA3, callback)` — pełna kontrola pętli przez
     pymoo, callback firuje per-generation z `budget.check_or_raise()`.
  2. Po konwergencji: Pareto-front zawiera N rozwiązań niezdominowanych.
     Selekcja `decision_mode="safety"` — wybieramy z Pareto rozwiązanie
     o najniższym `c_safety` (priorytet bezpieczeństwa nad gładkością/energią).

Reference: Deb & Jain (2014), "An Evolutionary Many-Objective Optimization
  Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
  Part I: Solving Problems With Box Constraints", IEEE T. Evol. Comp. 18(4).
"""
from __future__ import annotations

import logging
import time
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# Pymoo lazy imports — startup czas niezerowy.
try:
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.callback import Callback
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    _PYMOO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYMOO_AVAILABLE = False

from src.algorithms.avoidance.budget import BudgetExceeded, TimeBudget
from src.algorithms.avoidance.interfaces import (
    IPathOptimizer,
    OptimizationResult,
    PathProblem,
)


logger = logging.getLogger(__name__)


class _BudgetCallback:
    """Pymoo callback wywołujący `budget.check_or_raise()` per generację I
    capture'ujący best-so-far (regression fix 2026-05-03).

    Bez capture'owania best-so-far, gdy `BudgetExceeded` fires, NSGA-3 wracał
    `waypoints=None` mimo że populacja zawierała feasible candidates. Mealpy
    (SSA/OOA) nie ma tego problemu bo używa `max_time` natywnie i graceful'nie
    finalizuje. Tu capture'ujemy `algorithm.pop` po każdej generacji i przy
    BudgetExceeded zwracamy najlepszego non-sentinel osobnika.

    `BudgetExceeded` propaguje przez pymoo `minimize()` do top-level handlera.
    """

    def __init__(self, budget: TimeBudget) -> None:
        self.budget = budget
        self.generations_seen = 0
        # Capture best-so-far na koniec każdej gen (przed budget check).
        self.best_X: NDArray[np.float64] | None = None
        self.best_F: NDArray[np.float64] | None = None

    def __call__(self, algorithm) -> None:
        self.generations_seen += 1
        # Snapshot populacji ZANIM rzucimy BudgetExceeded.
        try:
            pop = algorithm.pop
            if pop is not None:
                X = pop.get("X")
                F = pop.get("F")
                if X is not None and F is not None and len(F) > 0:
                    self.best_X = np.atleast_2d(np.asarray(X, dtype=np.float64))
                    self.best_F = np.atleast_2d(np.asarray(F, dtype=np.float64))
        except Exception:
            pass  # snapshot best-effort; nie blokuje searchu
        self.budget.check_or_raise()


def _build_pymoo_problem(
    path_problem: PathProblem,
    bounds_lb: NDArray[np.float64],
    bounds_ub: NDArray[np.float64],
):
    """Buduje pymoo Problem z 1 obj = scalar `WeightedSumFitness.evaluate()`.

    Fairness fix 2026-05-03 (Krok 1): NSGA-III dotychczas używał multi-obj
    `evaluate_components` (4-D Pareto) + `decision_mode="safety"` (lexsort tylko
    po c_safety, IGNORUJE c_energy/jerk/symmetry). Pozostałe 3 algorytmy
    (SSA/OOA/MSFFOA) używają scalar weighted-sum przez `evaluate()`. Różne
    funkcje celu = nie porównujemy tej samej rzeczy. Tu unifikujemy: NSGA-III
    też scalar weighted-sum. NSGA-III zachowuje swoją mechanikę (ref-point
    niching nadal pracuje na decision space, tylko obj wymiar = 1).
    """
    if not _PYMOO_AVAILABLE:
        raise ImportError(
            "NSGA3OnlineOptimizer wymaga pakietu `pymoo`. Dodaj do environment.yaml."
        )
    fitness = path_problem.fitness

    n_var = int(len(bounds_lb))

    class _Problem(PymooProblem):
        def __init__(self):
            super().__init__(
                n_var=n_var,
                n_obj=1,
                xl=bounds_lb,
                xu=bounds_ub,
            )

        def _evaluate(self, X: NDArray[np.float64], out: dict, *args, **kwargs) -> None:
            # X: shape (pop, n_var). Scalar fitness per osobnik.
            n = len(X)
            F = np.empty((n, 1), dtype=np.float64)
            for i in range(n):
                spline = path_problem.path_repr.decode_genes(
                    np.asarray(X[i], dtype=np.float64), path_problem.context
                )
                F[i, 0] = fitness.evaluate(
                    spline, path_problem.context, path_problem.predictor
                )
            out["F"] = F

    return _Problem()


class NSGA3OnlineOptimizer(IPathOptimizer):
    """Online NSGA-III na YZ-genach z fitness multi-obj (4 składowe).

    Selekcja końcowa: `decision_mode`:
      - "safety"     : minimum `c_safety` (default; priorytet bezpieczeństwa)
      - "weighted"   : najniższy weighted-sum z `WeightedSumFitness.w_*`
      - "knee_point" : geometryczny knee w 4D Pareto (compromise solution)
    """

    def __init__(
        self,
        n_inner_waypoints: int = 5,
        epoch: int = 10,
        pop_size: int = 20,
        n_partitions: int = 4,
        decision_mode: Literal["safety", "weighted", "knee_point"] = "safety",
        min_compute_time_s: float = 0.05,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        """
        :param n_partitions: parametr `get_reference_directions("das-dennis", n_obj, n_partitions)`.
            Dla n_obj=4, n_partitions=4 → C(4+4-1, 4-1) = 35 ref dir.
            Mniejsze n_partitions = mniej ref dir = mniejszy pop_size potrzebny.
        :param decision_mode: jak wybrać single solution z Pareto-front. "safety"
            domyślnie — analogicznie do offline NSGA-3 strategii (Deb 2014).
        """
        if not _PYMOO_AVAILABLE:
            raise ImportError(
                "NSGA3OnlineOptimizer wymaga pakietu `pymoo`. Dodaj do environment.yaml."
            )
        if decision_mode not in ("safety", "weighted", "knee_point"):
            raise ValueError(
                f"decision_mode must be one of {{safety, weighted, knee_point}}, "
                f"got {decision_mode!r}."
            )
        self.n_inner_waypoints = int(n_inner_waypoints)
        self.epoch = int(epoch)
        self.pop_size = int(pop_size)
        self.n_partitions = int(n_partitions)
        self.decision_mode = decision_mode
        self.min_compute_time_s = float(min_compute_time_s)
        self.rng = rng

    def optimize(self, problem: PathProblem, budget: TimeBudget) -> OptimizationResult:
        t_start = time.perf_counter()

        if budget.remaining < self.min_compute_time_s:
            return OptimizationResult(
                waypoints=None,
                elapsed_s=time.perf_counter() - t_start,
                status="timed_out",
                extra={"reason": "budget_below_min_compute_time"},
            )

        try:
            lb, ub = problem.path_repr.gene_bounds(problem.context)
            pymoo_problem = _build_pymoo_problem(problem, lb, ub)

            # Fairness fix 2026-05-03 (Krok 1): n_obj=1 (scalar weighted-sum).
            # Dla 1-obj Das-Dennis(1, p) = 1 ref_dir (p irrelevant). To OK —
            # NSGA-III mechanika (ref-point niching) zachowana w decision space,
            # ale obj space jest 1-D więc niching nie wpływa na selekcję.
            ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=1)

            algorithm = NSGA3(
                ref_dirs=ref_dirs,
                pop_size=self.pop_size,
            )
            callback = _BudgetCallback(budget)

            res = minimize(
                pymoo_problem,
                algorithm,
                termination=("n_gen", self.epoch),
                seed=self.rng,
                callback=callback,
                verbose=False,
            )

            # res.F : (n_pareto, 4) — multi-obj values for Pareto front.
            # res.X : (n_pareto, n_var) — corresponding decision vectors.
            if res.X is None or res.F is None or len(res.F) == 0:
                return OptimizationResult(
                    waypoints=None,
                    elapsed_s=time.perf_counter() - t_start,
                    status="failed",
                    extra={"reason": "pymoo_returned_empty_pareto_front"},
                )

            # Pymoo res.X może być 1D (single solution) lub 2D (Pareto front).
            X = np.atleast_2d(res.X)
            F = np.atleast_2d(res.F)

            # Filtr sentinel-cost (regression fix 2026-05-02): candidates z
            # `decode_genes returning None` mają fitness `[1e9, 1e9, 1e9, 1e9]`.
            # `_select_from_pareto` po lexsort wybierałby ich nawet jeśli istnieją
            # feasible candidates z normalnym fitness. Zostawiamy tylko non-sentinel.
            SENTINEL_THRESHOLD = 1e8
            feasible_mask = F[:, 0] < SENTINEL_THRESHOLD
            if not np.any(feasible_mask):
                return OptimizationResult(
                    waypoints=None,
                    elapsed_s=time.perf_counter() - t_start,
                    status="failed",
                    extra={
                        "reason": "no_feasible_candidate_in_pareto",
                        "pareto_front_size": int(len(F)),
                        "generations_completed": int(callback.generations_seen),
                    },
                )
            X_feasible = X[feasible_mask]
            F_feasible = F[feasible_mask]

            # n_obj=1: pick lowest scalar fitness (argmin).
            best_idx = int(np.argmin(F_feasible[:, 0]))
            best_x = X_feasible[best_idx]
            best_F = F_feasible[best_idx]

            best_spline = problem.path_repr.decode_genes(best_x, problem.context)
            if best_spline is None:
                # Defensywne: feasible po fitness ale decode wrócił None
                # (np. gene_bounds zmienione między evaluations) — bardzo rzadkie.
                return OptimizationResult(
                    waypoints=None,
                    elapsed_s=time.perf_counter() - t_start,
                    status="failed",
                    extra={"reason": "best_decode_returned_none_post_filter"},
                )

            elapsed = time.perf_counter() - t_start
            generations = int(callback.generations_seen)
            return OptimizationResult(
                waypoints=np.asarray(best_spline.waypoints, dtype=np.float64),
                elapsed_s=elapsed,
                status="ok",
                extra={
                    "algorithm": "NSGA3",
                    "best_fitness": float(best_F[0]),
                    "evaluations_completed": generations * int(self.pop_size),
                    "generations_completed": generations,
                    "wallclock_s": elapsed,
                    "reason": "ok",
                },
            )

        except BudgetExceeded as e:
            elapsed = time.perf_counter() - t_start
            logger.warning(
                f"NSGA3OnlineOptimizer: d{problem.context.drone_id} — "
                f"BudgetExceeded po {elapsed:.3f}s ({e})"
            )
            # Best-so-far recovery (regression fix 2026-05-03): bez tego NSGA-3
            # zwracał None mimo że populacja miała feasible candidates. Mealpy
            # (SSA/OOA) graceful'nie kończy na max_time, my musimy zrobić to
            # ręcznie z capture'a w `_BudgetCallback`.
            try:
                if callback.best_X is not None and callback.best_F is not None:
                    F_so_far = np.atleast_2d(callback.best_F)
                    X_so_far = np.atleast_2d(callback.best_X)
                    SENTINEL_THRESHOLD = 1e8
                    feasible_mask = F_so_far[:, 0] < SENTINEL_THRESHOLD
                    if np.any(feasible_mask):
                        F_feas = F_so_far[feasible_mask]
                        X_feas = X_so_far[feasible_mask]
                        best_idx = int(np.argmin(F_feas[:, 0]))
                        best_x = X_feas[best_idx]
                        best_F_val = float(F_feas[best_idx, 0])
                        best_spline = problem.path_repr.decode_genes(
                            best_x, problem.context
                        )
                        if best_spline is not None:
                            generations = int(callback.generations_seen)
                            return OptimizationResult(
                                waypoints=np.asarray(
                                    best_spline.waypoints, dtype=np.float64
                                ),
                                elapsed_s=elapsed,
                                status="ok",
                                extra={
                                    "algorithm": "NSGA3",
                                    "best_fitness": best_F_val,
                                    "evaluations_completed": generations * int(self.pop_size),
                                    "generations_completed": generations,
                                    "wallclock_s": elapsed,
                                    "reason": "budget_exceeded_returned_best_so_far",
                                },
                            )
            except Exception as recover_err:
                logger.warning(
                    f"NSGA3OnlineOptimizer: best-so-far recovery failed: "
                    f"{recover_err}"
                )
            return OptimizationResult(
                waypoints=None,
                elapsed_s=elapsed,
                status="timed_out",
                extra={"reason": "cooperative_budget_exceeded"},
            )
        except Exception as e:
            elapsed = time.perf_counter() - t_start
            logger.error(
                f"NSGA3OnlineOptimizer: d{problem.context.drone_id} — "
                f"nieoczekiwany wyjątek: {e}",
                exc_info=True,
            )
            return OptimizationResult(
                waypoints=None,
                elapsed_s=elapsed,
                status="failed",
                extra={"reason": f"exception: {type(e).__name__}: {e}"},
            )

    def _select_from_pareto(
        self,
        F: NDArray[np.float64],
        fitness,
    ) -> int:
        """Z Pareto-front (F shape (n, 4)) wybierz indeks single solution."""
        if self.decision_mode == "safety":
            # Najniższe c_safety; tie-break po c_energy.
            order = np.lexsort((F[:, 1], F[:, 0]))
            return int(order[0])

        if self.decision_mode == "weighted":
            # Stosujemy wagi z `WeightedSumFitness` (jeśli dostępne).
            w = np.array([
                getattr(fitness, "w_safety", 1.0),
                getattr(fitness, "w_energy", 1.0),
                getattr(fitness, "w_jerk", 1.0),
                getattr(fitness, "w_symmetry", 1.0),
            ], dtype=np.float64)
            scores = F @ w
            return int(np.argmin(scores))

        if self.decision_mode == "knee_point":
            # Knee point: znormalizuj F do [0,1]^4, znajdź punkt o min |F_norm|.
            F_min = F.min(axis=0)
            F_max = F.max(axis=0)
            denom = np.maximum(F_max - F_min, 1e-9)
            F_norm = (F - F_min) / denom
            dists = np.linalg.norm(F_norm, axis=1)
            return int(np.argmin(dists))

        raise ValueError(f"Unknown decision_mode={self.decision_mode!r}")
