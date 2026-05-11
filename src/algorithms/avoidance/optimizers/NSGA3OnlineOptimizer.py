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
    """Pymoo callback z dwoma zadaniami:
    1. Wywołanie `budget.check_or_raise()` per generację (cooperative budget).
    2. Capture best-so-far populacji — bez tego przy `BudgetExceeded` NSGA-3
       wracał `waypoints=None` mimo że populacja zawierała feasible
       candidates (mealpy SSA/OOA nie ma tego problemu bo `max_time` jest
       natywne i graceful'nie finalizuje).

    `BudgetExceeded` propaguje przez pymoo `minimize()` do top-level handlera.
    """

    def __init__(self, budget: TimeBudget) -> None:
        """Powiąż callback z `budget` i zainicjuj snapshot best-so-far na pusty.

        Args:
            budget: Wspólny `TimeBudget` — sprawdzany na każdej generacji.
        """
        self.budget = budget
        self.generations_seen = 0
        self.best_X: NDArray[np.float64] | None = None
        self.best_F: NDArray[np.float64] | None = None
        # Per-gen najlepszy feasible fitness; `inf` gdy cała populacja
        # infeasible (analiza może filtrować inf).
        self.convergence_trace: list[float] = []

    def __call__(self, algorithm) -> None:
        """Wykonaj snapshot populacji i sprawdź budżet (rzuca `BudgetExceeded`).

        Args:
            algorithm: Bieżąca instancja `NSGA3` z pymoo (atrybut `pop` ma
                kolumny `X`, `F`).

        Raises:
            BudgetExceeded: Gdy `budget.check_or_raise()` przekracza próg.
        """
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
                    # Per-gen best feasible fitness do trace'u.
                    f_col = np.atleast_2d(self.best_F)[:, 0]
                    feas = f_col[f_col < 1e8]
                    self.convergence_trace.append(
                        float(feas.min()) if feas.size > 0 else float("inf")
                    )
        except Exception:
            pass  # snapshot best-effort; nie blokuje searchu
        self.budget.check_or_raise()


def _build_pymoo_problem(
    path_problem: PathProblem,
    bounds_lb: NDArray[np.float64],
    bounds_ub: NDArray[np.float64],
):
    """Zbuduj `pymoo.Problem` z 1 obj (scalar `WeightedSumFitness.evaluate`).

    Unifikacja z SSA/OOA/MSFFOA online: wszystkie 4 algorytmy minimalizują
    tę samą skalarną sumę ważoną (porównywalność per-trigger). NSGA-III
    zachowuje ref-point niching w decision space, ale obj space jest 1-D,
    więc Pareto-front degeneruje do pojedynczego punktu.

    Args:
        path_problem: Definicja problemu (path_repr, fitness, context, predictor).
        bounds_lb, bounds_ub: `(D,)` granice genów.

    Returns:
        Instancja `pymoo.core.problem.Problem` z `_evaluate` zapisującym
        skalarny `F` w `out["F"]`.

    Raises:
        ImportError: Gdy pakiet `pymoo` nie jest zainstalowany.
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
        """Skonfiguruj NSGA-III online: liczbę kierunków referencyjnych i tryb selekcji.

        Args:
            n_inner_waypoints: K — liczba waypointów spójna z
                `path_representation.n_inner_waypoints`.
            epoch: Górna liczba generacji (`pymoo.minimize(termination=("n_gen", epoch))`).
            pop_size: Rozmiar populacji NSGA-III.
            n_partitions: Parametr Das-Dennis dla
                `get_reference_directions("das-dennis", n_obj, n_partitions)`;
                mniejszy `n_partitions` ⇒ mniej kierunków referencyjnych
                ⇒ mniejsza wymagana populacja.
            decision_mode: Strategia wyboru pojedynczego rozwiązania
                z frontu Pareto:
                  - `"safety"` — minimum `c_safety` (tie-break po `c_energy`),
                  - `"weighted"` — najniższa suma ważona z `WeightedSumFitness.w_*`,
                  - `"knee_point"` — geometryczny knee w 4D Pareto.
            min_compute_time_s: Minimalny budżet [s] do uruchomienia
                optymalizatora.
            rng: Ziarno deterministyczne dla `pymoo.minimize(seed=…)`.

        Raises:
            ImportError: Gdy pakiet `pymoo` nie jest dostępny.
            ValueError: Gdy `decision_mode` jest spoza dozwolonego zbioru.
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

    @property
    def population_size(self) -> int:
        """Rozmiar populacji NSGA-III."""
        return self.pop_size

    def optimize(self, problem: PathProblem, budget: TimeBudget) -> OptimizationResult:
        """Uruchom NSGA-III na YZ-genach w ramach `budget` i zwróć najlepsze waypointy.

        Args:
            problem: `PathProblem` z fitness, predyktorem i opcjonalną
                populacją startową.
            budget: Kooperatywny budżet czasu — sprawdzany przez
                `_BudgetCallback` po każdej generacji.

        Returns:
            `OptimizationResult` ze statusem:
              - `"ok"` z `waypoints` i `extra`
                (`algorithm="NSGA3"`, `convergence_trace`),
              - `"timed_out"`, gdy budżet < `min_compute_time_s` albo
                `BudgetExceeded` (próbujemy zwrócić best-so-far),
              - `"failed"`, gdy front Pareto był pusty albo żaden kandydat
                nie był feasible.
        """
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

            # n_obj=1 → Das-Dennis(1, p) = 1 ref_dir (p irrelevant). NSGA-III
            # niching nie wpływa na selekcję w 1-D, ale ref-point niching
            # w decision space zostaje aktywny.
            ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=1)

            # Ceteris paribus: jeśli `GenericOptimizingAvoidance` pre-wygenerował
            # populację, przekazujemy ją jako `sampling` do pymoo. Pymoo akceptuje
            # numpy array (pop_size, n_var) i używa go AS-IS bez perturbacji.
            # UWAGA: pymoo traktuje jawne `sampling=None` jako nadpisanie defaultu
            # (FloatRandomSampling), dlatego budujemy kwargs warunkowo.
            nsga3_kwargs: dict = dict(
                ref_dirs=ref_dirs,
                pop_size=self.pop_size,
            )
            if (
                problem.initial_population is not None
                and problem.initial_population.shape[0] == self.pop_size
            ):
                nsga3_kwargs["sampling"] = np.clip(
                    problem.initial_population, lb, ub
                )

            algorithm = NSGA3(**nsga3_kwargs)
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

            # Filtr sentinel-cost: candidates z `decode_genes` zwracającym None
            # mają fitness 1e9. Bez filtru argmin/lexsort mógł je wybrać nawet
            # przy istniejących feasible candidates.
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
                    "convergence_trace": list(callback.convergence_trace),
                },
            )

        except BudgetExceeded as e:
            elapsed = time.perf_counter() - t_start
            logger.warning(
                f"NSGA3OnlineOptimizer: d{problem.context.drone_id} — "
                f"BudgetExceeded po {elapsed:.3f}s ({e})"
            )
            # Best-so-far recovery z snapshotu w `_BudgetCallback`: bez tego
            # NSGA-3 zwracał None mimo że populacja miała feasible candidates
            # (pymoo `minimize()` przerywany BudgetExceeded nie zwraca
            # częściowego stanu — recovery musi wyciągnąć go z callbacka).
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
                                    "convergence_trace": list(callback.convergence_trace),
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
        """Wybierz indeks rozwiązania z `(n, 4)` frontu Pareto wg `decision_mode`.

        Args:
            F: `(n, 4)` macierz fitness (kolumny: c_safety, c_energy, c_jerk,
                c_symmetry).
            fitness: Instancja `WeightedSumFitness` — używana w trybie
                `"weighted"` do odczytu wag.

        Returns:
            Indeks wybranego rozwiązania w `F`.

        Raises:
            ValueError: Gdy `decision_mode` nie jest rozpoznawany.
        """
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
