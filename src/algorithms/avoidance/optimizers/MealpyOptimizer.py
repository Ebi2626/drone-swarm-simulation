"""IPathOptimizer wrapper na mealpy (SSA, OOA, …) — Faza 2.

Wykorzystuje natywną terminację czasową mealpy (`termination={"max_time": s}`),
zamiast cooperative `budget.check_or_raise()` — mealpy honoruje to
deterministycznie po każdej generacji, więc nie potrzebujemy hooka w hot-loopie.

Jeśli mealpy zignoruje `max_time` (bug w bibliotece) — zewnętrzny SIGALRM
`hard_deadline()` w `GenericOptimizingAvoidance` przerwie symulację. To
zabezpieczenie jest naszym jedynym defense line dla bibliotek third-party.

Reference (mealpy API):
- Thieu Nguyen et al. (2023), "MEALPY: An open-source library for latest
  meta-heuristic algorithms in Python", Journal of Systems Architecture.
- mealpy.optimizer.Optimizer.solve(problem, termination={"max_time": …})
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray
try:
    from mealpy import FloatVar
    from mealpy import Problem as MealpyProblem
    _MEALPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MEALPY_AVAILABLE = False

from src.algorithms.avoidance.budget import TimeBudget
from src.algorithms.avoidance.interfaces import (
    IPathOptimizer,
    OptimizationResult,
    PathProblem,
)


logger = logging.getLogger(__name__)


class _AvoidanceMealpyProblem:
    """Adapter `mealpy.Problem` dla online avoidance.

    Wrapper jest stworzony dynamicznie wewnątrz `MealpyOptimizer.optimize`
    żeby uniknąć importu mealpy na poziomie modułu (lazy). Jeśli mealpy
    niedostępne, fabryka rzuca `ImportError`.
    """


def _make_mealpy_problem(
    path_problem: PathProblem,
    bounds_lb: NDArray[np.float64],
    bounds_ub: NDArray[np.float64],
):
    """Buduje konkretną instancję `mealpy.Problem` (closure nad PathProblem)."""
    if not _MEALPY_AVAILABLE:
        raise ImportError(
            "MealpyOptimizer wymaga pakietu `mealpy`. Dodaj do environment.yaml."
        )

    class _Problem(MealpyProblem):
        """Problem mealpy dla online avoidance.

        `obj_func(x)` dekoduje geny → BSpline (przez `path_repr.decode_genes`)
        → ocena `WeightedSumFitness.evaluate(spline)`. Nieprawidłowe dekodowanie
        → ekstremalna kara (1e9) by mealpy je wybrakował.
        """

        def __init__(self, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
            super().__init__(**kwargs)

        def obj_func(self, x: NDArray[np.float64]) -> float:  # type: ignore[override]
            spline = path_problem.path_repr.decode_genes(
                np.asarray(x, dtype=np.float64), path_problem.context
            )
            return float(
                path_problem.fitness.evaluate(
                    spline, path_problem.context, path_problem.predictor
                )
            )

    return _Problem(
        bounds=FloatVar(lb=bounds_lb.tolist(), ub=bounds_ub.tolist()),
        minmax="min",
        log_to=None,
    )


class MealpyOptimizer(IPathOptimizer):
    """Generyczny adapter `IPathOptimizer` na mealpy (SSA, OOA, …).

    Każdy mealpy-based config różni się tylko `algorithm_factory` w yamlu
    (Hydra `_partial_: true` na funkcji konstruującej `mealpy.Optimizer`).

    Kontrakt budżetu:
      - `budget.remaining` przekazywane do `model.solve(termination={"max_time"})`.
        Mealpy honoruje to natywnie — primary defense.
      - Jeśli `budget.remaining` < `min_compute_time_s`, optimizer zwraca
        `status="timed_out"` natychmiast (nie ma sensu odpalać meta-heuristics
        z budżetem 10 ms).
    """

    def __init__(
        self,
        algorithm_factory: Callable[..., Any],
        n_inner_waypoints: int = 5,  # Re-deklaracja dla spójności z BSplineYZGenes
        epoch: int = 10,
        pop_size: int = 20,
        min_compute_time_s: float = 0.05,
        rng: np.random.Generator | int | None = None,
        algorithm_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        :param algorithm_factory: callable zwracający instancję `mealpy.Optimizer`
            podany przez Hydrę z `_partial_: true` (np. `mealpy.swarm_based.SSA.OriginalSSA`).
            Sygnatura wywołania: `algorithm_factory(epoch=…, pop_size=…, **kwargs)`.
        :param n_inner_waypoints: K — musi być spójne z `path_representation.n_inner_waypoints`
            w yamlu. Trzymane tu redundantnie do walidacji.
        :param epoch: max_epoch dla mealpy (cap górny — natywne `max_time` powinno
            zwykle przerwać wcześniej w online).
        :param pop_size: rozmiar populacji.
        :param min_compute_time_s: minimalna ilość czasu by w ogóle uruchamiać
            meta-heuristic. Poniżej zwracamy `timed_out` od razu.
        :param rng: rng reproducibility (przekazywany do mealpy `solve(seed=…)`).
        :param algorithm_kwargs: dodatkowe kwargs przekazywane do `algorithm_factory`
            (np. dla OriginalSSA: `ST=0.8, PD=0.2, SD=0.1`).
        """
        if not _MEALPY_AVAILABLE:
            raise ImportError(
                "MealpyOptimizer wymaga pakietu `mealpy`. Dodaj do environment.yaml."
            )
        self.algorithm_factory = algorithm_factory
        self.n_inner_waypoints = int(n_inner_waypoints)
        self.epoch = int(epoch)
        self.pop_size = int(pop_size)
        self.min_compute_time_s = float(min_compute_time_s)
        self.rng = rng
        self.algorithm_kwargs = dict(algorithm_kwargs or {})

    def optimize(self, problem: PathProblem, budget: TimeBudget) -> OptimizationResult:
        t_start = time.perf_counter()

        if budget.remaining < self.min_compute_time_s:
            return OptimizationResult(
                waypoints=None,
                elapsed_s=time.perf_counter() - t_start,
                status="timed_out",
                extra={
                    "reason": "budget_below_min_compute_time",
                    "remaining_s": budget.remaining,
                    "min_required_s": self.min_compute_time_s,
                },
            )

        try:
            lb, ub = problem.path_repr.gene_bounds(problem.context)
            mealpy_problem = _make_mealpy_problem(problem, lb, ub)

            algo = self.algorithm_factory(
                epoch=self.epoch,
                pop_size=self.pop_size,
                **self.algorithm_kwargs,
            )

            # Mealpy `termination` jako dict. `max_time` ma pierwszeństwo —
            # mealpy sprawdza WSZYSTKIE warunki disjunctively, więc kończy
            # gdy KTÓRYKOLWIEK z {max_epoch, max_time} zostanie spełniony.
            termination = {
                "max_epoch": self.epoch,
                "max_time": max(self.min_compute_time_s, budget.remaining),
            }

            best_agent = algo.solve(
                problem=mealpy_problem,
                termination=termination,
                seed=self.rng,
            )

            best_x = np.asarray(best_agent.solution, dtype=np.float64)
            best_fitness = float(best_agent.target.fitness)

            # Filtr sentinel-cost (regression fix 2026-05-02): jeśli best_fitness
            # przekroczył próg sentinel (1e9 × min waga), to znaczy że WSZYSTKIE
            # candidates w populacji były infeasible — `decode_genes` na nich
            # wracał None i fitness leciał na sentinel. Zwracamy `no_feasible`
            # żeby drone wiedział że NIE znaleźliśmy planu (kontynuacja TRACKING).
            SENTINEL_THRESHOLD = 1e8
            if best_fitness > SENTINEL_THRESHOLD:
                return OptimizationResult(
                    waypoints=None,
                    elapsed_s=time.perf_counter() - t_start,
                    status="failed",
                    extra={
                        "reason": "no_feasible_candidate_in_population",
                        "best_fitness": best_fitness,
                    },
                )

            # Dekodujemy raz jeszcze by wyekstrahować waypointy do `OptimizationResult`.
            # `decode_genes` zwraca BSpline; potrzebujemy `spline.waypoints` (sparse).
            best_spline = problem.path_repr.decode_genes(best_x, problem.context)
            if best_spline is None:
                # Best gene zdekodowany w mealpy dał fitness, ale teraz None?
                # Numerycznie możliwe (race na floating point). Zwrot fail.
                logger.error(
                    f"MealpyOptimizer: d{problem.context.drone_id} — "
                    f"best_x decode_genes zwrócił None mimo zoptymalizowanego "
                    f"fitness={best_fitness}. Bug?"
                )
                return OptimizationResult(
                    waypoints=None,
                    elapsed_s=time.perf_counter() - t_start,
                    status="failed",
                    extra={"reason": "best_decode_returned_none"},
                )

            # Common-contract `extra` dict (Krok 4 fairness 2026-05-03):
            # `evaluations_completed`, `generations_completed`, `best_fitness`,
            # `wallclock_s`, `algorithm`, `reason` muszą być w każdym
            # `OptimizationResult.extra` z 4 optymalizatorów.
            elapsed = time.perf_counter() - t_start
            n_eval = int(getattr(algo.history, "list_global_best_fit", [None]).__len__())
            n_eval = n_eval * int(self.pop_size) if n_eval > 0 else 0
            return OptimizationResult(
                waypoints=np.asarray(best_spline.waypoints, dtype=np.float64),
                elapsed_s=elapsed,
                status="ok",
                extra={
                    "algorithm": type(algo).__name__,
                    "best_fitness": float(best_fitness),
                    "evaluations_completed": n_eval,
                    "generations_completed": n_eval // int(self.pop_size) if n_eval > 0 else 0,
                    "wallclock_s": elapsed,
                    "reason": "ok",
                },
            )

        except Exception as e:
            elapsed = time.perf_counter() - t_start
            logger.error(
                f"MealpyOptimizer: d{problem.context.drone_id} — "
                f"nieoczekiwany wyjątek: {e}",
                exc_info=True,
            )
            return OptimizationResult(
                waypoints=None,
                elapsed_s=elapsed,
                status="failed",
                extra={"reason": f"exception: {type(e).__name__}: {e}"},
            )
