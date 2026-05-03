"""IPathOptimizer dla MSFOA (Multiple Swarm Fruit Fly Optimization Algorithm).

Reimplementacja paperu Shi, Zhang & Xia (2020), "Multiple Swarm Fruit Fly
Optimization Algorithm Based Path Planning Method for Multi-UAVs", Applied
Sciences 10(8):2822, na **płaskim wektorze YZ-genów** (zamiast 3D tensora
z `core_msffoa.MSFFOAOptimizer`).

Mapowanie sekcji paperu → kod:
  Sec. 1     : podział populacji na G podrojów + walidacja `coe1 + coe2 = 1`.
  Sec. 2 Eq.7: faza globalna `X = x_best + step_global · sin(2·U(-1,1))`.
  Sec. 2 Eq.8: faza lokalna  `X = x_best + step_local · U(-1,1)`.
  Sec. 4 Eq.14: krzyżowanie międzyrojowe `coe1·leader_random + coe2·X_g,i`.
  Sec. 4 Eq.18-19: elityzm liderów roju (update tylko jeśli winner < best historycznie).

Adaptacje vs paper:
  - Anizotropowe `step_global_frac` / `step_local_frac` per gen — nie trzeba
    rozróżniać Y/Z (jeden frac dla obu osi w online, bo bounds są symetryczne
    wokół 0).
  - Brak smell-space transform (Sec. 3) — operujemy bezpośrednio na YZ-deltach
    BSplineYZGenes. Skalowanie kroku rekompensuje brak smell-to-physical mappingu.
  - Threshold (granica między fazą globalną a lokalną) skalibrowany dynamicznie
    z `initial_fitness * threshold_ratio` — paper podaje threshold jako stały
    parametr, ale bezwzględna wartość zależy od skali fitness, którą znamy
    dopiero po inicjalizacji.

Kontrakt budżetu:
  - `budget.check_or_raise()` wywoływane na początku każdej generacji.
  - `BudgetExceeded` → `OptimizationResult(status="timed_out", waypoints=best_so_far)`.
"""
from __future__ import annotations

import logging
import time

import numpy as np
from numpy.typing import NDArray

from src.algorithms.avoidance.budget import BudgetExceeded, TimeBudget
from src.algorithms.avoidance.interfaces import (
    IPathOptimizer,
    OptimizationResult,
    PathProblem,
)


logger = logging.getLogger(__name__)


class MSFFOAOnlineOptimizer(IPathOptimizer):
    """Online MSFOA na YZ-genach. Reference: Shi et al. 2020 (literatura w pliku).

    Parametry domyślne dobrane do online (1 s budżet, K=5 inner waypts):
      - pop_size=20, n_swarms=4 → P=5 fly/swarm
      - max_generations=10 (cooperative budget zwykle przerwie wcześniej)
      - step_global_frac=0.10, step_local_frac=0.03 (10% i 3% zakresu genów —
        większe niż offline, bo online ma mniej generacji do konwergencji).
    """

    def __init__(
        self,
        n_inner_waypoints: int = 5,
        pop_size: int = 20,
        n_swarms: int = 4,
        max_generations: int = 10,
        coe1: float = 0.8,
        coe2: float = 0.2,
        threshold_ratio: float = 0.5,
        step_global_frac: float = 0.10,
        step_local_frac: float = 0.03,
        min_compute_time_s: float = 0.05,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        if abs((coe1 + coe2) - 1.0) > 1e-6:
            raise ValueError(
                f"MSFOA paper (Shi et al. 2020) wymaga coe1 + coe2 = 1, "
                f"otrzymano coe1={coe1}, coe2={coe2}."
            )
        if n_swarms < 1:
            raise ValueError(f"n_swarms must be ≥ 1, got {n_swarms}.")
        if pop_size < n_swarms or pop_size % n_swarms != 0:
            raise ValueError(
                f"pop_size ({pop_size}) must be ≥ n_swarms ({n_swarms}) and "
                f"divisible by n_swarms."
            )
        if step_global_frac <= 0 or step_local_frac <= 0:
            raise ValueError("step_global_frac and step_local_frac must be > 0.")

        self.n_inner_waypoints = int(n_inner_waypoints)
        self.pop_size = int(pop_size)
        self.G = int(n_swarms)
        self.P = self.pop_size // self.G
        self.max_generations = int(max_generations)
        self.coe1 = float(coe1)
        self.coe2 = float(coe2)
        self.threshold_ratio = float(threshold_ratio)
        self.step_global_frac = float(step_global_frac)
        self.step_local_frac = float(step_local_frac)
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

        rng = np.random.default_rng(self.rng)
        ctx = problem.context
        path_repr = problem.path_repr

        try:
            lb, ub = path_repr.gene_bounds(ctx)
            K = int(path_repr.gene_dim(ctx))
            assert lb.shape == (K,) and ub.shape == (K,), "gene_bounds shape mismatch"

            # === Inicjalizacja populacji w jednolitym rozkładzie U(lb, ub) ============
            pop = lb[None, :] + rng.uniform(0.0, 1.0, size=(self.pop_size, K)) * (ub - lb)[None, :]
            fits = self._eval_batch(pop, problem)

            # Reshape na (G, P, K) dla operacji per-swarm.
            pop_swarm = pop.reshape(self.G, self.P, K)
            fits_swarm = fits.reshape(self.G, self.P)

            best_idx_per_swarm = np.argmin(fits_swarm, axis=1)  # (G,)
            swarm_best_pos = pop_swarm[np.arange(self.G), best_idx_per_swarm].copy()  # (G, K)
            swarm_best_fit = fits_swarm[np.arange(self.G), best_idx_per_swarm].copy()  # (G,)

            global_best_idx = int(np.argmin(swarm_best_fit))
            global_best_fit = float(swarm_best_fit[global_best_idx])
            global_best_pos = swarm_best_pos[global_best_idx].copy()

            # Threshold dynamiczny — separuje fazę globalną od lokalnej.
            threshold = max(1e-3, global_best_fit * self.threshold_ratio)

            # Anizotropowe step (per-gene zakres = ub - lb).
            step_global = (ub - lb) * self.step_global_frac  # (K,)
            step_local = (ub - lb) * self.step_local_frac  # (K,)

            generations_completed = 0

            # === Pętla generacji =================================================
            for gen in range(self.max_generations):
                # COOPERATIVE BUDGET CHECK
                budget.check_or_raise()

                # ---- Phase 1: search around swarm_best (Eq. 7-8) ----
                rand_vals = rng.uniform(-1.0, 1.0, size=(self.G, self.P, K))

                is_global = swarm_best_fit > threshold  # (G,) bool
                # Per-swarm: globalna lub lokalna eksploracja.
                step_global_val = step_global[None, None, :] * np.sin(2.0 * rand_vals)
                step_local_val = step_local[None, None, :] * rand_vals
                steps = np.where(
                    is_global[:, None, None], step_global_val, step_local_val
                )
                old_pop = swarm_best_pos[:, None, :] + steps  # (G, P, K)
                old_pop = np.clip(old_pop, lb, ub)

                old_fits = self._eval_batch(
                    old_pop.reshape(self.pop_size, K), problem
                ).reshape(self.G, self.P)

                # ---- Phase 2: cross-swarm leader crossover (Eq. 14) ----
                rand_leader_idx = rng.integers(0, self.G, size=(self.G, self.P))
                selected_leaders = swarm_best_pos[rand_leader_idx]  # (G, P, K)
                new_pop = self.coe1 * selected_leaders + self.coe2 * old_pop
                new_pop = np.clip(new_pop, lb, ub)

                new_fits = self._eval_batch(
                    new_pop.reshape(self.pop_size, K), problem
                ).reshape(self.G, self.P)

                # ---- Phase 3: competition + elitism (Eq. 18-19) ----
                old_best_local_idx = np.argmin(old_fits, axis=1)
                old_best_pos = old_pop[np.arange(self.G), old_best_local_idx]
                old_best_fit_local = old_fits[np.arange(self.G), old_best_local_idx]

                new_best_local_idx = np.argmin(new_fits, axis=1)
                new_best_pos = new_pop[np.arange(self.G), new_best_local_idx]
                new_best_fit_local = new_fits[np.arange(self.G), new_best_local_idx]

                win_is_new = new_best_fit_local < old_best_fit_local  # (G,)
                winner_fit = np.where(win_is_new, new_best_fit_local, old_best_fit_local)
                winner_pos = np.where(
                    win_is_new[:, None], new_best_pos, old_best_pos
                )

                # Elitism: lider zastępowany TYLKO gdy winner < swarm_best historycznie.
                update_mask = winner_fit < swarm_best_fit  # (G,)
                swarm_best_fit = np.where(update_mask, winner_fit, swarm_best_fit)
                swarm_best_pos = np.where(update_mask[:, None], winner_pos, swarm_best_pos)

                # Update global best.
                cur_global_idx = int(np.argmin(swarm_best_fit))
                if swarm_best_fit[cur_global_idx] < global_best_fit:
                    global_best_fit = float(swarm_best_fit[cur_global_idx])
                    global_best_pos = swarm_best_pos[cur_global_idx].copy()

                generations_completed = gen + 1

            # === Wynik ===========================================================
            # Filtr sentinel-cost (regression fix 2026-05-02): jeśli global_best_fit
            # > sentinel threshold → wszystkie candidates infeasible (decode_genes
            # zwracał None), wracamy `no_feasible` zamiast próbować decode best.
            SENTINEL_THRESHOLD = 1e8
            if global_best_fit > SENTINEL_THRESHOLD:
                return OptimizationResult(
                    waypoints=None,
                    elapsed_s=time.perf_counter() - t_start,
                    status="failed",
                    extra={
                        "reason": "no_feasible_candidate_in_population",
                        "best_fitness": global_best_fit,
                        "generations_completed": generations_completed,
                    },
                )

            best_spline = path_repr.decode_genes(global_best_pos, ctx)
            if best_spline is None:
                logger.error(
                    f"MSFFOAOnlineOptimizer: d{ctx.drone_id} — best gene decode_genes "
                    f"zwrócił None mimo fitness={global_best_fit}. Bug?"
                )
                return OptimizationResult(
                    waypoints=None,
                    elapsed_s=time.perf_counter() - t_start,
                    status="failed",
                    extra={"reason": "best_decode_returned_none"},
                )

            # Common-contract `extra` dict (Krok 4 fairness 2026-05-03).
            elapsed = time.perf_counter() - t_start
            return OptimizationResult(
                waypoints=np.asarray(best_spline.waypoints, dtype=np.float64),
                elapsed_s=elapsed,
                status="ok",
                extra={
                    "algorithm": "MSFOA",
                    "best_fitness": float(global_best_fit),
                    "evaluations_completed": int(generations_completed) * int(self.pop_size),
                    "generations_completed": int(generations_completed),
                    "wallclock_s": elapsed,
                    "reason": "ok",
                    # MSFFOA-specific (extra-extra, optional):
                    "n_swarms": self.G,
                },
            )

        except BudgetExceeded as e:
            elapsed = time.perf_counter() - t_start
            logger.warning(
                f"MSFFOAOnlineOptimizer: d{ctx.drone_id} — BudgetExceeded "
                f"po {elapsed:.3f}s ({e})"
            )
            # Best-so-far recovery (regression fix 2026-05-03): zwracamy
            # `status="ok"` jeśli mamy feasible best — `GenericOptimizingAvoidance`
            # odrzuca wszystko poza "ok", więc bez tego poprawne best-so-far
            # waypoints były marnowane. Sentinel filter odrzuca infeasible.
            SENTINEL_THRESHOLD = 1e8
            try:
                if global_best_fit < SENTINEL_THRESHOLD:
                    best_spline = path_repr.decode_genes(global_best_pos, ctx)
                    if best_spline is not None:
                        return OptimizationResult(
                            waypoints=np.asarray(
                                best_spline.waypoints, dtype=np.float64
                            ),
                            elapsed_s=elapsed,
                            status="ok",
                            extra={
                                "algorithm": "MSFOA",
                                "best_fitness": float(global_best_fit),
                                "evaluations_completed": int(generations_completed) * int(self.pop_size),
                                "generations_completed": int(generations_completed),
                                "wallclock_s": elapsed,
                                "reason": "budget_exceeded_returned_best_so_far",
                                "n_swarms": self.G,
                            },
                        )
            except Exception as recover_err:
                logger.warning(
                    f"MSFFOAOnlineOptimizer: best-so-far recovery failed: "
                    f"{recover_err}"
                )
            return OptimizationResult(
                waypoints=None,
                elapsed_s=elapsed,
                status="timed_out",
                extra={
                    "reason": "cooperative_budget_exceeded",
                    "best_fitness": global_best_fit,
                    "generations_completed": generations_completed,
                },
            )

        except Exception as e:
            elapsed = time.perf_counter() - t_start
            logger.error(
                f"MSFFOAOnlineOptimizer: d{ctx.drone_id} — nieoczekiwany "
                f"wyjątek: {e}",
                exc_info=True,
            )
            return OptimizationResult(
                waypoints=None,
                elapsed_s=elapsed,
                status="failed",
                extra={"reason": f"exception: {type(e).__name__}: {e}"},
            )

    @staticmethod
    def _eval_batch(
        pop: NDArray[np.float64],
        problem: PathProblem,
    ) -> NDArray[np.float64]:
        """Ocena batcha — sekwencyjna (online: pop_size~20, koszt znikomy).

        :param pop: (N, K) macierz genów.
        :return: (N,) fitness wartości (lower = better, 1e9 dla niezdekodowalnych).
        """
        out = np.empty(len(pop), dtype=np.float64)
        for i, x in enumerate(pop):
            spline = problem.path_repr.decode_genes(x, problem.context)
            out[i] = float(
                problem.fitness.evaluate(
                    spline, problem.context, problem.predictor
                )
            )
        return out
