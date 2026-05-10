from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray
from hydra.core.hydra_config import HydraConfig

from mealpy import FloatVar
from mealpy import Problem as MealpyProblem
from mealpy.swarm_based.OOA import OriginalOOA
from mealpy.utils.target import Target  # noqa: F401  # re-export-safe; not strictly required, kept for symmetry with SSA

from src.algorithms.abstraction.trajectory.objective_constrains import (
    VectorizedEvaluator,
)
from src.algorithms.abstraction.trajectory.strategies.nsga3_swarm_strategy import (
    SwarmOptimizationProblem,
)
from src.algorithms.abstraction.trajectory.strategies.soo_adapter import (
    TrajectorySOOAdapter,
)
from src.algorithms.abstraction.trajectory.strategies.shared.bspline_utils import (
    generate_bspline_batch,
)
from src.algorithms.abstraction.trajectory.strategies.shared.StraightLineNoiseSampling import (
    StraightLineNoiseSampling,
)
from src.algorithms.abstraction.trajectory.strategies.timing_utils import (
    TimingCollector,
)
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.utils.optimization_history_writer import OptimizationHistoryWriter
from src.utils.SeedRegistry import SeedRegistry

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


class OOAProblemAdapter(MealpyProblem):
    """
    Problem mealpy dla OOA z bezwzględnym wymuszaniem granic (Hard Clipping).
    """

    def __init__(
        self,
        *,
        bounds: FloatVar,
        evaluator: VectorizedEvaluator,
        scalar_adapter: TrajectorySOOAdapter,
        start_pos: NDArray[np.float64],
        target_pos: NDArray[np.float64],
        n_drones: int,
        n_inner: int,
        n_output_samples: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(bounds=bounds, minmax="min", **kwargs)

        self.evaluator = evaluator
        self.scalar_adapter = scalar_adapter
        self.n_drones = n_drones
        self.n_inner = n_inner
        self.n_output_samples = n_output_samples

        self._starts_bc = start_pos[np.newaxis, :, np.newaxis, :]
        self._targets_bc = target_pos[np.newaxis, :, np.newaxis, :]

        self._lb = np.asarray(bounds.lb, dtype=np.float64)
        self._ub = np.asarray(bounds.ub, dtype=np.float64)

    def amend_position(self, position: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        NADRYWANIE MEALPY: Gwarantuje, że przy każdej aktualizacji wektora decyzyjnego,
        agent nie wyleci w nieskończoność. Bez tego OOA bywa numerycznie niestabilne.
        """
        return np.clip(np.nan_to_num(position, nan=self._ub), self._lb, self._ub)

    def _decode_inner(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        pop_size = population.shape[0]
        return population.reshape(pop_size, self.n_drones, self.n_inner, 3)

    def evaluate_population(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        inner = self._decode_inner(population)
        fitness = np.asarray(self.scalar_adapter(inner), dtype=np.float64).reshape(-1)
        return fitness

    def evaluate_objectives(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Surowe F do logowania historii.
        Uwaga: VectorizedEvaluator przyjmuje 'inner' points, a nie 'dense'!
        Przekazanie gęstych punktów powoduje niezgodność rozmiarów macierzy.
        """
        inner = self._decode_inner(population)
        out: Dict[str, Any] = {}
        # Przekazujemy inner, bo evaluator zainicjalizowano z n_inner_points=n_inner
        self.evaluator.evaluate(inner, out)
        return np.asarray(out["F"], dtype=np.float64)

    def evaluate_full(self, population: NDArray[np.float64]) -> Dict[str, np.ndarray]:
        """Zwraca {'F', 'G'} dla per-gen feasibility / CV w loggerze."""
        inner = self._decode_inner(population)
        out: Dict[str, Any] = {}
        self.evaluator.evaluate(inner, out)
        return {
            "F": np.asarray(out["F"], dtype=np.float64),
            "G": np.asarray(out["G"], dtype=np.float64) if "G" in out else None,
        }

    def obj_func(self, x: np.ndarray) -> float:
        x_safe = np.clip(np.nan_to_num(x, nan=self._ub), self._lb, self._ub)
        return float(self.evaluate_population(x_safe[np.newaxis, :])[0])


class LoggedOriginalOOA(OriginalOOA):
    """OOA z logowaniem per-gen + cache F/G na targetach.

    Zmiana wydajnościowa (2026-05-07):
    OOA z paperu (`mealpy.swarm_based.OOA.OriginalOOA`) ma
    `is_parallelizable=False` i w `evolve` woła `generate_agent` per osobnik
    (dwie fazy → 2× pop_size pojedynczych ewaluacji per generację). Nie
    batchujemy tego — wymagałoby zmiany semantyki `evolve`. Eliminujemy za to
    redundantną re-ewaluację `self.pop` w loggerze poprzez nadpisanie
    `get_target` tak, by dla każdego osobnika doczepić F/G (z cache
    `scalar_adapter.last_objectives/last_constraints`) bezpośrednio do
    `target._F_row/_G_row`. `evolve(...)` po `super().evolve(epoch)` zbiera te
    F/G z `self.pop` zamiast wołać `evaluate_full(decisions)` (1× pop_size NFE
    per generację — ~33% całego budżetu ewaluacji OOA).

    Niezmienione: konstruktor, `solve(...)`, sygnatury i pola — testy
    (`test_ooa_strategy.py`) nie operują na wewnętrznych metodach.
    """

    def __init__(
        self,
        *args: Any,
        history_writer: Optional[OptimizationHistoryWriter] = None,
        history_problem: Optional[OOAProblemAdapter] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._history_writer = history_writer
        self._history_problem = history_problem

    def generate_agent(self, solution: np.ndarray = None) -> Any:  # type: ignore[override]
        """Po standardowym `generate_agent` doczepia F/G z cache scalar_adapter
        bezpośrednio do `agent._F_row/_G_row`. Atrybut przetrwa `agent.copy()`
        (target.copy() nie kopiuje custom atrybutów, więc cache na targecie
        gubi się po fazie 2 SSA — dla OOA mealpy także używa `agent.copy()`
        wewnętrznie).
        """
        agent = super().generate_agent(solution)
        scalar_adapter = getattr(self._history_problem, "scalar_adapter", None) if self._history_problem is not None else None
        if scalar_adapter is None:
            return agent
        F_full = getattr(scalar_adapter, "last_objectives", None)
        G_full = getattr(scalar_adapter, "last_constraints", None)
        if isinstance(F_full, np.ndarray) and F_full.ndim == 2 and F_full.shape[0] == 1:
            try:
                agent._F_row = F_full[0].copy()
            except Exception:
                pass
        if isinstance(G_full, np.ndarray) and G_full.ndim == 2 and G_full.shape[0] == 1:
            try:
                agent._G_row = G_full[0].copy()
            except Exception:
                pass
        return agent

    def evolve(self, epoch: int) -> None:
        import time

        gen_t0 = time.monotonic()
        super().evolve(epoch)
        gen_elapsed = time.monotonic() - gen_t0

        if self._history_writer is None or self._history_problem is None:
            return

        try:
            decisions = np.vstack(
                [self._history_problem.amend_position(np.asarray(agent.solution, dtype=np.float64)) for agent in self.pop]
            )
            n_eval = int(self._history_problem.evaluator.individuals_evaluated)

            # Preferowana ścieżka: F/G zacache'owane na targetach przez
            # nadpisany `get_target`. Eliminuje extra `evaluate_full(decisions)`.
            F_rows: List[np.ndarray] = []
            G_rows: List[np.ndarray] = []
            for agent in self.pop:
                f_row = getattr(agent, "_F_row", None)
                g_row = getattr(agent, "_G_row", None)
                if isinstance(f_row, np.ndarray):
                    F_rows.append(f_row)
                if isinstance(g_row, np.ndarray):
                    G_rows.append(g_row)

            if len(F_rows) == len(self.pop):
                F = np.vstack(F_rows)
                G = np.vstack(G_rows) if len(G_rows) == len(self.pop) else None
            else:
                # Fallback (np. testy z mockami) — re-eval pełnej populacji.
                full = self._history_problem.evaluate_full(decisions)
                F = full["F"]
                G = full.get("G")

            from src.utils.per_gen_metrics import per_gen_metrics_from_FG

            self._history_writer.put_generation_data(
                per_gen_metrics_from_FG(
                    objectives=F,
                    constraints=G,
                    decisions=decisions,
                    elapsed_s=gen_elapsed,
                    eval_count_cumulative=n_eval,
                )
            )
        except Exception as e:
            logger.warning(f"[OOA] Warning: history logging failed at epoch {epoch}: {e}")


def osprey_swarm_strategy(
    *,
    start_positions: NDArray[np.float64],
    target_positions: NDArray[np.float64],
    obstacles_data: Union[Any, List[Any]],
    world_data: WorldData,
    number_of_waypoints: int,
    drone_swarm_size: int,
    algorithm_params: Optional[Dict[str, Any]] = None,
    timing: Optional["TimingCollector"] = None,
    seeds: SeedRegistry = None,
) -> NDArray[np.float64]:
    
    params = algorithm_params or {}

    local_timing = False
    if timing is None:
        try:
            timing = TimingCollector("OOA_Swarm")
            local_timing = True
        except Exception:
            timing = None

    _measure = timing.measure if timing is not None else lambda *a, **kw: nullcontext()

    writer: Optional[OptimizationHistoryWriter] = None

    try:
        with _measure("total_optimization"):
            pop_size: int = int(params.get("pop_size", 200))
            max_generations: int = int(params.get("epochs", params.get("n_gen", 500)))
            n_inner: int = int(params.get("n_inner_waypoints", max(5, int(number_of_waypoints * 0.1))))

            if "objective_weights" in params:
                weights = np.asarray(params["objective_weights"], dtype=np.float64)
            else:
                weights = np.asarray(
                    [
                        params.get("w_path_length", 0.05),
                        params.get("w_collision_risk", 100.0),
                        params.get("w_elevation", 0.1),
                    ],
                    dtype=np.float64,
                )

            penalty_weight: float = float(params.get("penalty_weight", 1.0))
            noise_std_xy = float(params.get("noise_std_xy", 2.0))
            noise_std_z = float(params.get("noise_std_z", 0.3))

            n_workers: int = int(params.get("n_workers", 1))
            mode: str = "thread" if n_workers > 1 else "swarm"

            try:
                output_dir = HydraConfig.get().runtime.output_dir
            except ValueError:
                output_dir = os.getcwd()

            writer = OptimizationHistoryWriter(output_dir=os.path.join(output_dir, "optimization_history"))

            with _measure("initialization"):
                evaluator = VectorizedEvaluator(
                    obstacles=obstacles_data,
                    start_pos=start_positions,
                    target_pos=target_positions,
                    n_inner_points=n_inner,
                    params=params,
                )

                scalar_adapter = TrajectorySOOAdapter(
                    evaluator=evaluator,
                    start_positions=start_positions,
                    target_positions=target_positions,
                    n_drones=drone_swarm_size,
                    n_inner=n_inner,
                    weights=weights,
                    penalty_weight=penalty_weight,
                )

                scalar_adapter._f_ref = np.maximum(scalar_adapter._f_ref, 1.0)
                logger.info(f"[OOA] F_ref (normalization scales): {scalar_adapter._f_ref}")

                problem_ref = SwarmOptimizationProblem(
                    n_drones=drone_swarm_size,
                    n_inner_points=n_inner,
                    world_data=world_data,
                    evaluator=evaluator,
                    start_pos=start_positions,
                    target_pos=target_positions,
                    min_altitude=params.get("min_altitude"),
                    max_altitude=params.get("max_altitude"),
                )

                xl = np.asarray(problem_ref.xl, dtype=np.float64)
                xu = np.asarray(problem_ref.xu, dtype=np.float64)

                sampling = StraightLineNoiseSampling(
                    start_pos=start_positions,
                    target_pos=target_positions,
                    n_inner_points=n_inner,
                    n_drones=drone_swarm_size,
                    noise_std=noise_std_xy,
                    noise_std_z=noise_std_z,
                    rng=seeds.rng("sampling")
                )

                # Mealpy oczekuje listy 1D wektorów per starting solution.
                starting_solutions_raw = np.asarray(sampling._do(problem_ref, pop_size), dtype=np.float64)
                starting_solutions = [sol for sol in starting_solutions_raw]

                mealpy_problem = OOAProblemAdapter(
                    bounds=FloatVar(lb=xl, ub=xu),
                    evaluator=evaluator,
                    scalar_adapter=scalar_adapter,
                    start_pos=start_positions,
                    target_pos=target_positions,
                    n_drones=drone_swarm_size,
                    n_inner=n_inner,
                    n_output_samples=number_of_waypoints,
                    log_to="console",
                )

                logger.info(
                    f"[OOA B-Spline] Start. Pop: {pop_size}, Epochs: {max_generations}, "
                    f"Control Pts: {n_inner}, Weights: {weights}, Penalty: {penalty_weight}, "
                    f"Mode: {mode}, Workers: {n_workers}"
                )

                model = LoggedOriginalOOA(
                    epoch=max_generations,
                    pop_size=pop_size,
                    history_writer=writer,
                    history_problem=mealpy_problem,
                )

            with _measure("optimization"):
                best_agent = model.solve(
                    problem=mealpy_problem,
                    mode=mode,
                    n_workers=n_workers,
                    starting_solutions=starting_solutions,
                    seed=seeds.seed("optimizer")
                )

                best_x = mealpy_problem.amend_position(np.asarray(best_agent.solution, dtype=np.float64))
                best_fitness = float(best_agent.target.fitness)

                logger.info(f"[OOA] Optimization Finished. Best Fitness: {best_fitness:.4f}")

            with _measure("decision_and_reconstruction"):
                inner = best_x.reshape(1, drone_swarm_size, n_inner, 3)
                starts = start_positions[np.newaxis, :, np.newaxis, :]
                targets = target_positions[np.newaxis, :, np.newaxis, :]
                sparse_trajectory = np.concatenate([starts, inner, targets], axis=2)

                logger.info("[OOA] Applying B-Spline Post-Processing...")
                final_dense_traj = generate_bspline_batch(
                    sparse_trajectory,
                    num_samples=number_of_waypoints,
                )

                return np.asarray(final_dense_traj[0], dtype=np.float64)

    except Exception as e:
        logger.warning(f"[OOA] Optimization error: {e}. Returning straight-line fallback.")

    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if local_timing and timing is not None:
            try:
                out_dir = HydraConfig.get().runtime.output_dir
                timing.save_csv(os.path.join(out_dir, "optimization_timings.csv"))
            except Exception as e:
                pass

    with _measure("fallback", success=False):
        logger.warning("[OOA] Fallback: generating straight-line trajectory.")
        t_line = np.linspace(0, 1, number_of_waypoints)
        out = np.empty((drone_swarm_size, number_of_waypoints, 3), dtype=np.float64)
        min_safe_alt = params.get("min_safe_altitude", 1.0)

        for d in range(drone_swarm_size):
            for axis in range(2):
                out[d, :, axis] = np.interp(
                    t_line,
                    [0, 1],
                    [start_positions[d, axis], target_positions[d, axis]],
                )
            z_start = max(float(start_positions[d, 2]), min_safe_alt)
            z_target = max(float(target_positions[d, 2]), min_safe_alt)
            out[d, :, 2] = np.interp(t_line, [0, 1], [z_start, z_target])

        return out