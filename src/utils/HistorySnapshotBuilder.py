from __future__ import annotations

import time
import logging
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


class HistorySnapshotBuilder:
    """Buduje per-generację snapshoty (`payload`) historii optymalizacji do `h5_writer`.

    Łączy decyzje (`decisions_2d`), fitness skalarny, macierz objectivów,
    constraints, feasibility, leaderów i `extras` w jednolity dict zgodny
    ze schematem `OptimizationHistoryWriter`.
    """

    def __init__(
        self,
        *,
        history_writer: Any | None,
        logger: logging.Logger | None = None,
        label: str = "OPT",
    ) -> None:
        """Powiąż builder z writerem h5 i loggerem.

        Args:
            history_writer: Obiekt z `put_generation_data(payload)` lub `None`
                (write to no-op).
            logger: Logger; `None` ⇒ `logging.getLogger(__name__)`.
            label: Etykieta używana w komunikatach diagnostycznych.
        """
        self.history_writer = history_writer
        self.logger = logger or logging.getLogger(__name__)
        self.label = label

    def write(self, payload: dict[str, np.ndarray]) -> None:
        """Wyślij `payload` do writera h5 (no-op, gdy writer = `None`)."""
        if self.history_writer is None:
            return
        self.history_writer.put_generation_data(payload)

    def build_payload(
        self,
        *,
        decisions: NDArray[np.float64],
        scalar_fitness: NDArray[np.float64],
        objectives: Optional[NDArray[np.float64]] = None,
        gen: Optional[int] = None,
        gen_start_time: Optional[float] = None,
        fitness_owner: Any = None,
        evaluator_out: Optional[dict[str, Any]] = None,
        extras: Optional[dict[str, Any]] = None,
        leader_decisions: Optional[NDArray[np.float64]] = None,
        leader_scalar_fitness: Optional[NDArray[np.float64]] = None,
        global_best_scalar_fitness: Optional[float] = None,
    ) -> dict[str, np.ndarray]:
        """Złóż per-generację snapshot zgodny ze schematem `OptimizationHistoryWriter`.

        Args:
            decisions: `(N, D)` lub `(N, k, …)` macierz decyzji (flattenowana
                do `(N, D)` w razie potrzeby).
            scalar_fitness: `(N,)` skalarny fitness.
            objectives: `(N, M)` macierz objectivów (jeśli `None`, pobierana
                z `fitness_owner.last_objectives` lub fallback do `scalar_fitness.reshape(-1, 1)`).
            gen: Indeks generacji.
            gen_start_time: `time.perf_counter()` z początku generacji
                (do pomiaru `elapsed_s`).
            fitness_owner: Obiekt z zacache'owanymi `last_*` dla fallbacków.
            evaluator_out: Słownik z evaluatora (preferowane źródło `G`/`CV`/`feasible_mask`).
            extras: Dodatkowe pola dorzucane do `payload`.
            leader_decisions, leader_scalar_fitness: Macierz liderów (gdy algorytm
                je wyróżnia, np. MSFOA per-rój).
            global_best_scalar_fitness: Najlepszy fitness historycznie (skalar).

        Returns:
            Słownik gotowy do `OptimizationHistoryWriter.put_generation_data`.

        Raises:
            ValueError: Gdy populacje `decisions` i `scalar_fitness` mają
                niespójne rozmiary.
        """
        decisions_2d = self._flatten_population(decisions)
        scalar_fitness_1d = np.asarray(scalar_fitness, dtype=np.float64).reshape(-1)

        if decisions_2d.shape[0] != scalar_fitness_1d.shape[0]:
            raise ValueError(
                f"Population mismatch: decisions rows={decisions_2d.shape[0]}, "
                f"fitness size={scalar_fitness_1d.shape[0]}"
            )

        if objectives is None:
            objectives = self._get_cached_attr(
                fitness_owner,
                ("last_objectives", "raw_objectives", "objectives_matrix"),
            )

        if objectives is not None:
            objectives_arr = np.asarray(objectives, dtype=np.float64)
            if objectives_arr.shape[0] != decisions_2d.shape[0]:
                # Mismatch — kasuje zacache'owane objectives i podstawia
                # scalar fitness jako "objectives" o shape (N, 1). Logujemy,
                # bo fallback maskuje rozjazd między fitness_owner.last_objectives
                # a aktualną populacją (ETL poprzedni HV/IGD policzy na
                # 1-D scalarze, nie multi-objective wektorze).
                self.logger.warning(
                    "[%s] objectives row mismatch: objectives.shape[0]=%d, "
                    "decisions_2d.shape[0]=%d — fallback do scalar fitness "
                    "jako (-1, 1). HV/IGD w ETL będzie meaningless.",
                    self.label, objectives_arr.shape[0], decisions_2d.shape[0],
                )
                objectives_arr = scalar_fitness_1d.reshape(-1, 1)
        else:
            objectives_arr = scalar_fitness_1d.reshape(-1, 1)

        best_idx = int(np.argmin(scalar_fitness_1d))

        payload: dict[str, np.ndarray] = {
            "decisions_matrix": decisions_2d.copy(),
            "scalar_fitness": scalar_fitness_1d.copy(),
            "objectives_matrix": objectives_arr.copy(),
            "best_idx": np.array([best_idx], dtype=np.int32),
            "best_solution": decisions_2d[best_idx].copy(),
            "best_scalar_fitness": np.array([scalar_fitness_1d[best_idx]], dtype=np.float64),
            "best_objectives": objectives_arr[best_idx].copy(),
        }

        if gen is not None:
            payload["generation"] = np.array([gen], dtype=np.int32)

        if gen_start_time is not None:
            payload["elapsed_s"] = np.array(
                [time.perf_counter() - gen_start_time], dtype=np.float64
            )

        # Feasibility: preferowane źródło to `constraints_matrix` (gdy dostępne,
        # daje też pełen `violation_matrix` / `total_cv` / `weakest_cv`).
        # Fallback: zacache'owany `feasible_mask` z `_extract_feasible_mask`
        # (pochodzi z evaluator_out lub fitness_owner.last_feasible_mask).
        feasible_mask: Optional[np.ndarray] = None
        constraints_matrix = self._extract_constraints_matrix(evaluator_out, fitness_owner)
        if constraints_matrix is not None:
            constraints_matrix = np.asarray(constraints_matrix, dtype=np.float64)
            if constraints_matrix.ndim == 1:
                constraints_matrix = constraints_matrix.reshape(-1, 1)

            if constraints_matrix.shape[0] == decisions_2d.shape[0]:
                violation_matrix = np.maximum(constraints_matrix, 0.0)
                feasible_mask = np.all(constraints_matrix <= 0.0, axis=1)
                total_cv = np.sum(violation_matrix, axis=1)
                weakest_cv = np.max(violation_matrix, axis=1)

                payload["constraints_matrix"] = constraints_matrix.copy()
                payload["constraint_violation_matrix"] = violation_matrix.copy()
                payload["total_constraint_violation"] = total_cv.copy()
                payload["weakest_constraint_violation"] = weakest_cv.copy()

        if feasible_mask is None:
            cached_mask = self._extract_feasible_mask(evaluator_out, fitness_owner)
            if cached_mask is not None and cached_mask.shape[0] == decisions_2d.shape[0]:
                feasible_mask = cached_mask

        if feasible_mask is not None:
            payload["feasible_mask"] = feasible_mask.astype(bool).copy()
            payload["feasible_count"] = np.array(
                [int(np.count_nonzero(feasible_mask))], dtype=np.int32
            )
            payload["feasible_ratio"] = np.array(
                [float(np.mean(feasible_mask))], dtype=np.float64
            )

        if leader_decisions is not None:
            payload["leader_decisions_matrix"] = self._flatten_population(leader_decisions).copy()

        if leader_scalar_fitness is not None:
            payload["leader_scalar_fitness"] = np.asarray(
                leader_scalar_fitness, dtype=np.float64
            ).reshape(-1).copy()

        if global_best_scalar_fitness is not None:
            payload["global_best_scalar_fitness"] = np.array(
                [float(global_best_scalar_fitness)], dtype=np.float64
            )

        if extras:
            for key, value in extras.items():
                arr = np.asarray(value)
                payload[key] = arr.copy() if arr.ndim > 0 else np.array([arr.item()])

        return payload

    def evaluate_problem_state(
        self,
        *,
        problem: Any,
        decisions_2d: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Wywołaj `problem.evaluator` na `decisions_2d` i zwróć surowy `out` dict.

        Args:
            problem: Obiekt z atrybutami `_decode_inner` i `evaluator`.
            decisions_2d: `(N, D)` populacja do oceny.

        Returns:
            Słownik `out` z evaluatora (`F`, `G`, `CV`, `feasible_mask`); pusty
            przy błędzie / braku odpowiednich atrybutów.
        """
        out: dict[str, Any] = {}
        try:
            if hasattr(problem, "_decode_inner") and hasattr(problem, "evaluator"):
                inner = problem._decode_inner(decisions_2d)
                problem.evaluator.evaluate(inner, out)
        except Exception as e:
            self.logger.debug(f"[{self.label}] evaluator side-channel failed: {e}")
        return out

    def _flatten_population(self, pop: NDArray[np.float64]) -> NDArray[np.float64]:
        """Spłaszcz populację `(N, …)` do `(N, D)`; rzuca `ValueError` przy `<2D`."""
        arr = np.asarray(pop, dtype=np.float64)
        if arr.ndim == 2:
            return arr
        if arr.ndim >= 3:
            return arr.reshape(arr.shape[0], -1)
        raise ValueError(f"Unsupported population shape: {arr.shape}")

    def _get_cached_attr(self, owner: Any, names: tuple[str, ...]) -> Optional[np.ndarray]:
        if owner is None:
            return None
        for name in names:
            val = getattr(owner, name, None)
            if val is not None:
                return np.asarray(val)
        return None

    def _extract_feasible_mask(
        self,
        evaluator_out: Optional[dict[str, Any]],
        fitness_owner: Any,
    ) -> Optional[np.ndarray]:
        if evaluator_out:
            for key in ("feasible_mask", "is_feasible", "feasible"):
                if key in evaluator_out and evaluator_out[key] is not None:
                    return np.asarray(evaluator_out[key]).reshape(-1).astype(bool)

        cached = self._get_cached_attr(
            fitness_owner,
            ("last_feasible_mask", "feasible_mask", "is_feasible"),
        )
        if cached is not None:
            return np.asarray(cached).reshape(-1).astype(bool)

        cv = self._extract_constraint_violation(evaluator_out, fitness_owner)
        if cv is not None:
            if cv.ndim == 1:
                return cv <= 0.0
            return np.sum(cv, axis=1) <= 0.0

        return None
    
    def _extract_constraints_matrix(
        self,
        evaluator_out: Optional[dict[str, Any]],
        fitness_owner: Any,
    ) -> Optional[np.ndarray]:
        if evaluator_out:
            for key in ("G", "constraints_matrix", "constraint_values"):
                if key in evaluator_out and evaluator_out[key] is not None:
                    return np.asarray(evaluator_out[key], dtype=np.float64)

        cached = self._get_cached_attr(
            fitness_owner,
            ("last_constraints_matrix", "last_constraints", "constraints_matrix", "G"),
        )
        if cached is not None:
            return np.asarray(cached, dtype=np.float64)

        return None

    def _extract_constraint_violation(
        self,
        evaluator_out: Optional[dict[str, Any]],
        fitness_owner: Any,
    ) -> Optional[np.ndarray]:
        if evaluator_out:
            for key in ("CV", "constraint_violation", "constraint_violations"):
                if key in evaluator_out and evaluator_out[key] is not None:
                    return np.asarray(evaluator_out[key], dtype=np.float64)

        cached = self._get_cached_attr(
            fitness_owner,
            ("last_constraint_violation", "constraint_violation", "last_cv", "CV"),
        )
        if cached is not None:
            return np.asarray(cached, dtype=np.float64)

        return None