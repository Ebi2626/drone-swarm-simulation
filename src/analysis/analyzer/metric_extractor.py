"""Metric extractor — DB → pandas DataFrames.

Wszystkie zapytania SQL wracają w formacie tidy (long-form), gdzie każdy
wiersz to jedna obserwacja indeksowana przez (optimizer, environment,
seed[, iteration]). To jest standard pandas/seaborn (Wickham 2014, "Tidy
Data") — łatwo przejść na seaborn `hue=optimizer, col=environment`.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


logger = logging.getLogger(__name__)


class MetricExtractor:
    """Czyta metryki z `analysis.db` i zwraca DataFrames."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Brak bazy: {self.db_path}")

    # ------------------------------------------------------------------
    # Per-run (offline + online aggregates)
    # ------------------------------------------------------------------

    def run_summary(self) -> pd.DataFrame:
        """Per-run zestawienie: jedna obserwacja per run_id.

        Kolumny include: optimizer, environment, seed, avoidance, success,
        final_objective, hypervolume, igd_plus, gd_final, spread_final,
        spacing_final, r2_final, convergence_speed_gen, auc_best_so_far,
        oraz wszystkie online aggregates.
        """
        query = """
        SELECT
            r.run_id,
            r.optimizer_algo            AS optimizer,
            r.environment,
            r.avoidance_algo            AS avoidance,
            r.seed,
            r.algorithm_pair,
            m.drone_count,
            m.success,
            m.collision_count,
            m.evasion_event_count,
            m.obstacle_count,
            m.final_objective,
            m.final_objective_f1_trajectory,
            m.final_objective_f2_height_angle,
            m.total_threat_cost,
            m.total_turn_penalty,
            m.total_coordination_cost,
            m.total_path_length_2d,
            m.total_path_length_3d,
            m.hypervolume,
            m.hypervolume_normalized,
            m.igd_plus,
            m.gd_final,
            m.spread_final,
            m.spacing_final,
            m.r2_final,
            m.convergence_speed_gen,
            m.auc_best_so_far,
            m.front_size_last_gen,
            m.nondominated_count,
            m.min_inter_uav_distance_m,
            m.mean_inter_uav_distance_m,
            m.total_inter_uav_safety_violations,
            m.mean_energy_indicator,
            m.mean_smoothness_indicator
        FROM runs r
        LEFT JOIN run_metrics m ON m.run_id = r.run_id
        ORDER BY r.environment, r.optimizer_algo, r.seed
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def iteration_history(
        self,
        metrics: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Per-iteration time series. Jeden wiersz per (run_id, iteration).

        Args:
            metrics: opcjonalnie subset kolumn (zawsze zachowujemy
                run_id, optimizer, environment, seed, iteration).
        """
        cols = (
            metrics
            if metrics
            else [
                "best_so_far",
                "current_best",
                "current_mean",
                "current_std",
                "feasible_ratio",
                "diversity_metric",
                "elapsed_s",
                "eval_count_cumulative",
                "constraint_violation_best",
                "constraint_violation_mean",
                "nondominated_solutions",
                "nondominated_ratio",
                "hypervolume",
                "igd_plus",
                "gd",
                "spread",
                "spacing",
                "r2_indicator",
            ]
        )
        col_str = ", ".join(f"im.{c}" for c in cols)
        query = f"""
        SELECT
            r.run_id,
            r.optimizer_algo            AS optimizer,
            r.environment,
            r.seed,
            im.iteration,
            {col_str}
        FROM runs r
        JOIN iteration_metrics im ON im.run_id = r.run_id
        ORDER BY r.environment, r.optimizer_algo, r.seed, im.iteration
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def online_summary(self) -> pd.DataFrame:
        """Per-run online avoidance summary z view'a `vw_run_online_summary`."""
        query = """
        SELECT
            v.run_id,
            v.algorithm                 AS online_algorithm,
            v.environment,
            v.seed,
            v.total_evasion_triggers,
            v.avg_wallclock_s,
            v.max_wallclock_s,
            v.budget_exceeded_count,
            v.budget_violation_rate,
            v.avg_generations_completed,
            v.avg_evaluations_completed,
            v.avg_best_fitness,
            v.successful_rejoins,
            v.avg_pos_err_m,
            v.avg_vel_err_mps,
            v.avg_time_to_rejoin_s,
            v.min_inter_uav_distance_m,
            v.mean_inter_uav_distance_m,
            v.total_inter_uav_safety_violations,
            v.mean_energy_indicator,
            v.mean_smoothness_indicator
        FROM vw_run_online_summary v
        ORDER BY v.environment, v.algorithm, v.seed
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def pareto_front_last_gen(self) -> pd.DataFrame:
        """Last-gen feasible front per run, w formie long-form.

        Filtruje populację last-gen przez `feasible_mask` (gdy dostępny w h5),
        a następnie liczy non-dominated front. Bez tego filtru "pareto front"
        zawierałby infeasible solutions, co przekłamuje porównanie cross-algorytm
        (algorytm który zostawia więcej infeasible w pop ma "większy front").
        Spójne z `populate_offline_objectives._extract_best_feasible_F`.

        Kolumny: run_id, optimizer, environment, seed, point_idx,
        objective_j, value. Używane do plotów Pareto front.
        Wymaga `optimization_history.h5` na dysku.
        """
        # Ten extractor czyta h5 bezpośrednio (run_metrics nie przechowuje
        # całego fronta). Robimy to w pythonie z h5py.
        import numpy as np

        with sqlite3.connect(self.db_path) as conn:
            runs = conn.execute(
                """
                SELECT run_id, source_path, optimizer_algo, environment, seed
                FROM runs ORDER BY environment, optimizer_algo, seed
                """
            ).fetchall()

        try:
            import h5py
        except ImportError:
            return pd.DataFrame(
                columns=["run_id", "optimizer", "environment", "seed", "point_idx", "objective_j", "value"]
            )

        rows = []
        for run_id, source_path, optimizer, environment, seed in runs:
            h5_path = Path(source_path) / "optimization_history" / "optimization_history.h5"
            if not h5_path.exists():
                continue
            try:
                with h5py.File(h5_path, "r") as f:
                    if "objectives_matrix" not in f:
                        continue
                    obj = f["objectives_matrix"]
                    if obj.shape[0] == 0:
                        continue
                    last_idx = obj.shape[0] - 1
                    F = np.asarray(obj[last_idx], dtype=float)
                    if F.ndim == 1:
                        F = F[:, None]

                    # Feasibility filter — preferowany `feasible_mask`,
                    # fallback na `constraint_violation` ≤ FEASIBILITY_EPS.
                    feasible = _extract_feasible_mask_from_h5(f, last_idx, F.shape[0])
                    if feasible is not None:
                        F_feas = F[feasible]
                        if F_feas.shape[0] == 0:
                            # 0 feasible w last gen — pomijamy żeby nie produkować
                            # mylącego "infeasible-only front".
                            logger.warning(
                                "pareto_front_last_gen: run %r ma 0 feasible w last-gen — pomijam.",
                                run_id,
                            )
                            continue
                        F = F_feas

                    front = _non_dominated(F)
                    for p in range(front.shape[0]):
                        for j in range(front.shape[1]):
                            rows.append((run_id, optimizer, environment, seed, p, j, float(front[p, j])))
            except (OSError, KeyError, ValueError) as e:
                # Specyficzne wyjątki: błędny/uszkodzony h5 (OSError),
                # brak datasetu (KeyError), zły shape (ValueError). Logujemy
                # żeby ETL nie maskował korupcji bez ostrzeżenia.
                logger.warning(
                    "pareto_front_last_gen: pomijam run %r (h5=%s) — %s",
                    run_id, h5_path, e,
                )
                continue

        return pd.DataFrame(
            rows,
            columns=["run_id", "optimizer", "environment", "seed", "point_idx", "objective_j", "value"],
        )


def _extract_feasible_mask_from_h5(f, gen_idx: int, pop_size: int):
    """Wyciąga feasible_mask z h5 dla generacji `gen_idx`.

    Preferuje `feasible_mask`. Fallback: liczy z `constraint_violation`/`CV` przy
    pomocy `FEASIBILITY_EPS`. Zwraca None gdy brak danych — caller pozostaje
    przy nie-filtrowanym F.
    """
    import numpy as np
    if "feasible_mask" in f:
        try:
            fm = np.asarray(f["feasible_mask"][gen_idx]).reshape(-1).astype(bool)
            if fm.shape[0] == pop_size:
                return fm
        except (OSError, KeyError, ValueError):
            pass
    for name in ("constraint_violation", "CV", "constraint_violations", "last_cv"):
        if name in f:
            try:
                cv = np.asarray(f[name][gen_idx], dtype=np.float64)
                if cv.ndim == 1:
                    cv_total = np.maximum(cv, 0.0)
                else:
                    cv_total = np.sum(np.maximum(cv, 0.0), axis=1)
                if cv_total.shape[0] == pop_size:
                    # Spójna tolerancja z `per_gen_metrics.FEASIBILITY_EPS`.
                    from src.utils.per_gen_metrics import FEASIBILITY_EPS
                    return cv_total <= FEASIBILITY_EPS
            except (OSError, KeyError, ValueError):
                pass
    return None


def _non_dominated(F):
    import numpy as np

    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        leq = np.all(F <= F[i], axis=1)
        lt = np.any(F < F[i], axis=1)
        dominators = leq & lt
        dominators[i] = False
        if np.any(dominators):
            keep[i] = False
    return F[keep]
