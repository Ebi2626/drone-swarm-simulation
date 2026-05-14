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


# Docelowa odległość między najbliższymi sąsiadami w roju [m], wyprowadzona
# z geometrii misji: drony rozmieszczone na osi X co 5 m w pozycjach
# startowych (`initial_xyzs`) i docelowych (`end_xyzs`) — zob.
# `configs/environment/{forest,urban}.yaml`. W idealnej formacji NN-distance
# pozostaje 5 m przez cały czas misji. `swarm_cohesion_deviation` mierzy
# sumę odchyleń worst-case kompresji i dispersji od tej wartości.
SWARM_COHESION_TARGET_NN_M = 5.0


class MetricExtractor:
    """Czyta metryki z `analysis.db` i zwraca DataFrames."""

    def __init__(self, db_path: str | Path) -> None:
        """Powiąż ekstraktor z istniejącą bazą `analysis.db`.

        Args:
            db_path: Ścieżka do pliku bazy SQLite.

        Raises:
            FileNotFoundError: Gdy plik bazy nie istnieje.
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Brak bazy: {self.db_path}")

    def run_summary(self) -> pd.DataFrame:
        """Zwróć tidy DataFrame z per-run zestawieniem (jedna obserwacja per `run_id`).

        Kolumny: `optimizer`, `environment`, `seed`, `avoidance`, `success`,
        `final_objective`, MOO indicators (HV/IGD+/GD/spread/spacing/R2)
        oraz online aggregates (`collision_count`, `min_inter_uav_distance_m`,
        `mean_*_indicator`, …).

        Returns:
            Tidy DataFrame indeksowany po `run_id`.
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
            m.tracking_phase_collisions,
            m.evasion_phase_collisions,
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
            m.total_wallclock_offline_s,
            m.front_size_last_gen,
            m.nondominated_count,
            m.min_inter_uav_distance_m,
            m.max_inter_uav_distance_m,
            m.mean_inter_uav_distance_m,
            m.total_inter_uav_safety_violations,
            m.mean_energy_indicator,
            m.mean_smoothness_indicator,
            m.avg_wallclock_online_s,
            m.max_wallclock_online_s,
            m.mean_online_best_fitness,
            m.median_online_best_fitness,
            m.mean_online_generations_completed,
            m.mean_online_evaluations_completed,
            m.online_optimization_task_count,
            -- §3.1.3.3 docs/Praca magisterska.md — trajektoria fazy online
            m.mean_evasion_arc_length_m,
            m.median_evasion_arc_length_m,
            m.mean_evasion_plan_duration_s,
            m.mean_pos_err_at_rejoin_m,
            m.mean_vel_err_at_rejoin_mps,
            m.mean_time_to_rejoin_s,
            m.rejoin_success_rate,
            m.rejoin_completion_rate,
            m.rejoin_quality,
            m.budget_violation_rate,
            m.online_success_rate,
            m.online_sp1
        FROM runs r
        LEFT JOIN run_metrics m ON m.run_id = r.run_id
        ORDER BY r.environment, r.optimizer_algo, r.seed
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
        # Derywowane metryki jakości trajektorii (offline, best feasible
        # solution last generation — F-vector z populate_offline_objectives).
        # Mapowanie: F[0]=f1 (długość+kształt), F[1]=f2 (wysokość+kąt),
        # F[2]=f3 (zagrożenie), F[3]=f4 (zakręty), F[4]=f5 (koordynacja).
        if "final_objective_f1_trajectory" in df.columns:
            # Długość trajektorii (kara długości polilinii + odchylenia bocznego).
            df["trajectory_length_f1"] = df["final_objective_f1_trajectory"]
        if {"final_objective_f2_height_angle", "total_turn_penalty"}.issubset(df.columns):
            # Gładkość trajektorii — kara wysokości/kąta (f2) + zakręty (f4).
            # Suma surowych wartości; obie składowe operują na podobnej skali
            # (kąty radianowe + odchylenia metryczne).
            df["trajectory_smoothness_f2_f4"] = (
                df["final_objective_f2_height_angle"] + df["total_turn_penalty"]
            )
        if {"total_threat_cost", "total_coordination_cost"}.issubset(df.columns):
            # Bezpieczeństwo trajektorii — kara zagrożenia (f3) + koordynacji (f5).
            # f3: hinge penalty za penetrację stref przeszkód.
            # f5: exponential penalty za zbliżanie się dronów < d_min.
            # Skale są różne — interpretować z ostrożnością cross-scenariusz.
            df["trajectory_safety_f3_f5"] = (
                df["total_threat_cost"] + df["total_coordination_cost"]
            )
        # swarm_cohesion_deviation = suma odchyleń worst-case kompresji
        # i dispersji od docelowego NN-spacing (5 m). Per
        # `docs/Praca magisterska.md` §3.1.3.1 — metryka oceny *offline*
        # zaplanowanej trajektorii (mierzona w fizycznej symulacji, ale
        # ocenia jakość planowania). Niska wartość ⇒ rój utrzymuje
        # formację zbliżoną do startowej; wysoka ⇒ "lot poszarpany"
        # z dużymi wahaniami separacji.
        if "min_inter_uav_distance_m" in df.columns and "max_inter_uav_distance_m" in df.columns:
            df["swarm_cohesion_deviation"] = (
                (df["min_inter_uav_distance_m"] - SWARM_COHESION_TARGET_NN_M).abs()
                + (df["max_inter_uav_distance_m"] - SWARM_COHESION_TARGET_NN_M).abs()
            )
        return df

    def iteration_history(
        self,
        metrics: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Zwróć tidy time series — jeden wiersz per `(run_id, iteration)`.

        Args:
            metrics: Opcjonalny subset kolumn metryk; klucze tożsamości
                (`run_id, optimizer, environment, seed, iteration`) są
                zawsze obecne. `None` ⇒ pełen zestaw domyślnych kolumn.

        Returns:
            Tidy DataFrame z kolumnami tożsamości + wybranymi metrykami.
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
        """Zwróć per-run podsumowanie online avoidance z widoku `vw_run_online_summary`.

        Returns:
            DataFrame z agregatami: liczba triggerów uniku, wallclock, błędy
            rejoin, dystanse między dronami i wskaźniki energii / gładkości.
        """
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
        """Zwróć long-form feasible front Pareto z ostatniej generacji per run.

        Czyta `optimization_history.h5` per run, filtruje populację przez
        `feasible_mask` (lub `constraint_violation` z `FEASIBILITY_EPS`)
        i liczy non-dominated front. Bez filtru rankingi cross-algorytm są
        zaburzone przez infeasible solutions.

        Returns:
            DataFrame z kolumnami `run_id, optimizer, environment, seed,
            point_idx, objective_j, value`. Pusty, gdy brak `h5py` lub
            żaden run nie ma feasible kandydatów w last-gen.
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
    """Wyciągnij maskę feasible z h5 dla generacji `gen_idx`.

    Args:
        f: Otwarty `h5py.File`.
        gen_idx: Indeks generacji.
        pop_size: Oczekiwany rozmiar populacji (do walidacji shape).

    Returns:
        `(pop_size,)` maska boolowska albo `None`, gdy brak danych — caller
        pozostaje wtedy przy niefiltrowanej macierzy `F`.
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
    """Zwróć podzbiór `F` zawierający rozwiązania niezdominowane (`O(n²)`)."""
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
