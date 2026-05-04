from __future__ import annotations

import json
import sqlite3
from collections import defaultdict


def populate_iteration_metrics(conn: sqlite3.Connection, run_id: str) -> None:
    cur = conn.execute(
        """
        SELECT
            generation,
            source_name,
            metric_name,
            metric_value
        FROM optimization_generation_stats
        WHERE run_id = ?
        ORDER BY generation, source_name, metric_name
        """,
        (run_id,),
    )

    def _empty_entry() -> dict:
        return {
            "population_size": None,
            "best_so_far": None,
            "current_best": None,
            "current_mean": None,
            "current_std": None,
            "current_worst": None,
            "feasible_solutions": None,
            "feasible_ratio": None,
            "diversity_metric": None,
            "elapsed_s": None,
            "eval_count_cumulative": None,
            "constraint_violation_best": None,
            "constraint_violation_mean": None,
            "constraint_violation_worst": None,
            "nondominated_solutions": None,
            "nondominated_ratio": None,
            "hypervolume": None,
            "igd_plus": None,
            "extra": {},
        }

    by_generation: dict[int, dict] = defaultdict(_empty_entry)

    for generation, source_name, metric_name, metric_value in cur.fetchall():
        generation = int(generation)
        metric_value = float(metric_value)
        entry = by_generation[generation]

        handled = False

        # =========================================================
        # Wspólne metryki SOO / MOO
        # =========================================================
        if metric_name == "population_size":
            entry["population_size"] = int(metric_value)
            handled = True

        elif metric_name == "feasible_solutions":
            entry["feasible_solutions"] = int(metric_value)
            handled = True

        elif metric_name == "feasible_ratio":
            entry["feasible_ratio"] = metric_value
            handled = True

        elif metric_name == "diversity_metric":
            entry["diversity_metric"] = metric_value
            handled = True

        elif metric_name == "elapsed_s":
            entry["elapsed_s"] = metric_value
            handled = True

        elif metric_name == "eval_count_cumulative":
            entry["eval_count_cumulative"] = int(metric_value)
            handled = True

        # =========================================================
        # Skalarne metryki przebiegu
        # =========================================================
        elif metric_name == "best_so_far_obj0":
            entry["best_so_far"] = metric_value
            handled = True

        elif metric_name in ("objective_0_min", "scalar_fitness_min", "best_scalar_fitness"):
            entry["current_best"] = metric_value
            handled = True

        elif metric_name in ("objective_0_mean", "scalar_fitness_mean"):
            entry["current_mean"] = metric_value
            handled = True

        elif metric_name in ("objective_0_std", "scalar_fitness_std"):
            entry["current_std"] = metric_value
            handled = True

        elif metric_name in ("objective_0_max", "scalar_fitness_max"):
            entry["current_worst"] = metric_value
            handled = True

        # =========================================================
        # Naruszenia ograniczeń
        # =========================================================
        elif metric_name in ("constraint_violation_min", "cv_min", "constraint_violation_best"):
            entry["constraint_violation_best"] = metric_value
            handled = True

        elif metric_name in ("constraint_violation_mean", "cv_mean"):
            entry["constraint_violation_mean"] = metric_value
            handled = True

        elif metric_name in ("constraint_violation_max", "cv_max", "constraint_violation_worst"):
            entry["constraint_violation_worst"] = metric_value
            handled = True

        # =========================================================
        # Metryki specyficzne dla Pareto / NSGA-III
        # =========================================================
        elif metric_name in ("nondominated_solutions", "nd_count", "rank0_count"):
            entry["nondominated_solutions"] = int(metric_value)
            handled = True

        elif metric_name in ("nondominated_ratio", "nd_ratio", "rank0_ratio"):
            entry["nondominated_ratio"] = metric_value
            handled = True

        elif metric_name == "hypervolume":
            entry["hypervolume"] = metric_value
            handled = True

        elif metric_name in ("igd_plus", "igd+"):
            entry["igd_plus"] = metric_value
            handled = True

        # =========================================================
        # Fallback: zachowaj wszystko, czego nie mapujemy jawnie
        # =========================================================
        if not handled:
            entry["extra"].setdefault(source_name, {})[metric_name] = metric_value

    conn.execute(
        "DELETE FROM iteration_metrics WHERE run_id = ?",
        (run_id,),
    )

    if not by_generation:
        return

    rows_to_insert = []
    for generation in sorted(by_generation):
        entry = by_generation[generation]

        # Pochodne wspólne
        if (
            entry["feasible_ratio"] is None
            and entry["feasible_solutions"] is not None
            and entry["population_size"] not in (None, 0)
        ):
            entry["feasible_ratio"] = entry["feasible_solutions"] / entry["population_size"]

        if (
            entry["nondominated_ratio"] is None
            and entry["nondominated_solutions"] is not None
            and entry["population_size"] not in (None, 0)
        ):
            entry["nondominated_ratio"] = entry["nondominated_solutions"] / entry["population_size"]

        extra_json = (
            json.dumps(entry["extra"], ensure_ascii=False, sort_keys=True)
            if entry["extra"]
            else None
        )

        rows_to_insert.append(
            (
                run_id,
                generation,
                entry["population_size"],
                entry["feasible_solutions"],
                entry["feasible_ratio"],
                entry["diversity_metric"],
                entry["elapsed_s"],
                entry["eval_count_cumulative"],
                entry["constraint_violation_best"],
                entry["constraint_violation_mean"],
                entry["constraint_violation_worst"],
                entry["best_so_far"],
                entry["current_best"],
                entry["current_mean"],
                entry["current_std"],
                entry["current_worst"],
                entry["nondominated_solutions"],
                entry["nondominated_ratio"],
                entry["hypervolume"],
                entry["igd_plus"],
                extra_json,
            )
        )

    conn.executemany(
        """
        INSERT INTO iteration_metrics (
            run_id,
            iteration,
            population_size,
            feasible_solutions,
            feasible_ratio,
            diversity_metric,
            elapsed_s,
            eval_count_cumulative,
            constraint_violation_best,
            constraint_violation_mean,
            constraint_violation_worst,
            best_so_far,
            current_best,
            current_mean,
            current_std,
            current_worst,
            nondominated_solutions,
            nondominated_ratio,
            hypervolume,
            igd_plus,
            extra_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows_to_insert,
    )