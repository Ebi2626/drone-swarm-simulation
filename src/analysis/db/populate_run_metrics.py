"""Agregat per-run metryk do tabeli `run_metrics`.

Schema scope: `VectorizedEvaluator` ma 5-obj F-vector mapowane na kolumny
`final_objective_f1_trajectory / f2_height_angle / total_threat_cost /
total_turn_penalty / total_coordination_cost`. **Te pola NIE są wpisywane
przez ten populator** — domena `populate_offline_objectives` (UPDATE z
h5 F-vector). Wiersz powstaje tutaj z NULL'ami w tych kolumnach, eliminując
ordering dependency (poprzednie wersje wpisywały jawne NULL'e i nadpisywały
wartości z h5 gdy kolejność populatorów była zaburzona).
"""
import json
import sqlite3


def populate_run_metrics(conn: sqlite3.Connection, run_id: str) -> None:
    """Zbuduj agregat `run_metrics` dla `run_id` z wcześniej załadowanych tabel.

    Args:
        conn: Aktywne połączenie do `analysis.db`.
        run_id: Identyfikator runa.

    Efekty uboczne:
        Wstawia / aktualizuje rekord w `run_metrics`. Pola F-vector
        (`final_objective*`) pozostają `NULL` — wypełnia je
        `populate_offline_objectives`.
    """
    cur = conn.execute(
        """
        WITH
        uav AS (
            SELECT
                COUNT(*) AS uav_rows,
                MIN(COALESCE(success, 1)) AS all_uav_success
            FROM uav_metrics
            WHERE run_id = ?
        ),
        traj AS (
            SELECT
                COUNT(*) AS traj_rows,
                COUNT(DISTINCT uav_id) AS drone_count,
                CASE WHEN COUNT(path_length_2d) > 0 THEN total(path_length_2d) END AS total_path_length_2d,
                CASE WHEN COUNT(path_length_3d) > 0 THEN total(path_length_3d) END AS total_path_length_3d
            FROM trajectory_metrics
            WHERE run_id = ?
              AND source_name = 'trajectory_samples'
        ),
        coll AS (
            SELECT COUNT(*) AS collision_count
            FROM collisions
            WHERE run_id = ?
        ),
        evas AS (
            SELECT COUNT(*) AS evasion_event_count
            FROM evasion_events
            WHERE run_id = ?
        ),
        obs AS (
            SELECT COUNT(*) AS obstacle_count
            FROM generated_obstacles
            WHERE run_id = ?
        ),
        best_gen AS (
            SELECT generation AS best_iteration
            FROM optimization_generation_stats
            WHERE run_id = ?
              AND source_name = 'objectives_matrix'
              AND metric_name = 'best_so_far_obj0'
            ORDER BY metric_value ASC, generation ASC
            LIMIT 1
        ),
        last_gen AS (
            SELECT MAX(generation) AS generation
            FROM optimization_generation_stats
            WHERE run_id = ?
        ),
        moo AS (
            SELECT
                MAX(CASE WHEN ogs.metric_name IN ('nondominated_solutions', 'nd_count', 'rank0_count')
                         THEN ogs.metric_value END) AS nondominated_count,
                MAX(CASE WHEN ogs.metric_name = 'hypervolume'
                         THEN ogs.metric_value END) AS hypervolume,
                MAX(CASE WHEN ogs.metric_name IN ('igd_plus', 'igd+')
                         THEN ogs.metric_value END) AS igd_plus,
                MAX(CASE WHEN ogs.metric_name = 'front_size'
                         THEN ogs.metric_value END) AS front_size_last_gen,
                MAX(CASE WHEN ogs.metric_name = 'hypervolume_normalized'
                         THEN ogs.metric_value END) AS hypervolume_normalized
            FROM optimization_generation_stats ogs
            JOIN last_gen lg
              ON lg.generation = ogs.generation
            WHERE ogs.run_id = ?
        ),
        online_uav AS (
            SELECT
                MIN(min_inter_uav_distance_m)        AS min_inter_uav_distance_m,
                AVG(mean_inter_uav_distance_m)       AS mean_inter_uav_distance_m,
                CASE WHEN COUNT(inter_uav_safety_violation_count) > 0
                     THEN total(inter_uav_safety_violation_count)
                END                                   AS total_inter_uav_safety_violations,
                AVG(energy_indicator)                AS mean_energy_indicator,
                AVG(smoothness_indicator)            AS mean_smoothness_indicator
            FROM uav_online_metrics
            WHERE run_id = ?
        ),
        moo_quality_final AS (
            SELECT
                im.gd            AS gd_final,
                im.spread        AS spread_final,
                im.spacing       AS spacing_final,
                im.r2_indicator  AS r2_final
            FROM iteration_metrics im
            JOIN last_gen lg ON im.iteration = lg.generation
            WHERE im.run_id = ?
        ),
        drones_fallback AS (
            SELECT COUNT(DISTINCT drone_id) AS drone_count
            FROM trajectory_samples
            WHERE run_id = ?
        )
        SELECT
            COALESCE(
                NULLIF((SELECT drone_count FROM traj), 0),
                NULLIF((SELECT drone_count FROM drones_fallback), 0)
            ) AS drone_count,

            CASE
                WHEN (SELECT collision_count FROM coll) > 0 THEN 0
                WHEN (SELECT uav_rows FROM uav) > 0 THEN COALESCE((SELECT all_uav_success FROM uav), 1)
                ELSE 1
            END AS success,

            (SELECT total_path_length_2d FROM traj) AS total_path_length_2d,
            (SELECT total_path_length_3d FROM traj) AS total_path_length_3d,
            (SELECT collision_count FROM coll) AS collision_count,
            (SELECT evasion_event_count FROM evas) AS evasion_event_count,
            (SELECT obstacle_count FROM obs) AS obstacle_count,
            (SELECT best_iteration FROM best_gen) AS best_iteration,
            (SELECT nondominated_count FROM moo) AS nondominated_count,
            (SELECT hypervolume FROM moo) AS hypervolume,
            (SELECT igd_plus FROM moo) AS igd_plus,
            (SELECT front_size_last_gen FROM moo) AS front_size_last_gen,
            (SELECT hypervolume_normalized FROM moo) AS hypervolume_normalized,
            (SELECT min_inter_uav_distance_m FROM online_uav) AS min_inter_uav_distance_m,
            (SELECT mean_inter_uav_distance_m FROM online_uav) AS mean_inter_uav_distance_m,
            (SELECT total_inter_uav_safety_violations FROM online_uav) AS total_inter_uav_safety_violations,
            (SELECT mean_energy_indicator FROM online_uav) AS mean_energy_indicator,
            (SELECT mean_smoothness_indicator FROM online_uav) AS mean_smoothness_indicator,
            (SELECT gd_final FROM moo_quality_final) AS gd_final,
            (SELECT spread_final FROM moo_quality_final) AS spread_final,
            (SELECT spacing_final FROM moo_quality_final) AS spacing_final,
            (SELECT r2_final FROM moo_quality_final) AS r2_final,
            (SELECT uav_rows FROM uav) AS uav_rows,
            (SELECT traj_rows FROM traj) AS traj_rows
        """,
        # 11 placeholders: uav, traj, coll, evas, obs, best_gen, last_gen,
        # moo, online_uav, moo_quality_final, drones_fallback.
        (run_id,) * 11,
    )

    row = cur.fetchone()
    columns = [desc[0] for desc in cur.description]
    data = dict(zip(columns, row))

    summary = {
        "path_source": "trajectory_samples",
        "best_iteration_source": "optimization_generation_stats.best_so_far_obj0",
        "moo_metrics_source": "optimization_generation_stats[last_generation]",
        "success_rule": "collision_count == 0 and all_uav_success_if_present",
        "uav_rows_used": data["uav_rows"],
        "trajectory_metric_rows_used": data["traj_rows"],
    }

    total_violations = data["total_inter_uav_safety_violations"]
    total_violations_int = (
        int(total_violations) if total_violations is not None else None
    )

    convergence_speed_gen, auc_best_so_far = _convergence_speed_and_auc(conn, run_id)

    # `final_objective`, `total_threat_cost`, `total_turn_penalty`,
    # `total_coordination_cost`, `final_objective_f*` NIE są w tym INSERT —
    # domena `populate_offline_objectives` (UPDATE z h5).
    conn.execute(
        """
        INSERT INTO run_metrics (
            run_id,
            drone_count,
            success,
            total_path_length_2d,
            total_path_length_3d,
            collision_count,
            evasion_event_count,
            obstacle_count,
            best_iteration,
            nondominated_count,
            hypervolume,
            igd_plus,
            front_size_last_gen,
            hypervolume_normalized,
            min_inter_uav_distance_m,
            mean_inter_uav_distance_m,
            total_inter_uav_safety_violations,
            mean_energy_indicator,
            mean_smoothness_indicator,
            gd_final,
            spread_final,
            spacing_final,
            r2_final,
            convergence_speed_gen,
            auc_best_so_far,
            summary_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            drone_count = excluded.drone_count,
            success = excluded.success,
            total_path_length_2d = excluded.total_path_length_2d,
            total_path_length_3d = excluded.total_path_length_3d,
            collision_count = excluded.collision_count,
            evasion_event_count = excluded.evasion_event_count,
            obstacle_count = excluded.obstacle_count,
            best_iteration = excluded.best_iteration,
            nondominated_count = excluded.nondominated_count,
            hypervolume = excluded.hypervolume,
            igd_plus = excluded.igd_plus,
            front_size_last_gen = excluded.front_size_last_gen,
            hypervolume_normalized = excluded.hypervolume_normalized,
            min_inter_uav_distance_m = excluded.min_inter_uav_distance_m,
            mean_inter_uav_distance_m = excluded.mean_inter_uav_distance_m,
            total_inter_uav_safety_violations = excluded.total_inter_uav_safety_violations,
            mean_energy_indicator = excluded.mean_energy_indicator,
            mean_smoothness_indicator = excluded.mean_smoothness_indicator,
            gd_final = excluded.gd_final,
            spread_final = excluded.spread_final,
            spacing_final = excluded.spacing_final,
            r2_final = excluded.r2_final,
            convergence_speed_gen = excluded.convergence_speed_gen,
            auc_best_so_far = excluded.auc_best_so_far,
            summary_json = excluded.summary_json
        """,
        (
            run_id,
            data["drone_count"],
            data["success"],
            data["total_path_length_2d"],
            data["total_path_length_3d"],
            data["collision_count"],
            data["evasion_event_count"],
            data["obstacle_count"],
            data["best_iteration"],
            int(data["nondominated_count"]) if data["nondominated_count"] is not None else None,
            data["hypervolume"],
            data["igd_plus"],
            int(data["front_size_last_gen"]) if data["front_size_last_gen"] is not None else None,
            data["hypervolume_normalized"],
            data["min_inter_uav_distance_m"],
            data["mean_inter_uav_distance_m"],
            total_violations_int,
            data["mean_energy_indicator"],
            data["mean_smoothness_indicator"],
            data["gd_final"],
            data["spread_final"],
            data["spacing_final"],
            data["r2_final"],
            convergence_speed_gen,
            auc_best_so_far,
            json.dumps(summary, ensure_ascii=False, sort_keys=True),
        ),
    )


def _convergence_speed_and_auc(
    conn: sqlite3.Connection, run_id: str
) -> tuple[int | None, float | None]:
    """Liczy z `iteration_metrics`:
    - convergence_speed_gen: pierwsza generacja gdy HV ≥ 0.9 · HV(last_gen).
      NULL gdy brak HV lub last_gen HV < final*0.9.
    - auc_best_so_far: ∫ best_so_far(g) dg (trapez), znormalizowane przez
      (last_gen − first_gen). Lower = lepiej (mniej kosztu over time).
    """
    rows = conn.execute(
        """
        SELECT iteration, hypervolume, best_so_far
        FROM iteration_metrics
        WHERE run_id = ?
        ORDER BY iteration ASC
        """,
        (run_id,),
    ).fetchall()
    if not rows:
        return None, None

    iters = [int(r[0]) for r in rows]
    hvs = [r[1] for r in rows]
    bsf = [r[2] for r in rows]

    convergence_speed_gen: int | None = None
    valid_hvs = [(i, h) for i, h in zip(iters, hvs) if h is not None]
    if valid_hvs:
        final_hv = valid_hvs[-1][1]
        if final_hv > 0:
            threshold = 0.9 * final_hv
            for i, h in valid_hvs:
                if h >= threshold:
                    convergence_speed_gen = i
                    break

    auc_best_so_far: float | None = None
    valid_bsf = [(i, b) for i, b in zip(iters, bsf) if b is not None]
    if len(valid_bsf) >= 2:
        xs = [p[0] for p in valid_bsf]
        ys = [p[1] for p in valid_bsf]
        area = 0.0
        for k in range(1, len(xs)):
            dx = xs[k] - xs[k - 1]
            area += 0.5 * (ys[k] + ys[k - 1]) * dx
        span = xs[-1] - xs[0]
        if span > 0:
            auc_best_so_far = area / span

    return convergence_speed_gen, auc_best_so_far
