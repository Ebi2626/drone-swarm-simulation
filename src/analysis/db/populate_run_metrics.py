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
        coll_phase AS (
            -- Per kolizja: znajdź ostatni event ∈ {trigger, rejoin} dla danego
            -- drona przed sim_time kolizji. Jeśli to 'trigger' (bez następnego
            -- 'rejoin') → drone był w trakcie uniku → evasion_phase. W p.p.
            -- (event=rejoin lub brak eventów) → drone wykonywał plan offline
            -- → tracking_phase. Suma = collision_count.
            SELECT
                SUM(CASE WHEN last_state = 'trigger' THEN 1 ELSE 0 END)
                    AS evasion_phase_collisions,
                SUM(CASE WHEN last_state IS NULL OR last_state = 'rejoin' THEN 1 ELSE 0 END)
                    AS tracking_phase_collisions
            FROM (
                SELECT
                    c.collision_id,
                    (SELECT e.event_type FROM evasion_events e
                     WHERE e.run_id = c.run_id
                       AND e.drone_id = c.drone_id
                       AND e.sim_time <= c.sim_time
                       AND e.event_type IN ('trigger', 'rejoin')
                     ORDER BY e.sim_time DESC
                     LIMIT 1) AS last_state
                FROM collisions c
                WHERE c.run_id = ?
            )
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
                MAX(max_inter_uav_distance_m)        AS max_inter_uav_distance_m,
                AVG(mean_inter_uav_distance_m)       AS mean_inter_uav_distance_m,
                CASE WHEN COUNT(inter_uav_safety_violation_count) > 0
                     THEN total(inter_uav_safety_violation_count)
                END                                   AS total_inter_uav_safety_violations,
                AVG(energy_indicator)                AS mean_energy_indicator,
                AVG(smoothness_indicator)            AS mean_smoothness_indicator
            FROM uav_online_metrics
            WHERE run_id = ?
        ),
        online_optimization AS (
            -- 2026-05-13: agregaty per-task online optimizer'a + jakość
            -- manewru unikowego. Realizuje §3.1.3.3 (trajektoria online)
            -- i §3.1.3.4 (algorytm online) docs/Praca magisterska.md.
            --
            -- §3.1.3.4 — algorytm online:
            --   AVG(best_fitness) — wartość optymalizacji w budżecie czasowym.
            --   AVG(generations_completed) / AVG(evaluations_completed) —
            --     realne zużycie budżetu (efektywność per-iteracja).
            --   COUNT(*) — liczba triggerów evasion (informational; cross-algo
            --     nieporównywalne bo różne strategie generują różne ilości).
            --
            -- §3.1.3.3 — trajektoria online (plan-level):
            --   AVG(plan_arc_length_m) — długość manewru B-spline.
            --   AVG(plan_total_duration_s) — czas trwania manewru.
            --   AVG(pos_err_at_rejoin_m) / AVG(vel_err_at_rejoin_mps) —
            --     skuteczność powrotu na nominalną trajektorię offline.
            --   AVG(time_to_rejoin_s) — czas powrotu na nominalną.
            --   rejoin_success_rate = SUM(outcome='rejoined_ok')/COUNT(*) —
            --     niezawodność powrotu (proporcja sukcesów).
            SELECT
                AVG(wallclock_s)                     AS avg_wallclock_online_s,
                MAX(wallclock_s)                     AS max_wallclock_online_s,
                AVG(best_fitness)                    AS mean_online_best_fitness,
                AVG(generations_completed)           AS mean_online_generations_completed,
                AVG(evaluations_completed)           AS mean_online_evaluations_completed,
                COUNT(*)                             AS online_optimization_task_count,
                AVG(plan_arc_length_m)               AS mean_evasion_arc_length_m,
                AVG(plan_total_duration_s)           AS mean_evasion_plan_duration_s,
                AVG(pos_err_at_rejoin_m)             AS mean_pos_err_at_rejoin_m,
                AVG(vel_err_at_rejoin_mps)           AS mean_vel_err_at_rejoin_mps,
                AVG(time_to_rejoin_s)                AS mean_time_to_rejoin_s,
                CAST(SUM(CASE WHEN outcome = 'rejoined_ok' THEN 1 ELSE 0 END) AS REAL)
                    / NULLIF(COUNT(*), 0)            AS rejoin_success_rate,
                -- §3.1.3.3 metryka GŁÓWNA: completion na poziomie EPIZODU.
                -- Mianownik wyklucza `pending` (replan supersession, stan
                -- pośredni — nie porażka). T_fail = {never_rejoined,
                -- collided_*}.
                CAST(SUM(CASE WHEN outcome = 'rejoined_ok' THEN 1 ELSE 0 END) AS REAL)
                    / NULLIF(
                        SUM(CASE
                            WHEN outcome IN (
                                'rejoined_ok', 'never_rejoined',
                                'collided_drone', 'collided_ground',
                                'collided_obstacle'
                            ) THEN 1 ELSE 0 END),
                        0
                    )                                AS rejoin_completion_rate,
                -- §3.1.3.4: proporcja triggerów, w których algorytm nie
                -- dostarczył feasible planu w budżecie (status != 'ok' —
                -- obejmuje status='failed' = brak feasible candidate
                -- w populacji oraz status='timed_out' = budget exceeded
                -- bez best-so-far). Safety metric hard real-time (lower=
                -- better). UWAGA: NIE używamy `wallclock_s >= time_budget_s`
                -- bo wszystkie algorytmy saturują budżet i mają OS-jitter
                -- overshoot ~10-50ms (false 100% violation rate).
                CAST(SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) AS REAL)
                    / NULLIF(COUNT(*), 0)            AS budget_violation_rate,
                -- §3.1.3.4 — online_success_rate i SP1 (Auger & Hansen 2005).
                -- online_success_rate = 1 - BVR = SUM(status='ok')/COUNT.
                -- online_sp1 = RT_succ / p_s
                --            = AVG(evals WHERE status='ok') / success_rate
                --            = oczekiwana liczba NFE na jeden dostarczony
                --              plan (Auger & Hansen 2005). LOWER=BETTER.
                -- UWAGA: nie ERT (Hansen 2009 BBOB) bo failed online tasks
                -- abortują z RT_unsucc=0 → ERT degeneruje do RT_succ,
                -- traci semantykę. SP1 uogólnia ERT na scenariusze
                -- early-abort bez restartów (failure liczona jako stracona
                -- próba o oczekiwanym koszcie RT_succ).
                CAST(SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS REAL)
                    / NULLIF(COUNT(*), 0)            AS online_success_rate,
                AVG(CASE WHEN status = 'ok' THEN evaluations_completed END)
                    / NULLIF(
                        CAST(SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS REAL)
                            / NULLIF(COUNT(*), 0),
                        0
                    )                                AS online_sp1
            FROM online_optimization_tasks
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

            -- success = 1 iff trasa zaplanowana offline jest bezkolizyjna
            -- w wykonaniu fizycznym (tracking_phase_collisions == 0).
            -- Kolizje w fazie uniku (evasion_phase) NIE wpływają na success —
            -- te są mierzone osobno przez `is_online_failure`. Zob.
            -- `reports/failure_success_methodology.md` §1.
            CASE
                WHEN COALESCE((SELECT tracking_phase_collisions FROM coll_phase), 0) > 0 THEN 0
                ELSE 1
            END AS success,

            (SELECT total_path_length_2d FROM traj) AS total_path_length_2d,
            (SELECT total_path_length_3d FROM traj) AS total_path_length_3d,
            (SELECT collision_count FROM coll) AS collision_count,
            (SELECT COALESCE(tracking_phase_collisions, 0) FROM coll_phase) AS tracking_phase_collisions,
            (SELECT COALESCE(evasion_phase_collisions, 0) FROM coll_phase) AS evasion_phase_collisions,
            (SELECT evasion_event_count FROM evas) AS evasion_event_count,
            (SELECT obstacle_count FROM obs) AS obstacle_count,
            (SELECT best_iteration FROM best_gen) AS best_iteration,
            (SELECT nondominated_count FROM moo) AS nondominated_count,
            (SELECT hypervolume FROM moo) AS hypervolume,
            (SELECT igd_plus FROM moo) AS igd_plus,
            (SELECT front_size_last_gen FROM moo) AS front_size_last_gen,
            (SELECT hypervolume_normalized FROM moo) AS hypervolume_normalized,
            (SELECT min_inter_uav_distance_m FROM online_uav) AS min_inter_uav_distance_m,
            (SELECT max_inter_uav_distance_m FROM online_uav) AS max_inter_uav_distance_m,
            (SELECT mean_inter_uav_distance_m FROM online_uav) AS mean_inter_uav_distance_m,
            (SELECT total_inter_uav_safety_violations FROM online_uav) AS total_inter_uav_safety_violations,
            (SELECT mean_energy_indicator FROM online_uav) AS mean_energy_indicator,
            (SELECT mean_smoothness_indicator FROM online_uav) AS mean_smoothness_indicator,
            (SELECT avg_wallclock_online_s FROM online_optimization) AS avg_wallclock_online_s,
            (SELECT max_wallclock_online_s FROM online_optimization) AS max_wallclock_online_s,
            (SELECT mean_online_best_fitness FROM online_optimization) AS mean_online_best_fitness,
            (SELECT mean_online_generations_completed FROM online_optimization) AS mean_online_generations_completed,
            (SELECT mean_online_evaluations_completed FROM online_optimization) AS mean_online_evaluations_completed,
            (SELECT online_optimization_task_count FROM online_optimization) AS online_optimization_task_count,
            (SELECT mean_evasion_arc_length_m FROM online_optimization) AS mean_evasion_arc_length_m,
            (SELECT mean_evasion_plan_duration_s FROM online_optimization) AS mean_evasion_plan_duration_s,
            (SELECT mean_pos_err_at_rejoin_m FROM online_optimization) AS mean_pos_err_at_rejoin_m,
            (SELECT mean_vel_err_at_rejoin_mps FROM online_optimization) AS mean_vel_err_at_rejoin_mps,
            (SELECT mean_time_to_rejoin_s FROM online_optimization) AS mean_time_to_rejoin_s,
            (SELECT rejoin_success_rate FROM online_optimization) AS rejoin_success_rate,
            (SELECT rejoin_completion_rate FROM online_optimization) AS rejoin_completion_rate,
            (SELECT budget_violation_rate FROM online_optimization) AS budget_violation_rate,
            (SELECT online_success_rate FROM online_optimization) AS online_success_rate,
            (SELECT online_sp1 FROM online_optimization) AS online_sp1,
            (SELECT gd_final FROM moo_quality_final) AS gd_final,
            (SELECT spread_final FROM moo_quality_final) AS spread_final,
            (SELECT spacing_final FROM moo_quality_final) AS spacing_final,
            (SELECT r2_final FROM moo_quality_final) AS r2_final,
            (SELECT uav_rows FROM uav) AS uav_rows,
            (SELECT traj_rows FROM traj) AS traj_rows
        """,
        # 13 placeholders: uav, traj, coll, coll_phase, evas, obs, best_gen,
        # last_gen, moo, online_uav, online_wallclock, moo_quality_final,
        # drones_fallback.
        (run_id,) * 13,
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

    convergence_speed_gen, auc_best_so_far, total_wallclock_offline_s = (
        _convergence_speed_and_auc(conn, run_id)
    )

    # Median(best_fitness) liczona w Pythonie — SQLite brak natywnego MEDIAN.
    # Wartość komplementarna dla `mean_online_best_fitness` (skewed
    # rozkłady triggerów dają mean/median split → diagnostyka).
    median_online_best_fitness = _compute_online_median_best_fitness(conn, run_id)

    # §3.1.3.3: median(plan_arc_length_m) per run — odporna na pojedyncze
    # długie manewry (np. obejście dużej przeszkody) zaburzające mean.
    median_evasion_arc_length_m = _compute_online_median(
        conn, run_id, "plan_arc_length_m",
    )

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
            tracking_phase_collisions,
            evasion_phase_collisions,
            evasion_event_count,
            obstacle_count,
            best_iteration,
            nondominated_count,
            hypervolume,
            igd_plus,
            front_size_last_gen,
            hypervolume_normalized,
            min_inter_uav_distance_m,
            max_inter_uav_distance_m,
            mean_inter_uav_distance_m,
            total_inter_uav_safety_violations,
            mean_energy_indicator,
            mean_smoothness_indicator,
            avg_wallclock_online_s,
            max_wallclock_online_s,
            mean_online_best_fitness,
            median_online_best_fitness,
            mean_online_generations_completed,
            mean_online_evaluations_completed,
            online_optimization_task_count,
            mean_evasion_arc_length_m,
            median_evasion_arc_length_m,
            mean_evasion_plan_duration_s,
            mean_pos_err_at_rejoin_m,
            mean_vel_err_at_rejoin_mps,
            mean_time_to_rejoin_s,
            rejoin_success_rate,
            rejoin_completion_rate,
            budget_violation_rate,
            online_success_rate,
            online_sp1,
            gd_final,
            spread_final,
            spacing_final,
            r2_final,
            convergence_speed_gen,
            auc_best_so_far,
            total_wallclock_offline_s,
            summary_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            drone_count = excluded.drone_count,
            success = excluded.success,
            total_path_length_2d = excluded.total_path_length_2d,
            total_path_length_3d = excluded.total_path_length_3d,
            collision_count = excluded.collision_count,
            tracking_phase_collisions = excluded.tracking_phase_collisions,
            evasion_phase_collisions = excluded.evasion_phase_collisions,
            evasion_event_count = excluded.evasion_event_count,
            obstacle_count = excluded.obstacle_count,
            best_iteration = excluded.best_iteration,
            nondominated_count = excluded.nondominated_count,
            hypervolume = excluded.hypervolume,
            igd_plus = excluded.igd_plus,
            front_size_last_gen = excluded.front_size_last_gen,
            hypervolume_normalized = excluded.hypervolume_normalized,
            min_inter_uav_distance_m = excluded.min_inter_uav_distance_m,
            max_inter_uav_distance_m = excluded.max_inter_uav_distance_m,
            mean_inter_uav_distance_m = excluded.mean_inter_uav_distance_m,
            total_inter_uav_safety_violations = excluded.total_inter_uav_safety_violations,
            mean_energy_indicator = excluded.mean_energy_indicator,
            mean_smoothness_indicator = excluded.mean_smoothness_indicator,
            avg_wallclock_online_s = excluded.avg_wallclock_online_s,
            max_wallclock_online_s = excluded.max_wallclock_online_s,
            mean_online_best_fitness = excluded.mean_online_best_fitness,
            median_online_best_fitness = excluded.median_online_best_fitness,
            mean_online_generations_completed = excluded.mean_online_generations_completed,
            mean_online_evaluations_completed = excluded.mean_online_evaluations_completed,
            online_optimization_task_count = excluded.online_optimization_task_count,
            mean_evasion_arc_length_m = excluded.mean_evasion_arc_length_m,
            median_evasion_arc_length_m = excluded.median_evasion_arc_length_m,
            mean_evasion_plan_duration_s = excluded.mean_evasion_plan_duration_s,
            mean_pos_err_at_rejoin_m = excluded.mean_pos_err_at_rejoin_m,
            mean_vel_err_at_rejoin_mps = excluded.mean_vel_err_at_rejoin_mps,
            mean_time_to_rejoin_s = excluded.mean_time_to_rejoin_s,
            rejoin_success_rate = excluded.rejoin_success_rate,
            rejoin_completion_rate = excluded.rejoin_completion_rate,
            budget_violation_rate = excluded.budget_violation_rate,
            online_success_rate = excluded.online_success_rate,
            online_sp1 = excluded.online_sp1,
            gd_final = excluded.gd_final,
            spread_final = excluded.spread_final,
            spacing_final = excluded.spacing_final,
            r2_final = excluded.r2_final,
            convergence_speed_gen = excluded.convergence_speed_gen,
            auc_best_so_far = excluded.auc_best_so_far,
            total_wallclock_offline_s = excluded.total_wallclock_offline_s,
            summary_json = excluded.summary_json
        """,
        (
            run_id,
            data["drone_count"],
            data["success"],
            data["total_path_length_2d"],
            data["total_path_length_3d"],
            data["collision_count"],
            data["tracking_phase_collisions"],
            data["evasion_phase_collisions"],
            data["evasion_event_count"],
            data["obstacle_count"],
            data["best_iteration"],
            int(data["nondominated_count"]) if data["nondominated_count"] is not None else None,
            data["hypervolume"],
            data["igd_plus"],
            int(data["front_size_last_gen"]) if data["front_size_last_gen"] is not None else None,
            data["hypervolume_normalized"],
            data["min_inter_uav_distance_m"],
            data["max_inter_uav_distance_m"],
            data["mean_inter_uav_distance_m"],
            total_violations_int,
            data["mean_energy_indicator"],
            data["mean_smoothness_indicator"],
            data["avg_wallclock_online_s"],
            data["max_wallclock_online_s"],
            data["mean_online_best_fitness"],
            median_online_best_fitness,
            data["mean_online_generations_completed"],
            data["mean_online_evaluations_completed"],
            (
                int(data["online_optimization_task_count"])
                if data["online_optimization_task_count"] is not None else None
            ),
            data["mean_evasion_arc_length_m"],
            median_evasion_arc_length_m,
            data["mean_evasion_plan_duration_s"],
            data["mean_pos_err_at_rejoin_m"],
            data["mean_vel_err_at_rejoin_mps"],
            data["mean_time_to_rejoin_s"],
            data["rejoin_success_rate"],
            data["rejoin_completion_rate"],
            data["budget_violation_rate"],
            data["online_success_rate"],
            data["online_sp1"],
            data["gd_final"],
            data["spread_final"],
            data["spacing_final"],
            data["r2_final"],
            convergence_speed_gen,
            auc_best_so_far,
            total_wallclock_offline_s,
            json.dumps(summary, ensure_ascii=False, sort_keys=True),
        ),
    )


def _compute_online_median_best_fitness(
    conn: sqlite3.Connection, run_id: str,
) -> float | None:
    """Median(best_fitness) over `online_optimization_tasks` per run.

    SQLite brak natywnego MEDIAN — liczymy w Pythonie ze sortowanej listy.
    Returns `None` gdy brak rekordów / wszystkie `best_fitness` NULL.
    """
    return _compute_online_median(conn, run_id, "best_fitness")


def _compute_online_median(
    conn: sqlite3.Connection, run_id: str, column: str,
) -> float | None:
    """Median(`column`) over `online_optimization_tasks` per run.

    Generic helper — SQLite brak natywnego MEDIAN. Whitelist nazw kolumn,
    by uniknąć SQL injection (column name nie może być parametryzowany).
    """
    _ALLOWED_COLUMNS = {
        "best_fitness", "plan_arc_length_m", "plan_total_duration_s",
        "pos_err_at_rejoin_m", "vel_err_at_rejoin_mps", "time_to_rejoin_s",
        "wallclock_s", "generations_completed", "evaluations_completed",
    }
    if column not in _ALLOWED_COLUMNS:
        raise ValueError(f"_compute_online_median: kolumna {column!r} poza whitelistą.")
    import statistics
    rows = conn.execute(
        f"SELECT {column} FROM online_optimization_tasks "
        f"WHERE run_id = ? AND {column} IS NOT NULL",
        (run_id,),
    ).fetchall()
    values = [float(r[0]) for r in rows]
    if not values:
        return None
    return float(statistics.median(values))


def _convergence_speed_and_auc(
    conn: sqlite3.Connection, run_id: str
) -> tuple[int | None, float | None, float | None]:
    """Liczy z `iteration_metrics`:
    - convergence_speed_gen: pierwsza generacja gdy HV ≥ 0.9 · HV(last_gen).
      NULL gdy brak HV lub last_gen HV < final*0.9.
    - auc_best_so_far: ∫ best_so_far(g) dg (trapez), znormalizowane przez
      (last_gen − first_gen). Lower = lepiej (mniej kosztu over time).
    - total_wallclock_offline_s: SUM(elapsed_s) — fizyczny czas wallclock
      całej fazy offline. NULL gdy żadna iteracja nie ma `elapsed_s`.
    """
    rows = conn.execute(
        """
        SELECT iteration, hypervolume, best_so_far, elapsed_s
        FROM iteration_metrics
        WHERE run_id = ?
        ORDER BY iteration ASC
        """,
        (run_id,),
    ).fetchall()
    if not rows:
        return None, None, None

    iters = [int(r[0]) for r in rows]
    hvs = [r[1] for r in rows]
    bsf = [r[2] for r in rows]
    elapsed = [r[3] for r in rows]

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

    valid_elapsed = [e for e in elapsed if e is not None]
    total_wallclock_offline_s: float | None = (
        float(sum(valid_elapsed)) if valid_elapsed else None
    )

    return convergence_speed_gen, auc_best_so_far, total_wallclock_offline_s
