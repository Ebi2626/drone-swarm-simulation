import json
import sqlite3

def populate_run_metrics(conn: sqlite3.Connection, run_id: str) -> None:
    cur = conn.execute(
        """
        WITH
        uav AS (
            SELECT
                COUNT(*) AS uav_rows,
                MIN(COALESCE(success, 1)) AS all_uav_success,
                CASE WHEN COUNT(final_objective) > 0 THEN total(final_objective) END AS final_objective,
                CASE WHEN COUNT(energy_cost) > 0 THEN total(energy_cost) END AS total_energy_cost,
                CASE WHEN COUNT(smoothness_cost) > 0 THEN total(smoothness_cost) END AS total_smoothness_cost,
                CASE WHEN COUNT(threat_cost) > 0 THEN total(threat_cost) END AS total_threat_cost,
                CASE WHEN COUNT(altitude_cost) > 0 THEN total(altitude_cost) END AS total_altitude_cost,
                CASE WHEN COUNT(terrain_penalty) > 0 THEN total(terrain_penalty) END AS total_terrain_penalty,
                CASE WHEN COUNT(turn_penalty) > 0 THEN total(turn_penalty) END AS total_turn_penalty,
                CASE WHEN COUNT(climb_penalty) > 0 THEN total(climb_penalty) END AS total_climb_penalty,
                CASE WHEN COUNT(collision_penalty) > 0 THEN total(collision_penalty) END AS total_collision_penalty
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
                MAX(CASE WHEN ogs.metric_name = 'feasible_nondominated_count'
                         THEN ogs.metric_value END) AS feasible_nondominated_count,
                MAX(CASE WHEN ogs.metric_name = 'hypervolume'
                         THEN ogs.metric_value END) AS hypervolume,
                MAX(CASE WHEN ogs.metric_name IN ('igd_plus', 'igd+')
                         THEN ogs.metric_value END) AS igd_plus
            FROM optimization_generation_stats ogs
            JOIN last_gen lg
              ON lg.generation = ogs.generation
            WHERE ogs.run_id = ?
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

            (SELECT final_objective FROM uav) AS final_objective,
            (SELECT total_path_length_2d FROM traj) AS total_path_length_2d,
            (SELECT total_path_length_3d FROM traj) AS total_path_length_3d,
            (SELECT total_energy_cost FROM uav) AS total_energy_cost,
            (SELECT total_smoothness_cost FROM uav) AS total_smoothness_cost,
            (SELECT total_threat_cost FROM uav) AS total_threat_cost,
            (SELECT total_altitude_cost FROM uav) AS total_altitude_cost,
            (SELECT total_terrain_penalty FROM uav) AS total_terrain_penalty,
            (SELECT total_turn_penalty FROM uav) AS total_turn_penalty,
            (SELECT total_climb_penalty FROM uav) AS total_climb_penalty,
            (SELECT total_collision_penalty FROM uav) AS total_collision_penalty,
            (SELECT collision_count FROM coll) AS collision_count,
            (SELECT evasion_event_count FROM evas) AS evasion_event_count,
            (SELECT obstacle_count FROM obs) AS obstacle_count,
            (SELECT best_iteration FROM best_gen) AS best_iteration,
            (SELECT nondominated_count FROM moo) AS nondominated_count,
            (SELECT feasible_nondominated_count FROM moo) AS feasible_nondominated_count,
            (SELECT hypervolume FROM moo) AS hypervolume,
            (SELECT igd_plus FROM moo) AS igd_plus,
            (SELECT uav_rows FROM uav) AS uav_rows,
            (SELECT traj_rows FROM traj) AS traj_rows
        """,
        (run_id, run_id, run_id, run_id, run_id, run_id, run_id, run_id, run_id),
    )

    row = cur.fetchone()
    columns = [desc[0] for desc in cur.description]
    data = dict(zip(columns, row))

    decision_mode = conn.execute(
        """
        SELECT value
        FROM meta
        WHERE key = ?
        """,
        (f"{run_id}.decision_mode",),
    ).fetchone()
    decision_mode = decision_mode[0] if decision_mode else None

    selected_solution_index = conn.execute(
        """
        SELECT value
        FROM meta
        WHERE key = ?
        """,
        (f"{run_id}.selected_solution_index",),
    ).fetchone()
    selected_solution_index = int(selected_solution_index[0]) if selected_solution_index else None

    reference_point = conn.execute(
        """
        SELECT value
        FROM meta
        WHERE key = ?
        """,
        (f"{run_id}.reference_point_json",),
    ).fetchone()
    reference_point_json = reference_point[0] if reference_point else None

    objective_components = {
        "energy_cost": data["total_energy_cost"],
        "smoothness_cost": data["total_smoothness_cost"],
        "threat_cost": data["total_threat_cost"],
        "altitude_cost": data["total_altitude_cost"],
        "terrain_penalty": data["total_terrain_penalty"],
        "turn_penalty": data["total_turn_penalty"],
        "climb_penalty": data["total_climb_penalty"],
        "collision_penalty": data["total_collision_penalty"],
    }

    summary = {
        "path_source": "trajectory_samples",
        "best_iteration_source": "optimization_generation_stats.best_so_far_obj0",
        "moo_metrics_source": "optimization_generation_stats[last_generation]",
        "success_rule": "collision_count == 0 and all_uav_success_if_present",
        "uav_rows_used": data["uav_rows"],
        "trajectory_metric_rows_used": data["traj_rows"],
    }

    conn.execute(
        """
        INSERT INTO run_metrics (
            run_id,
            drone_count,
            success,
            final_objective,
            total_path_length_2d,
            total_path_length_3d,
            total_energy_cost,
            total_smoothness_cost,
            total_threat_cost,
            total_altitude_cost,
            total_terrain_penalty,
            total_turn_penalty,
            total_climb_penalty,
            total_collision_penalty,
            collision_count,
            evasion_event_count,
            obstacle_count,
            best_iteration,
            decision_mode,
            selected_solution_index,
            nondominated_count,
            feasible_nondominated_count,
            hypervolume,
            igd_plus,
            reference_point_json,
            objective_components_json,
            summary_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            drone_count = excluded.drone_count,
            success = excluded.success,
            final_objective = excluded.final_objective,
            total_path_length_2d = excluded.total_path_length_2d,
            total_path_length_3d = excluded.total_path_length_3d,
            total_energy_cost = excluded.total_energy_cost,
            total_smoothness_cost = excluded.total_smoothness_cost,
            total_threat_cost = excluded.total_threat_cost,
            total_altitude_cost = excluded.total_altitude_cost,
            total_terrain_penalty = excluded.total_terrain_penalty,
            total_turn_penalty = excluded.total_turn_penalty,
            total_climb_penalty = excluded.total_climb_penalty,
            total_collision_penalty = excluded.total_collision_penalty,
            collision_count = excluded.collision_count,
            evasion_event_count = excluded.evasion_event_count,
            obstacle_count = excluded.obstacle_count,
            best_iteration = excluded.best_iteration,
            decision_mode = excluded.decision_mode,
            selected_solution_index = excluded.selected_solution_index,
            nondominated_count = excluded.nondominated_count,
            feasible_nondominated_count = excluded.feasible_nondominated_count,
            hypervolume = excluded.hypervolume,
            igd_plus = excluded.igd_plus,
            reference_point_json = excluded.reference_point_json,
            objective_components_json = excluded.objective_components_json,
            summary_json = excluded.summary_json
        """,
        (
            run_id,
            data["drone_count"],
            data["success"],
            data["final_objective"],
            data["total_path_length_2d"],
            data["total_path_length_3d"],
            data["total_energy_cost"],
            data["total_smoothness_cost"],
            data["total_threat_cost"],
            data["total_altitude_cost"],
            data["total_terrain_penalty"],
            data["total_turn_penalty"],
            data["total_climb_penalty"],
            data["total_collision_penalty"],
            data["collision_count"],
            data["evasion_event_count"],
            data["obstacle_count"],
            data["best_iteration"],
            decision_mode,
            selected_solution_index,
            int(data["nondominated_count"]) if data["nondominated_count"] is not None else None,
            int(data["feasible_nondominated_count"]) if data["feasible_nondominated_count"] is not None else None,
            data["hypervolume"],
            data["igd_plus"],
            reference_point_json,
            json.dumps(objective_components, ensure_ascii=False, sort_keys=True),
            json.dumps(summary, ensure_ascii=False, sort_keys=True),
        ),
    )