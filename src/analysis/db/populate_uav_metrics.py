from __future__ import annotations

import json
import sqlite3


def populate_uav_metrics(conn: sqlite3.Connection, run_id: str) -> None:
    cur = conn.execute(
        """
        WITH
        uavs AS (
            SELECT uav_id
            FROM trajectory_metrics
            WHERE run_id = ?

            UNION

            SELECT drone_id AS uav_id
            FROM trajectory_samples
            WHERE run_id = ?

            UNION

            SELECT drone_id AS uav_id
            FROM counted_trajectory_points
            WHERE run_id = ?

            UNION

            SELECT drone_id AS uav_id
            FROM collisions
            WHERE run_id = ?

            UNION

            SELECT drone_id AS uav_id
            FROM evasion_events
            WHERE run_id = ?
        ),
        traj_actual AS (
            SELECT
                uav_id,
                point_count,
                path_length_2d,
                path_length_3d,
                min_altitude,
                max_altitude,
                mean_altitude
            FROM trajectory_metrics
            WHERE run_id = ?
              AND source_name = 'trajectory_samples'
        ),
        traj_planned AS (
            SELECT
                uav_id,
                point_count,
                path_length_2d,
                path_length_3d,
                min_altitude,
                max_altitude,
                mean_altitude
            FROM trajectory_metrics
            WHERE run_id = ?
              AND source_name = 'counted_trajectory_points'
        ),
        coll AS (
            SELECT
                drone_id AS uav_id,
                COUNT(*) AS collision_count
            FROM collisions
            WHERE run_id = ?
            GROUP BY drone_id
        ),
        evas AS (
            SELECT
                drone_id AS uav_id,
                COUNT(*) AS evasion_event_count
            FROM evasion_events
            WHERE run_id = ?
            GROUP BY drone_id
        )
        SELECT
            u.uav_id,
            CASE
                WHEN COALESCE(c.collision_count, 0) > 0 THEN 0
                ELSE 1
            END AS success,

            COALESCE(a.path_length_2d, p.path_length_2d) AS path_length_2d,
            COALESCE(a.path_length_3d, p.path_length_3d) AS path_length_3d,

            COALESCE(c.collision_count, 0) AS collision_count,
            COALESCE(e.evasion_event_count, 0) AS evasion_event_count,

            a.point_count AS actual_point_count,
            p.point_count AS planned_point_count,

            a.min_altitude AS actual_min_altitude,
            a.max_altitude AS actual_max_altitude,
            a.mean_altitude AS actual_mean_altitude,

            p.min_altitude AS planned_min_altitude,
            p.max_altitude AS planned_max_altitude,
            p.mean_altitude AS planned_mean_altitude
        FROM uavs u
        LEFT JOIN traj_actual a
            ON a.uav_id = u.uav_id
        LEFT JOIN traj_planned p
            ON p.uav_id = u.uav_id
        LEFT JOIN coll c
            ON c.uav_id = u.uav_id
        LEFT JOIN evas e
            ON e.uav_id = u.uav_id
        ORDER BY u.uav_id
        """,
        (
            run_id,  # uavs / trajectory_metrics
            run_id,  # uavs / trajectory_samples
            run_id,  # uavs / counted_trajectory_points
            run_id,  # uavs / collisions
            run_id,  # uavs / evasion_events
            run_id,  # traj_actual
            run_id,  # traj_planned
            run_id,  # coll
            run_id,  # evas
        ),
    )

    source_rows = cur.fetchall()

    rows_to_insert = []
    for row in source_rows:
        (
            uav_id,
            success,
            path_length_2d,
            path_length_3d,
            collision_count,
            evasion_event_count,
            actual_point_count,
            planned_point_count,
            actual_min_altitude,
            actual_max_altitude,
            actual_mean_altitude,
            planned_min_altitude,
            planned_max_altitude,
            planned_mean_altitude,
        ) = row

        extra_json = json.dumps(
            {
                "path_source_preference": "trajectory_samples_then_counted_trajectory_points",
                "actual_point_count": actual_point_count,
                "planned_point_count": planned_point_count,
                "actual_altitude": {
                    "min": actual_min_altitude,
                    "max": actual_max_altitude,
                    "mean": actual_mean_altitude,
                },
                "planned_altitude": {
                    "min": planned_min_altitude,
                    "max": planned_max_altitude,
                    "mean": planned_mean_altitude,
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        rows_to_insert.append(
            (
                run_id,
                uav_id,
                success,
                None,                 # final_objective
                path_length_2d,
                path_length_3d,
                None,                 # energy_cost
                None,                 # smoothness_cost
                None,                 # threat_cost
                None,                 # altitude_cost
                None,                 # terrain_penalty
                None,                 # turn_penalty
                None,                 # climb_penalty
                None,                 # collision_penalty
                collision_count,
                evasion_event_count,
                extra_json,
            )
        )

    conn.execute(
        "DELETE FROM uav_metrics WHERE run_id = ?",
        (run_id,),
    )

    if not rows_to_insert:
        return

    conn.executemany(
        """
        INSERT INTO uav_metrics (
            run_id,
            uav_id,
            success,
            final_objective,
            path_length_2d,
            path_length_3d,
            energy_cost,
            smoothness_cost,
            threat_cost,
            altitude_cost,
            terrain_penalty,
            turn_penalty,
            climb_penalty,
            collision_penalty,
            collision_count,
            evasion_event_count,
            extra_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows_to_insert,
    )