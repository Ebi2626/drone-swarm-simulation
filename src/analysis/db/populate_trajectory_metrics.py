from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict


def _compute_metrics(points: list[tuple[float, float, float]]) -> dict:
    point_count = len(points)

    if point_count == 0:
        return {
            "point_count": 0,
            "path_length_2d": None,
            "path_length_3d": None,
            "min_altitude": None,
            "max_altitude": None,
            "mean_altitude": None,
        }

    zs = [p[2] for p in points]

    path_length_2d = 0.0
    path_length_3d = 0.0

    for i in range(1, point_count):
        x0, y0, z0 = points[i - 1]
        x1, y1, z1 = points[i]

        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0

        path_length_2d += math.hypot(dx, dy)
        path_length_3d += math.sqrt(dx * dx + dy * dy + dz * dz)

    return {
        "point_count": point_count,
        "path_length_2d": path_length_2d,
        "path_length_3d": path_length_3d,
        "min_altitude": min(zs),
        "max_altitude": max(zs),
        "mean_altitude": sum(zs) / point_count,
    }


def _load_grouped_points(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple,
) -> dict[int, list[tuple[float, float, float]]]:
    grouped: dict[int, list[tuple[float, float, float]]] = defaultdict(list)

    cur = conn.execute(sql, params)
    for uav_id, x, y, z in cur:
        grouped[int(uav_id)].append((float(x), float(y), float(z)))

    return dict(grouped)


def _populate_one_source(
    conn: sqlite3.Connection,
    run_id: str,
    source_name: str,
    grouped_points: dict[int, list[tuple[float, float, float]]],
) -> None:
    conn.execute(
        """
        DELETE FROM trajectory_metrics
        WHERE run_id = ? AND source_name = ?
        """,
        (run_id, source_name),
    )

    rows_to_insert = []

    for uav_id in sorted(grouped_points):
        metrics = _compute_metrics(grouped_points[uav_id])

        extra_json = json.dumps(
            {
                "source_name": source_name,
                "computation": "polyline_metrics_from_ordered_points",
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        rows_to_insert.append(
            (
                run_id,
                source_name,
                uav_id,
                metrics["point_count"],
                metrics["path_length_2d"],
                metrics["path_length_3d"],
                metrics["min_altitude"],
                metrics["max_altitude"],
                metrics["mean_altitude"],
                extra_json,
            )
        )

    if rows_to_insert:
        conn.executemany(
            """
            INSERT INTO trajectory_metrics (
                run_id,
                source_name,
                uav_id,
                point_count,
                path_length_2d,
                path_length_3d,
                min_altitude,
                max_altitude,
                mean_altitude,
                extra_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )


def populate_trajectory_metrics(conn: sqlite3.Connection, run_id: str) -> None:
    counted_points = _load_grouped_points(
        conn,
        """
        SELECT drone_id, x, y, z
        FROM counted_trajectory_points
        WHERE run_id = ?
        ORDER BY drone_id, waypoint_id
        """,
        (run_id,),
    )

    actual_points = _load_grouped_points(
        conn,
        """
        SELECT drone_id, x, y, z
        FROM trajectory_samples
        WHERE run_id = ?
        ORDER BY drone_id, sample_index
        """,
        (run_id,),
    )

    _populate_one_source(
        conn=conn,
        run_id=run_id,
        source_name="counted_trajectory_points",
        grouped_points=counted_points,
    )

    _populate_one_source(
        conn=conn,
        run_id=run_id,
        source_name="trajectory_samples",
        grouped_points=actual_points,
    )