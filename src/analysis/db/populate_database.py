# src/analysis/db/populate_database.py
from pathlib import Path
import sqlite3
import csv

from .utils import list_run_directories, parse_run_dir_name


def populate_database(experiment_dir: str | Path) -> Path:
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    db_path = experiment_dir / "analysis.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row

        for run_dir in list_run_directories(experiment_dir):
            run_meta = parse_run_dir_name(run_dir.name)
            run_id = run_dir.name

            conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id,
                    run_dir_name,
                    source_path,
                    optimizer_algo,
                    avoidance_algo,
                    environment,
                    seed,
                    algorithm_pair,
                    aggregation_status,
                    aggregated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'aggregated', CURRENT_TIMESTAMP)
                """,
                (
                    run_id,
                    run_dir.name,
                    str(run_dir),
                    run_meta["optimizer"],
                    run_meta["avoidance"],
                    run_meta["environment"],
                    run_meta["seed"],
                    run_meta["algorithm_pair"],
                ),
            )

            _register_run_files(conn, run_id, run_dir)
            _load_optimization_timings(conn, run_id, run_dir / "optimization_timings.csv")
            _load_collisions(conn, run_id, run_dir / "collisions.csv")
            _load_evasion_events(conn, run_id, run_dir / "evasion_events.csv")
            _load_world_boundaries(conn, run_id, run_dir / "world_boundaries.csv")
            _load_generated_obstacles(conn, run_id, run_dir / "generated_obstacles.csv")
            _load_counted_trajectories(conn, run_id, run_dir / "counted_trajectories.csv")
            _load_trajectories(conn, run_id, run_dir / "trajectories.csv")

            # tu później:
            # _load_iteration_metrics_from_h5(...)
            # _load_trajectory_points(...)
            # _compute_and_store_run_metrics(...)

        conn.commit()

    return db_path


def _register_run_files(conn: sqlite3.Connection, run_id: str, run_dir: Path) -> None:
    file_map = {
        "collisions_csv": "collisions.csv",
        "counted_trajectories_csv": "counted_trajectories.csv",
        "evasion_events_csv": "evasion_events.csv",
        "generated_obstacles_csv": "generated_obstacles.csv",
        "lidar_hits_h5": "lidar_hits.h5",
        "main_log": "main.log",
        "optimization_history_h5": "optimization_history/optimization_history.h5",
        "optimization_timings_csv": "optimization_timings.csv",
        "trajectories_csv": "trajectories.csv",
        "world_boundaries_csv": "world_boundaries.csv",
    }

    for role, relative_path in file_map.items():
        full_path = run_dir / relative_path
        conn.execute(
            """
            INSERT OR REPLACE INTO run_files (
                run_id, file_role, relative_path, file_format, exists_flag, size_bytes
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                role,
                relative_path,
                full_path.suffix.lstrip(".") or None,
                int(full_path.exists()),
                full_path.stat().st_size if full_path.exists() else None,
            ),
        )


def _load_optimization_timings(conn: sqlite3.Connection, run_id: str, csv_path: Path) -> None:
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: brak nagłówków CSV")

        required = {
            "algorithm_name",
            "stage_name",
            "wall_time_s",
            "cpu_time_s",
            "timestamp_utc",
            "success",
        }
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path}: brak wymaganych kolumn: {sorted(missing)}; "
                f"dostępne: {reader.fieldnames}"
            )

        rows = []
        for line_no, row in enumerate(reader, start=2):
            algorithm_name = (row.get("algorithm_name") or "").strip() or None
            stage_name = (row.get("stage_name") or "").strip()
            wall_time_s = _to_float(row.get("wall_time_s"))
            cpu_time_s = _to_float(row.get("cpu_time_s"))
            timestamp_utc = (row.get("timestamp_utc") or "").strip() or None
            success = _to_int_bool(row.get("success"))

            if not stage_name:
                raise ValueError(f"{csv_path}:{line_no}: puste stage_name")

            if wall_time_s is None:
                raise ValueError(
                    f"{csv_path}:{line_no}: niepoprawne wall_time_s={row.get('wall_time_s')!r}"
                )

            rows.append(
                (
                    run_id,
                    algorithm_name,
                    stage_name,
                    wall_time_s,
                    cpu_time_s,
                    timestamp_utc,
                    success,
                )
            )

    conn.executemany(
        """
        INSERT OR REPLACE INTO optimization_timings (
            run_id, algorithm_name, stage_name, wall_time_s, cpu_time_s, timestamp_utc, success
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _load_collisions(conn: sqlite3.Connection, run_id: str, csv_path: Path) -> None:
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: brak nagłówków CSV")

        required = {"time", "drone_id", "other_body_id"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path}: brak wymaganych kolumn: {sorted(missing)}; "
                f"dostępne: {reader.fieldnames}"
            )

        rows = []
        for line_no, row in enumerate(reader, start=2):
            sim_time = _to_float(row.get("time"))
            drone_id = _to_int(row.get("drone_id"))
            other_body_id = _to_int(row.get("other_body_id"))

            if sim_time is None:
                raise ValueError(f"{csv_path}:{line_no}: niepoprawne time={row.get('time')!r}")
            if drone_id is None:
                raise ValueError(f"{csv_path}:{line_no}: niepoprawne drone_id={row.get('drone_id')!r}")
            if other_body_id is None:
                raise ValueError(f"{csv_path}:{line_no}: niepoprawne other_body_id={row.get('other_body_id')!r}")

            rows.append(
                (
                    run_id,
                    line_no - 2,
                    sim_time,
                    drone_id,
                    other_body_id,
                )
            )

    conn.executemany(
        """
        INSERT OR REPLACE INTO collisions (
            run_id, event_index, sim_time, drone_id, other_body_id
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )


def _load_evasion_events(conn: sqlite3.Connection, run_id: str, csv_path: Path) -> None:
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: brak nagłówków CSV")

        required = {
            "time", "drone_id", "event_type", "mode",
            "ttc", "dist_to_threat",
            "threat_x", "threat_y", "threat_z",
            "threat_vx", "threat_vy", "threat_vz",
            "rejoin_x", "rejoin_y", "rejoin_z", "rejoin_arc",
            "astar_success", "fallback_used",
            "pos_error_at_rejoin", "vel_error_at_rejoin",
            "planning_wall_time_s", "notes",
        }
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path}: brak wymaganych kolumn: {sorted(missing)}; "
                f"dostępne: {reader.fieldnames}"
            )

        rows = []
        for line_no, row in enumerate(reader, start=2):
            sim_time = _to_float_nullable(row.get("time"))
            drone_id = _to_int_nullable(row.get("drone_id"))
            event_type = _to_str_nullable(row.get("event_type"))
            mode = _to_int_nullable(row.get("mode"))

            if sim_time is None:
                raise ValueError(f"{csv_path}:{line_no}: niepoprawne time={row.get('time')!r}")
            if drone_id is None:
                raise ValueError(f"{csv_path}:{line_no}: niepoprawne drone_id={row.get('drone_id')!r}")
            if not event_type:
                raise ValueError(f"{csv_path}:{line_no}: puste event_type")

            rows.append(
                (
                    run_id,
                    line_no - 2,
                    sim_time,
                    drone_id,
                    event_type,
                    mode,
                    _to_float_nullable(row.get("ttc")),
                    _to_float_nullable(row.get("dist_to_threat")),
                    _to_float_nullable(row.get("threat_x")),
                    _to_float_nullable(row.get("threat_y")),
                    _to_float_nullable(row.get("threat_z")),
                    _to_float_nullable(row.get("threat_vx")),
                    _to_float_nullable(row.get("threat_vy")),
                    _to_float_nullable(row.get("threat_vz")),
                    _to_float_nullable(row.get("rejoin_x")),
                    _to_float_nullable(row.get("rejoin_y")),
                    _to_float_nullable(row.get("rejoin_z")),
                    _to_float_nullable(row.get("rejoin_arc")),
                    _to_int_bool_nullable(row.get("astar_success")),
                    _to_int_bool_nullable(row.get("fallback_used")),
                    _to_float_nullable(row.get("pos_error_at_rejoin")),
                    _to_float_nullable(row.get("vel_error_at_rejoin")),
                    _to_float_nullable(row.get("planning_wall_time_s")),
                    _to_str_nullable(row.get("notes")),
                )
            )

    conn.executemany(
        """
        INSERT OR REPLACE INTO evasion_events (
            run_id,
            event_index,
            sim_time,
            drone_id,
            event_type,
            mode,
            ttc,
            dist_to_threat,
            threat_x,
            threat_y,
            threat_z,
            threat_vx,
            threat_vy,
            threat_vz,
            rejoin_x,
            rejoin_y,
            rejoin_z,
            rejoin_arc,
            astar_success,
            fallback_used,
            pos_error_at_rejoin,
            vel_error_at_rejoin,
            planning_wall_time_s,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _load_world_boundaries(conn: sqlite3.Connection, run_id: str, csv_path: Path) -> None:
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: brak nagłówków CSV")

        required = {"Axis", "Dimension", "Min_Bound", "Max_Bound", "Center"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path}: brak wymaganych kolumn: {sorted(missing)}; "
                f"dostępne: {reader.fieldnames}"
            )

        axes = {}
        for line_no, row in enumerate(reader, start=2):
            axis = _to_str_nullable(row.get("Axis"))
            if axis not in {"X", "Y", "Z"}:
                raise ValueError(f"{csv_path}:{line_no}: niepoprawna oś {axis!r}")

            axes[axis] = {
                "dimension": _to_float_nullable(row.get("Dimension")),
                "min_bound": _to_float_nullable(row.get("Min_Bound")),
                "max_bound": _to_float_nullable(row.get("Max_Bound")),
                "center": _to_float_nullable(row.get("Center")),
            }

    conn.execute(
        """
        INSERT OR REPLACE INTO world_boundaries (
            run_id,
            x_dimension, x_min_bound, x_max_bound, x_center,
            y_dimension, y_min_bound, y_max_bound, y_center,
            z_dimension, z_min_bound, z_max_bound, z_center
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            axes.get("X", {}).get("dimension"),
            axes.get("X", {}).get("min_bound"),
            axes.get("X", {}).get("max_bound"),
            axes.get("X", {}).get("center"),
            axes.get("Y", {}).get("dimension"),
            axes.get("Y", {}).get("min_bound"),
            axes.get("Y", {}).get("max_bound"),
            axes.get("Y", {}).get("center"),
            axes.get("Z", {}).get("dimension"),
            axes.get("Z", {}).get("min_bound"),
            axes.get("Z", {}).get("max_bound"),
            axes.get("Z", {}).get("center"),
        ),
    )


def _load_generated_obstacles(conn: sqlite3.Connection, run_id: str, csv_path: Path) -> None:
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: brak nagłówków CSV")

        required = {"x", "y", "z", "radius", "height", "unused_dim"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path}: brak wymaganych kolumn: {sorted(missing)}; "
                f"dostępne: {reader.fieldnames}"
            )

        rows = []
        for line_no, row in enumerate(reader, start=2):
            x = _to_float_nullable(row.get("x"))
            y = _to_float_nullable(row.get("y"))
            z = _to_float_nullable(row.get("z"))

            if x is None or y is None or z is None:
                raise ValueError(
                    f"{csv_path}:{line_no}: niepoprawne współrzędne x={row.get('x')!r}, "
                    f"y={row.get('y')!r}, z={row.get('z')!r}"
                )

            rows.append(
                (
                    run_id,
                    line_no - 2,
                    x,
                    y,
                    z,
                    _to_float_nullable(row.get("radius")),
                    _to_float_nullable(row.get("height")),
                    _to_float_nullable(row.get("unused_dim")),
                )
            )

    conn.executemany(
        """
        INSERT OR REPLACE INTO generated_obstacles (
            run_id, obstacle_index, x, y, z, radius, height, unused_dim
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _load_counted_trajectories(conn: sqlite3.Connection, run_id: str, csv_path: Path) -> None:
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: brak nagłówków CSV")

        required = {"drone_id", "waypoint_id", "x", "y", "z"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path}: brak wymaganych kolumn: {sorted(missing)}; "
                f"dostępne: {reader.fieldnames}"
            )

        rows = []
        for line_no, row in enumerate(reader, start=2):
            drone_id = _to_int_nullable(row.get("drone_id"))
            waypoint_id = _to_int_nullable(row.get("waypoint_id"))
            x = _to_float_nullable(row.get("x"))
            y = _to_float_nullable(row.get("y"))
            z = _to_float_nullable(row.get("z"))

            if drone_id is None:
                raise ValueError(f"{csv_path}:{line_no}: niepoprawne drone_id={row.get('drone_id')!r}")
            if waypoint_id is None:
                raise ValueError(f"{csv_path}:{line_no}: niepoprawne waypoint_id={row.get('waypoint_id')!r}")
            if x is None or y is None or z is None:
                raise ValueError(
                    f"{csv_path}:{line_no}: niepoprawne współrzędne "
                    f"x={row.get('x')!r}, y={row.get('y')!r}, z={row.get('z')!r}"
                )

            rows.append((run_id, drone_id, waypoint_id, x, y, z))

    conn.executemany(
        """
        INSERT OR REPLACE INTO counted_trajectory_points (
            run_id, drone_id, waypoint_id, x, y, z
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _load_trajectories(conn: sqlite3.Connection, run_id: str, csv_path: Path) -> None:
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: brak nagłówków CSV")

        required = {"time", "drone_id", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path}: brak wymaganych kolumn: {sorted(missing)}; "
                f"dostępne: {reader.fieldnames}"
            )

        rows = []
        for line_no, row in enumerate(reader, start=2):
            sim_time = _to_float_nullable(row.get("time"))
            drone_id = _to_int_nullable(row.get("drone_id"))
            x = _to_float_nullable(row.get("x"))
            y = _to_float_nullable(row.get("y"))
            z = _to_float_nullable(row.get("z"))
            roll = _to_float_nullable(row.get("roll"))
            pitch = _to_float_nullable(row.get("pitch"))
            yaw = _to_float_nullable(row.get("yaw"))
            vx = _to_float_nullable(row.get("vx"))
            vy = _to_float_nullable(row.get("vy"))
            vz = _to_float_nullable(row.get("vz"))

            if sim_time is None:
                raise ValueError(f"{csv_path}:{line_no}: niepoprawne time={row.get('time')!r}")
            if drone_id is None:
                raise ValueError(f"{csv_path}:{line_no}: niepoprawne drone_id={row.get('drone_id')!r}")
            if x is None or y is None or z is None:
                raise ValueError(
                    f"{csv_path}:{line_no}: niepoprawne współrzędne "
                    f"x={row.get('x')!r}, y={row.get('y')!r}, z={row.get('z')!r}"
                )

            rows.append(
                (
                    run_id,
                    sim_time,
                    drone_id,
                    x,
                    y,
                    z,
                    roll,
                    pitch,
                    yaw,
                    vx,
                    vy,
                    vz,
                )
            )

    conn.executemany(
        """
        INSERT OR REPLACE INTO trajectory_samples (
            run_id, sim_time, drone_id, x, y, z, roll, pitch, yaw, vx, vy, vz
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _to_int(value):
    return None if value in (None, "") else int(value)


def _to_float(value):
    return None if value in (None, "") else float(value)

def _to_bool(value):
    if value is None:
        return None

    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    return None

def _to_int_bool(value):
    if value is None:
        return None

    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return 1
    if value in {"false", "0", "no", "n"}:
        return 0
    return None

def _to_str_nullable(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() == "nan":
        return None
    return value

def _to_int_nullable(value):
    value = _to_str_nullable(value)
    return None if value is None else int(value)

def _to_float_nullable(value):
    value = _to_str_nullable(value)
    return None if value is None else float(value)

def _to_int_bool_nullable(value):
    value = _to_str_nullable(value)
    if value is None:
        return None
    if value.lower() in {"true", "1", "yes", "y"}:
        return 1
    if value.lower() in {"false", "0", "no", "n"}:
        return 0
    return None

def _to_int_nullable(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() == "nan":
        return None
    return int(value)

def _to_float_nullable(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() == "nan":
        return None
    return float(value)