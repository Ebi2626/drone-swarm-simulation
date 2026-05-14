"""Populator bazy `analysis.db` — łączy wszystkie loadery i populatory metryk.

Per run wykonuje 4 fazy: surowe pliki źródłowe (CSV/h5), tabele pochodne
(trajectory/uav/iteration/online_safety), agregat `run_metrics` i F-vector
z `optimization_history.h5`. Status każdego runa zapisywany jest w
`runs.aggregation_status` — pojedynczy błąd nie blokuje pozostałych runów.
"""
import logging
from pathlib import Path
import sqlite3
import csv

from src.analysis.db.populate_trajectory_metrics import populate_trajectory_metrics
from src.analysis.db.populate_uav_metrics import populate_uav_metrics
from src.analysis.db.populate_run_metrics import populate_run_metrics
from src.analysis.db.populate_iteration_metrics import populate_iteration_metrics
from src.analysis.db.populate_online_metrics import populate_online_metrics
from src.analysis.db.populate_online_safety_metrics import populate_online_safety_metrics
from src.analysis.db.populate_offline_objectives import populate_offline_objectives
from src.analysis.db.populate_final_objective_aggregated import (
    populate_final_objective_aggregated,
)
from src.analysis.db.populate_rejoin_quality import populate_rejoin_quality
from src.analysis.db.populate_moo_quality import populate_moo_quality

from src.analysis.db.utils import _to_float, _to_float_nullable, _to_int, _to_int_bool, _to_int_bool_nullable, _to_int_nullable, _to_str_nullable, list_run_directories, parse_run_dir_name
from src.utils.per_gen_metrics import FEASIBILITY_EPS


logger = logging.getLogger(__name__)


def populate_database(experiment_dir: str | Path) -> Path:
    """Załaduj wszystkie runy z `experiment_dir` do `analysis.db`.

    Każdy run jest najpierw oznaczany `aggregation_status='discovered'`,
    następnie przechodzi pełen pipeline (raw load → derived tables →
    aggregates → F-vector). Wynik kończy się statusem `'aggregated'` albo
    `'failed'` z `aggregation_error`. Pojedynczy fail nie przerywa pętli.

    Args:
        experiment_dir: Katalog eksperymentu zawierający per-run podkatalogi.

    Returns:
        Ścieżkę do `analysis.db`.
    """
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    db_path = experiment_dir / "analysis.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row

        for run_dir in list_run_directories(experiment_dir):
            run_meta = parse_run_dir_name(run_dir.name)
            run_id = run_dir.name

            # Wstawiamy/aktualizujemy wpis w `runs` z status='discovered'
            # ZANIM zaczniemy ładować dane. Jeśli któryś z populatorów rzuci
            # wyjątkiem, przerwiemy i zapiszemy aggregation_status='failed'
            # z aggregation_error w except poniżej.
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, run_dir_name, source_path, optimizer_algo,
                    avoidance_algo, environment, seed, algorithm_pair,
                    aggregation_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'discovered')
                ON CONFLICT(run_id) DO UPDATE SET
                    run_dir_name = excluded.run_dir_name,
                    source_path = excluded.source_path,
                    optimizer_algo = excluded.optimizer_algo,
                    avoidance_algo = excluded.avoidance_algo,
                    environment = excluded.environment,
                    seed = excluded.seed,
                    algorithm_pair = excluded.algorithm_pair,
                    aggregation_status = 'discovered',
                    aggregation_error = NULL
                """,
                (
                    run_id, run_dir.name, str(run_dir),
                    run_meta["optimizer"], run_meta["avoidance"],
                    run_meta["environment"], run_meta["seed"],
                    run_meta["algorithm_pair"],
                ),
            )

            try:
                _register_run_files(conn, run_id, run_dir)

                # 1. Dane źródłowe
                _load_optimization_timings(conn, run_id, run_dir / "optimization_timings.csv")
                _load_collisions(conn, run_id, run_dir / "collisions.csv")
                _load_evasion_events(conn, run_id, run_dir / "evasion_events.csv")
                _load_world_boundaries(conn, run_id, run_dir / "world_boundaries.csv")
                _load_generated_obstacles(conn, run_id, run_dir / "generated_obstacles.csv")
                _load_counted_trajectories(conn, run_id, run_dir / "counted_trajectories.csv")
                _load_trajectories(conn, run_id, run_dir / "trajectories.csv")
                _load_optimization_history(conn, run_id, run_dir / "optimization_history/optimization_history.h5")
                populate_online_metrics(conn, run_id, run_dir)

                populate_moo_quality(
                    conn, run_id,
                    run_dir / "optimization_history" / "optimization_history.h5",
                    reference_set=None,
                )

                # 2. Tabele pochodne
                populate_trajectory_metrics(conn, run_id)
                populate_iteration_metrics(conn, run_id)
                populate_uav_metrics(conn, run_id)
                populate_online_safety_metrics(conn, run_id)

                # 3. Agregat końcowy
                populate_run_metrics(conn, run_id)

                # 4. F-vector z h5 — UPDATE per-obj kolumn F[0..4] +
                # final_objective_weights_json. `final_objective` SAMOSTOI
                # zostaje NULL aż do post-pass `populate_final_objective_aggregated`
                # (potrzebuje per-env median F_ref).
                populate_offline_objectives(
                    conn, run_id,
                    run_dir / "optimization_history" / "optimization_history.h5",
                    run_dir=run_dir,
                )

                # Sukces — oznacz jako 'aggregated' z aggregated_at.
                conn.execute(
                    """UPDATE runs SET
                        aggregation_status = 'aggregated',
                        aggregated_at = CURRENT_TIMESTAMP,
                        aggregation_error = NULL
                       WHERE run_id = ?""",
                    (run_id,),
                )
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                logger.error(
                    f"populate_database: błąd agregacji run_id={run_id!r}: {err_msg}",
                    exc_info=True,
                )
                conn.execute(
                    """UPDATE runs SET
                        aggregation_status = 'failed',
                        aggregation_error = ?
                       WHERE run_id = ?""",
                    (err_msg[:1000], run_id),  # truncate na 1KB żeby nie spuchnąć DB
                )
                # Kontynuujemy z kolejnymi runami — pojedynczy fail nie blokuje
                # całej agregacji (status w runs pozwala później zidentyfikować
                # i ewentualnie ponownie uruchomić).

        # 5. Post-pass agregacji cross-run: per-environment median F_ref +
        # weighted normalized sum → `run_metrics.final_objective` (2026-05-13).
        # Musi być POZA pętlą per-run, bo potrzebuje wszystkich F_best
        # do policzenia median per-env.
        try:
            populate_final_objective_aggregated(conn)
        except Exception as e:
            # Nie blokujemy całego pipeline'u jeśli post-pass padnie —
            # per-obj kolumny i weights_json są już zapisane, brakuje
            # tylko skalarnego agregatu.
            logger.error(
                "populate_database: błąd post-pass final_objective_aggregated: "
                "%s: %s — `final_objective` może być NULL.",
                type(e).__name__, e, exc_info=True,
            )

        # 6. Post-pass agregacji cross-run dla §3.1.3.3 — TOPSIS composite
        # `rejoin_quality` (Skuteczność powrotu, Hwang & Yoon 1981).
        # Wymaga per-env median pos_err/vel_err/time_to_rejoin → cross-run.
        try:
            populate_rejoin_quality(conn)
        except Exception as e:
            logger.error(
                "populate_database: błąd post-pass rejoin_quality: "
                "%s: %s — `rejoin_quality` może być NULL.",
                type(e).__name__, e, exc_info=True,
            )

        conn.commit()

    return db_path


def _count_csv_rows(path: Path) -> int | None:
    """Zlicza wiersze danych w CSV (n_lines − 1 dla nagłówka). Zwraca None
    gdy plik nie istnieje albo jest pusty."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)  # minus header
    except Exception:
        return None


def _register_run_files(conn: sqlite3.Connection, run_id: str, run_dir: Path) -> None:
    """Rejestruje pliki źródłowe runu w `run_files`.

    Kolumny:
    - `relative_path`, `file_format`, `exists_flag`, `size_bytes` —
      podstawowe metadata systemu plików
    - `row_count` — liczba wierszy danych dla CSV (NULL dla h5/log)
    - `modified_at` — ISO 8601 timestamp z `stat().st_mtime`
    """
    from datetime import datetime, timezone

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
        "online_optimization_csv": "online_optimization.csv",
        "convergence_traces_csv": "convergence_traces.csv",
    }

    for role, relative_path in file_map.items():
        full_path = run_dir / relative_path
        exists = full_path.exists()
        file_ext = full_path.suffix.lstrip(".") or None

        size_bytes = full_path.stat().st_size if exists else None
        modified_at = (
            datetime.fromtimestamp(full_path.stat().st_mtime, tz=timezone.utc)
            .isoformat(timespec="seconds")
            if exists else None
        )
        # row_count tylko dla CSV (h5 ma własną granularność datasetów,
        # log ma niespecyficzną semantykę linii).
        row_count = _count_csv_rows(full_path) if file_ext == "csv" else None

        conn.execute(
            """
            INSERT OR REPLACE INTO run_files (
                run_id, file_role, relative_path, file_format, exists_flag,
                size_bytes, row_count, modified_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, role, relative_path, file_ext, int(exists),
                size_bytes, row_count, modified_at,
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
    """Ładuje `evasion_events.csv` do tabeli `evasion_events`.

    Kolumny:
    - `ttc_source` ('oracle_discrete' | 'continuous') — jednoznacznie określa
      źródło `ttc` (oracle dyskretyzowane z lookahead vs naive `dist/closing_speed`).
    - `preferred_axis` ('right'|'left'|'up'|'down'|NULL) — notacja kierunkowa
      zgodna z `AxisChooser`. NULL gdy oś nie określona (avoidance nie wstawił
      `axis_chosen` do extra dict).

    Backward-compat: legacy CSV z dodatkowymi kolumnami są ignorowane;
    brak `ttc_source` / `preferred_axis` w starych CSV → NULL w DB.
    """
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
            "fallback_used",
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

            # `preferred_axis`: nowsze CSV mają jako osobną kolumnę. Starsze
            # — wyciągamy z `notes` (`"axis=right"` pattern). "unknown" → NULL.
            # Notacja kierunkowa: 'right'/'left'/'up'/'down' zgodnie z
            # `AxisChooser._choose`.
            preferred_axis = _to_str_nullable(row.get("preferred_axis"))
            notes_str = _to_str_nullable(row.get("notes"))
            valid_axes = ("right", "left", "up", "down")
            if not preferred_axis and notes_str and notes_str.startswith("axis="):
                axis_candidate = notes_str[len("axis="):].strip()
                if axis_candidate in valid_axes:
                    preferred_axis = axis_candidate
                # "unknown" / inne → preferred_axis = None
            if preferred_axis not in valid_axes:
                preferred_axis = None
            # Jeśli notes zawierało wyłącznie "axis=..." i wyekstraktowaliśmy
            # axis (lub było "axis=unknown"), wyczyść notes — nie ma już
            # informacyjnej wartości.
            if notes_str and notes_str.startswith("axis="):
                notes_str = None

            ttc_source = _to_str_nullable(row.get("ttc_source"))
            if ttc_source not in ("oracle_discrete", "continuous"):
                ttc_source = None

            rows.append(
                (
                    run_id,
                    line_no - 2,
                    sim_time,
                    drone_id,
                    event_type,
                    mode,
                    _to_float_nullable(row.get("ttc")),
                    ttc_source,
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
                    preferred_axis,
                    _to_int_bool_nullable(row.get("fallback_used")),
                    _to_float_nullable(row.get("pos_error_at_rejoin")),
                    _to_float_nullable(row.get("vel_error_at_rejoin")),
                    _to_float_nullable(row.get("planning_wall_time_s")),
                    notes_str,
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
            ttc_source,
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
            preferred_axis,
            fallback_used,
            pos_error_at_rejoin,
            vel_error_at_rejoin,
            planning_wall_time_s,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    """Ładuje `generated_obstacles.csv` do tabeli `generated_obstacles`.

    Semantyka per shape_type:
    - Cylinder (forest): zapisujemy `shape_type='cylinder'`, `radius`,
      `height`; `length=NULL, width=NULL`.
    - Box (urban): zapisujemy `shape_type='box'`, `length`, `width`,
      `height`; `radius=NULL`.

    Nie ma już kolumny `unused_dim`. Backward-compat dla starych CSV
    cylindra (z `unused_dim` jako 6. kolumną) — populator po prostu jej
    nie czyta.
    """
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: brak nagłówków CSV")

        fieldnames = set(reader.fieldnames)

        cylinder_required = {"x", "y", "z", "radius", "height"}
        box_required = {"x", "y", "z", "length", "width", "height"}

        if cylinder_required.issubset(fieldnames):
            obstacle_mode = "cylinder"
        elif box_required.issubset(fieldnames):
            obstacle_mode = "box"
        else:
            raise ValueError(
                f"{csv_path}: nieobsługiwany format przeszkód. "
                f"Oczekiwano kolumn cylindra {sorted(cylinder_required)} "
                f"lub boxa {sorted(box_required)}; dostępne: {reader.fieldnames}"
            )

        rows = []
        for line_no, row in enumerate(reader, start=2):
            x = _to_float_nullable(row.get("x"))
            y = _to_float_nullable(row.get("y"))
            z = _to_float_nullable(row.get("z"))

            if x is None or y is None or z is None:
                raise ValueError(
                    f"{csv_path}:{line_no}: niepoprawne współrzędne "
                    f"x={row.get('x')!r}, y={row.get('y')!r}, z={row.get('z')!r}"
                )

            if obstacle_mode == "cylinder":
                radius = _to_float_nullable(row.get("radius"))
                height = _to_float_nullable(row.get("height"))

                if radius is None or height is None:
                    raise ValueError(
                        f"{csv_path}:{line_no}: niepoprawne parametry cylindra "
                        f"radius={row.get('radius')!r}, height={row.get('height')!r}"
                    )

                rows.append(
                    (
                        run_id,
                        line_no - 2,
                        x, y, z,
                        "cylinder",
                        radius,
                        None,    # length
                        None,    # width
                        height,
                    )
                )

            else:  # obstacle_mode == "box"
                length = _to_float_nullable(row.get("length"))
                width = _to_float_nullable(row.get("width"))
                height = _to_float_nullable(row.get("height"))

                if length is None or width is None or height is None:
                    raise ValueError(
                        f"{csv_path}:{line_no}: niepoprawne parametry prostopadłościanu "
                        f"length={row.get('length')!r}, width={row.get('width')!r}, "
                        f"height={row.get('height')!r}"
                    )

                rows.append(
                    (
                        run_id,
                        line_no - 2,
                        x, y, z,
                        "box",
                        None,    # radius
                        length,
                        width,
                        height,
                    )
                )

    conn.execute(
        "DELETE FROM generated_obstacles WHERE run_id = ?",
        (run_id,),
    )

    if rows:
        conn.executemany(
            """
            INSERT INTO generated_obstacles (
                run_id, obstacle_index,
                x, y, z,
                shape_type,
                radius, length, width, height
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        sample_index_by_drone = {}

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

            sample_index = sample_index_by_drone.get(drone_id, 0)
            sample_index_by_drone[drone_id] = sample_index + 1

            rows.append(
                (
                    run_id,
                    sample_index,
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
            run_id, sample_index, sim_time, drone_id, x, y, z, roll, pitch, yaw, vx, vy, vz
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _load_optimization_history(conn: sqlite3.Connection, run_id: str, h5_path: Path) -> None:
    if not h5_path.exists():
        return

    import h5py
    import numpy as np

    def _safe_array(ds, gen):
        if ds is None:
            return None
        arr = np.asarray(ds[gen])
        return arr

    def _compute_diversity_metric(decisions: np.ndarray) -> float | None:
        if decisions is None:
            return None
        arr = np.asarray(decisions, dtype=np.float64)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[0] <= 1:
            return 0.0
        return float(np.mean(np.std(arr, axis=0)))

    def _total_cv(cv: np.ndarray | None) -> np.ndarray | None:
        if cv is None:
            return None
        arr = np.asarray(cv, dtype=np.float64)
        if arr.ndim == 1:
            return np.maximum(arr, 0.0)
        return np.sum(np.maximum(arr, 0.0), axis=1)

    def _is_nondominated(F: np.ndarray) -> np.ndarray:
        F = np.asarray(F, dtype=np.float64)
        n = F.shape[0]
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
            if np.any(dominates_i):
                nd[i] = False
                continue
            dominated_by_i = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
            nd[dominated_by_i] = False
            nd[i] = True
        return nd

    def _compute_hv(F: np.ndarray, ref_point: np.ndarray | None) -> float | None:
        if ref_point is None:
            return None
        try:
            from pymoo.indicators.hv import HV
        except Exception:
            return None
        arr = np.asarray(F, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        try:
            return float(HV(ref_point=np.asarray(ref_point, dtype=np.float64))(arr))
        except Exception:
            return None

    with h5py.File(h5_path, "r") as f:
        if "objectives_matrix" not in f:
            return

        objectives_ds = f["objectives_matrix"]
        decisions_ds = f["decisions_matrix"] if "decisions_matrix" in f else None
        feasible_mask_ds = f["feasible_mask"] if "feasible_mask" in f else None
        cv_ds = None
        for name in ("constraint_violation", "CV", "constraint_violations", "last_cv"):
            if name in f:
                cv_ds = f[name]
                break

        elapsed_ds = f["elapsed_s"] if "elapsed_s" in f else None
        eval_ds = f["eval_count_cumulative"] if "eval_count_cumulative" in f else f.get("n_eval")
        rank_ds = f["nd_rank"] if "nd_rank" in f else f.get("rank")

        ref_point = None
        if "reference_point" in f:
            ref_point = np.asarray(f["reference_point"], dtype=np.float64)
        elif "reference_point" in f.attrs:
            ref_point = np.asarray(f.attrs["reference_point"], dtype=np.float64)

        generation_count = objectives_ds.shape[0]
        rows = []
        prev_best = None

        for gen in range(generation_count):
            obj = np.asarray(objectives_ds[gen], dtype=np.float64)
            if obj.ndim == 1:
                obj = obj[:, np.newaxis]

            population_size = obj.shape[0]
            objective_count = obj.shape[1]

            rows.append((run_id, gen, "objectives_matrix", "population_size", float(population_size)))
            rows.append((run_id, gen, "objectives_matrix", "objective_count", float(objective_count)))

            for j in range(objective_count):
                col = obj[:, j]
                rows.append((run_id, gen, "objectives_matrix", f"objective_{j}_min", float(np.min(col))))
                rows.append((run_id, gen, "objectives_matrix", f"objective_{j}_mean", float(np.mean(col))))
                rows.append((run_id, gen, "objectives_matrix", f"objective_{j}_std", float(np.std(col))))
                rows.append((run_id, gen, "objectives_matrix", f"objective_{j}_max", float(np.max(col))))
                rows.append((run_id, gen, "objectives_matrix", f"objective_{j}_median", float(np.median(col))))

            best_now = float(np.min(obj[:, 0]))
            best_so_far = best_now if prev_best is None else min(prev_best, best_now)
            improvement = 0.0 if prev_best is None else (prev_best - best_now)

            rows.append((run_id, gen, "objectives_matrix", "best_so_far_obj0", best_so_far))
            rows.append((run_id, gen, "objectives_matrix", "improvement_vs_prev_obj0", improvement))
            prev_best = best_so_far

            decisions = _safe_array(decisions_ds, gen)
            if decisions is not None:
                diversity = _compute_diversity_metric(decisions)
                if diversity is not None:
                    rows.append((run_id, gen, "decisions_matrix", "diversity_metric", diversity))

            feasible_mask = _safe_array(feasible_mask_ds, gen)
            if feasible_mask is not None:
                feasible_mask = np.asarray(feasible_mask).reshape(-1).astype(bool)

            cv = _safe_array(cv_ds, gen)
            cv_total = _total_cv(cv)

            if feasible_mask is None and cv_total is not None:
                # Spójna tolerancja z `per_gen_metrics.FEASIBILITY_EPS` —
                # `≤ 0.0` traktowałoby CV=1e-12 (floating-point szum) jako
                # infeasible, mimo że `per_gen_metrics_from_FG` ją zaliczyła.
                feasible_mask = cv_total <= FEASIBILITY_EPS

            if feasible_mask is not None:
                feasible_solutions = int(np.count_nonzero(feasible_mask))
                feasible_ratio = float(feasible_solutions / population_size) if population_size > 0 else None
                rows.append((run_id, gen, "constraints", "feasible_solutions", float(feasible_solutions)))
                if feasible_ratio is not None:
                    rows.append((run_id, gen, "constraints", "feasible_ratio", feasible_ratio))

            if cv_total is not None and cv_total.size > 0:
                rows.append((run_id, gen, "constraints", "constraint_violation_min", float(np.min(cv_total))))
                rows.append((run_id, gen, "constraints", "constraint_violation_mean", float(np.mean(cv_total))))
                rows.append((run_id, gen, "constraints", "constraint_violation_max", float(np.max(cv_total))))

            elapsed = _safe_array(elapsed_ds, gen)
            if elapsed is not None:
                elapsed_val = float(np.asarray(elapsed).reshape(-1)[0])
                rows.append((run_id, gen, "runtime", "elapsed_s", elapsed_val))

            eval_count = _safe_array(eval_ds, gen)
            if eval_count is not None:
                eval_val = int(np.asarray(eval_count).reshape(-1)[0])
                rows.append((run_id, gen, "runtime", "eval_count_cumulative", float(eval_val)))

            nd_rank = _safe_array(rank_ds, gen)
            if nd_rank is not None:
                nd_rank = np.asarray(nd_rank).reshape(-1)
                nd_count = int(np.count_nonzero(nd_rank == 0))
            elif objective_count > 1:
                nd_mask = _is_nondominated(obj)
                if feasible_mask is not None:
                    nd_mask = nd_mask & feasible_mask
                nd_count = int(np.count_nonzero(nd_mask))
            else:
                nd_count = None

            if nd_count is not None:
                rows.append((run_id, gen, "pareto", "nondominated_solutions", float(nd_count)))
                if population_size > 0:
                    rows.append((run_id, gen, "pareto", "nondominated_ratio", float(nd_count / population_size)))

            if objective_count > 1:
                hv = _compute_hv(obj if feasible_mask is None else obj[feasible_mask], ref_point)
                if hv is not None:
                    rows.append((run_id, gen, "pareto", "hypervolume", hv))

            # IGD+ tylko jeśli masz reference set zapisany jawnie.
            # Przykład:
            # if "reference_set" in f:
            #     igd_plus = _compute_igd_plus(obj, np.asarray(f["reference_set"]))
            #     rows.append((run_id, gen, "pareto", "igd_plus", float(igd_plus)))

            if len(rows) >= 5000:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO optimization_generation_stats
                    (run_id, generation, source_name, metric_name, metric_value)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                rows.clear()

        if rows:
            conn.executemany(
                """
                INSERT OR REPLACE INTO optimization_generation_stats
                (run_id, generation, source_name, metric_name, metric_value)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
