# src/analysis/db/populate_online_metrics.py
import csv
import sqlite3
from pathlib import Path

from src.analysis.db.utils import _to_float_nullable, _to_int_nullable, _to_str_nullable


def populate_online_metrics(conn: sqlite3.Connection, run_id: str, run_dir: Path) -> None:
    """
    Główna funkcja ładująca logi z planowania trajektorii online (unikanie przeszkód).
    Odpowiada za zasilenie tabel: online_optimization_tasks oraz online_convergence_traces.
    """
    online_opt_csv = run_dir / "online_optimization.csv"
    convergence_csv = run_dir / "convergence_traces.csv"

    _load_online_optimization_tasks(conn, run_id, online_opt_csv)
    _load_online_convergence_traces(conn, run_id, convergence_csv)


def _load_online_optimization_tasks(conn: sqlite3.Connection, run_id: str, csv_path: Path) -> None:
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: brak nagłówków CSV")

        required = {
            "drone_id", "trigger_time", "algorithm", "status", "reason",
            "best_fitness", "evaluations_completed", "generations_completed",
            "wallclock_s", "time_budget_s", "chosen_axis", "plan_waypoints_json",
            "plan_total_duration_s", "plan_arc_length_m", "outcome",
            "pos_err_at_rejoin_m", "vel_err_at_rejoin_mps", "time_to_rejoin_s"
        }
        
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path}: brak wymaganych kolumn: {sorted(missing)}; "
                f"dostępne: {reader.fieldnames}"
            )

        rows = []
        for line_no, row in enumerate(reader, start=2):
            drone_id = _to_int_nullable(row.get("drone_id"))
            trigger_time = _to_float_nullable(row.get("trigger_time"))
            algorithm = _to_str_nullable(row.get("algorithm"))
            status = _to_str_nullable(row.get("status"))
            wallclock_s = _to_float_nullable(row.get("wallclock_s"))
            time_budget_s = _to_float_nullable(row.get("time_budget_s"))

            # Podstawowa walidacja kluczowych pól
            if drone_id is None:
                raise ValueError(f"{csv_path}:{line_no}: niepoprawne drone_id={row.get('drone_id')!r}")
            if trigger_time is None:
                raise ValueError(f"{csv_path}:{line_no}: niepoprawne trigger_time={row.get('trigger_time')!r}")
            if algorithm is None:
                raise ValueError(f"{csv_path}:{line_no}: brak nazwy algorytmu")

            # `chosen_axis`: notacja kierunkowa z AxisChooser
            # ('right'/'left'/'up'/'down'). Magic string "unknown" lub puste —
            # normalizujemy na NULL żeby spełnić CHECK constraint
            # (right|left|up|down|NULL).
            raw_axis = _to_str_nullable(row.get("chosen_axis"))
            if raw_axis not in ("right", "left", "up", "down"):
                raw_axis = None

            rows.append((
                run_id,
                drone_id,
                trigger_time,
                algorithm,
                status,
                _to_str_nullable(row.get("reason")),
                _to_float_nullable(row.get("best_fitness")),
                _to_int_nullable(row.get("evaluations_completed")),
                _to_int_nullable(row.get("generations_completed")),
                wallclock_s,
                time_budget_s,
                raw_axis,
                _to_str_nullable(row.get("plan_waypoints_json")),
                _to_float_nullable(row.get("plan_total_duration_s")),
                _to_float_nullable(row.get("plan_arc_length_m")),
                _to_str_nullable(row.get("outcome")),
                _to_float_nullable(row.get("pos_err_at_rejoin_m")),
                _to_float_nullable(row.get("vel_err_at_rejoin_mps")),
                _to_float_nullable(row.get("time_to_rejoin_s"))
            ))

    conn.executemany(
        """
        INSERT OR REPLACE INTO online_optimization_tasks (
            run_id, drone_id, trigger_time, algorithm, status, reason, 
            best_fitness, evaluations_completed, generations_completed, 
            wallclock_s, time_budget_s, chosen_axis, plan_waypoints_json, 
            plan_total_duration_s, plan_arc_length_m, outcome, 
            pos_err_at_rejoin_m, vel_err_at_rejoin_mps, time_to_rejoin_s
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _load_online_convergence_traces(conn: sqlite3.Connection, run_id: str, csv_path: Path) -> None:
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: brak nagłówków CSV")

        required = {"drone_id", "trigger_time", "algorithm", "generation", "best_fitness"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{csv_path}: brak wymaganych kolumn: {sorted(missing)}")

        rows = []
        for line_no, row in enumerate(reader, start=2):
            drone_id = _to_int_nullable(row.get("drone_id"))
            trigger_time = _to_float_nullable(row.get("trigger_time"))
            algorithm = _to_str_nullable(row.get("algorithm"))
            generation = _to_int_nullable(row.get("generation"))
            best_fitness = _to_float_nullable(row.get("best_fitness"))

            if None in (drone_id, trigger_time, algorithm, generation, best_fitness):
                raise ValueError(f"{csv_path}:{line_no}: brakuje jednego z wymaganych kluczy głównych lub miar")

            rows.append((
                run_id,
                drone_id,
                trigger_time,
                algorithm,
                generation,
                best_fitness
            ))

    # Optymalizacja dla potencjalnie bardzo dużej liczby rekordów śledzenia zbieżności
    batch_size = 5000
    for i in range(0, len(rows), batch_size):
        conn.executemany(
            """
            INSERT OR REPLACE INTO online_convergence_traces (
                run_id, drone_id, trigger_time, algorithm, generation, best_fitness
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows[i:i + batch_size],
        )
