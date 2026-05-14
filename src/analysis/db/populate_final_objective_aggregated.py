"""Post-pass: skalarny `run_metrics.final_objective` jako weighted normalized sum.

Wywoływany RAZ po pełnym przejściu `populate_offline_objectives` dla
wszystkich runów. Liczy per-environment median F_best[i] (zapis do
`offline_objective_normalization`) i UPDATE'uje `run_metrics.final_objective`
dla każdego runu jako:

    final_objective = Σ_{i=1..5} w_i · F_best[i] / F_ref_env[i]

gdzie:
- `w_i` z `run_metrics.final_objective_weights_json` (zapisane przez
  `populate_offline_objectives`).
- `F_best[i]` z per-obj kolumn `run_metrics` (`final_objective_f1_trajectory`,
  `final_objective_f2_height_angle`, `total_threat_cost`, `total_turn_penalty`,
  `total_coordination_cost`).
- `F_ref_env[i]` = median F_best[i] across all runs w danym environment.

Reference: Hwang & Yoon (1981) Multiple Attribute Decision Making §4.2
(weighted-sum scalarization). Median-based F_ref to pragmatic approximation
straight-line F_ref z `soo_adapter._compute_reference_scales`; zachowuje
rankings within-env-seed (Friedman/Nemenyi p-values identyczne).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


# Indeks F → kolumna run_metrics zawierająca F_best[i]. Definitive source
# (mirroruje `_F_TO_COLUMN_5OBJ` z populate_offline_objectives).
_F_COLUMNS: tuple[str, ...] = (
    "final_objective_f1_trajectory",
    "final_objective_f2_height_angle",
    "total_threat_cost",
    "total_turn_penalty",
    "total_coordination_cost",
)

# Próg minimum dla F_ref_env — gdy median jest ~0 (np. f3=threat w
# czystym korytarzu, wszystkie runy mają 0), używamy 1.0 jako neutralny
# mianownik. Identyczna logika jak w `soo_adapter._compute_reference_scales`
# (zero-component guard).
_F_REF_FLOOR: float = 1e-9


def populate_final_objective_aggregated(
    conn: sqlite3.Connection,
) -> None:
    """Post-pass UPDATE'ujący `run_metrics.final_objective` we wszystkich runach.

    Idempotentne — re-run regeneruje F_ref median i nadpisuje wartości.
    Nie wymaga argumentu `run_id` — operuje cross-run (per environment).

    Args:
        conn: Aktywne połączenie do bazy.

    Efekty uboczne:
        - DELETE FROM `offline_objective_normalization` + INSERT świeżych
          median F_ref_env.
        - UPDATE `run_metrics.final_objective` per row.
    """
    f_ref_per_env = _compute_f_ref_per_environment(conn)
    if not f_ref_per_env:
        logger.warning(
            "populate_final_objective_aggregated: brak runów z F_best — "
            "final_objective pozostanie NULL we wszystkich rekordach."
        )
        return

    _persist_normalization(conn, f_ref_per_env)
    _update_final_objective(conn, f_ref_per_env)


def _compute_f_ref_per_environment(
    conn: sqlite3.Connection,
) -> dict[str, np.ndarray]:
    """Zbierz F_best[i] per environment i policz median per (env, f_idx).

    Pomija runy z którymikolwiek per-obj kolumną NULL (niekompletny F_best).
    Zwracane słownik: env → `np.ndarray (5,)` median.
    """
    select_cols = ", ".join(_F_COLUMNS)
    query = f"""
        SELECT r.environment, {select_cols}
        FROM runs r
        JOIN run_metrics m ON m.run_id = r.run_id
        WHERE
            {" AND ".join(f"m.{c} IS NOT NULL" for c in _F_COLUMNS)}
    """
    by_env: dict[str, list[list[float]]] = {}
    for row in conn.execute(query):
        env = row[0]
        f_vec = [float(v) for v in row[1:]]
        by_env.setdefault(env, []).append(f_vec)

    f_ref_per_env: dict[str, np.ndarray] = {}
    for env, fvecs in by_env.items():
        arr = np.asarray(fvecs, dtype=np.float64)
        med = np.median(arr, axis=0)
        # Zero-component guard: F_ref ≤ 1e-9 → użyj 1.0 (neutralny mianownik).
        # Bez tego F[i]/0 = inf, co skażało by skalar.
        med = np.where(med <= _F_REF_FLOOR, 1.0, med)
        f_ref_per_env[env] = med
        logger.info(
            "populate_final_objective_aggregated: env=%r n_runs=%d "
            "F_ref_median=%s",
            env, arr.shape[0], med.tolist(),
        )
    return f_ref_per_env


def _persist_normalization(
    conn: sqlite3.Connection,
    f_ref_per_env: dict[str, np.ndarray],
) -> None:
    """Zapisz F_ref median per (env, f_idx) do `offline_objective_normalization`.

    Idempotentne — DELETE + INSERT (nie UPSERT, bo full refresh per run
    ETL pipeline'u; n_runs może się zmienić między runami i chcemy
    deterministycznych nadpisań).
    """
    n_runs_per_env = {
        env: int(conn.execute(
            f"""
            SELECT COUNT(*) FROM runs r
            JOIN run_metrics m ON m.run_id = r.run_id
            WHERE r.environment = ? AND {" AND ".join(f"m.{c} IS NOT NULL" for c in _F_COLUMNS)}
            """,
            (env,),
        ).fetchone()[0])
        for env in f_ref_per_env
    }

    conn.execute("DELETE FROM offline_objective_normalization")
    rows = []
    for env, med in f_ref_per_env.items():
        for f_idx, val in enumerate(med):
            rows.append((env, f_idx, float(val), n_runs_per_env[env]))
    conn.executemany(
        """
        INSERT INTO offline_objective_normalization
            (environment, f_idx, f_ref_median, n_runs)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )


def _update_final_objective(
    conn: sqlite3.Connection,
    f_ref_per_env: dict[str, np.ndarray],
) -> None:
    """UPDATE `run_metrics.final_objective` per row, używając wag + F_ref_env."""
    select_cols = ", ".join(_F_COLUMNS)
    query = f"""
        SELECT r.run_id, r.environment, m.final_objective_weights_json,
               {select_cols}
        FROM runs r
        JOIN run_metrics m ON m.run_id = r.run_id
        WHERE m.final_objective_weights_json IS NOT NULL
          AND {" AND ".join(f"m.{c} IS NOT NULL" for c in _F_COLUMNS)}
    """
    updates: list[tuple[float, str]] = []
    skipped_no_ref = 0
    skipped_bad_weights = 0

    for row in conn.execute(query):
        run_id = row[0]
        env = row[1]
        weights_json = row[2]
        f_best = np.asarray([float(v) for v in row[3:]], dtype=np.float64)

        try:
            weights = np.asarray(json.loads(weights_json), dtype=np.float64)
        except (ValueError, TypeError) as e:
            logger.warning(
                "populate_final_objective_aggregated: niepoprawny "
                "weights_json=%r dla run_id=%r (%s) — pomijam.",
                weights_json, run_id, e,
            )
            skipped_bad_weights += 1
            continue

        if weights.shape != (5,):
            logger.warning(
                "populate_final_objective_aggregated: weights.shape=%s "
                "dla run_id=%r — oczekiwano (5,). Pomijam.",
                weights.shape, run_id,
            )
            skipped_bad_weights += 1
            continue

        f_ref = f_ref_per_env.get(env)
        if f_ref is None:
            skipped_no_ref += 1
            continue

        # final_objective = Σ w_i · F[i] / F_ref_env[i]
        # Vectorized: weights · (f_best / f_ref)
        scalar = float(np.dot(weights, f_best / f_ref))
        updates.append((scalar, run_id))

    if updates:
        conn.executemany(
            "UPDATE run_metrics SET final_objective = ? WHERE run_id = ?",
            updates,
        )

    logger.info(
        "populate_final_objective_aggregated: zaktualizowano "
        "final_objective dla %d runów (skipped: no_ref=%d, bad_weights=%d).",
        len(updates), skipped_no_ref, skipped_bad_weights,
    )


def load_f_ref_per_environment(
    conn: sqlite3.Connection,
) -> dict[str, np.ndarray]:
    """Załaduj F_ref median z `offline_objective_normalization`.

    Helper dla testów i konsumentów chcących odtworzyć normalizację
    używaną w `final_objective`.

    Returns:
        Słownik env → `np.ndarray (5,)` median F_ref (lub pusty gdy
        normalizacja nie była jeszcze policzona).
    """
    result: dict[str, dict[int, float]] = {}
    for row in conn.execute(
        "SELECT environment, f_idx, f_ref_median "
        "FROM offline_objective_normalization "
        "ORDER BY environment, f_idx"
    ):
        env, f_idx, val = row[0], int(row[1]), float(row[2])
        result.setdefault(env, {})[f_idx] = val

    return {
        env: np.asarray([d[i] for i in range(5)], dtype=np.float64)
        for env, d in result.items()
        if len(d) == 5
    }
