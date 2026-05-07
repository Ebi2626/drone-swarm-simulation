"""Ekstrakcja best-feasible solution z `optimization_history.h5` (Faza 2).

Nowy `VectorizedEvaluator` ([objective_constrains.py](
src/algorithms/abstraction/trajectory/objective_constrains.py)) ma 5 funkcji
celu i 3 ograniczenia. F-vector last-gen best feasible solution dostarcza
wartości, które ETL dotychczas zostawiał `NULL` w `run_metrics`.

Mapowanie F → kolumny `run_metrics`:
  F[0] = f1 trajectory_cost (length + shape) → final_objective_f1_trajectory
  F[1] = f2 height_angle_cost                → final_objective_f2_height_angle
  F[2] = f3 threat_cost                      → total_threat_cost
  F[3] = f4 turn_cost                        → total_turn_penalty
  F[4] = f5 coordination_cost                → total_coordination_cost

Constraint vector G nie jest zapisywany do h5 przez żadną z 4 strategii
(NSGA3/MSFFOA/SSA/OOA) — pomijamy go.

Reference: Deb 2014 NSGA-III; Goldberg 1989 selecting feasible solutions
in constrained EAs.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


# Mapowanie indeksu w F → nazwa kolumny `run_metrics`. Definitive source.
_F_TO_COLUMN_5OBJ: tuple[str, ...] = (
    "final_objective_f1_trajectory",   # f1: length + shape
    "final_objective_f2_height_angle", # f2: height + angle
    "total_threat_cost",               # f3
    "total_turn_penalty",              # f4
    "total_coordination_cost",         # f5: coordination
)

# Legacy mapping (stary VectorizedEvaluator z 3 obj) — fallback graceful.
# Stary: F = [swarm_total_length, swarm_smoothness, swarm_collisions].
# Po refaktorze 2026-05-07 (`total_collision_penalty` usunięte ze schemy)
# 3-obj fallback mapuje tylko 2 pierwsze; trzeci pozostaje w
# `final_objectives_json`.
_F_TO_COLUMN_3OBJ: tuple[str, ...] = (
    "final_objective_f1_trajectory",   # legacy: total_length
    "final_objective_f2_height_angle", # legacy: smoothness (cluster jakkolwiek)
)


def populate_offline_objectives(
    conn: sqlite3.Connection,
    run_id: str,
    h5_path: Path,
) -> None:
    """Wyciąga best feasible F-vector z h5 i UPDATE'uje `run_metrics`.

    Idempotentne: re-run UPDATE'uje wartości, nie duplikuje wierszy.
    Brak h5 / pusty `objectives_matrix` → silent no-op (run_metrics
    pozostaje z NULL'ami — odpowiednik braku optymalizacji).
    """
    if not h5_path.exists():
        return

    selected_F = _extract_best_feasible_F(h5_path)
    if selected_F is None:
        logger.warning(
            f"populate_offline_objectives: {h5_path} brak `objectives_matrix` "
            f"lub pusty — pomijamy run_id={run_id!r}."
        )
        return

    n_obj = int(selected_F.shape[0])
    if n_obj == 5:
        column_map = _F_TO_COLUMN_5OBJ
    elif n_obj == 3:
        logger.warning(
            f"populate_offline_objectives: {h5_path} ma 3 obj "
            f"(legacy VectorizedEvaluator). Mapowanie legacy "
            f"[length, smoothness, collisions]."
        )
        column_map = _F_TO_COLUMN_3OBJ
    else:
        # Nieznana arity — zapisz tylko JSON, nie próbuj mapować.
        logger.warning(
            f"populate_offline_objectives: {h5_path} ma {n_obj} obj — "
            f"nieznane mapowanie. Zapisuję tylko `final_objectives_json`."
        )
        column_map = ()

    # final_objective = F[0] (trajectory cost) — zachowuje semantykę
    # `best_so_far_obj0` używaną w istniejących widokach analitycznych.
    final_objective_main = float(selected_F[0])

    set_clauses = ["final_objective = ?", "final_objectives_json = ?"]
    params: list = [final_objective_main, json.dumps(selected_F.tolist())]
    for i, column in enumerate(column_map):
        set_clauses.append(f"{column} = ?")
        params.append(float(selected_F[i]))
    params.append(run_id)

    sql = f"UPDATE run_metrics SET {', '.join(set_clauses)} WHERE run_id = ?"
    cur = conn.execute(sql, params)
    if cur.rowcount == 0:
        # WARNING (nie INFO) — wymagana kolejność w `populate_database.py` to
        # `populate_run_metrics` ⟶ `populate_offline_objectives`. Brak wiersza
        # tutaj oznacza naruszony invariant pipeline'u (np. populate_run_metrics
        # poległ albo kolejność została zaburzona). Final_objective dla tego
        # run_id pozostanie NULL.
        logger.warning(
            "populate_offline_objectives: brak wpisu w `run_metrics` dla "
            "run_id=%r — UPDATE pominięty (final_objective pozostanie NULL). "
            "Sprawdź czy `populate_run_metrics` wykonał się przed tym krokiem.",
            run_id,
        )


def _extract_best_feasible_F(h5_path: Path) -> Optional[np.ndarray]:
    """Z h5 wyciągnij F-vector best feasible solution z LAST generation.

    Strategia (decyzja użytkownika 2026-05-08): odczyt `best_idx` z h5.
    `HistorySnapshotBuilder.build_payload` (używany przez MSFFOA) zapisuje
    `best_idx = argmin(scalar_fitness)` — to jest "best po skalaryzacji
    algorytmu" (Big-M feasibility-first + weighted sum). Używamy go
    bezpośrednio, bez ponownego wyboru po f1.

    Pipeline:
    1. Jeśli h5 zawiera `best_idx` (HistorySnapshotBuilder writers): użyj go
       — F[best_idx] reprezentuje WIERNIE best po fitness algorytmu.
    2. Fallback (per-gen metrics SSA/OOA writers, h5 bez best_idx):
       feasibility-first wybór najpierw, potem argmin(f1) z feasible. Jeśli
       0 feasible — argmin(f1) z całej populacji + WARNING (final_objective
       reprezentuje infeasible).

    Zwraca `None` gdy:
    - h5 nie zawiera `objectives_matrix`,
    - last gen ma 0 individuals,
    - exception przy odczycie h5.
    """
    try:
        import h5py
    except ImportError:  # pragma: no cover
        logger.error("populate_offline_objectives: brakuje pakietu h5py.")
        return None

    try:
        with h5py.File(h5_path, "r") as f:
            if "objectives_matrix" not in f:
                return None
            obj_ds = f["objectives_matrix"]
            n_gens = obj_ds.shape[0]
            if n_gens == 0:
                return None
            last_gen_idx = n_gens - 1
            obj_last = np.asarray(obj_ds[last_gen_idx], dtype=np.float64)
            if obj_last.ndim == 1:
                obj_last = obj_last[:, np.newaxis]
            if obj_last.shape[0] == 0:
                return None

            # Preferowana ścieżka: `best_idx` zapisany przez algorytm
            # (HistorySnapshotBuilder.build_payload). Reprezentuje
            # argmin(scalar_fitness) — wierny ranking algorytmu.
            if "best_idx" in f:
                try:
                    best_idx_val = int(np.asarray(f["best_idx"][last_gen_idx]).reshape(-1)[0])
                    if 0 <= best_idx_val < obj_last.shape[0]:
                        return obj_last[best_idx_val]
                    logger.warning(
                        "_extract_best_feasible_F: %s best_idx=%d poza [0, %d) — "
                        "fallback na argmin(f1) z feasible.",
                        h5_path, best_idx_val, obj_last.shape[0],
                    )
                except (OSError, KeyError, ValueError, IndexError) as e:
                    logger.warning(
                        "_extract_best_feasible_F: %s best_idx odczyt failed (%s) — "
                        "fallback na argmin(f1).",
                        h5_path, e,
                    )

            # Fallback: feasibility-first + argmin(f1).
            # Feasibility — preferowana z `feasible_mask`; w przeciwnym wypadku
            # z `constraint_violation` / `CV`.
            feasible_mask = None
            for name in ("feasible_mask",):
                if name in f:
                    fm = np.asarray(f[name][last_gen_idx]).reshape(-1).astype(bool)
                    if fm.shape[0] == obj_last.shape[0]:
                        feasible_mask = fm
                        break

            if feasible_mask is None:
                for name in ("constraint_violation", "CV", "constraint_violations", "last_cv"):
                    if name in f:
                        cv = np.asarray(f[name][last_gen_idx], dtype=np.float64)
                        if cv.ndim == 1:
                            cv_total = np.maximum(cv, 0.0)
                        else:
                            cv_total = np.sum(np.maximum(cv, 0.0), axis=1)
                        if cv_total.shape[0] == obj_last.shape[0]:
                            from src.utils.per_gen_metrics import FEASIBILITY_EPS
                            feasible_mask = cv_total <= FEASIBILITY_EPS
                            break

            # Wybór best feasible po f1 (col 0). Jeśli nic feasible → użyj
            # wszystkich (best-effort recovery). UWAGA: w tym fallback'u
            # `final_objective` w `run_metrics` reprezentuje **najmniejszy
            # f1 w infeasible populacji**, nie prawdziwy "best feasible".
            if feasible_mask is None or not np.any(feasible_mask):
                logger.warning(
                    "_extract_best_feasible_F: %s — last gen ma 0 feasible "
                    "rozwiązań (feasible_mask=%s). Fallback na argmin(f1) "
                    "z całej populacji; `final_objective` reprezentuje "
                    "infeasible solution.",
                    h5_path,
                    "None" if feasible_mask is None else "all-False",
                )
                pool = obj_last
            else:
                pool = obj_last[feasible_mask]

            best_local_idx = int(np.argmin(pool[:, 0]))
            return pool[best_local_idx]

    except Exception as e:  # pragma: no cover
        logger.error(
            f"populate_offline_objectives: błąd odczytu {h5_path}: {e}",
            exc_info=True,
        )
        return None
