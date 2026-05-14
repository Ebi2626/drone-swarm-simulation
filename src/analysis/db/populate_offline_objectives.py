"""Ekstrakcja best-feasible solution z `optimization_history.h5` + załaduj wagi celów.

`VectorizedEvaluator` (`src/algorithms/abstraction/trajectory/objective_constrains.py`)
ma 5 funkcji celu i 3 ograniczenia. F-vector last-gen best feasible solution
dostarcza wartości do per-objective kolumn w `run_metrics`.

Mapowanie F → kolumny `run_metrics`:
  F[0] = f1 trajectory_cost (length + shape) → final_objective_f1_trajectory
  F[1] = f2 height_angle_cost                → final_objective_f2_height_angle
  F[2] = f3 threat_cost                      → total_threat_cost
  F[3] = f4 turn_cost                        → total_turn_penalty
  F[4] = f5 coordination_cost                → total_coordination_cost

Dodatkowo (2026-05-13): wczytuje `objective_weights` z per-run
`.hydra/config.yaml` (`optimizer.algorithm_params.objective_weights`) i
zapisuje do `run_metrics.final_objective_weights_json`. NSGA-III nie
ma tych wag w configu → canonical fallback `[0.05, 0.5, 0.8, 1.0, 0.25]`
(identyczne z SOO peers, by `final_objective` cross-algorithm comparison
był fair). Wagi zużywa post-pass `populate_final_objective_aggregated`
do obliczenia weighted-normalized-sum `final_objective`.

Constraint vector G nie jest zapisywany do h5 przez żadną z 4 strategii
(NSGA3/MSFFOA/SSA/OOA) — pomijamy go.

Reference: Deb 2014 NSGA-III; Goldberg 1989 selecting feasible solutions
in constrained EAs; Hwang & Yoon 1981 §4.2 weighted-sum MCDM scalarization.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


# Canonical objective weights stosowane gdy run nie ma swojego
# `objective_weights` w `.hydra/config.yaml` (typowo NSGA-III, który
# optymalizuje multi-objective bez skalaryzacji). Identyczne z SOO peers
# (`configs/optimizer/{msffoa,ooa,ssa}.yaml`), by `final_objective`
# weighted-normalized-sum był comparable cross-algorithm.
_CANONICAL_OBJECTIVE_WEIGHTS: tuple[float, ...] = (0.05, 0.5, 0.8, 1.0, 0.25)


# Mapowanie indeksu w F → nazwa kolumny `run_metrics`. Definitive source.
_F_TO_COLUMN_5OBJ: tuple[str, ...] = (
    "final_objective_f1_trajectory",   # f1: length + shape
    "final_objective_f2_height_angle", # f2: height + angle
    "total_threat_cost",               # f3
    "total_turn_penalty",              # f4
    "total_coordination_cost",         # f5: coordination
)

# Legacy 3-obj fallback (stary VectorizedEvaluator
# F=[swarm_total_length, swarm_smoothness, swarm_collisions]).
# Mapuje tylko 2 pierwsze kolumny; trzeci komponent zostaje w
# `final_objectives_json` (kolumna `total_collision_penalty` usunięta z schemy).
_F_TO_COLUMN_3OBJ: tuple[str, ...] = (
    "final_objective_f1_trajectory",   # legacy: total_length
    "final_objective_f2_height_angle", # legacy: smoothness (cluster jakkolwiek)
)


def populate_offline_objectives(
    conn: sqlite3.Connection,
    run_id: str,
    h5_path: Path,
    run_dir: Optional[Path] = None,
) -> None:
    """UPDATE'uj `run_metrics` per-objective wartościami z best feasible F-vector + wagami.

    Idempotentne — re-run nadpisuje wartości. Brak `h5_path`,
    `objectives_matrix` lub pusta last-gen ⇒ silent no-op (kolumny per-obj
    pozostają NULL, ale wagi mogą być zapisane gdy `run_dir` zawiera
    hydra config).

    Args:
        conn: Aktywne połączenie do bazy.
        run_id: Identyfikator runa.
        h5_path: Ścieżka do `optimization_history.h5`.
        run_dir: Katalog runu (zawiera `.hydra/config.yaml`). Gdy `None`,
            inferowany jako `h5_path.parent.parent` (kompatybilność wsteczna
            dla wywołań pre-2026-05-13).

    Efekty uboczne:
        UPDATE w `run_metrics` (`final_objectives_json`, per-obj kolumny
        zgodne z `_F_TO_COLUMN_*`, `final_objective_weights_json`).
        `final_objective` jest USTAWIANY w post-pass
        `populate_final_objective_aggregated`, nie tutaj.
    """
    # Infer run_dir dla backward-compat ze starymi wywołaniami.
    if run_dir is None:
        # h5_path: <run_dir>/optimization_history/optimization_history.h5
        run_dir = h5_path.parent.parent

    weights = _load_objective_weights(run_dir, run_id)

    if not h5_path.exists():
        # Brak h5 — zapisujemy same wagi, by post-pass mógł je zużyć
        # (choć w praktyce brak h5 = brak F → final_objective pozostanie NULL).
        _update_weights_only(conn, run_id, weights)
        return

    selected_F = _extract_best_feasible_F(h5_path)
    if selected_F is None:
        logger.warning(
            f"populate_offline_objectives: {h5_path} brak `objectives_matrix` "
            f"lub pusty — pomijamy F dla run_id={run_id!r}."
        )
        _update_weights_only(conn, run_id, weights)
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

    set_clauses = [
        "final_objectives_json = ?",
        "final_objective_weights_json = ?",
    ]
    params: list = [
        json.dumps(selected_F.tolist()),
        json.dumps(list(weights)),
    ]
    for i, column in enumerate(column_map):
        set_clauses.append(f"{column} = ?")
        params.append(float(selected_F[i]))
    params.append(run_id)

    sql = f"UPDATE run_metrics SET {', '.join(set_clauses)} WHERE run_id = ?"
    cur = conn.execute(sql, params)
    if cur.rowcount == 0:
        # WARNING (nie INFO) — wymagana kolejność w `populate_database.py` to
        # `populate_run_metrics` ⟶ `populate_offline_objectives`. Brak wiersza
        # tutaj oznacza naruszony invariant pipeline'u.
        logger.warning(
            "populate_offline_objectives: brak wpisu w `run_metrics` dla "
            "run_id=%r — UPDATE pominięty. Sprawdź czy `populate_run_metrics` "
            "wykonał się przed tym krokiem.",
            run_id,
        )


def _update_weights_only(
    conn: sqlite3.Connection,
    run_id: str,
    weights: tuple[float, ...],
) -> None:
    """UPDATE samych `final_objective_weights_json` — gdy F-vector niedostępny."""
    cur = conn.execute(
        "UPDATE run_metrics SET final_objective_weights_json = ? WHERE run_id = ?",
        (json.dumps(list(weights)), run_id),
    )
    if cur.rowcount == 0:
        logger.warning(
            "populate_offline_objectives: brak wpisu w `run_metrics` dla "
            "run_id=%r — weights_json nie zapisane.",
            run_id,
        )


def _load_objective_weights(
    run_dir: Path,
    run_id: str,
) -> tuple[float, ...]:
    """Wczytaj `optimizer.algorithm_params.objective_weights` z `.hydra/config.yaml`.

    Fallback: `_CANONICAL_OBJECTIVE_WEIGHTS` z WARNING (typowo NSGA-III,
    który nie ma wag — używa multi-objective natywnie). Fallback z INFO
    gdy plik konfigu nie istnieje (np. testy unit bez run_dir).

    Args:
        run_dir: Katalog runu zawierający `.hydra/config.yaml`.
        run_id: Do komunikatów diagnostycznych.

    Returns:
        `(5,)` krotka wag — z configu albo canonical fallback.
    """
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        logger.info(
            "populate_offline_objectives: brak %s dla run_id=%r — używam "
            "canonical weights %s.",
            config_path, run_id, _CANONICAL_OBJECTIVE_WEIGHTS,
        )
        return _CANONICAL_OBJECTIVE_WEIGHTS

    try:
        import yaml
    except ImportError:  # pragma: no cover
        logger.error(
            "populate_offline_objectives: brakuje pakietu PyYAML — fallback "
            "na canonical weights dla run_id=%r.", run_id,
        )
        return _CANONICAL_OBJECTIVE_WEIGHTS

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except Exception as e:
        logger.error(
            "populate_offline_objectives: błąd parsowania %s (%s) — "
            "fallback na canonical weights dla run_id=%r.",
            config_path, e, run_id,
        )
        return _CANONICAL_OBJECTIVE_WEIGHTS

    weights = (
        cfg.get("optimizer", {})
        .get("algorithm_params", {})
        .get("objective_weights")
    )
    if weights is None:
        # NSGA-III: brak wag w configu (multi-objective natywnie). To NIE
        # jest błąd — log INFO, fallback canonical.
        logger.info(
            "populate_offline_objectives: brak optimizer.algorithm_params."
            "objective_weights w %s dla run_id=%r (typowo NSGA-III) — "
            "canonical fallback %s.",
            config_path, run_id, _CANONICAL_OBJECTIVE_WEIGHTS,
        )
        return _CANONICAL_OBJECTIVE_WEIGHTS

    if not isinstance(weights, (list, tuple)) or len(weights) != 5:
        logger.warning(
            "populate_offline_objectives: nieprawidłowy kształt "
            "objective_weights=%r dla run_id=%r — oczekiwano 5 wartości. "
            "Canonical fallback.",
            weights, run_id,
        )
        return _CANONICAL_OBJECTIVE_WEIGHTS

    return tuple(float(w) for w in weights)


def _extract_best_feasible_F(h5_path: Path) -> Optional[np.ndarray]:
    """Wyciągnij F-vector best feasible solution z LAST generation w `h5_path`.

    Preferuje `best_idx` zapisany przez `HistorySnapshotBuilder.build_payload`
    (`argmin(scalar_fitness)`). Fallback: feasibility-first + `argmin(f1)`
    z `feasible_mask` lub `constraint_violation` / `CV`. Gdy 0 feasible —
    `argmin(f1)` z całej populacji z `WARNING` (final_objective reprezentuje
    infeasible).

    Args:
        h5_path: Ścieżka do `optimization_history.h5`.

    Returns:
        `(n_obj,)` F-vector wybranego rozwiązania lub `None`, gdy h5 nie
        zawiera `objectives_matrix`, last-gen jest pusta albo wystąpił błąd.
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
