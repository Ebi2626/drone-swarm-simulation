"""Buduje reference Pareto sets cross-run (Faza 3 plan.md).

W absencji prawdziwego "true Pareto front" (typowo używanego w benchmarkach
ZDT/DTLZ z znaną postacią analityczną) standard literaturowy dla problemów
real-world to **merged non-dominated front** zbudowany z all-feasible-ND
last-gen rozwiązań ze wszystkich runów (algorytm × seed) per (env, n_obj).
Reference: Riquelme, Lücken & Baran (2015) "Performance metrics in
multi-objective optimization", CLEI EJ 18(1) §4; także zalecane przez
Ishibuchi et al. (2018) "How to Specify a Reference Point in Hypervolume
Calculation for Fair Performance Comparison".

Po zbudowaniu R, `backfill_moo_quality_with_reference` re-liczy GD i IGD+
per generacja per run z R jako celem.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


# Margin ε w `r* = nadir + ε · (nadir − ideal)` (Ishibuchi 2018, Eq. 4).
# ε=0.1 to środek zalecanego pasma [0.05, 0.2]; dla 5-obj w forest/urban
# objective space jest dobrze unormowany przez nasze ograniczenia, więc
# margines 10% wystarcza by każde feasible rozwiązanie było zdominowane
# przez r* (warunek konieczny żeby HV było > 0).
DEFAULT_REF_POINT_MARGIN = 0.1


def build_reference_pareto_sets(conn: sqlite3.Connection, experiment_dir: Path) -> dict[tuple[str, int], np.ndarray]:
    """Buduje reference set R per (environment, n_obj) z last-gen feasible-ND
    fronts wszystkich runów. Wpisuje R do `reference_pareto_sets` ORAZ
    reference point r* do `reference_points`.

    Returns:
        dict[(env, n_obj) -> R: ndarray (|R|, n_obj)] dla wygody backfill'u.
    """
    runs = conn.execute(
        "SELECT run_id, source_path, environment FROM runs ORDER BY environment, run_id"
    ).fetchall()

    grouped_pts: dict[tuple[str, int], list[np.ndarray]] = {}

    for run_id, source_path, environment in runs:
        h5_path = Path(source_path) / "optimization_history" / "optimization_history.h5"
        if not h5_path.exists():
            continue
        last_front = _last_gen_feasible_front(h5_path)
        if last_front is None or last_front.shape[0] == 0:
            continue
        n_obj = last_front.shape[1]
        grouped_pts.setdefault((environment, n_obj), []).append(last_front)

    # Wyczyść stare ref-sety (idempotency). Reference points też — będą
    # przeliczone z aktualnego R.
    conn.execute("DELETE FROM reference_pareto_sets")
    conn.execute("DELETE FROM reference_points")

    result: dict[tuple[str, int], np.ndarray] = {}

    for (env, n_obj), fronts in grouped_pts.items():
        merged = np.vstack(fronts)
        # Re-non-dominated-sort merged set
        R = _non_dominated(merged)
        if R.shape[0] == 0:
            continue
        result[(env, n_obj)] = R

        rows = []
        for point_idx in range(R.shape[0]):
            for j in range(n_obj):
                rows.append((env, n_obj, point_idx, j, float(R[point_idx, j])))
        conn.executemany(
            """
            INSERT OR REPLACE INTO reference_pareto_sets
            (environment, n_obj, point_idx, objective_j, value)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

        # Reference point r* dla HV — Ishibuchi 2018 §4.
        # nadir + ε · (nadir − ideal). ε=0.1 (DEFAULT_REF_POINT_MARGIN).
        # Dla zdegenerowanego R (1 punkt) range=0 → r* = nadir + ε·|nadir|
        # by ref > nadir w sensie ścisłym (HV > 0 wymaga r > każde f).
        ideal = np.min(R, axis=0)
        nadir = np.max(R, axis=0)
        rng = nadir - ideal
        # Fallback gdy zakres = 0 (degenerate single-point R) — używamy
        # |nadir| jako proxy skali, z drobnym epsilon dla zera.
        rng = np.where(rng > 0, rng, np.maximum(np.abs(nadir), 1.0))
        r_star = nadir + DEFAULT_REF_POINT_MARGIN * rng

        ref_rows = [
            (
                env, n_obj, j,
                float(r_star[j]),
                float(ideal[j]),
                DEFAULT_REF_POINT_MARGIN,
                "nadir_plus_eps_range",
            )
            for j in range(n_obj)
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO reference_points
            (environment, n_obj, objective_j, value, ideal_value, margin, method)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ref_rows,
        )

        logger.info(
            f"build_reference_pareto: env={env} n_obj={n_obj} → |R|={R.shape[0]} "
            f"z {len(fronts)} runów; r*={np.array2string(r_star, precision=3)}"
        )

    return result


def load_reference_set(
    conn: sqlite3.Connection, environment: str, n_obj: int
) -> Optional[np.ndarray]:
    """Wczytuje R z DB. Zwraca None jeśli brak."""
    rows = conn.execute(
        """
        SELECT point_idx, objective_j, value
        FROM reference_pareto_sets
        WHERE environment = ? AND n_obj = ?
        ORDER BY point_idx, objective_j
        """,
        (environment, n_obj),
    ).fetchall()
    if not rows:
        return None
    point_idxs = sorted({r[0] for r in rows})
    R = np.zeros((len(point_idxs), n_obj), dtype=np.float64)
    idx_map = {p: i for i, p in enumerate(point_idxs)}
    for p, j, v in rows:
        R[idx_map[p], j] = v
    return R


def load_reference_point(
    conn: sqlite3.Connection, environment: str, n_obj: int
) -> Optional[np.ndarray]:
    """Wczytuje r* (nadir+ε·range) z DB. Zwraca None gdy brak."""
    rows = conn.execute(
        """
        SELECT objective_j, value FROM reference_points
        WHERE environment = ? AND n_obj = ?
        ORDER BY objective_j
        """,
        (environment, n_obj),
    ).fetchall()
    if not rows or len(rows) != n_obj:
        return None
    r = np.zeros(n_obj, dtype=np.float64)
    for j, v in rows:
        r[j] = v
    return r


def load_ideal_point(
    conn: sqlite3.Connection, environment: str, n_obj: int
) -> Optional[np.ndarray]:
    """Wczytuje z* = min(R, axis=0) z `reference_points.ideal_value`. Zwraca
    None gdy brak (np. backfill z legacy DB bez tej kolumny)."""
    rows = conn.execute(
        """
        SELECT objective_j, ideal_value FROM reference_points
        WHERE environment = ? AND n_obj = ?
        ORDER BY objective_j
        """,
        (environment, n_obj),
    ).fetchall()
    if not rows or len(rows) != n_obj:
        return None
    z = np.zeros(n_obj, dtype=np.float64)
    for j, v in rows:
        if v is None:
            return None
        z[j] = v
    return z


def backfill_moo_quality_with_reference(
    conn: sqlite3.Connection,
    reference_sets: dict[tuple[str, int], np.ndarray],
) -> None:
    """Re-liczy GD/IGD+/HV per generacja dla każdego runu i UPDATE'uje
    `iteration_metrics`. Następnie odświeża `run_metrics.gd_final`,
    `run_metrics.igd_plus`, `run_metrics.hypervolume` z last-gen.

    HV liczone z reference_point r* załadowanego z `reference_points`
    (Ishibuchi 2018). Jeśli r* nie istnieje dla danego (env, n_obj), HV
    pominięte (zachowanie wsteczne).
    """
    from src.analysis.db.populate_moo_quality import populate_moo_quality

    runs = conn.execute(
        "SELECT run_id, source_path, environment FROM runs ORDER BY run_id"
    ).fetchall()

    for run_id, source_path, environment in runs:
        h5_path = Path(source_path) / "optimization_history" / "optimization_history.h5"
        if not h5_path.exists():
            continue
        # Trzeba znać n_obj przed wyborem ref-set'u — zerkamy do h5.
        n_obj = _peek_n_obj(h5_path)
        if n_obj is None:
            continue
        R = reference_sets.get((environment, n_obj))
        if R is None:
            continue
        r_star = load_reference_point(conn, environment, n_obj)
        z_ideal = load_ideal_point(conn, environment, n_obj)
        # Re-run populate_moo_quality z R + r* + z* — INSERT OR REPLACE'ami
        # nadpisuje gd/igd+/hypervolume/hypervolume_normalized/front_size
        # w optimization_generation_stats.
        # `compute_baseline_metrics=False`: spread/spacing/r2 są już w DB
        # z initial populate — pomijamy redundantną pracę (~70% kosztu
        # backfillu).
        populate_moo_quality(
            conn, run_id, h5_path,
            reference_set=R, reference_point=r_star, ideal_point=z_ideal,
            compute_baseline_metrics=False,
        )

    # Re-trigger iteration_metrics + run_metrics dla wszystkich runów,
    # żeby pochwycić nowe GD/IGD+/HV. Po refaktorze 2026-05-08:
    # `populate_run_metrics` NIE odnosi się do `final_objective`/`total_threat_cost`/
    # `total_turn_penalty` (domena `populate_offline_objectives`), więc re-run
    # tej funkcji jest bezpieczny — nie wymazuje F-vector z poprzedniego cyklu.
    # `populate_offline_objectives` jest tu wywoływany ponownie żeby gwarantować
    # poprawne wartości po backfill (defensive idempotent re-write).
    from src.analysis.db.populate_iteration_metrics import populate_iteration_metrics
    from src.analysis.db.populate_run_metrics import populate_run_metrics
    from src.analysis.db.populate_offline_objectives import populate_offline_objectives

    runs_with_path = conn.execute(
        "SELECT run_id, source_path FROM runs"
    ).fetchall()
    for run_id, source_path in runs_with_path:
        populate_iteration_metrics(conn, run_id)
        populate_run_metrics(conn, run_id)
        h5_path_run = Path(source_path) / "optimization_history" / "optimization_history.h5"
        if h5_path_run.exists():
            populate_offline_objectives(conn, run_id, h5_path_run)


# ---------------------------------------------------------------------------
# Helpery
# ---------------------------------------------------------------------------


def _peek_n_obj(h5_path: Path) -> Optional[int]:
    try:
        import h5py
    except ImportError:  # pragma: no cover
        return None
    try:
        with h5py.File(h5_path, "r") as f:
            if "objectives_matrix" not in f:
                return None
            shape = f["objectives_matrix"].shape
            if len(shape) < 3:
                return None
            return int(shape[-1])
    except Exception:
        return None


def _last_gen_feasible_front(h5_path: Path) -> Optional[np.ndarray]:
    """Zwraca feasible non-dominated front z last gen lub None."""
    try:
        import h5py
    except ImportError:  # pragma: no cover
        return None
    try:
        with h5py.File(h5_path, "r") as f:
            if "objectives_matrix" not in f:
                return None
            obj_ds = f["objectives_matrix"]
            n_gens = obj_ds.shape[0]
            if n_gens == 0:
                return None
            F = np.asarray(obj_ds[n_gens - 1], dtype=np.float64)
            if F.ndim == 1:
                F = F[:, np.newaxis]
            if F.shape[0] == 0:
                return None

            # Feasibility (best-effort)
            fmask = None
            if "feasible_mask" in f:
                try:
                    fm = np.asarray(f["feasible_mask"][n_gens - 1]).reshape(-1).astype(bool)
                    if fm.shape[0] == F.shape[0]:
                        fmask = fm
                except Exception:
                    pass
            if fmask is None:
                for cv_name in ("constraint_violation", "CV", "constraint_violations", "last_cv"):
                    if cv_name in f:
                        try:
                            cv = np.asarray(f[cv_name][n_gens - 1], dtype=np.float64)
                            if cv.ndim == 1:
                                tot = np.maximum(cv, 0.0)
                            else:
                                tot = np.sum(np.maximum(cv, 0.0), axis=1)
                            if tot.shape[0] == F.shape[0]:
                                fmask = tot <= 1e-6
                            break
                        except Exception:
                            pass
            F_feas = F[fmask] if fmask is not None and np.any(fmask) else F
            return _non_dominated(F_feas)
    except Exception as e:  # pragma: no cover
        logger.error(f"_last_gen_feasible_front: {h5_path}: {e}")
        return None


def _non_dominated(F: np.ndarray) -> np.ndarray:
    """Pareto-front (rows of F nie zdominowane przez żadny inny row).

    Preferuje `pymoo.util.nds.non_dominated_sorting.NonDominatedSorting`
    (Cython-optimized). Fallback na O(N²) numpy. Krytyczne dla cross-run
    merged front, gdzie N może rosnąć do kilkudziesięciu tysięcy (wszystkie
    feasible-ND last-gen ze wszystkich runów per (env, n_obj)).
    """
    F = np.asarray(F, dtype=np.float64)
    n = F.shape[0]
    if n == 0:
        return F
    if n == 1:
        return F.copy()
    try:
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
        return F[idx]
    except Exception:  # pragma: no cover — fallback dla awarii pymoo
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            leq = np.all(F <= F[i], axis=1)
            lt = np.any(F < F[i], axis=1)
            dominators = leq & lt
            dominators[i] = False
            if np.any(dominators):
                keep[i] = False
        return F[keep]
