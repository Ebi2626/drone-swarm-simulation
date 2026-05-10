"""Wskaźniki jakości MOO per generation.

Wskaźniki (lower=better dla GD/IGD+/spread/spacing; r2 też lower=better):
- **Spread** Δ (Deb et al. 2002, NSGA-II paper, equation (15)):
  Δ = (d_f + d_l + Σ_{i=1..N-1}|d_i − d̄|) / (d_f + d_l + (N-1)·d̄)
  gdzie d_i to euklidesowa odl. między i-tym a (i+1) sąsiadującym
  rozwiązaniem na froncie (po sortowaniu po pierwszej osi),
  d_f, d_l to odl. od skrajnych wektorów ekstrema do końcówek fronta,
  d̄ = mean(d_i). Δ=0 idealnie równomierny.

- **Spacing** S (Schott 1995):
  S = sqrt( (1/(N-1)) · Σ (d_i − d̄)^2 )
  d_i = min_{j≠i} ‖f_i − f_j‖_1 (Manhattan distance do najbliższego sąsiada).

- **R2 indicator** (Hansen & Jaszkiewicz 1998):
  R2(A, W, z*) = (1/|W|) · Σ_{w ∈ W} min_{a ∈ A} max_j w_j·|a_j − z*_j|
  W = uniformly distributed weight vectors (simplex).

- **GD** (Van Veldhuizen 1999): mean(min ‖f_i − r_j‖_2 for r ∈ R)
- **IGD+** (Ishibuchi et al. 2015): mean(min d^+(r,f)) z d^+(r,f) =
  sqrt(Σ max(0, f_j − r_j)^2). Pareto-compliant, lower=better.

Reference: Riquelme, Lücken, Baran (2015) "Performance metrics in
multi-objective optimization", CLEI EJ 18(1).
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


def populate_moo_quality(
    conn: sqlite3.Connection,
    run_id: str,
    h5_path: Path,
    reference_set: Optional[np.ndarray] = None,
    reference_point: Optional[np.ndarray] = None,
    ideal_point: Optional[np.ndarray] = None,
    compute_baseline_metrics: bool = True,
) -> None:
    """Liczy spread/spacing/r2 (zawsze gdy compute_baseline_metrics) + gd/igd+
    (gdy reference_set!=None) + hypervolume (gdy reference_point!=None) per
    generacja i wpisuje do `optimization_generation_stats`.

    Args:
        reference_set: R per (env, n_obj) dla GD/IGD+. Riquelme et al. 2015.
        reference_point: r* per (env, n_obj) dla HV. Ishibuchi et al. 2018.
        ideal_point: z* = min(R, axis=0) per (env, n_obj). Wymagany dla
            `hypervolume_normalized = HV / Π(r* − z*)` (Riquelme 2015 §3.6,
            zapewnia cross-env porównywalność HV).
        compute_baseline_metrics: gdy False, pomija spread/spacing/r2
            (zakłada że są już w `optimization_generation_stats` z poprzedniego
            biegu). Używane w `backfill_moo_quality_with_reference` żeby
            uniknąć ~70% redundantnej pracy. Default True (kompatybilność
            wsteczna — niezmienione dla bezpośrednich wywołań z populate_database).

    Idempotentne: re-run nadpisuje (INSERT OR REPLACE).
    `populate_iteration_metrics` musi być wywołany PO tej funkcji żeby
    pochwycić nowe metryki do tablicy `iteration_metrics`.
    """
    if not h5_path.exists():
        return

    try:
        import h5py
    except ImportError:  # pragma: no cover
        logger.error("populate_moo_quality: brakuje h5py.")
        return

    rows: list[tuple] = []

    try:
        with h5py.File(h5_path, "r") as f:
            if "objectives_matrix" not in f:
                return
            obj_ds = f["objectives_matrix"]
            n_gens = obj_ds.shape[0]
            if n_gens == 0:
                return

            # feasibility (opcjonalnie)
            feasible_mask_ds = f["feasible_mask"] if "feasible_mask" in f else None
            cv_ds = None
            for name in ("constraint_violation", "CV", "constraint_violations", "last_cv"):
                if name in f:
                    cv_ds = f[name]
                    break

            # Wagi dla R2 — uniformly distributed simplex (Das-Dennis-like).
            # Cache, bo n_obj jest stałe per run.
            r2_weights_cache: dict[int, np.ndarray] = {}

            for gen in range(n_gens):
                F = np.asarray(obj_ds[gen], dtype=np.float64)
                if F.ndim == 1:
                    F = F[:, np.newaxis]
                if F.shape[0] == 0:
                    continue
                n_obj = F.shape[1]

                # Feasibility filter (jeśli możliwe)
                fmask = _feasibility_mask(feasible_mask_ds, cv_ds, gen, F.shape[0])
                F_feas = F[fmask] if fmask is not None and np.any(fmask) else F

                # Pareto front feasible'a
                front = _non_dominated(F_feas)
                if front.shape[0] == 0:
                    # Diagnostic: empty feasible front (rare but informative).
                    rows.append((run_id, gen, "moo_quality", "front_size", 0.0))
                    continue

                # Diagnostic: |F_feas ∩ ND| per gen.
                rows.append((run_id, gen, "moo_quality", "front_size", float(front.shape[0])))

                # --- Spread / Spacing / R2 (baseline indicators) ---
                # W backfillu (compute_baseline_metrics=False) te metryki są
                # już w DB z initial populate — re-liczenie to ~70% pracy do
                # wyrzucenia.
                if compute_baseline_metrics:
                    spread = _spread_metric(front)
                    if spread is not None:
                        rows.append((run_id, gen, "moo_quality", "spread", float(spread)))

                    spacing = _spacing_metric(front)
                    if spacing is not None:
                        rows.append((run_id, gen, "moo_quality", "spacing", float(spacing)))

                    if n_obj not in r2_weights_cache:
                        r2_weights_cache[n_obj] = _generate_simplex_weights(n_obj, n_partitions=8)
                    # Przekazujemy `ideal_point` (globalny z* z reference set
                    # per env) gdy dostępny — gwarancja ceteris paribus dla
                    # cross-algorithm porównań. Brak ideal_point → fallback
                    # na local z* (mniej miarodajny, ale niezerowy).
                    r2 = _r2_indicator(
                        front, r2_weights_cache[n_obj],
                        z_star=ideal_point if ideal_point is not None
                        and ideal_point.shape[0] == n_obj else None,
                    )
                    if r2 is not None:
                        rows.append((run_id, gen, "moo_quality", "r2_indicator", float(r2)))

                # --- GD / IGD+ (wymagają reference_set) ---
                if reference_set is not None and reference_set.shape[1] == n_obj and reference_set.shape[0] > 0:
                    gd = _generational_distance(front, reference_set)
                    igd_plus = _igd_plus(front, reference_set)
                    if gd is not None:
                        rows.append((run_id, gen, "moo_quality", "gd", float(gd)))
                    if igd_plus is not None:
                        rows.append((run_id, gen, "moo_quality", "igd_plus", float(igd_plus)))

                # --- HV (wymaga reference_point) ---
                if reference_point is not None and reference_point.shape[0] == n_obj:
                    hv = _hypervolume(front, reference_point)
                    if hv is not None:
                        rows.append((run_id, gen, "moo_quality", "hypervolume", float(hv)))
                        # HV_norm = HV / Π(r* − z*). Cross-env comparable.
                        # Wymaga ideal_point z reference_pareto_sets (min per oś).
                        if (
                            ideal_point is not None
                            and ideal_point.shape[0] == n_obj
                        ):
                            box = np.asarray(reference_point, dtype=np.float64) - np.asarray(
                                ideal_point, dtype=np.float64
                            )
                            if np.all(box > 0):
                                denom = float(np.prod(box))
                                if denom > 0:
                                    # Cap at 1.0 — numerical drift / small front
                                    # subsampling może dać HV>denom o ~ε.
                                    hv_norm = min(1.0, max(0.0, float(hv) / denom))
                                    rows.append(
                                        (run_id, gen, "moo_quality", "hypervolume_normalized", hv_norm)
                                    )

    except Exception as e:  # pragma: no cover
        logger.error(f"populate_moo_quality: {h5_path}: {e}", exc_info=True)
        return

    if rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO optimization_generation_stats
            (run_id, generation, source_name, metric_name, metric_value)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )


# ---------------------------------------------------------------------------
# Helpery — feasibility, dominance, wskaźniki
# ---------------------------------------------------------------------------


def _feasibility_mask(feasible_mask_ds, cv_ds, gen: int, n_pop: int) -> Optional[np.ndarray]:
    if feasible_mask_ds is not None:
        try:
            fm = np.asarray(feasible_mask_ds[gen]).reshape(-1).astype(bool)
            if fm.shape[0] == n_pop:
                return fm
        except Exception:
            pass
    if cv_ds is not None:
        try:
            cv = np.asarray(cv_ds[gen], dtype=np.float64)
            if cv.ndim == 1:
                tot = np.maximum(cv, 0.0)
            else:
                tot = np.sum(np.maximum(cv, 0.0), axis=1)
            if tot.shape[0] == n_pop:
                return tot <= 1e-6
        except Exception:
            pass
    return None


def _non_dominated(F: np.ndarray) -> np.ndarray:
    """Zwraca Pareto-front (rows of F nie zdominowane przez żadny inny row).

    Preferuje `pymoo.util.nds.non_dominated_sorting.NonDominatedSorting`
    (Cython-optimized, ~20-50× szybsze niż naiwne O(N²) w Pythonie). Fallback
    na O(N²) numpy gdy pymoo niedostępny lub awaryjnie.

    Reference: Deb, Pratap, Agarwal & Meyarivan (2002) "A fast elitist
    non-dominated sorting genetic algorithm for multi-objective optimization:
    NSGA-II", IEEE Trans. Evol. Comput. 6(2):182–197 sec. III.A.
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


def _spread_metric(front: np.ndarray) -> Optional[float]:
    """Δ-metric (Deb et al. 2002). Definicja oryginalna jest dla 2 obj —
    dla M>2 stosujemy uogólnienie po sortowaniu wzdłuż pierwszej osi
    (Riquelme et al. 2015 Sec. 3.2).

    UWAGA: bez referencyjnych ekstremów (`d_f`, `d_l`) wynik degeneruje
    do `Σ|d_i − d_mean| / ((n-1)·d_mean)` — to jest miara JEDNORODNOŚCI
    odstępów, semantycznie bliższa SPACING niż klasycznemu SPREAD'owi
    (który mierzy także rozciąg do extremes). Wyniki w DB w kolumnie
    `spread` są spójne ranking-wise ALE niezgodne ze ścisłą definicją
    Deb 2002. Patrz `code-review.md` (uproszczenie udokumentowane).
    """
    n = front.shape[0]
    if n < 2:
        return None
    sorted_front = front[np.argsort(front[:, 0])]
    diffs = np.linalg.norm(np.diff(sorted_front, axis=0), axis=1)
    if diffs.size == 0:
        return None
    d_mean = float(np.mean(diffs))
    # d_f, d_l: odl. od skrajnych ekstremów. W braku referencji ekstremalnych
    # wektorów ustawione na 0 (uproszczenie zgodne z Riquelme 2015 §3.2).
    # Konsekwencja: wzór redukuje się do measure odstępów (zob. docstring).
    d_f = 0.0
    d_l = 0.0
    num = d_f + d_l + float(np.sum(np.abs(diffs - d_mean)))
    den = d_f + d_l + (n - 1) * d_mean
    if den <= 1e-12:
        return None
    return num / den


def _spacing_metric(front: np.ndarray) -> Optional[float]:
    """Schott 1995. d_i = min_{j≠i} ‖f_i − f_j‖_1; S = sqrt(var(d_i))."""
    n = front.shape[0]
    if n < 2:
        return None
    # Manhattan pairwise distances
    diff = front[:, np.newaxis, :] - front[np.newaxis, :, :]
    d = np.sum(np.abs(diff), axis=-1)
    np.fill_diagonal(d, np.inf)
    d_min = np.min(d, axis=1)
    d_mean = float(np.mean(d_min))
    if n - 1 <= 0:
        return None
    return float(np.sqrt(np.mean((d_min - d_mean) ** 2)))


def _generate_simplex_weights(n_obj: int, n_partitions: int = 8) -> np.ndarray:
    """Das-Dennis simplex weights — uniformly distributed na simplex'ie,
    używane do R2 indicator. Wagi sumują się do 1.
    """
    from itertools import combinations_with_replacement

    pts = []
    for combo in combinations_with_replacement(range(n_partitions + 1), n_obj):
        if sum(combo) == n_partitions:
            pts.append([c / n_partitions for c in combo])
    if not pts:
        return np.eye(n_obj)
    return np.asarray(pts, dtype=np.float64)


def _r2_indicator(
    front: np.ndarray,
    weights: np.ndarray,
    z_star: Optional[np.ndarray] = None,
) -> Optional[float]:
    """R2(A, W, z*) = (1/|W|) Σ_w min_a max_j w_j · |a_j − z*_j|.

    Args:
        front: Pareto front (|A|, n_obj).
        weights: simplex weights (|W|, n_obj).
        z_star: ideal point (n_obj,). Gdy podany — używany **globalnie**
            dla cross-run comparability (Hansen 1998, Brockhoff 2012).
            Gdy None — fallback na `min(front, axis=0)` (lokalny ideal),
            wartość R2 NIE jest porównywalna między runami.

    Strategia ceteris paribus: przekazujemy `z_star` z
    `reference_pareto_sets.ideal_point` (per env, n_obj) — ten sam ideal dla
    wszystkich algorytmów porównywanych w danym środowisku. Bez tego R2 jest
    porównywalne tylko w obrębie pojedynczego runa, niewłaściwe dla benchmark
    cross-algorithm.
    """
    if front.shape[0] == 0 or weights.shape[0] == 0:
        return None
    if z_star is None:
        z_star = np.min(front, axis=0)
    z_star = np.asarray(z_star, dtype=np.float64).reshape(-1)
    if z_star.shape[0] != front.shape[1]:
        return None
    # |W| × |A| × n_obj
    diffs = np.abs(front[np.newaxis, :, :] - z_star[np.newaxis, np.newaxis, :])
    weighted = weights[:, np.newaxis, :] * diffs
    cheby = np.max(weighted, axis=-1)        # |W| × |A|
    min_per_w = np.min(cheby, axis=-1)       # |W|
    return float(np.mean(min_per_w))


def _generational_distance(front: np.ndarray, ref_set: np.ndarray, p: float = 2.0) -> Optional[float]:
    """GD = (1/|A|) · (Σ_a min_r ‖a − r‖_2^p)^(1/p). Niższe = bliższe R."""
    if front.shape[0] == 0 or ref_set.shape[0] == 0:
        return None
    diff = front[:, np.newaxis, :] - ref_set[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    min_per_a = np.min(dists, axis=1)
    return float(np.power(np.mean(np.power(min_per_a, p)), 1.0 / p))


# Próg sub-samplingu frontu przed HV. pymoo `HV` (algorytm WFG) jest
# implementowany w pure-Python (nie Cython!) z złożonością O(M·N^(M-1)).
# W 5D dla N=200 to ~30s/call; N=500 → ~17h/call. Stąd cap na 200.
#
# UWAGA NAUKOWA: zwracana wartość HV po sub-sampling jest **konserwatywnym
# lower-bound** prawdziwego HV (subset_volume ≤ full_volume zawsze, dla
# Pareto-frontów). Cytat z Bringmann & Friedrich (2010) dotyczy uniform
# random sampling W OBJ-SPACE (Monte Carlo volume estimation, gdzie estymata
# JEST unbiased), nie sub-samplingu z fronta. Nasz schemat (extreme points
# + uniform stride) jest deterministic, ale zaniża HV proporcjonalnie do
# liczby odrzuconych punktów. Dla cross-algorithm comparison: ranking
# pozostaje wiarygodny gdy wszystkie fronts są jednakowo "duże" przed
# subsampling. Inaczej algorytm produkujący 1000 feasibles dostaje większy
# bias niż produkujący 200 feasibles.
#
# Reference: Ishibuchi, Imada, Setoguchi & Nojima (2018) Sec. V eksplicytnie
# zalecają |S| ≤ 100 dla M ≥ 4 dla tractable cross-algorithm comparison
# i AKCEPTUJĄ rezultującą stratę precyzji.
HV_FRONT_SUBSAMPLE_THRESHOLD = 200


def _hypervolume(front: np.ndarray, ref_point: np.ndarray) -> Optional[float]:
    """HV(F, r*) — objętość obj-space zdominowana przez F, ograniczona r*.
    Zitzler & Thiele (1998); implementacja przez `pymoo.indicators.hv.HV`.
    Higher = lepiej.

    Wymaga r* > każde f komponentowo (Ishibuchi 2018) — w p.p. HV=0
    lub ujemne (degenerate). Filtrujemy tylko pkt zdominowane przez r*
    (czyli f < r* component-wise) — to jest zgodne z definicją HV.

    Dla `|F_valid| > HV_FRONT_SUBSAMPLE_THRESHOLD` aplikujemy deterministic
    uniform sub-sampling (extreme points + równomierny stride) — pymoo HV
    w 5D+ z większymi frontami staje się intractable (godziny/call).
    """
    if front.shape[0] == 0 or ref_point.shape[0] == 0:
        return None
    if front.shape[1] != ref_point.shape[0]:
        return None
    # Tylko punkty zdominowane przez r* przyczyniają się do HV.
    valid = np.all(front < ref_point[np.newaxis, :], axis=1)
    if not np.any(valid):
        return 0.0
    F_valid = front[valid]

    F_for_hv = _maybe_subsample_for_hv(F_valid)

    try:
        from pymoo.indicators.hv import HV
    except ImportError:  # pragma: no cover
        logger.warning("pymoo niedostępne — HV pominięte.")
        return None
    try:
        return float(HV(ref_point=np.asarray(ref_point, dtype=np.float64))(F_for_hv))
    except Exception as e:  # pragma: no cover
        logger.error(f"_hypervolume: pymoo HV failed: {e}")
        return None


def _maybe_subsample_for_hv(F: np.ndarray) -> np.ndarray:
    """Sub-sample do `HV_FRONT_SUBSAMPLE_THRESHOLD` punktów dla HV w 5D+.

    Strategia: zachowaj M ekstremów (min per oś — kotwice frontu) +
    deterministic stride na pozostałych po sortowaniu po pierwszej osi.
    Daje stabilny ranking cross-run (deterministyczny, niezależny od RNG).
    """
    n, m = F.shape
    if n <= HV_FRONT_SUBSAMPLE_THRESHOLD:
        return F
    # Ekstremalne punkty (min per oś) — kluczowe dla bounding boxa HV.
    extreme_idx = np.unique(np.argmin(F, axis=0))
    # Stride po sortowaniu po f0 — równomierny przekrój.
    sort_order = np.argsort(F[:, 0])
    n_remain = HV_FRONT_SUBSAMPLE_THRESHOLD - len(extreme_idx)
    if n_remain <= 0:
        return F[extreme_idx]
    stride = max(1, n // n_remain)
    sampled_idx = sort_order[::stride][:n_remain]
    keep = np.unique(np.concatenate([extreme_idx, sampled_idx]))
    return F[keep]


def _igd_plus(front: np.ndarray, ref_set: np.ndarray) -> Optional[float]:
    """IGD+ (Ishibuchi 2015). Pareto-compliant.
    d^+(r, f) = sqrt(Σ_j max(0, f_j − r_j)^2)  — kara *tylko* gdy f gorsze od r.
    IGD+ = mean_r min_f d^+(r, f).
    """
    if front.shape[0] == 0 or ref_set.shape[0] == 0:
        return None
    # ref × front × n_obj
    diff = front[np.newaxis, :, :] - ref_set[:, np.newaxis, :]
    pos = np.maximum(0.0, diff)
    d_plus = np.sqrt(np.sum(pos ** 2, axis=-1))   # ref × front
    min_per_r = np.min(d_plus, axis=1)
    return float(np.mean(min_per_r))
