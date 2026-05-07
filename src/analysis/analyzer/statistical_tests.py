"""Testy statystyczne dla porównania meta-heurystyk.

Reference (gold standard):
- **Demšar (2006)** "Statistical Comparisons of Classifiers over Multiple
  Data Sets", JMLR 7:1-30. Zaleca:
  * Friedman test (≥3 algorytmów) + Nemenyi post-hoc.
  * Wilcoxon signed-rank dla par algorytmów.
  * Korekcja Holma na wielokrotne porównania.
- **Arcuri & Briand (2014)** "A Hitchhiker's guide to statistical tests for
  assessing randomized algorithms in software engineering", STVR 24(3).
  Zaleca:
  * Effect size — Vargha-Delaney A12 (non-parametric).
  * Bootstrap CI dla median/quantyli.

Konwencje:
- "lower is better" zakłada się dla większości metryk (final_objective, GD,
  IGD+, spread, spacing). Dla "higher is better" (HV, success rate)
  należy przekazać `higher_is_better=True`.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FriedmanResult:
    statistic: float
    p_value: float
    n_datasets: int        # liczba (env × seed) "dataset"-ów
    n_algorithms: int
    average_ranks: dict[str, float]
    cd_nemenyi: Optional[float]   # critical difference dla Nemenyi α=0.05


@dataclass
class WilcoxonPair:
    alg_a: str
    alg_b: str
    statistic: float
    p_value: float
    p_value_holm: float    # po korekcji Holm-Bonferroni
    n: int                 # liczba par
    median_diff: float


@dataclass
class A12Result:
    alg_a: str
    alg_b: str
    a12: float
    magnitude: str         # negligible/small/medium/large per Vargha-Delaney


# ----------------------------------------------------------------------
# Friedman + Nemenyi
# ----------------------------------------------------------------------


def friedman_with_nemenyi(
    df: pd.DataFrame,
    metric: str,
    higher_is_better: bool = False,
    group_col: str = "optimizer",
    block_cols: Iterable[str] = ("environment", "seed"),
    alpha: float = 0.05,
) -> FriedmanResult:
    """Friedman test + Nemenyi post-hoc.

    Args:
        df: long-form DataFrame z kolumnami `group_col`, `block_cols`, `metric`.
        higher_is_better: jeśli True, ranking odwracamy (większe = lepsze rank).
        block_cols: kolumny definiujące "datasets" (Demšar 2006 nomenclature).

    Returns:
        FriedmanResult z statistic, p_value, average ranks per algorytm,
        oraz critical difference (Nemenyi α=0.05).

    Raises:
        ValueError gdy < 2 algorytmów lub brak kompletnego bloku.
    """
    block_cols = list(block_cols)
    pivot = df.pivot_table(
        index=block_cols, columns=group_col, values=metric, aggfunc="mean"
    ).dropna()

    if pivot.shape[1] < 2:
        raise ValueError(f"Friedman wymaga ≥2 algorytmów, mam {pivot.shape[1]}.")
    if pivot.shape[0] < 2:
        raise ValueError(f"Friedman wymaga ≥2 datasetów, mam {pivot.shape[0]}.")

    # Ranks per row: ascending = lower rank (1 = lowest = best gdy lower=better).
    # Jeśli higher_is_better, ranks wyznaczamy na -wartości.
    values = pivot.values
    if higher_is_better:
        values = -values
    ranks = pd.DataFrame(values, index=pivot.index, columns=pivot.columns).rank(
        axis=1, method="average"
    )

    statistic, p_value = stats.friedmanchisquare(*[pivot[c].values for c in pivot.columns])

    avg_ranks = ranks.mean(axis=0).to_dict()

    # Nemenyi critical difference (Demšar 2006, eq. 13):
    # CD = q_alpha · sqrt(k(k+1) / (6N))
    k = pivot.shape[1]
    N = pivot.shape[0]
    q_alpha = _nemenyi_q_alpha(k, alpha)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N)) if q_alpha is not None else None

    return FriedmanResult(
        statistic=float(statistic),
        p_value=float(p_value),
        n_datasets=int(N),
        n_algorithms=int(k),
        average_ranks={str(k): float(v) for k, v in avg_ranks.items()},
        cd_nemenyi=float(cd) if cd is not None else None,
    )


# Hardcoded q_alpha values dla Nemenyi (Demšar 2006, Table 5b).
# k = liczba klasyfikatorów (≥ 2). α=0.05 / 0.10. Tabela do k=10.
_NEMENYI_Q_ALPHA_05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}
_NEMENYI_Q_ALPHA_10 = {
    2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
    7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920,
}


def _nemenyi_q_alpha(k: int, alpha: float) -> Optional[float]:
    table = _NEMENYI_Q_ALPHA_05 if abs(alpha - 0.05) < 1e-6 else (
        _NEMENYI_Q_ALPHA_10 if abs(alpha - 0.10) < 1e-6 else None
    )
    if table is None:
        return None
    return table.get(k)


# ----------------------------------------------------------------------
# Wilcoxon pairwise + Holm correction
# ----------------------------------------------------------------------


def wilcoxon_pairwise(
    df: pd.DataFrame,
    metric: str,
    group_col: str = "optimizer",
    block_cols: Iterable[str] = ("environment", "seed"),
) -> list[WilcoxonPair]:
    """Wilcoxon signed-rank dla każdej pary algorytmów + Holm correction.

    Returns:
        Lista par posortowana po surowym p-value (najmniejszy najpierw).
    """
    block_cols = list(block_cols)
    pivot = df.pivot_table(
        index=block_cols, columns=group_col, values=metric, aggfunc="mean"
    ).dropna()

    algs = sorted(pivot.columns.tolist())
    raw_results: list[tuple[str, str, float, float, int, float]] = []
    for a, b in combinations(algs, 2):
        x = pivot[a].values
        y = pivot[b].values
        n = len(x)
        if n < 1:
            continue
        if np.allclose(x, y):
            stat, p = float("nan"), 1.0
        else:
            try:
                stat, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                stat, p = float("nan"), 1.0
        median_diff = float(np.median(x - y))
        raw_results.append((a, b, float(stat), float(p), int(n), median_diff))

    # Holm-Bonferroni: sortuj po p, mnóż przez (m - i) ze stopniowym ograniczeniem ≤1.
    raw_results.sort(key=lambda r: r[3])
    m = len(raw_results)
    holm_corrected: list[float] = []
    running_max = 0.0
    for i, r in enumerate(raw_results):
        adj = min(1.0, r[3] * (m - i))
        # Holm: monotonia — adjusted p-value rośnie z rangą
        running_max = max(running_max, adj)
        holm_corrected.append(running_max)

    return [
        WilcoxonPair(
            alg_a=r[0],
            alg_b=r[1],
            statistic=r[2],
            p_value=r[3],
            p_value_holm=hp,
            n=r[4],
            median_diff=r[5],
        )
        for r, hp in zip(raw_results, holm_corrected)
    ]


# ----------------------------------------------------------------------
# Vargha-Delaney A12
# ----------------------------------------------------------------------


def vargha_delaney_a12(
    df: pd.DataFrame,
    metric: str,
    group_col: str = "optimizer",
    higher_is_better: bool = False,
) -> list[A12Result]:
    """Vargha-Delaney A12 — non-parametric effect size.

    A12 = P(X > Y) + 0.5 · P(X = Y).
    Jeśli `higher_is_better=False`, wynik interpretujemy względem reverse:
      A12(A vs B) = P(A < B) + 0.5·P(A = B) — A jest "lepszy" gdy A12 > 0.5.

    Magnitude (Vargha & Delaney 2000):
      |A12 − 0.5| < 0.06: negligible
      0.06 ≤ ... < 0.14: small
      0.14 ≤ ... < 0.21: medium
      ≥ 0.21:           large
    """
    algs = sorted(df[group_col].unique())
    results: list[A12Result] = []
    for a, b in combinations(algs, 2):
        x = df.loc[df[group_col] == a, metric].dropna().values
        y = df.loc[df[group_col] == b, metric].dropna().values
        if len(x) == 0 or len(y) == 0:
            continue
        a12 = _a12_value(x, y, higher_is_better)
        results.append(A12Result(alg_a=a, alg_b=b, a12=a12, magnitude=_a12_magnitude(a12)))
    return results


def _a12_value(x: np.ndarray, y: np.ndarray, higher_is_better: bool) -> float:
    """A12 = P(X better than Y) + 0.5 P(X == Y).
    "Better" = > gdy higher_is_better, < gdy lower_is_better.
    """
    n_x, n_y = len(x), len(y)
    if higher_is_better:
        wins = float((x[:, None] > y[None, :]).sum())
    else:
        wins = float((x[:, None] < y[None, :]).sum())
    ties = float((x[:, None] == y[None, :]).sum())
    return (wins + 0.5 * ties) / (n_x * n_y)


def _a12_magnitude(a12: float) -> str:
    diff = abs(a12 - 0.5)
    if diff < 0.06:
        return "negligible"
    if diff < 0.14:
        return "small"
    if diff < 0.21:
        return "medium"
    return "large"


# ----------------------------------------------------------------------
# Bootstrap CI
# ----------------------------------------------------------------------


def bootstrap_ci(
    values: Iterable[float],
    statistic: callable = np.median,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap percentile CI dla `statistic` (default: median).

    Returns:
        (point_estimate, ci_low, ci_high). CI używa percentyli z resampli.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(arr))
    if len(arr) == 1:
        return point, point, point
    rng = np.random.default_rng(rng_seed)
    resamples = rng.choice(arr, size=(n_resamples, len(arr)), replace=True)
    boot_stats = np.apply_along_axis(statistic, 1, resamples)
    alpha = 1.0 - confidence
    low = float(np.percentile(boot_stats, 100 * alpha / 2))
    high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, low, high


def summary_with_ci(
    df: pd.DataFrame,
    metric: str,
    group_cols: Iterable[str] = ("environment", "optimizer"),
    n_resamples: int = 10000,
) -> pd.DataFrame:
    """Buduje summary table per group: mean, std, median, IQR, CI95(median).

    Returns:
        DataFrame indexed by group_cols, columns:
        ['n', 'mean', 'std', 'median', 'q25', 'q75', 'ci95_low', 'ci95_high'].
    """
    group_cols = list(group_cols)
    records = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        values = sub[metric].dropna().values
        if len(values) == 0:
            continue
        median, lo, hi = bootstrap_ci(values, np.median, n_resamples=n_resamples)
        rec: dict = {c: k for c, k in zip(group_cols, keys if isinstance(keys, tuple) else (keys,))}
        rec["n"] = len(values)
        rec["mean"] = float(np.mean(values))
        rec["std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rec["median"] = median
        rec["q25"] = float(np.percentile(values, 25))
        rec["q75"] = float(np.percentile(values, 75))
        rec["ci95_low"] = lo
        rec["ci95_high"] = hi
        records.append(rec)
    return pd.DataFrame(records).sort_values(group_cols).reset_index(drop=True)
