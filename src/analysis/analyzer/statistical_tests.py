"""Testy statystyczne dla porównania meta-heurystyk.

Zgodnie z `reports/statistical_tests_methodology.md` używamy *tylko* trzech
narzędzi statystycznych, dobranych tak by każde odpowiadało na odrębne
pytanie naukowe bez nakładania się funkcji:

- **Friedman + Nemenyi** (Demšar 2006) — globalny test różnic + post-hoc
  parami dla metryk ciągłych. Operuje na rangach, więc odporny na
  niegausowskie ogony (kara Big-M, bimodalność).
- **Vargha-Delaney A12** (Vargha & Delaney 2000; Arcuri & Briand 2014) —
  miara wielkości efektu probabilistycznego dla par algorytmów.
- **Wilson 95% CI** (Wilson 1927; Newcombe 1998) — przedział ufności dla
  proporcji niepowodzeń (`failure_rate`).

Pozostałe metody (Wilcoxon+Holm, Bootstrap CI, Cochran's Q, McNemar+RD/OR,
ANOVA, Bonferroni) zostały świadomie odrzucone — pełne uzasadnienia
w `reports/statistical_tests_methodology.md` §5.

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
    """Wynik testu Friedmana z post-hoc Nemenyi.

    Args:
        statistic: Statystyka χ² Friedmana.
        p_value: p-value testu Friedmana.
        n_datasets: Liczba „dataset"-ów (`env × seed`).
        n_algorithms: Liczba porównywanych algorytmów.
        average_ranks: Średnia ranga per algorytm.
        cd_nemenyi: Critical difference Nemenyi (α=0.05) lub `None`,
            gdy `k` poza tablicą.
    """
    statistic: float
    p_value: float
    n_datasets: int
    n_algorithms: int
    average_ranks: dict[str, float]
    cd_nemenyi: Optional[float]


@dataclass
class A12Result:
    """Effect size Vargha-Delaney A12 dla pary algorytmów.

    Args:
        alg_a, alg_b: Nazwy porównywanych algorytmów.
        a12: Wartość A12 ∈ `[0, 1]`; `0.5` oznacza brak efektu.
        magnitude: Klasyfikacja `negligible / small / medium / large`
            (Vargha & Delaney 2000).
    """
    alg_a: str
    alg_b: str
    a12: float
    magnitude: str


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
    """Wykonaj test Friedmana z post-hoc Nemenyi (Demšar 2006).

    Args:
        df: Long-form DataFrame z kolumnami `group_col`, `block_cols`, `metric`.
        metric: Kolumna metryki.
        higher_is_better: `True` ⇒ ranking odwracamy (większe = lepsza ranga).
        group_col: Kolumna identyfikująca algorytm.
        block_cols: Kolumny definiujące „datasets".
        alpha: Poziom istotności dla Nemenyi (`0.05` lub `0.10`).

    Returns:
        `FriedmanResult` ze statystyką, p-value, średnimi rangami i CD Nemenyi.

    Raises:
        ValueError: Gdy mniej niż 2 algorytmy lub mniej niż 2 datasety.
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
    """Zwróć stałą `q_alpha` Nemenyi dla `k` algorytmów; `None` poza tablicą."""
    table = _NEMENYI_Q_ALPHA_05 if abs(alpha - 0.05) < 1e-6 else (
        _NEMENYI_Q_ALPHA_10 if abs(alpha - 0.10) < 1e-6 else None
    )
    if table is None:
        return None
    return table.get(k)


# ----------------------------------------------------------------------
# Vargha-Delaney A12
# ----------------------------------------------------------------------


def vargha_delaney_a12(
    df: pd.DataFrame,
    metric: str,
    group_col: str = "optimizer",
    higher_is_better: bool = False,
) -> list[A12Result]:
    """Wylicz Vargha-Delaney A12 dla każdej pary algorytmów.

    `A12 = P(X better than Y) + 0.5·P(X = Y)`. Klasyfikacja magnitude wg
    Vargha & Delaney (2000): `<0.06` negligible, `<0.14` small, `<0.21`
    medium, `≥0.21` large.

    Args:
        df: Long-form DataFrame z `group_col` i `metric`.
        metric: Kolumna metryki.
        group_col: Kolumna identyfikująca algorytm.
        higher_is_better: `True` ⇒ „better" = `>`; `False` ⇒ „better" = `<`.

    Returns:
        Lista `A12Result` dla wszystkich par algorytmów.
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
    """`A12 = P(X better Y) + 0.5·P(X = Y)` — `>` gdy `higher_is_better`, inaczej `<`."""
    n_x, n_y = len(x), len(y)
    if higher_is_better:
        wins = float((x[:, None] > y[None, :]).sum())
    else:
        wins = float((x[:, None] < y[None, :]).sum())
    ties = float((x[:, None] == y[None, :]).sum())
    return (wins + 0.5 * ties) / (n_x * n_y)


def _a12_magnitude(a12: float) -> str:
    """Klasyfikuj A12: `negligible / small / medium / large` (Vargha-Delaney 2000)."""
    diff = abs(a12 - 0.5)
    if diff < 0.06:
        return "negligible"
    if diff < 0.14:
        return "small"
    if diff < 0.21:
        return "medium"
    return "large"


# ----------------------------------------------------------------------
# Wilson score CI dla proporcji
# ----------------------------------------------------------------------


def wilson_proportion_ci(
    n_success: int,
    n_trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score 95% CI dla proporcji binomialnej (Wilson 1927).

    Lepszy niż klasyczny Wald CI dla `p̂` blisko 0 lub 1 (typowe w
    safety-critical). Newcombe (1998) §3 wprost zaleca Wilson jako *default*.

    Args:
        n_success: Liczba sukcesów (lub failure'ów — zależne od interpretacji).
        n_trials: Łączna liczba prób.
        confidence: Poziom ufności (default 0.95).

    Returns:
        `(ci_low, ci_high)` ∈ `[0, 1]`; `(NaN, NaN)` dla `n_trials = 0`.
    """
    if n_trials <= 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p_hat = n_success / n_trials
    z2 = z * z
    denom = 1.0 + z2 / n_trials
    center = (p_hat + z2 / (2.0 * n_trials)) / denom
    margin = (z / denom) * np.sqrt(p_hat * (1.0 - p_hat) / n_trials
                                   + z2 / (4.0 * n_trials ** 2))
    return max(0.0, center - margin), min(1.0, center + margin)


# ----------------------------------------------------------------------
# Summary statistics dla metryk ciągłych — mediana [Q1, Q3]
# ----------------------------------------------------------------------


def summary_stats(
    df: pd.DataFrame,
    metric: str,
    group_cols: Iterable[str] = ("environment", "optimizer"),
) -> pd.DataFrame:
    """Zwróć tabelę statystyk opisowych per grupa: `n, mean, std, min, max,
    median, q25, q75`.

    Para `(mediana, [Q1, Q3])` jest standardową formą raportowania rozkładów
    w literaturze metaheurystyk (Bartz-Beielstein et al. 2020 §4.2) — IQR
    obejmuje 50% obserwacji i jest odporny na ogony niegausowskie.
    `min/max` to ekstrema obserwacji (dla *lower-is-better* metryk: best/worst run);
    konwencja {best, worst, mean, median, std} z Bartz-Beielstein et al. (2020) §4.2.

    Args:
        df: Long-form DataFrame z kolumnami `group_cols` i `metric`.
        metric: Kolumna metryki.
        group_cols: Kolumny grupujące (np. `(environment, optimizer)`).

    Returns:
        DataFrame z kolumnami `[*group_cols, n, mean, std, min, max, median,
        q25, q75]` posortowany rosnąco po `group_cols`.
    """
    group_cols = list(group_cols)
    records = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        values = sub[metric].dropna().values
        if len(values) == 0:
            continue
        rec: dict = {c: k for c, k in zip(group_cols, keys if isinstance(keys, tuple) else (keys,))}
        rec["n"] = len(values)
        rec["mean"] = float(np.mean(values))
        rec["std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rec["min"] = float(np.min(values))
        rec["max"] = float(np.max(values))
        rec["median"] = float(np.median(values))
        rec["q25"] = float(np.percentile(values, 25))
        rec["q75"] = float(np.percentile(values, 75))
        records.append(rec)
    return pd.DataFrame(records).sort_values(group_cols).reset_index(drop=True)
