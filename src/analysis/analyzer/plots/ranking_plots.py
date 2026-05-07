"""Ranking heatmap: alg × env, kolor = średnia ranga w danym (env, metric).

Pomocne przy szybkiej diagnostyce który algorytm dominuje w którym
środowisku (Demšar 2006 §5).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.analyzer.plots._common import make_figure, save_and_close


def plot_ranking_heatmap(
    df_run: pd.DataFrame,
    metric: str,
    out_path: Path,
    higher_is_better: bool = False,
) -> Path | None:
    """Heatmap (env × optimizer) z średnią rangą po seedach w danym (env).

    Args:
        df_run: tidy run_summary.
        metric: kolumna do rangowania.
        out_path: docelowy PDF.
        higher_is_better: jeśli True, wartości odwracane przed rangowaniem.

    Returns:
        Ścieżka wynikowa lub None gdy brak danych.
    """
    if metric not in df_run.columns:
        return None
    data = df_run.dropna(subset=[metric, "optimizer", "environment", "seed"])
    if data.empty:
        return None

    # Ranga per (env, seed) → średnia po seedach w env.
    pivot = data.pivot_table(
        index=["environment", "seed"], columns="optimizer", values=metric, aggfunc="mean"
    ).dropna()
    if pivot.empty or pivot.shape[1] < 2:
        return None

    values = pivot.values
    if higher_is_better:
        values = -values
    ranks = pd.DataFrame(values, index=pivot.index, columns=pivot.columns).rank(
        axis=1, method="average"
    )
    avg_ranks = ranks.groupby(level="environment").mean()  # env × optimizer

    fig, ax = make_figure(figsize=(1.2 + 0.7 * avg_ranks.shape[1], 0.6 + 0.45 * avg_ranks.shape[0]))
    im = ax.imshow(avg_ranks.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(np.arange(avg_ranks.shape[1]))
    ax.set_xticklabels(avg_ranks.columns, rotation=15)
    ax.set_yticks(np.arange(avg_ranks.shape[0]))
    ax.set_yticklabels(avg_ranks.index)
    ax.set_title(f"avg rank — {metric} ({'higher=better' if higher_is_better else 'lower=better'})")
    for i in range(avg_ranks.shape[0]):
        for j in range(avg_ranks.shape[1]):
            ax.text(
                j, i, f"{avg_ranks.values[i, j]:.2f}",
                ha="center", va="center", fontsize=8, color="black",
            )
    fig.colorbar(im, ax=ax, label="rank (1=best)")

    save_and_close(fig, out_path)
    return out_path
