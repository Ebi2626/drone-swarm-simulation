"""Wykresy zbieżności — mean ± std band per algorytm, per środowisko.

Standard w meta-heuristics literature: `Engelbrecht (2007)` Sec 16, plus
`Beyer & Schwefel (2002)` "Evolution strategies".
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.analysis.analyzer.plots._common import make_figure, save_and_close


def plot_convergence(
    df_iter: pd.DataFrame,
    out_dir: Path,
    metrics: Iterable[str] = ("hypervolume", "best_so_far", "feasible_ratio"),
    higher_is_better: dict[str, bool] | None = None,
) -> list[Path]:
    """Per (environment, metric): mean curve ± 1 std band, kolor=optimizer.

    Args:
        df_iter: tidy DataFrame z `MetricExtractor.iteration_history`.
            Wymagane kolumny: optimizer, environment, seed, iteration,
            <metric>.
        out_dir: katalog wyjściowy. Stworzy się jeśli brak.
        metrics: które metryki rysować.
        higher_is_better: nieużywane bezpośrednio — to tylko adnotacja
            kierunku w tytule. Default: HV→True, reszta False.

    Returns:
        Lista wygenerowanych ścieżek (PDF).
    """
    if higher_is_better is None:
        higher_is_better = {
            "hypervolume": True,
            "feasible_ratio": True,
            "nondominated_ratio": True,
        }

    out_paths: list[Path] = []
    envs = sorted(df_iter["environment"].dropna().unique())

    for env in envs:
        sub_env = df_iter[df_iter["environment"] == env]
        for metric in metrics:
            if metric not in sub_env.columns:
                continue
            data = sub_env.dropna(subset=[metric])
            if data.empty:
                continue

            fig, ax = make_figure()
            for optimizer, sub in data.groupby("optimizer"):
                pivot = sub.pivot_table(
                    index="iteration", columns="seed", values=metric, aggfunc="mean"
                )
                if pivot.empty:
                    continue
                xs = pivot.index.values
                vals = pivot.values
                mean = np.nanmean(vals, axis=1)
                std = np.nanstd(vals, axis=1, ddof=0)
                line, = ax.plot(xs, mean, label=str(optimizer))
                ax.fill_between(xs, mean - std, mean + std, alpha=0.18, color=line.get_color())

            direction = "higher=better" if higher_is_better.get(metric, False) else "lower=better"
            ax.set_xlabel("generation")
            ax.set_ylabel(metric)
            ax.set_title(f"{env} — {metric} ({direction})")
            ax.legend(frameon=False)

            out_path = out_dir / f"convergence_{env}_{metric}.pdf"
            save_and_close(fig, out_path)
            out_paths.append(out_path)

    return out_paths
