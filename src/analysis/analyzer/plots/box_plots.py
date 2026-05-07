"""Box-and-whiskers per metryka per algorytm.

Standard w empirycznym porównaniu meta-heurystyk (Demšar 2006 §3.4).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.analysis.analyzer.plots._common import make_figure, save_and_close


def plot_boxplots(
    df_run: pd.DataFrame,
    out_dir: Path,
    metrics: Iterable[str],
) -> list[Path]:
    """Per (environment, metric): boxplot grupy `optimizer`.

    Args:
        df_run: tidy DataFrame z `MetricExtractor.run_summary`.
        out_dir: katalog wyjściowy.
        metrics: które kolumny rysować.

    Returns:
        Lista wygenerowanych ścieżek (PDF).
    """
    out_paths: list[Path] = []
    envs = sorted(df_run["environment"].dropna().unique())

    for env in envs:
        sub_env = df_run[df_run["environment"] == env]
        for metric in metrics:
            if metric not in sub_env.columns:
                continue
            data = sub_env.dropna(subset=[metric])
            if data.empty:
                continue
            grouped = data.groupby("optimizer")[metric]
            optimizers = sorted(grouped.groups.keys())
            arrays = [grouped.get_group(o).values for o in optimizers]
            if not any(len(a) > 0 for a in arrays):
                continue

            fig, ax = make_figure()
            ax.boxplot(
                arrays,
                tick_labels=optimizers,
                showmeans=True,
                meanline=True,
                widths=0.55,
            )
            ax.set_ylabel(metric)
            ax.set_title(f"{env} — {metric}")
            ax.tick_params(axis="x", rotation=15)

            out_path = out_dir / f"boxplot_{env}_{metric}.pdf"
            save_and_close(fig, out_path)
            out_paths.append(out_path)

    return out_paths
