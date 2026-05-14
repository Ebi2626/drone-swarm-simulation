"""Bar plots: failure rate per (env, alg).

Standard plot dla failure-rate w empirycznym porównaniu (Engelbrecht 2007).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.analyzer.plots._common import make_figure, save_and_close


def plot_failure_rate_bars(
    df_run: pd.DataFrame,
    out_dir: Path,
    failure_col: str = "is_offline_failure",
    file_suffix: str = "offline",
) -> list[Path]:
    """Per environment: failure rate per optimizer.

    Failure rate = liczba runów z `failure_col=1` / liczba runów. Wywołujący
    decyduje czy plotuje offline failure (`is_offline_failure`: HV=0 lub
    front_size_last_gen=0) czy online (`is_online_failure`: collision_count>0).

    Reference: Liefooghe & Verel (2014) "On the impact of operators and
    parameters on the performance of single and multi-objective evolutionary
    algorithms" — failure rate jest standardowym uzupełnieniem mean/median
    summary statistics, oddającym tail-risk algorytmu.
    """
    if failure_col not in df_run.columns:
        return []

    out_paths: list[Path] = []
    envs = sorted(df_run["environment"].dropna().unique())

    for env in envs:
        sub_env = df_run[df_run["environment"] == env]
        if sub_env.empty:
            continue
        agg = sub_env.groupby("optimizer")[failure_col].mean().sort_index()
        if agg.empty:
            continue
        fig, ax = make_figure()
        xs = np.arange(len(agg))
        ax.bar(xs, agg.values, color="#7c2d12")
        ax.set_xticks(xs)
        ax.set_xticklabels(agg.index, rotation=15)
        ax.set_ylim(0, max(1.05, float(agg.max()) + 0.1))
        ax.set_ylabel(f"{file_suffix} failure rate")
        ax.set_title(f"{env} — {file_suffix} failure rate")
        for x, v in zip(xs, agg.values):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        out_path = out_dir / f"bar_{env}_failure_rate_{file_suffix}.pdf"
        save_and_close(fig, out_path)
        out_paths.append(out_path)

    return out_paths
