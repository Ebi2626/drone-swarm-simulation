"""Scatter plots: HV vs runtime, HV vs n_eval (per algorytm).

Standardowy diagnostic plot w meta-heuristic literature — Pareto efficiency
budżetu obliczeniowego (Hansen et al. 2009 "Real-Parameter Black-Box
Optimization Benchmarking").
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.analyzer.plots._common import make_figure, save_and_close


def plot_scatter(
    df_run: pd.DataFrame,
    df_iter: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    """HV (z `df_run`) vs runtime/n_eval (z last-row `df_iter`).

    Args:
        df_run: per-run summary (run_summary).
        df_iter: per-iteration time series (iteration_history).

    Returns:
        Lista wygenerowanych ścieżek.
    """
    out_paths: list[Path] = []
    if "hypervolume" not in df_run.columns:
        return out_paths

    # Last-iteration runtime + eval count per run.
    cols = [c for c in ("elapsed_s", "eval_count_cumulative") if c in df_iter.columns]
    if not cols:
        return out_paths
    last = (
        df_iter.sort_values(["run_id", "iteration"])
        .groupby("run_id")[cols]
        .last()
        .reset_index()
    )
    merged = df_run.merge(last, on="run_id", how="left")

    for x_col, x_label, fname in (
        ("elapsed_s", "elapsed (s)", "hv_vs_runtime"),
        ("eval_count_cumulative", "evaluations", "hv_vs_evals"),
    ):
        if x_col not in merged.columns:
            continue
        data = merged.dropna(subset=[x_col, "hypervolume"])
        if data.empty:
            continue
        fig, ax = make_figure()
        for optimizer, sub in data.groupby("optimizer"):
            ax.scatter(sub[x_col], sub["hypervolume"], label=str(optimizer), alpha=0.75)
        ax.set_xlabel(x_label)
        ax.set_ylabel("hypervolume (higher=better)")
        ax.set_title(f"HV vs {x_label}")
        ax.legend(frameon=False)
        out_path = out_dir / f"scatter_{fname}.pdf"
        save_and_close(fig, out_path)
        out_paths.append(out_path)

    return out_paths
