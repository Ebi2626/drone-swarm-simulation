"""2D projekcje fronta Pareto per (env, pair-of-objectives).

Jeśli n_obj=2, jeden wykres. Dla n_obj>2 generujemy wszystkie pary
(i, j), i < j (Riquelme et al. 2015 §3.5).
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from src.analysis.analyzer.plots._common import make_figure, save_and_close


def plot_pareto_projections(df_pareto: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Per (environment, i_obj, j_obj): scatter wszystkich punktów front per
    optimizer.

    Args:
        df_pareto: long-form z `MetricExtractor.pareto_front_last_gen`.
            Kolumny: run_id, optimizer, environment, seed, point_idx,
            objective_j, value.

    Returns:
        Lista ścieżek wygenerowanych PDF.
    """
    out_paths: list[Path] = []
    if df_pareto.empty:
        return out_paths

    # Pivot do (run_id, point_idx) × objective_j.
    wide = (
        df_pareto.pivot_table(
            index=["run_id", "optimizer", "environment", "seed", "point_idx"],
            columns="objective_j",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    obj_cols = sorted(c for c in wide.columns if isinstance(c, (int, float)))
    if len(obj_cols) < 2:
        return out_paths

    envs = sorted(wide["environment"].dropna().unique())
    for env in envs:
        sub_env = wide[wide["environment"] == env]
        if sub_env.empty:
            continue
        for i, j in combinations(obj_cols, 2):
            data = sub_env.dropna(subset=[i, j])
            if data.empty:
                continue
            fig, ax = make_figure()
            for optimizer, sub in data.groupby("optimizer"):
                ax.scatter(sub[i], sub[j], label=str(optimizer), alpha=0.6, s=14)
            ax.set_xlabel(f"f{i}")
            ax.set_ylabel(f"f{j}")
            ax.set_title(f"{env} — Pareto projection f{i} vs f{j}")
            ax.legend(frameon=False)
            out_path = out_dir / f"pareto_{env}_f{i}_vs_f{j}.pdf"
            save_and_close(fig, out_path)
            out_paths.append(out_path)

    return out_paths
