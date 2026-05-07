#!/usr/bin/env python3
"""
Budget Calibration Analysis — Convergence vs NFE for offline optimization.

Reads per-iteration metrics from the ETL database (analysis.db) and produces:
  1. Convergence curves: metric vs eval_count_cumulative (NFE), mean ± std
  2. NFE threshold table: first NFE where metric reaches 90%/95%/99% of final
  3. Ranking stability heatmap: Friedman rank at various NFE fractions
  4. Budget summary CSV + LaTeX table

This script addresses experiment_restrictions.md:
  - §6.3: Sensitivity analysis (convergence vs computational budget)
  - §6.4: Ranking stability under budget variation
  - §2.3: Empirical evidence for SSA budget calibration

Usage:
    python src/analysis/budget_calibration_analysis.py results/<exp_id>

Requirements:
    - ETL must have been run first (ExperimentAggregator.aggregate())
    - analysis.db must exist in the experiment directory

References:
    - Hansen et al. (2009) BBOB: NFE as standard budget metric
    - Demšar (2006) §4: Friedman + Nemenyi for algorithm comparison
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Project imports ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.analysis.analyzer.metric_extractor import MetricExtractor
from src.analysis.analyzer.plots._common import make_figure, save_and_close

# ── Constants ────────────────────────────────────────────────────────────

# Metrics to analyze: (column_name, higher_is_better)
CONVERGENCE_METRICS: dict[str, bool] = {
    "best_so_far": False,       # SOO scalar fitness — lower = better
    "hypervolume": True,        # MOO indicator (NSGA-III) — higher = better
    "feasible_ratio": True,     # fraction of feasible solutions — higher = better
}

# NFE thresholds: fraction of final metric value
THRESHOLDS = [0.90, 0.95, 0.99]

# NFE fractions for ranking stability analysis
NFE_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.0]

# Visual settings per optimizer (consistent across plots)
OPTIMIZER_COLORS = {
    "msffoa": "#1f77b4",
    "ooa": "#ff7f0e",
    "ssa": "#2ca02c",
    "nsga-3": "#d62728",
}

OPTIMIZER_LABELS = {
    "msffoa": "MSFFOA",
    "ooa": "OOA",
    "ssa": "SSA",
    "nsga-3": "NSGA-III",
}


# ── Core functions ───────────────────────────────────────────────────────

def compute_nfe_thresholds(
    df_iter: pd.DataFrame,
    metric: str,
    higher_is_better: bool,
    thresholds: list[float] = THRESHOLDS,
) -> pd.DataFrame:
    """Find first NFE where metric reaches X% of its final value.

    Convergence "90%" znaczy: pokonano 90% drogi z `initial_val` do
    `final_val`. Wzór ujednolicony dla obu kierunków:

        target = initial_val + threshold * (final_val − initial_val)

    Dla `higher_is_better=True` szukamy pierwszego `val ≥ target`.
    Dla `higher_is_better=False` szukamy pierwszego `val ≤ target`.

    Wzór jest **bezpieczny** dla negatywnych objectivów (np. R2 może być
    ujemny w niektórych konwencjach) — wcześniejsza wersja `target = thr ·
    final` dawała `vals >= target` natychmiast triggerujące przy
    ujemnym `final`.

    Returns DataFrame with columns:
        optimizer, environment, seed, threshold, nfe_at_threshold
    """
    rows = []
    groups = df_iter.groupby(["optimizer", "environment", "seed"])

    for (opt, env, seed), grp in groups:
        grp = grp.sort_values("eval_count_cumulative")
        nfe = grp["eval_count_cumulative"].values
        vals = grp[metric].values

        if len(vals) == 0 or np.all(np.isnan(vals)):
            continue

        # Remove NaN rows
        mask = ~np.isnan(vals)
        nfe = nfe[mask]
        vals = vals[mask]

        if len(vals) == 0:
            continue

        initial_val = vals[0]
        final_val = vals[-1]

        # Edge case: stagnacja (initial == final). Każdy próg "osiągnięty"
        # od pierwszej iteracji — raportujemy NFE[0].
        if np.isclose(initial_val, final_val):
            for thr in thresholds:
                rows.append({
                    "optimizer": opt,
                    "environment": env,
                    "seed": seed,
                    "threshold": thr,
                    "nfe_at_threshold": int(nfe[0]),
                })
            continue

        for thr in thresholds:
            # Ujednolicona formuła: target = initial + thr · (final − initial).
            # Robust dla obu kierunków + negatywnych wartości.
            target = initial_val + thr * (final_val - initial_val)
            if higher_is_better:
                idx = np.where(vals >= target)[0]
            else:
                idx = np.where(vals <= target)[0]

            nfe_at = int(nfe[idx[0]]) if len(idx) > 0 else int(nfe[-1])
            rows.append({
                "optimizer": opt,
                "environment": env,
                "seed": seed,
                "threshold": thr,
                "nfe_at_threshold": nfe_at,
            })

    return pd.DataFrame(rows)


def _interpolate_metric_on_nfe_grid(
    df_iter: pd.DataFrame,
    metric: str,
    n_points: int = 200,
) -> pd.DataFrame:
    """Interpolate metric values onto a common NFE grid.

    Different algorithms report NFE at different iteration counts. To compare
    them on the same x-axis, we interpolate each run's metric onto a shared
    grid of `n_points` evenly spaced NFE values from 0 to max NFE observed.

    Returns DataFrame with columns:
        optimizer, environment, seed, nfe_grid, {metric}
    """
    max_nfe = df_iter["eval_count_cumulative"].max()
    nfe_grid = np.linspace(0, max_nfe, n_points)

    rows = []
    groups = df_iter.groupby(["optimizer", "environment", "seed"])

    for (opt, env, seed), grp in groups:
        grp = grp.sort_values("eval_count_cumulative")
        nfe = grp["eval_count_cumulative"].values.astype(float)
        vals = grp[metric].values.astype(float)

        mask = ~np.isnan(vals)
        nfe = nfe[mask]
        vals = vals[mask]

        if len(nfe) < 2:
            continue

        interp_vals = np.interp(nfe_grid, nfe, vals)
        for i, (n, v) in enumerate(zip(nfe_grid, interp_vals)):
            rows.append({
                "optimizer": opt,
                "environment": env,
                "seed": seed,
                "nfe_grid": n,
                metric: v,
            })

    return pd.DataFrame(rows)


def plot_convergence_vs_nfe(
    df_iter: pd.DataFrame,
    metric: str,
    higher_is_better: bool,
    out_dir: Path,
) -> None:
    """Plot convergence curves: metric vs NFE, mean ± 1 std.

    One plot per environment, all optimizers overlaid.
    """
    interp_df = _interpolate_metric_on_nfe_grid(df_iter, metric)

    for env, env_df in interp_df.groupby("environment"):
        fig, ax = make_figure(figsize=(8, 5))

        for opt, opt_df in env_df.groupby("optimizer"):
            agg = opt_df.groupby("nfe_grid")[metric].agg(["mean", "std"])
            nfe = agg.index.values
            mean = agg["mean"].values
            std = agg["std"].fillna(0).values

            color = OPTIMIZER_COLORS.get(opt, "#333333")
            label = OPTIMIZER_LABELS.get(opt, opt)

            ax.plot(nfe, mean, color=color, label=label)
            ax.fill_between(nfe, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_xlabel("NFE (Number of Function Evaluations)")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Convergence: {metric} vs NFE — {env}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        save_and_close(fig, out_dir / f"convergence_vs_nfe_{metric}_{env}.pdf")


def compute_ranking_at_nfe_slices(
    df_iter: pd.DataFrame,
    metric: str,
    higher_is_better: bool,
    fractions: list[float] = NFE_FRACTIONS,
) -> pd.DataFrame:
    """Compute Friedman-style ranking at various NFE budget fractions.

    At each fraction of max NFE, we extract the metric value (via interpolation)
    and rank the optimizers per (environment, seed). Then average ranks across
    seeds to get mean rank per (environment, nfe_fraction, optimizer).

    Returns DataFrame with columns:
        environment, nfe_fraction, optimizer, avg_rank
    """
    interp_df = _interpolate_metric_on_nfe_grid(df_iter, metric)
    max_nfe = interp_df["nfe_grid"].max()

    rows = []
    for frac in fractions:
        target_nfe = frac * max_nfe
        # Find closest NFE grid point
        all_nfe = interp_df["nfe_grid"].unique()
        closest_nfe = all_nfe[np.argmin(np.abs(all_nfe - target_nfe))]

        slice_df = interp_df[interp_df["nfe_grid"] == closest_nfe].copy()

        for env, env_df in slice_df.groupby("environment"):
            for seed, seed_df in env_df.groupby("seed"):
                if higher_is_better:
                    seed_df = seed_df.copy()
                    seed_df["rank"] = seed_df[metric].rank(ascending=False)
                else:
                    seed_df = seed_df.copy()
                    seed_df["rank"] = seed_df[metric].rank(ascending=True)

                for _, row in seed_df.iterrows():
                    rows.append({
                        "environment": env,
                        "nfe_fraction": frac,
                        "optimizer": row["optimizer"],
                        "seed": seed,
                        "rank": row["rank"],
                    })

    ranking_df = pd.DataFrame(rows)
    # Average ranks across seeds
    avg_ranks = (
        ranking_df
        .groupby(["environment", "nfe_fraction", "optimizer"])["rank"]
        .mean()
        .reset_index()
        .rename(columns={"rank": "avg_rank"})
    )
    return avg_ranks


def plot_ranking_stability_heatmap(
    rankings_df: pd.DataFrame,
    out_dir: Path,
    metric: str,
) -> None:
    """Heatmap: x=NFE fraction, y=optimizer, color=avg rank.

    One heatmap per environment. Demonstrates ranking stability (§6.4):
    if rankings don't change across budget fractions, the comparison is robust.
    """
    import matplotlib.pyplot as plt

    for env, env_df in rankings_df.groupby("environment"):
        pivot = env_df.pivot_table(
            index="optimizer", columns="nfe_fraction", values="avg_rank",
        )
        # Sort optimizers by final rank (best at top)
        final_col = pivot.columns[-1]
        pivot = pivot.sort_values(final_col)

        # Rename index labels
        pivot.index = [OPTIMIZER_LABELS.get(o, o) for o in pivot.index]

        fig, ax = make_figure(figsize=(8, 3.5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=4)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{f:.0%}" for f in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("NFE Budget Fraction")
        ax.set_ylabel("Algorithm")
        ax.set_title(f"Ranking Stability: {metric} — {env}")

        # Annotate cells with rank values
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]:.1f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if pivot.values[i, j] > 2.5 else "black")

        fig.colorbar(im, ax=ax, label="Avg Rank", shrink=0.8)
        save_and_close(fig, out_dir / f"ranking_stability_{metric}_{env}.pdf")


def generate_budget_summary_table(
    df_thresholds: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Aggregate NFE thresholds into a summary table.

    Output: budget_summary.csv + budget_summary.tex with columns:
        optimizer, environment, threshold, nfe_median, nfe_q25, nfe_q75
    """
    summary = (
        df_thresholds
        .groupby(["optimizer", "environment", "threshold"])["nfe_at_threshold"]
        .agg(
            nfe_median="median",
            nfe_q25=lambda x: x.quantile(0.25),
            nfe_q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )

    # Rename optimizers for readability
    summary["optimizer"] = summary["optimizer"].map(
        lambda x: OPTIMIZER_LABELS.get(x, x)
    )

    # Sort for consistent output
    summary = summary.sort_values(
        ["environment", "optimizer", "threshold"]
    ).reset_index(drop=True)

    # CSV
    csv_path = out_dir / "budget_summary.csv"
    summary.to_csv(csv_path, index=False)

    # LaTeX table
    tex_path = out_dir / "budget_summary.tex"
    # Format NFE as integers with thousands separator
    tex_df = summary.copy()
    for col in ["nfe_median", "nfe_q25", "nfe_q75"]:
        tex_df[col] = tex_df[col].apply(lambda x: f"{int(x):,}")

    tex_df.columns = [
        "Algorithm", "Environment", "Threshold",
        "NFE (median)", "NFE (Q25)", "NFE (Q75)",
    ]
    latex_str = tex_df.to_latex(index=False, escape=True, column_format="llrrrr")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by budget_calibration_analysis.py\n")
        f.write(latex_str)

    print(f"  Budget summary → {csv_path}")
    print(f"  LaTeX table    → {tex_path}")


# ── Main pipeline ────────────────────────────────────────────────────────

def main(experiment_dir: str | Path) -> None:
    experiment_dir = Path(experiment_dir)
    db_path = experiment_dir / "analysis.db"

    if not db_path.exists():
        print(
            f"Error: {db_path} not found. Run ETL first:\n"
            f"  python -c \"from src.analysis.ExperimentAggregator import "
            f"ExperimentAggregator; ExperimentAggregator().aggregate('{experiment_dir}')\"",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = experiment_dir / "budget_calibration_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading iteration metrics from {db_path}...")
    extractor = MetricExtractor(db_path)
    df_iter = extractor.iteration_history(
        metrics=["best_so_far", "feasible_ratio", "eval_count_cumulative",
                 "hypervolume"]
    )

    if df_iter.empty:
        print("Error: No iteration metrics found in database.", file=sys.stderr)
        sys.exit(1)

    n_runs = df_iter["run_id"].nunique()
    print(f"  Loaded {len(df_iter)} rows from {n_runs} runs")

    # Collect all threshold results for the summary table
    all_thresholds: list[pd.DataFrame] = []

    for metric, higher_is_better in CONVERGENCE_METRICS.items():
        # Skip metrics that are all-NaN (e.g. hypervolume for SOO algorithms)
        if df_iter[metric].isna().all():
            print(f"  Skipping {metric} (all NaN)")
            continue

        print(f"\n  Analyzing: {metric} (higher_is_better={higher_is_better})")

        # 1. Convergence curves
        print(f"    Plotting convergence curves...")
        plot_convergence_vs_nfe(df_iter, metric, higher_is_better, out_dir)

        # 2. NFE thresholds
        print(f"    Computing NFE thresholds...")
        df_thr = compute_nfe_thresholds(df_iter, metric, higher_is_better)
        if not df_thr.empty:
            all_thresholds.append(df_thr)

        # 3. Ranking stability
        print(f"    Computing ranking stability...")
        rankings = compute_ranking_at_nfe_slices(
            df_iter, metric, higher_is_better
        )
        if not rankings.empty:
            print(f"    Plotting ranking stability heatmap...")
            plot_ranking_stability_heatmap(rankings, out_dir, metric)

    # 4. Budget summary table (aggregated across all metrics)
    if all_thresholds:
        print(f"\nGenerating budget summary table...")
        combined_thresholds = pd.concat(all_thresholds, ignore_index=True)
        generate_budget_summary_table(combined_thresholds, out_dir)

    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Budget calibration analysis — convergence vs NFE"
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment results directory (must contain analysis.db)",
    )
    args = parser.parse_args()
    main(args.experiment_dir)
