"""Wspólny config matplotlib dla wszystkich plotów analyzera.

Wymusza headless backend ("Agg") — zgodnie z CLAUDE.md (Skill: Headless
Physics Test). Plot funkcje akceptują `out_path: Path` i zapisują PDF;
do testów można też przekazać `ax: matplotlib.axes.Axes` żeby skomponować
subplots.
"""
from __future__ import annotations

import os

import matplotlib

# Headless backend; bez tego plt.savefig na maszynach bez DISPLAY pada.
if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg"):
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402

# Konsystentny styl publikacyjny (mała czcionka, serif).
_RC = {
    "figure.figsize": (6.0, 4.0),
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
}


def apply_style() -> None:
    plt.rcParams.update(_RC)


def make_figure(figsize: tuple[float, float] | None = None):
    apply_style()
    if figsize is None:
        figsize = _RC["figure.figsize"]
    return plt.subplots(figsize=figsize)


def save_and_close(fig, out_path) -> None:
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    # Save PNG alongside PDF for report embedding (imread handles PNG, not PDF).
    if out_path.suffix.lower() == ".pdf":
        fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
