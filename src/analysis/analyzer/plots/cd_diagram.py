"""Critical Difference diagram (Demsar 2006, SS3.2).

Wizualizacja: os z srednimi rangami z testu Friedmana; grube poziome
kreski (clique bars) lacza algorytmy, ktorych srednie rangi roznia sie
o mniej niz CD — sa "statystycznie nierozroznialne" (Nemenyi alpha=0.05).

Layout (top to bottom):
  1. Title
  2. Rank axis with CD bar, tick marks and numbers
  3. Algorithm labels on two sides (best=right, worst=left),
     connected by elbow lines; text faces OUTWARD (away from axis)
  4. Clique bars just below labels, with thin vertical drops from rank
     positions so the reader sees which algorithms they connect
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis.analyzer.plots._common import apply_style, save_and_close
import matplotlib.pyplot as plt


def plot_cd_diagram(
    average_ranks: dict[str, float],
    cd: float,
    out_path: Path,
    title: str | None = None,
) -> Path:
    """Rysuje CD diagram.

    Args:
        average_ranks: {algorithm -> avg_rank}.
        cd: critical difference (z Nemenyi).
        out_path: docelowy PDF.
        title: opcjonalny tytul.

    Returns:
        Sciezka do zapisu.
    """
    apply_style()
    if not average_ranks:
        raise ValueError("plot_cd_diagram: pusty average_ranks")

    items = sorted(average_ranks.items(), key=lambda kv: kv[1])
    names = [n for n, _ in items]
    ranks = np.array([r for _, r in items], dtype=float)
    k = len(names)
    half = (k + 1) // 2  # best-ranked count (right side)

    # --- Compute maximal cliques ---
    cliques: list[tuple[float, float]] = []
    for i in range(k):
        for j in range(i + 1, k):
            if ranks[j] - ranks[i] < cd:
                cliques.append((ranks[i], ranks[j]))
    maximal: list[tuple[float, float]] = []
    for c in cliques:
        if not any(
            other != c and other[0] <= c[0] and other[1] >= c[1]
            for other in cliques
        ):
            maximal.append(c)
    maximal = sorted(set(maximal))
    n_cliques = len(maximal)

    # =================================================================
    # COORDINATE SYSTEM
    # =================================================================
    # X-axis: rank values (INVERTED — rank 1 on the right of screen).
    # Y-axis: layout zones, higher = higher on screen.
    #
    # With invert_xaxis():
    #   screen-right = low  data-x  (best ranks)
    #   screen-left  = high data-x  (worst ranks)
    #
    # Text alignment rules (inverted axis!):
    #   ha="left"  → text grows toward LOWER data-x → RIGHTWARD on screen
    #   ha="right" → text grows toward HIGHER data-x → LEFTWARD on screen
    # =================================================================

    rank_lo, rank_hi = float(min(ranks)), float(max(ranks))
    tick_lo = int(np.floor(rank_lo))
    tick_hi = int(np.ceil(rank_hi))

    # --- Y positions ---
    axis_y = 10.0
    label_spacing = 1.2
    label_top_y = axis_y - 1.2
    n_rows = max(half, k - half)
    label_bot_y = label_top_y - (n_rows - 1) * label_spacing

    clique_top_y = label_bot_y - 1.2
    clique_spacing = 0.6
    clique_bot_y = clique_top_y - max(n_cliques - 1, 0) * clique_spacing

    # --- X positions (use tick bounds to ensure axis covers all marks) ---
    # Right-side labels (best): anchor far right (low data-x), ha="left"
    # Left-side labels (worst): anchor far left (high data-x), ha="right"
    right_anchor_x = tick_lo - 1.2   # far right on screen
    left_anchor_x = tick_hi + 1.2    # far left on screen

    x_pad = 2.0  # extra padding for text
    x_lo = tick_lo - x_pad
    x_hi = tick_hi + x_pad
    y_lo = clique_bot_y - 0.8
    y_hi = axis_y + 1.8

    # --- Figure ---
    fig_w = 8.0
    fig_h = max(3.5, 2.0 + 0.6 * k + 0.4 * n_cliques)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.invert_xaxis()

    # =================================================================
    # 1. RANK AXIS + CD BAR
    # =================================================================
    ax.plot([tick_lo - 0.3, tick_hi + 0.3], [axis_y, axis_y],
            color="black", linewidth=1.2, solid_capstyle="butt")
    for x in range(tick_lo, tick_hi + 1):
        ax.plot([x, x], [axis_y, axis_y + 0.3], color="black", linewidth=0.8)
        ax.text(x, axis_y + 0.45, str(x), ha="center", va="bottom", fontsize=9)

    # CD bar — placed above the axis, aligned to the right (best-rank end)
    cd_y = axis_y + 1.1
    cd_x0 = rank_lo
    cd_x1 = rank_lo + cd  # extends toward worse ranks
    ax.plot([cd_x0, cd_x1], [cd_y, cd_y], color="black", linewidth=2.0)
    ax.plot([cd_x0, cd_x0], [cd_y - 0.12, cd_y + 0.12],
            color="black", linewidth=2.0)
    ax.plot([cd_x1, cd_x1], [cd_y - 0.12, cd_y + 0.12],
            color="black", linewidth=2.0)
    ax.text((cd_x0 + cd_x1) / 2, cd_y + 0.22, f"CD = {cd:.2f}",
            ha="center", va="bottom", fontsize=8)

    # =================================================================
    # 2. ALGORITHM LABELS + ELBOW CONNECTORS
    # =================================================================
    # Best half → right side; worst half → left side.
    # Elbow: vertical drop from axis, then horizontal toward the label.
    # Text faces OUTWARD (away from diagram centre).

    for i, (name, r) in enumerate(zip(names, ranks)):
        if i < half:
            # --- Right side (best, low rank) ---
            label_y = label_top_y - i * label_spacing
            # Vertical drop
            ax.plot([r, r], [axis_y, label_y],
                    color="black", linewidth=0.6)
            # Horizontal toward right margin
            ax.plot([r, right_anchor_x + 0.35], [label_y, label_y],
                    color="black", linewidth=0.6)
            # Label: ha="left" → text extends rightward on screen (AWAY)
            ax.text(right_anchor_x + 0.25, label_y,
                    f"{name} ({r:.2f})",
                    ha="left", va="center", fontsize=9)
        else:
            # --- Left side (worst, high rank) ---
            j = i - half
            label_y = label_top_y - j * label_spacing
            # Vertical drop
            ax.plot([r, r], [axis_y, label_y],
                    color="black", linewidth=0.6)
            # Horizontal toward left margin
            ax.plot([r, left_anchor_x - 0.35], [label_y, label_y],
                    color="black", linewidth=0.6)
            # Label: ha="right" → text extends leftward on screen (AWAY)
            ax.text(left_anchor_x - 0.25, label_y,
                    f"{name} ({r:.2f})",
                    ha="right", va="center", fontsize=9)

    # =================================================================
    # 3. CLIQUE BARS (thick lines connecting non-different groups)
    # =================================================================
    # Each clique bar sits below the labels. Thin vertical guides drop
    # from the clique endpoints to help the reader trace which rank
    # positions are connected.

    for idx, (a, b) in enumerate(maximal):
        cy = clique_top_y - idx * clique_spacing
        # Thick horizontal bar
        ax.plot([a, b], [cy, cy],
                color="black", linewidth=4.0, solid_capstyle="round")
        # Thin vertical guides from label zone to clique bar
        guide_top = label_bot_y - 0.3
        ax.plot([a, a], [guide_top, cy], color="gray", linewidth=0.4,
                linestyle=":")
        ax.plot([b, b], [guide_top, cy], color="gray", linewidth=0.4,
                linestyle=":")

    # =================================================================
    # CLEANUP
    # =================================================================
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, pad=10)

    save_and_close(fig, out_path)
    return out_path
