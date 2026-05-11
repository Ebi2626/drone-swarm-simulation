"""LaTeX exporter — `pandas.to_latex` z konwencjami booktabs.

Reference: Mittelbach & Goossens (2004) "The LaTeX Companion", §3.5
(booktabs `\\toprule/\\midrule/\\bottomrule`).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_latex(
    df: pd.DataFrame,
    out_path: Path,
    caption: str | None = None,
    label: str | None = None,
    float_format: str = "%.4f",
    index: bool = False,
) -> None:
    """Zapisz `df` jako tabelę LaTeX (`booktabs`) w `out_path`.

    Args:
        df: DataFrame do eksportu.
        out_path: Docelowa ścieżka `.tex` (katalogi nadrzędne tworzone automatycznie).
        caption: Treść `\\caption` lub `None`, gdy podpis ma być pominięty.
        label: Identyfikator `\\label` lub `None`.
        float_format: Format liczb zmiennoprzecinkowych.
        index: Czy włączyć indeks DataFrame jako kolumnę.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = df.to_latex(
        index=index,
        float_format=float_format,
        caption=caption,
        label=label,
        escape=True,
        bold_rows=False,
    )
    out_path.write_text(body, encoding="utf-8")
