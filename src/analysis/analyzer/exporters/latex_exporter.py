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
    """Eksport DataFrame do .tex (booktabs).

    Args:
        df: DataFrame to export.
        out_path: docelowa ścieżka .tex.
        caption: opis tabeli (\\caption).
        label: identyfikator tabeli (\\label).
        float_format: format liczb zmiennoprzecinkowych.
        index: czy włączyć index.
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
