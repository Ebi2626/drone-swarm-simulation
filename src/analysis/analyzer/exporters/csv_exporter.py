"""CSV exporter — `pandas.to_csv` z konsystentnymi opcjami."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_csv(df: pd.DataFrame, out_path: Path, float_format: str = "%.6f") -> None:
    """Zapisz `df` do CSV w `out_path` (tworzy katalogi nadrzędne).

    Args:
        df: DataFrame do zapisu.
        out_path: Ścieżka docelowego pliku.
        float_format: Format liczb zmiennoprzecinkowych (np. `%.6f`).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format=float_format)
