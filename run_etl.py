"""ETL pipeline: agregacja surowych wynikow eksperymentu + analiza statystyczna.

Uzycie:
    python run_etl.py results/exp_20260506_377919c3_per_env_test

Pipeline:
    1. ExperimentAggregator  — zbiera CSV/HDF5 z runow do analysis.db
    2. ExperimentAnalyzer    — testy statystyczne, wykresy, raport (MD + PDF)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.analysis.ExperimentAggregator import ExperimentAggregator
from src.analysis.analyzer.ExperimentAnalyzer import ExperimentAnalyzer


def main(experiment_dir: str) -> None:
    path = Path(experiment_dir)
    if not path.is_dir():
        print(f"Blad: katalog nie istnieje: {path}", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    agg = ExperimentAggregator()
    agg.aggregate(str(path))

    anl = ExperimentAnalyzer()
    anl.analyze(str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ETL pipeline: agregacja + analiza eksperymentu",
    )
    parser.add_argument(
        "experiment_dir",
        help="Sciezka do katalogu eksperymentu (np. results/exp_20260506_...)",
    )
    args = parser.parse_args()
    main(args.experiment_dir)
