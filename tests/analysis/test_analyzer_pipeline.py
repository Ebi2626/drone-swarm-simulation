"""Integration test pełnego pipeline'u `ExperimentAnalyzer`.

Tworzymy syntetyczny eksperyment z 2 algorytmami × 1 środowisko × 4 seedy,
populate'ujemy DB przez `ExperimentAggregator`, a następnie odpalamy
`ExperimentAnalyzer.analyze`. Weryfikujemy, że kluczowe artefakty się
materializują i są nie-puste.
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

# Force matplotlib Agg backend zanim ExperimentAnalyzer go zaimportuje.
os.environ.setdefault("MPLBACKEND", "Agg")


def _write_run(exp_dir: Path, run_dir_name: str, algorithm_pair_seed: int,
               last_gen: np.ndarray) -> None:
    """Minimalny katalog runa kompatybilny z `populate_database`."""
    run_dir = exp_dir / run_dir_name
    run_dir.mkdir(parents=True)

    with (run_dir / "world_boundaries.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Axis", "Dimension", "Min_Bound", "Max_Bound", "Center"])
        for axis in ("X", "Y", "Z"):
            w.writerow([axis, 100, 0, 100, 50])

    # 2 drony, 10 sampli, constant velocity.
    with (run_dir / "trajectories.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "drone_id", "x", "y", "z",
                    "roll", "pitch", "yaw", "vx", "vy", "vz"])
        for i in range(10):
            t = round(i * 0.1, 3)
            x = round(i * 0.1, 3)
            w.writerow([t, 0, x, 0.0, 5.0, 0, 0, 0, 1.0, 0.0, 0.0])
            w.writerow([t, 1, x, 5.0, 5.0, 0, 0, 0, 1.0, 0.0, 0.0])

    # h5 z `last_gen` Pareto front w gen=1.
    pop, n_obj = last_gen.shape
    obj = np.full((2, pop, n_obj), 999.0, dtype=np.float64)
    obj[1] = last_gen
    h5_dir = run_dir / "optimization_history"
    h5_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_dir / "optimization_history.h5", "w") as h:
        h.create_dataset("objectives_matrix", data=obj)


@pytest.fixture
def populated_experiment(tmp_path: Path):
    from src.analysis.ExperimentAggregator import ExperimentAggregator

    exp_dir = tmp_path / "exp"

    # 2 algorytmy × 4 seedy. Algorytm "msffoa" generalnie z lepszymi (niższymi)
    # objective values; "ssa" z gorszymi — żeby Friedman/Wilcoxon miały co
    # rozróżnić.
    rng = np.random.default_rng(0)
    for seed in range(4):
        front_msffoa = np.array([
            [1.0 + 0.1 * seed, 4.0 + 0.05 * seed, 2.0],
            [2.0, 2.0 + 0.05 * seed, 1.5],
            [3.0 + 0.1 * seed, 1.0, 2.5],
        ])
        front_ssa = np.array([
            [3.0 + 0.1 * seed, 6.0 + 0.05 * seed, 3.0],
            [4.0, 4.0 + 0.05 * seed, 3.5],
            [5.0 + 0.1 * seed, 3.0, 4.5],
        ])
        _write_run(
            exp_dir,
            f"msffoa_forest_msffoa_seed{seed}",
            seed,
            front_msffoa,
        )
        _write_run(
            exp_dir,
            f"ssa_forest_ssa_seed{seed}",
            seed,
            front_ssa,
        )

    ExperimentAggregator().aggregate(exp_dir)
    yield exp_dir


class TestAnalyzerPipeline:
    def test_produces_tables_and_plots(self, populated_experiment: Path) -> None:
        from src.analysis.analyzer import ExperimentAnalyzer

        out = ExperimentAnalyzer().analyze(populated_experiment)
        assert out.exists()
        tables = out / "tables"
        plots = out / "plots"
        assert tables.exists() and plots.exists()

        # ≥1 summary CSV powinno być wygenerowane (np. dla hypervolume).
        summary_csvs = list(tables.glob("summary_*.csv"))
        assert summary_csvs, "Brak żadnego summary_<metric>.csv"

        # Boxplot dla forest+jakaś metryka.
        boxplots = list((plots / "boxplots").glob("boxplot_forest_*.pdf"))
        assert boxplots, "Brak boxplot PDF"

        # Convergence curves
        convergence = list((plots / "convergence").glob("convergence_forest_*.pdf"))
        assert convergence, "Brak convergence PDF"

        # Bar success rate
        bar = list((plots / "bar").glob("bar_forest_*.pdf"))
        assert bar, "Brak bar PDF"

    def test_friedman_csv_has_expected_columns(self, populated_experiment: Path) -> None:
        from src.analysis.analyzer import ExperimentAnalyzer
        import pandas as pd

        out = ExperimentAnalyzer().analyze(populated_experiment)
        # Per-env split: pliki mają prefiks `{env}_` (np. `forest_friedman_*.csv`).
        friedman_csvs = list((out / "tables").glob("*friedman_*.csv"))
        if not friedman_csvs:
            pytest.skip("Brak *friedman_*.csv (zbyt mało metryk z >= 2 alg).")
        df = pd.read_csv(friedman_csvs[0])
        for col in ("optimizer", "avg_rank", "statistic", "p_value", "cd_nemenyi"):
            assert col in df.columns

    def test_idempotent_rerun(self, populated_experiment: Path) -> None:
        from src.analysis.analyzer import ExperimentAnalyzer

        out1 = ExperimentAnalyzer().analyze(populated_experiment)
        files_1 = sorted(p.relative_to(out1) for p in out1.rglob("*.csv"))
        out2 = ExperimentAnalyzer().analyze(populated_experiment)
        files_2 = sorted(p.relative_to(out2) for p in out2.rglob("*.csv"))
        assert files_1 == files_2  # ten sam zestaw artefaktów
