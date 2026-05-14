"""Wykresy zbieżności — mean ± std band per algorytm, per środowisko.

Standard w meta-heuristics literature: `Engelbrecht (2007)` Sec 16, plus
`Beyer & Schwefel (2002)` "Evolution strategies".
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.analysis.analyzer.plots._common import make_figure, save_and_close


def plot_convergence(
    df_iter: pd.DataFrame,
    out_dir: Path,
    metrics: Iterable[str] = ("hypervolume", "best_so_far", "feasible_ratio"),
    higher_is_better: dict[str, bool] | None = None,
) -> list[Path]:
    """Per (environment, metric): mean curve ± 1 std band, kolor=optimizer.

    Args:
        df_iter: tidy DataFrame z `MetricExtractor.iteration_history`.
            Wymagane kolumny: optimizer, environment, seed, iteration,
            <metric>.
        out_dir: katalog wyjściowy. Stworzy się jeśli brak.
        metrics: które metryki rysować.
        higher_is_better: nieużywane bezpośrednio — to tylko adnotacja
            kierunku w tytule. Default: HV→True, reszta False.

    Returns:
        Lista wygenerowanych ścieżek (PDF).
    """
    if higher_is_better is None:
        higher_is_better = {
            "hypervolume": True,
            "feasible_ratio": True,
            "nondominated_ratio": True,
        }

    out_paths: list[Path] = []
    envs = sorted(df_iter["environment"].dropna().unique())

    for env in envs:
        sub_env = df_iter[df_iter["environment"] == env]
        for metric in metrics:
            if metric not in sub_env.columns:
                continue
            data = sub_env.dropna(subset=[metric])
            if data.empty:
                continue

            fig, ax = make_figure()
            for optimizer, sub in data.groupby("optimizer"):
                pivot = sub.pivot_table(
                    index="iteration", columns="seed", values=metric, aggfunc="mean"
                )
                if pivot.empty:
                    continue
                xs = pivot.index.values
                vals = pivot.values
                mean = np.nanmean(vals, axis=1)
                std = np.nanstd(vals, axis=1, ddof=0)
                line, = ax.plot(xs, mean, label=str(optimizer))
                ax.fill_between(xs, mean - std, mean + std, alpha=0.18, color=line.get_color())

            direction = "higher=better" if higher_is_better.get(metric, False) else "lower=better"
            ax.set_xlabel("generation")
            ax.set_ylabel(metric)
            ax.set_title(f"{env} — {metric} ({direction})")
            ax.legend(frameon=False)

            out_path = out_dir / f"convergence_{env}_{metric}.pdf"
            save_and_close(fig, out_path)
            out_paths.append(out_path)

    return out_paths


def plot_online_convergence(
    db_path: Path,
    out_dir: Path,
) -> list[Path]:
    """Per-environment krzywa zbieżności online optimizer'a (§3.1.3.2).

    Każde zdarzenie unikowe (trigger) uruchamia osobne zadanie optymalizacji
    z budżetem czasowym (≈0.5 s). Trace per generacja jest w
    `online_convergence_traces`. Agregacja: mean ± std `best_fitness` per
    (environment, algorithm, generation) across wszystkich triggerów ze
    wszystkich runów.

    UWAGA na interpretację: różne triggery mają różną liczbę zakończonych
    generacji (zależnie od trudności zadania i tempa algorytmu w budżecie).
    Mean dla generacji `g` to średnia po WSZYSTKICH triggerach które osiągnęły
    przynajmniej `g`-tą generację. Na późniejszych generacjach `n` spada
    (krzywa szumna) — to zamierzone, oddaje rzeczywistość budżetu czasowego.

    Args:
        db_path: Ścieżka do `analysis.db`.
        out_dir: Katalog wyjściowy plotów.

    Returns:
        Lista wygenerowanych ścieżek PDF (po jednej per environment).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []

    query = """
        SELECT
            r.environment,
            r.avoidance_algo                  AS algorithm,
            t.run_id,
            t.drone_id,
            t.trigger_time,
            t.generation,
            t.best_fitness
        FROM online_convergence_traces t
        JOIN runs r ON r.run_id = t.run_id
        WHERE t.best_fitness IS NOT NULL
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        return out_paths

    envs = sorted(df["environment"].dropna().unique())
    for env in envs:
        sub_env = df[df["environment"] == env]
        if sub_env.empty:
            continue

        fig, ax = make_figure()
        for algo, sub in sub_env.groupby("algorithm"):
            # Każdy trigger to (run_id, drone_id, trigger_time). Per
            # generation agregacja: median + IQR (Q1, Q3) across triggerów.
            # Median+IQR zamiast mean±std — odporne na ekstremalne outliers
            # typowe dla początkowych generacji (random init może dać
            # fitness rzędu 1e9 dla niefeasible candidates).
            agg = sub.groupby("generation")["best_fitness"].agg(
                median="median",
                q1=lambda x: x.quantile(0.25),
                q3=lambda x: x.quantile(0.75),
                n="count",
            ).reset_index()
            # Pomijamy generations z tylko 1 obserwacją — IQR niereprezentatywne.
            agg = agg[agg["n"] >= 2]
            if agg.empty:
                continue
            xs = agg["generation"].values
            median = agg["median"].values
            q1 = agg["q1"].values
            q3 = agg["q3"].values
            line, = ax.plot(xs, median, label=str(algo))
            ax.fill_between(xs, q1, q3, alpha=0.18, color=line.get_color())

        ax.set_xlabel("generation (online task)")
        ax.set_ylabel("best_fitness — mediana per generacja (lower=better)")
        ax.set_title(
            f"{env} — Online optimizer convergence "
            f"(§3.1.3.2 Krzywa optymalizacji)"
        )
        ax.legend(frameon=False, title="avoidance algo")

        out_path = out_dir / f"online_convergence_{env}.pdf"
        save_and_close(fig, out_path)
        out_paths.append(out_path)

    return out_paths
