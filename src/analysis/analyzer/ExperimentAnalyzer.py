"""ExperimentAnalyzer — orkestrator pełnej analizy porównawczej.

Wymaga populated `analysis.db` (zob. `ExperimentAggregator`). Generuje:
- `analysis_output/tables/` — CSV + LaTeX summary, Wilcoxon, Friedman, A12.
- `analysis_output/plots/{convergence, boxplots, cd_diagrams, pareto,
  scatter, bar, rankings}/`.

Reference: Demšar (2006); Arcuri & Briand (2014); Riquelme et al. (2015).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.analysis.analyzer.exporters import export_csv, export_latex
from src.analysis.analyzer.metric_extractor import MetricExtractor
from src.analysis.analyzer.plots import (
    plot_boxplots,
    plot_cd_diagram,
    plot_convergence,
    plot_failure_rate_bars,
    plot_pareto_projections,
    plot_ranking_heatmap,
    plot_scatter,
    plot_success_and_collision_bars,
)
from src.analysis.analyzer.statistical_tests import (
    friedman_with_nemenyi,
    summary_with_ci,
    vargha_delaney_a12,
    wilcoxon_pairwise,
)


# Próg dla "low-power warning" w summary (Demšar 2006 §4: Wilcoxon
# z N<6 ma p-value floor 1/2^N — z N=4 to 0.0625, z N=5 to 0.0312).
LOW_POWER_N_THRESHOLD = 6


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Konfiguracja: które metryki są "higher=better"
# ----------------------------------------------------------------------

# Lower is better dla większości MOO indicator-ów (GD/IGD+/spread/spacing/R2).
# Higher is better dla HV i success rate.
HIGHER_IS_BETTER = {
    "hypervolume": True,
    "hypervolume_normalized": True,
    "success": True,
    "feasible_ratio": True,
    "nondominated_count": True,
    "front_size_last_gen": True,
    "min_inter_uav_distance_m": True,
    "mean_inter_uav_distance_m": True,
}

# Default zestawy metryk per faza analizy.
#
# OFFLINE: zależą tylko od (optimizer, environment, seed). Wartości
# dla danego (opt, env, seed) są identyczne dla 4 różnych avoidance
# algorithms — pseudo-replication grozi inflated N. ExperimentAnalyzer
# deduplicuje per (env, optimizer, seed) PRZED testami statystycznymi,
# więc Friedman/Wilcoxon używają prawdziwego N = n_envs × n_seeds.
#
# ONLINE: zależą od (optimizer, environment, seed, avoidance) — pełny
# product (n_avoid × n_env × n_seed) tworzy prawdziwe dataset.
OFFLINE_METRICS = (
    "hypervolume",
    "hypervolume_normalized",
    "igd_plus",
    "gd_final",
    "spread_final",
    "spacing_final",
    "r2_final",
    "convergence_speed_gen",
    "auc_best_so_far",
    "final_objective",
    "front_size_last_gen",
)

ONLINE_METRICS = (
    "collision_count",
    "evasion_event_count",
    "min_inter_uav_distance_m",
    "mean_inter_uav_distance_m",
    "total_inter_uav_safety_violations",
    "mean_energy_indicator",
    "mean_smoothness_indicator",
)

ITER_METRICS = (
    "hypervolume",
    "hypervolume_normalized",
    "best_so_far",
    "feasible_ratio",
    "front_size",
    "igd_plus",
    "gd",
    "spread",
    "spacing",
    "r2_indicator",
)


@dataclass
class AnalyzerConfig:
    metrics_offline: Iterable[str] = field(default_factory=lambda: OFFLINE_METRICS)
    metrics_online: Iterable[str] = field(default_factory=lambda: ONLINE_METRICS)
    metrics_iter: Iterable[str] = field(default_factory=lambda: ITER_METRICS)
    bootstrap_resamples: int = 10000
    output_subdir: str = "analysis_output"


class ExperimentAnalyzer:
    """High-level facade — `analyze(experiment_dir)` produkuje pełny raport."""

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        self.cfg = config or AnalyzerConfig()

    def analyze(self, experiment_dir: str | Path) -> Path:
        experiment_dir = Path(experiment_dir).expanduser().resolve()
        db_path = experiment_dir / "analysis.db"
        if not db_path.exists():
            raise FileNotFoundError(
                f"Brak analysis.db w {experiment_dir}. Uruchom najpierw "
                f"ExperimentAggregator.aggregate(experiment_dir)."
            )

        out_dir = experiment_dir / self.cfg.output_subdir
        tables_dir = out_dir / "tables"
        plots_dir = out_dir / "plots"

        extractor = MetricExtractor(db_path)
        df_run = extractor.run_summary()
        df_iter = extractor.iteration_history(metrics=list(self.cfg.metrics_iter))
        try:
            df_online = extractor.online_summary()
        except Exception:  # widok może nie istnieć dla minimalistycznych runów
            df_online = pd.DataFrame()
        try:
            df_pareto = extractor.pareto_front_last_gen()
        except Exception:
            df_pareto = pd.DataFrame()

        if df_run.empty:
            logger.warning("ExperimentAnalyzer: pusty run_summary — abort.")
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir

        # Compute is_failure flag używany w summary tables i bar plots.
        df_run = _compute_failure_flag(df_run)

        # ============================================================
        # 1. Tabele zbiorcze
        # ============================================================
        self._build_tables(df_run, df_online, tables_dir)

        # ============================================================
        # 2. Wykresy
        # ============================================================
        self._build_plots(df_run, df_iter, df_pareto, plots_dir)

        # ============================================================
        # 3. Raport zbiorczy (PDF + Markdown)
        # ============================================================
        from src.analysis.analyzer.report_generator import ReportGenerator

        try:
            report_gen = ReportGenerator(experiment_dir)
            md_path, pdf_path = report_gen.generate()
            logger.info("Report: %s, %s", md_path, pdf_path)
        except Exception:
            logger.warning("ReportGenerator failed — skipping report.", exc_info=True)

        logger.info(f"ExperimentAnalyzer: wygenerowano analizę w {out_dir}")
        return out_dir

    # ------------------------------------------------------------------
    # Helpery
    # ------------------------------------------------------------------

    def _build_tables(
        self,
        df_run: pd.DataFrame,
        df_online: pd.DataFrame,
        tables_dir: Path,
    ) -> None:
        # Offline: dedup per (env, optimizer, seed) — wartości metryk offline
        # nie zależą od `avoidance`, ale full join daje 4 duplikaty per seed
        # (po jednym dla każdego avoidance). Wilcoxon/Friedman traktowałyby
        # je jako niezależne obserwacje → false-positive risk (Demšar 2006
        # §3.1: testy zakładają niezależne datasets).
        df_offline = _dedup_offline(df_run)
        offline_blocks = ("environment", "seed")

        for metric in self.cfg.metrics_offline:
            if metric not in df_offline.columns:
                continue
            sub = df_offline.dropna(subset=[metric])
            if sub.empty:
                continue
            higher = HIGHER_IS_BETTER.get(metric, False)
            self._emit_metric_tables(
                sub, metric, higher, offline_blocks, tables_dir,
            )

        # Online: pełny product (env, opt, seed, avoidance) — `avoidance`
        # wpływa na metryki online, więc jest legitymowanym factor-em.
        online_blocks = ("environment", "seed", "avoidance")
        for metric in self.cfg.metrics_online:
            if metric not in df_run.columns:
                continue
            sub = df_run.dropna(subset=[metric])
            if sub.empty:
                continue
            higher = HIGHER_IS_BETTER.get(metric, False)
            self._emit_metric_tables(
                sub, metric, higher, online_blocks, tables_dir,
                file_prefix="online_",
            )

        # Failure rate per (env, opt) — odporne uzupełnienie mean/median
        # statistik (tail-risk). Split offline/online — patrz docstring
        # `_compute_failure_flag`.
        for flag_col, suffix, caption in (
            (
                "is_offline_failure",
                "offline",
                "Offline failure rate (HV=0 OR front_size_last_gen=0).",
            ),
            (
                "is_online_failure",
                "online",
                "Online failure rate (collision_count>0).",
            ),
        ):
            if flag_col not in df_run.columns:
                continue
            # Offline metryki nie zależą od `avoidance` — dedup'ujemy zanim
            # liczymy failure rate, inaczej n_runs = 8 zamiast 2 per (env,opt).
            base = _dedup_offline(df_run) if flag_col == "is_offline_failure" else df_run
            failure_summary = (
                base.groupby(["environment", "optimizer"], dropna=False)[flag_col]
                .agg(["count", "sum", "mean"])
                .reset_index()
                .rename(columns={
                    "count": "n_runs",
                    "sum": "n_failures",
                    "mean": "failure_rate",
                })
            )
            export_csv(failure_summary, tables_dir / f"failure_rate_{suffix}.csv")
            export_latex(
                failure_summary, tables_dir / f"failure_rate_{suffix}.tex",
                caption=caption,
                label=f"tab:failure-rate-{suffix}",
            )

        if not df_online.empty:
            export_csv(df_online, tables_dir / "online_summary.csv")

    def _emit_metric_tables(
        self,
        sub: pd.DataFrame,
        metric: str,
        higher: bool,
        block_cols: tuple[str, ...],
        tables_dir: Path,
        file_prefix: str = "",
    ) -> None:
        """Generuje summary/wilcoxon/friedman/a12 dla pojedynczej metryki.
        `block_cols` definiuje "datasets" w sensie Demšar (2006).
        """
        summary = summary_with_ci(
            sub, metric=metric,
            group_cols=("environment", "optimizer"),
            n_resamples=self.cfg.bootstrap_resamples,
        )
        if not summary.empty:
            # Low-power warning: jeśli min(n) < threshold, dorzuć kolumnę
            # ostrzegawczą (Demšar 2006 §4: Wilcoxon p_min = 1/2^N).
            summary["low_power_warning"] = summary["n"] < LOW_POWER_N_THRESHOLD
            export_csv(summary, tables_dir / f"{file_prefix}summary_{metric}.csv")
            export_latex(
                summary, tables_dir / f"{file_prefix}summary_{metric}.tex",
                caption=f"Summary statistics — {metric}",
                label=f"tab:{file_prefix}summary-{metric}",
            )

        # Wilcoxon pairwise
        try:
            pairs = wilcoxon_pairwise(sub, metric=metric, block_cols=block_cols)
            if pairs:
                df_pairs = pd.DataFrame([p.__dict__ for p in pairs])
                export_csv(df_pairs, tables_dir / f"{file_prefix}wilcoxon_{metric}.csv")
        except (ValueError, KeyError):
            pass

        # Friedman + Nemenyi
        try:
            fr = friedman_with_nemenyi(
                sub, metric=metric, higher_is_better=higher, block_cols=block_cols,
            )
            rank_rows = pd.DataFrame(
                {
                    "optimizer": list(fr.average_ranks.keys()),
                    "avg_rank": list(fr.average_ranks.values()),
                }
            )
            rank_rows["statistic"] = fr.statistic
            rank_rows["p_value"] = fr.p_value
            rank_rows["cd_nemenyi"] = fr.cd_nemenyi
            rank_rows["n_datasets"] = fr.n_datasets
            export_csv(rank_rows, tables_dir / f"{file_prefix}friedman_{metric}.csv")
        except ValueError:
            pass

        # A12
        try:
            a12 = vargha_delaney_a12(sub, metric=metric, higher_is_better=higher)
            if a12:
                export_csv(
                    pd.DataFrame([a.__dict__ for a in a12]),
                    tables_dir / f"{file_prefix}a12_{metric}.csv",
                )
        except (ValueError, KeyError):
            pass

    def _build_plots(
        self,
        df_run: pd.DataFrame,
        df_iter: pd.DataFrame,
        df_pareto: pd.DataFrame,
        plots_dir: Path,
    ) -> None:
        if not df_iter.empty:
            plot_convergence(
                df_iter, plots_dir / "convergence",
                metrics=[m for m in self.cfg.metrics_iter if m in df_iter.columns],
                higher_is_better={m: HIGHER_IS_BETTER.get(m, False) for m in self.cfg.metrics_iter},
            )

        # Boxploty offline na zdedupowanym DF (jedna obserwacja per
        # (env, opt, seed)); online — pełny.
        df_offline_for_box = _dedup_offline(df_run)
        plot_boxplots(
            df_offline_for_box, plots_dir / "boxplots",
            metrics=[m for m in self.cfg.metrics_offline if m in df_offline_for_box.columns],
        )
        plot_boxplots(
            df_run, plots_dir / "boxplots",
            metrics=[m for m in self.cfg.metrics_online if m in df_run.columns],
        )

        plot_success_and_collision_bars(df_run, plots_dir / "bar")
        if "is_offline_failure" in df_run.columns:
            plot_failure_rate_bars(
                _dedup_offline(df_run), plots_dir / "bar",
                failure_col="is_offline_failure",
                file_suffix="offline",
            )
        if "is_online_failure" in df_run.columns:
            plot_failure_rate_bars(
                df_run, plots_dir / "bar",
                failure_col="is_online_failure",
                file_suffix="online",
            )
        if not df_pareto.empty:
            plot_pareto_projections(df_pareto, plots_dir / "pareto")

        if not df_iter.empty:
            plot_scatter(df_run, df_iter, plots_dir / "scatter")

        # CD diagrams + ranking heatmap per metryka. Offline używa
        # zdedupowanego (env, opt, seed); online — pełnego (env, seed, avoidance).
        df_offline = _dedup_offline(df_run)
        for metric in self.cfg.metrics_offline:
            if metric not in df_offline.columns:
                continue
            sub = df_offline.dropna(subset=[metric])
            if sub.empty:
                continue
            higher = HIGHER_IS_BETTER.get(metric, False)
            try:
                fr = friedman_with_nemenyi(
                    sub, metric=metric, higher_is_better=higher,
                    block_cols=("environment", "seed"),
                )
                if fr.cd_nemenyi is not None:
                    plot_cd_diagram(
                        fr.average_ranks, fr.cd_nemenyi,
                        plots_dir / "cd_diagrams" / f"cd_{metric}.pdf",
                        title=f"CD — {metric} ({'higher=better' if higher else 'lower=better'})",
                    )
            except ValueError:
                pass
            plot_ranking_heatmap(
                df_offline, metric,
                plots_dir / "rankings" / f"ranking_{metric}.pdf",
                higher_is_better=higher,
            )


# ----------------------------------------------------------------------
# Module-level helpery
# ----------------------------------------------------------------------


def _dedup_offline(df_run: pd.DataFrame) -> pd.DataFrame:
    """Dedup DataFrame per (environment, optimizer, seed). Wartości metryk
    offline (HV, IGD+, GD, ...) zależą TYLKO od (opt, env, seed) — `avoidance`
    jest irrelewantny dla offline phase. Bez dedup'u pivot z `block_cols=
    (env, seed)` aggreguje 4 identyczne wartości per (env, opt, seed) z
    `aggfunc=mean` (no-op), ALE każdy row to wciąż osobne dataset z
    perspektywy Wilcoxon → fałszywie zawyżone N.

    Reference: Demšar (2006) §3.1 — testy zakładają niezależne datasets.
    """
    if df_run.empty:
        return df_run
    keys = [c for c in ("environment", "optimizer", "seed") if c in df_run.columns]
    if not keys:
        return df_run
    return df_run.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)


def _compute_failure_flag(df_run: pd.DataFrame) -> pd.DataFrame:
    """Dodaje binarne kolumny `is_offline_failure` i `is_online_failure` per run.

    OFFLINE failure ⟺ przynajmniej jeden:
    - `front_size_last_gen` ∈ {NULL, 0} — brak feasible non-dominated solutions
    - `hypervolume` ∈ {NULL, 0} — front nie dominuje r* (degenerate / wszystkie
      feasible solutions outside ref-point box)

    ONLINE failure ⟺ `collision_count > 0`.

    Powody splitu (Kamień 2): plan analizy raportował q25=0 dla HV msffoa
    urban → tail-failure offline phase (algorytm nie znalazł żadnych
    feasible-ND rozwiązań pokrywających r*). Online collisions to inny
    failure mode (avoidance phase) — łączenie obu obscure'owałoby root
    cause.

    Reference: Liefooghe & Verel (2014). Failure rate to standardowy
    proxy tail-risk algorytmu, uzupełniający mean/median (które maskują
    katastrofalne runy).
    """
    if df_run.empty:
        return df_run
    df = df_run.copy()
    fs = df.get("front_size_last_gen")
    hv = df.get("hypervolume")
    cc = df.get("collision_count")
    offline_fail = pd.Series(False, index=df.index)
    if fs is not None:
        offline_fail = offline_fail | fs.isna() | (fs == 0)
    if hv is not None:
        offline_fail = offline_fail | hv.isna() | (hv == 0)
    df["is_offline_failure"] = offline_fail.astype(int)

    online_fail = pd.Series(False, index=df.index)
    if cc is not None:
        online_fail = online_fail | (cc.fillna(0) > 0)
    df["is_online_failure"] = online_fail.astype(int)
    return df
