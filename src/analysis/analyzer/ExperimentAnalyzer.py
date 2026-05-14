"""ExperimentAnalyzer — orkestrator pełnej analizy porównawczej.

Wymaga populated `analysis.db` (zob. `ExperimentAggregator`). Generuje:
- `analysis_output/tables/` — CSV + LaTeX: summary, Friedman + Nemenyi, A12,
  failure_rate z Wilson 95% CI.
- `analysis_output/plots/{convergence, boxplots, cd_diagrams, pareto,
  scatter, bar, rankings}/`.

Statystyka: trzy narzędzia (Friedman + Nemenyi, Vargha-Delaney A12,
Wilson 95% CI) — pełne uzasadnienie metodologiczne w
`reports/statistical_tests_methodology.md`.
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
)
from src.analysis.analyzer.statistical_tests import (
    friedman_with_nemenyi,
    summary_stats,
    vargha_delaney_a12,
    wilson_proportion_ci,
)


# Próg dla "low-power warning" w summary. Friedman z bardzo małym N traci
# moc statystyczną (rozkład χ² jest asymptotyczny). Przy N<6 wynik testu
# należy interpretować z ostrożnością.
LOW_POWER_N_THRESHOLD = 6


logger = logging.getLogger(__name__)


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
    # §3.1.3.3: proporcje sukcesów powrotu na nominalną trajektorię.
    # `rejoin_completion_rate` — metryka GŁÓWNA (mianownik = epizody);
    # `rejoin_success_rate` — surowa (mianownik = wszystkie triggery
    # włącznie z pending z replanu).
    "rejoin_success_rate": True,
    "rejoin_completion_rate": True,
    # Bezpieczeństwo fazy online (collision-based, per-run binary). Identyczna
    # semantyka co §3.2 — różni się tylko prezentacją (Mean/Std/Median vs Wilson CI).
    "online_success_rate": True,
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
    # MOO / Pareto-aware (drugorzędne — głównie diagnostyka NSGA-III)
    "hypervolume",
    "hypervolume_normalized",
    "igd_plus",
    "gd_final",
    "spread_final",
    "spacing_final",
    "r2_final",
    "front_size_last_gen",
    # Konwergencja / efektywność
    "convergence_speed_gen",
    "auc_best_so_far",
    "total_wallclock_offline_s",
    # Skalarna jakość rozwiązania
    "final_objective",
    # Jakość zaplanowanej trajektorii — agregaty F-składowych
    "trajectory_length_f1",          # f1 (długość + odchylenie)
    "trajectory_smoothness_f2_f4",   # f2 + f4 (wysokość/kąt + zakręty)
    "trajectory_safety_f3_f5",       # f3 + f5 (zagrożenie + koordynacja)
    # Spójność roju — per docs/Praca magisterska.md §3.1.3.1, ocena jakości
    # offline-zaplanowanej trajektorii (MAE od 5 m NN-spacing). Wartość
    # mierzona w PyBullet podczas wykonywania trajektorii — z paired design
    # eksperymentu (opt == avoidance, 1 run per (env, opt, seed))
    # `_dedup_offline` jest no-op, więc traktowanie jako OFFLINE poprawne.
    "swarm_cohesion_deviation",
    # Bezpieczeństwo zaplanowanej trasy (zmierzone w PyBullet podczas tracking)
    "tracking_phase_collisions",     # kolizje poza okresem uniku — failure offline
)

ONLINE_METRICS = (
    # Bezpieczeństwo fazy online — JEDYNA metryka safety online:
    # `online_success_rate` (per-run binary: 1 = run bez kolizji w fazie
    # evasion, 0 = ≥1 kolizja). Renderowana w §2.1 (Mean/Std/Median/Q1/Q3),
    # §2.2 (Friedman), §2.3 (A12) ORAZ §3.2 (Wilson CI). Wartość kolumny
    # nadpisywana w `_compute_failure_flag` z `1 - is_online_failure`
    # (kolumna DB `run_metrics.online_success_rate` = `1 - BVR` używana
    # tylko wewnątrz DB do obliczenia SP1).
    # `evasion_phase_collisions` pozostaje jako surowy licznik dla §3.2 base.
    # `collision_count` i `evasion_event_count` jako audyt/informacyjne.
    "online_success_rate",
    "evasion_phase_collisions",
    "collision_count",                # raw total (informational, legacy)
    "evasion_event_count",
    "min_inter_uav_distance_m",
    "max_inter_uav_distance_m",
    "mean_inter_uav_distance_m",
    "total_inter_uav_safety_violations",
    # Jakość trajektorii fizycznej
    "mean_energy_indicator",
    "mean_smoothness_indicator",
    # Efektywność obliczeń online (audit; wallclock saturuje budżet
    # ~0.5s dla wszystkich algorytmów → ZERO discriminative info; pozostawione
    # w bazie dla audytowalności).
    "avg_wallclock_online_s",
    "max_wallclock_online_s",
    # §3.1.3.4 (docs/Praca magisterska.md) — ocena ALGORYTMÓW fazy ONLINE:
    #   - "Wartość optymalizacji w budżecie czasowym" → mean_online_best_fitness
    #   - "Prędkość optymalizacji" → mean_online_evaluations_completed (NFE,
    #      standard BBOB 2009; gens jako informacyjne)
    #   - "Niezawodność hard real-time" → budget_violation_rate (lower=lepiej)
    #   - "Efektywność łączona" → online_sp1 = RT_succ / (1 - BVR)
    #     (Success Performance 1, Auger & Hansen 2005). NOTE: kolumna DB
    #     `run_metrics.online_success_rate` (= 1 - BVR) jest używana wewnątrz
    #     DB tylko jako składnik SP1; na poziomie DataFrame ta sama nazwa
    #     reprezentuje collision-based safety (zob. komentarz wyżej).
    "mean_online_best_fitness",
    "mean_online_evaluations_completed",
    "mean_online_generations_completed",
    "budget_violation_rate",
    "online_sp1",
    # §3.1.3.3 (docs/Praca magisterska.md) — TRAJEKTORIA fazy online:
    #   - Geometria manewru: mean_evasion_arc_length_m
    #     (mean_evasion_plan_duration_s świadomie pominięte — silnie
    #      skorelowane z arc, powiela informację geometryczną; pozostaje
    #      w DB dla audytu)
    #   - Skuteczność powrotu: rejoin_quality (TOPSIS composite —
    #      odległość euklidesowa od ideału w 3D znormalizowanej przestrzeni
    #      [pos_err, vel_err, time_to_rejoin]; Hwang & Yoon 1981).
    #      Oryginalne 3 składniki (mean_pos_err_at_rejoin_m,
    #      mean_vel_err_at_rejoin_mps, mean_time_to_rejoin_s) pozostają
    #      w DB dla audytu, ale nie w głównej analizie.
    #   - Niezawodność: rejoin_completion_rate (PRIMARY),
    #     rejoin_success_rate (secondary, surowa z pending w mianowniku)
    "mean_evasion_arc_length_m",
    "rejoin_quality",
    "rejoin_completion_rate",
    "rejoin_success_rate",
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
    """Konfiguracja `ExperimentAnalyzer` — listy metryk i katalog wyjściowy.

    Args:
        metrics_offline: Metryki offline (offline phase, MOO indicators).
        metrics_online: Metryki online (collisions, distances, energy).
        metrics_iter: Metryki iteracyjne (krzywe konwergencji per generacja).
        output_subdir: Nazwa katalogu wyjściowego w katalogu eksperymentu.
    """
    metrics_offline: Iterable[str] = field(default_factory=lambda: OFFLINE_METRICS)
    metrics_online: Iterable[str] = field(default_factory=lambda: ONLINE_METRICS)
    metrics_iter: Iterable[str] = field(default_factory=lambda: ITER_METRICS)
    output_subdir: str = "analysis_output"


class ExperimentAnalyzer:
    """High-level facade — `analyze(experiment_dir)` produkuje pełny raport."""

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        """Skonfiguruj analizator (domyślny `AnalyzerConfig`, gdy `None`)."""
        self.cfg = config or AnalyzerConfig()

    def analyze(self, experiment_dir: str | Path) -> Path:
        """Wygeneruj kompletny zestaw tabel, wykresów i raport dla eksperymentu.

        Args:
            experiment_dir: Katalog eksperymentu zawierający `analysis.db`
                (po `ExperimentAggregator.aggregate`).

        Returns:
            Ścieżka do `analysis_output/` (lub odpowiedniego `output_subdir`).

        Raises:
            FileNotFoundError: Gdy `analysis.db` nie istnieje.

        Efekty uboczne:
            Tworzy podkatalogi `tables/` i `plots/{convergence, boxplots,
            cd_diagrams, pareto, scatter, bar, rankings}/` oraz raport PDF/MD.
        """
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

        self._build_tables(df_run, df_online, tables_dir)
        self._build_plots(df_run, df_iter, df_pareto, plots_dir)

        from src.analysis.analyzer.report_generator import ReportGenerator

        try:
            report_gen = ReportGenerator(experiment_dir)
            md_path, pdf_path = report_gen.generate()
            logger.info("Report: %s, %s", md_path, pdf_path)
        except Exception:
            logger.warning("ReportGenerator failed — skipping report.", exc_info=True)

        logger.info(f"ExperimentAnalyzer: wygenerowano analizę w {out_dir}")
        return out_dir

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

        # Parametry algorytmów różnią się między środowiskami (pop_size,
        # n_gen, wagi skalaryzacji, marginesy bezpieczeństwa — zob.
        # docs/ceteris_paribus.md §1). Stąd Wilcoxon/Friedman/A12 emitujemy
        # *per-environment* — pooling cross-env mieszałby heterogeniczne
        # konfiguracje. Summary CSV pozostaje cross-env (env w wierszach) —
        # każdy wiersz dotyczy jednej (env, optimizer), więc bez mieszania.
        per_env_block_cols = ("seed",)

        for metric in self.cfg.metrics_offline:
            if metric not in df_offline.columns:
                continue
            sub = df_offline.dropna(subset=[metric])
            if sub.empty:
                continue
            higher = HIGHER_IS_BETTER.get(metric, False)
            self._emit_summary_stats(sub, metric, tables_dir)
            for env in sorted(sub["environment"].dropna().unique()):
                sub_env = sub[sub["environment"] == env]
                if sub_env.empty:
                    continue
                self._emit_pairwise_tests(
                    sub_env, metric, higher, per_env_block_cols,
                    tables_dir, file_prefix=f"{env}_",
                )

        # Online: pełny product (env, opt, seed, avoidance). W eksperymencie
        # paired (optimizer == avoidance) factor porównawczy to `optimizer`,
        # block per-env to `(seed,)` — w każdym (env, seed) mamy 4 obs
        # po jednej na każdy z 4 algorytmów. Per-env split z tych samych
        # powodów co offline + dodatkowo: metryki online (collision_count,
        # min_inter_uav_distance) silnie zależą od geometrii środowiska,
        # więc cross-env aggregacja sama w sobie miesza różne reżimy
        # fizyczne (forest sparse, urban dense).
        for metric in self.cfg.metrics_online:
            if metric not in df_run.columns:
                continue
            sub = df_run.dropna(subset=[metric])
            if sub.empty:
                continue
            higher = HIGHER_IS_BETTER.get(metric, False)
            self._emit_summary_stats(sub, metric, tables_dir, file_prefix="online_")
            for env in sorted(sub["environment"].dropna().unique()):
                sub_env = sub[sub["environment"] == env]
                if sub_env.empty:
                    continue
                self._emit_pairwise_tests(
                    sub_env, metric, higher, per_env_block_cols,
                    tables_dir, file_prefix=f"online_{env}_",
                )

        # Failure rate per (env, opt) — odporne uzupełnienie mean/median
        # statistik (tail-risk). Trzy kategorie:
        #   - offline failure: kolizja podczas tracking (plan offline kolizyjny)
        #   - online failure: kolizja podczas evasion (algorytm unikania zawiódł)
        #   - hv_degenerate (diagnostyka): HV=0 lub pusty front — głównie
        #     informatywne dla NSGA-III; dla SOO to tautologia konstrukcji.
        for flag_col, suffix, caption in (
            (
                "is_offline_failure",
                "offline",
                "Offline failure rate (tracking-phase collisions > 0).",
            ),
            (
                "is_online_failure",
                "online",
                "Online failure rate (evasion-phase collisions > 0).",
            ),
            (
                "is_hv_degenerate",
                "hv_degenerate",
                "HV-degenerate rate (HV=0 OR empty Pareto front); diagnostic only.",
            ),
        ):
            if flag_col not in df_run.columns:
                continue
            # Offline-fazowe i HV-related nie zależą od `avoidance` — dedup'ujemy.
            # Online-fazowe (kolizje podczas evasion) zależą od avoidance.
            base = (
                _dedup_offline(df_run)
                if flag_col in ("is_offline_failure", "is_hv_degenerate")
                else df_run
            )
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
            # Wilson 95% score CI dla każdej proporcji (Newcombe 1998).
            ci_lows: list[float] = []
            ci_highs: list[float] = []
            for _, row in failure_summary.iterrows():
                lo, hi = wilson_proportion_ci(
                    int(row["n_failures"]), int(row["n_runs"]),
                )
                ci_lows.append(lo)
                ci_highs.append(hi)
            failure_summary["wilson_ci95_low"] = ci_lows
            failure_summary["wilson_ci95_high"] = ci_highs
            export_csv(failure_summary, tables_dir / f"failure_rate_{suffix}.csv")
            export_latex(
                failure_summary, tables_dir / f"failure_rate_{suffix}.tex",
                caption=caption,
                label=f"tab:failure-rate-{suffix}",
            )

        if not df_online.empty:
            export_csv(df_online, tables_dir / "online_summary.csv")

    def _emit_summary_stats(
        self,
        sub: pd.DataFrame,
        metric: str,
        tables_dir: Path,
        file_prefix: str = "",
    ) -> None:
        """Emit summary CSV/LaTeX cross-environment — env pozostaje kolumną wiersza.

        Summary jest jedyną tabelą, gdzie pooling cross-env jest poprawny: każdy
        wiersz dotyczy jednego (env, optimizer), więc nie ma mieszania
        heterogenicznych konfiguracji.
        """
        summary = summary_stats(
            sub, metric=metric,
            group_cols=("environment", "optimizer"),
        )
        if summary.empty:
            return
        # Low-power warning: Friedman z bardzo małym N ma asymptotyczny rozkład χ²
        # i traci moc statystyczną — flag pomaga czytelnikowi w interpretacji.
        summary["low_power_warning"] = summary["n"] < LOW_POWER_N_THRESHOLD
        export_csv(summary, tables_dir / f"{file_prefix}summary_{metric}.csv")
        export_latex(
            summary, tables_dir / f"{file_prefix}summary_{metric}.tex",
            caption=f"Summary statistics — {metric}",
            label=f"tab:{file_prefix}summary-{metric}",
        )

    def _emit_pairwise_tests(
        self,
        sub: pd.DataFrame,
        metric: str,
        higher: bool,
        block_cols: tuple[str, ...],
        tables_dir: Path,
        file_prefix: str = "",
    ) -> None:
        """Emit Friedman + Nemenyi (global) + A12 (effect size) dla *jednej*
        podgrupy (zwykle per-env).

        Caller jest odpowiedzialny za pre-filtrowanie `sub` do pojedynczego
        środowiska i odpowiednie `file_prefix` (np. `forest_`, `online_urban_`).
        Pooling cross-env jest niepoprawny w eksperymencie, gdzie parametry
        algorytmu różnią się między środowiskami — zob.
        [docs/ceteris_paribus.md](../docs/ceteris_paribus.md) §1.
        """
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

        # §3.1.3.2 (docs/Praca magisterska.md) Krzywa optymalizacji online —
        # best_fitness per generation w online optimization tasks (osobny od
        # offline convergence: różne źródło danych, różna granularność —
        # per-trigger vs per-run). plots_dir = <exp>/analysis_output/plots,
        # więc db = <exp>/analysis.db = plots_dir.parent.parent / analysis.db.
        try:
            from src.analysis.analyzer.plots.convergence_plots import (
                plot_online_convergence,
            )
            db_path = plots_dir.parent.parent / "analysis.db"
            plot_online_convergence(db_path, plots_dir / "convergence")
        except Exception:
            logger.warning(
                "ExperimentAnalyzer: plot_online_convergence failed — "
                "online convergence curve unavailable.",
                exc_info=True,
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

        # CD diagrams per (env, metric). Pooling cross-env byłby mieszaniem
        # heterogenicznych konfiguracji algorytmu — zob. `_build_tables`
        # i docs/ceteris_paribus.md §1.
        # Ranking heatmap pozostaje cross-env (env na osi Y), bo wewnętrznie
        # rangi liczone są per (env, seed) — heatmap pokazuje per-env rangi
        # w jednym obrazie dla ułatwienia porównania międzyśrodowiskowego.
        df_offline = _dedup_offline(df_run)
        offline_envs = sorted(df_offline["environment"].dropna().unique())
        for metric in self.cfg.metrics_offline:
            if metric not in df_offline.columns:
                continue
            sub = df_offline.dropna(subset=[metric])
            if sub.empty:
                continue
            higher = HIGHER_IS_BETTER.get(metric, False)
            for env in offline_envs:
                sub_env = sub[sub["environment"] == env]
                if sub_env.empty:
                    continue
                try:
                    fr = friedman_with_nemenyi(
                        sub_env, metric=metric, higher_is_better=higher,
                        block_cols=("seed",),
                    )
                    if fr.cd_nemenyi is not None:
                        plot_cd_diagram(
                            fr.average_ranks, fr.cd_nemenyi,
                            plots_dir / "cd_diagrams" / f"cd_{env}_{metric}.pdf",
                            title=(
                                f"CD — {metric} [{env}] "
                                f"({'higher=better' if higher else 'lower=better'})"
                            ),
                        )
                except ValueError:
                    pass
            plot_ranking_heatmap(
                df_offline, metric,
                plots_dir / "rankings" / f"ranking_{metric}.pdf",
                higher_is_better=higher,
            )

        # CD diagrams dla online — per-env, na pełnym df_run (paired runs
        # dają 4 obs per (env, seed) — Friedman w 1-D = poprawny). Wcześniej
        # pooled cross-env Friedman dla online cicho failował z powodu
        # nakładania się avoidance × seed indeksu — per-env naprawia to.
        online_envs = sorted(df_run["environment"].dropna().unique())
        for metric in self.cfg.metrics_online:
            if metric not in df_run.columns:
                continue
            sub = df_run.dropna(subset=[metric])
            if sub.empty:
                continue
            higher = HIGHER_IS_BETTER.get(metric, False)
            for env in online_envs:
                sub_env = sub[sub["environment"] == env]
                if sub_env.empty:
                    continue
                try:
                    fr = friedman_with_nemenyi(
                        sub_env, metric=metric, higher_is_better=higher,
                        block_cols=("seed",),
                    )
                    if fr.cd_nemenyi is not None:
                        plot_cd_diagram(
                            fr.average_ranks, fr.cd_nemenyi,
                            plots_dir / "cd_diagrams" / f"cd_online_{env}_{metric}.pdf",
                            title=(
                                f"CD — {metric} [online, {env}] "
                                f"({'higher=better' if higher else 'lower=better'})"
                            ),
                        )
                except ValueError:
                    pass


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
    """Dodaje binarne kolumny `is_offline_failure`, `is_online_failure`
    i `is_hv_degenerate` per run.

    **OFFLINE failure** ⟺ `tracking_phase_collisions > 0` — drone miał
    kolizję podczas wykonywania trasy zaplanowanej offline (poza okresem
    aktywnego uniku). Sygnalizuje, że trajektoria z fazy planowania była
    kolizyjna z perspektywy fizycznej symulacji.

    **ONLINE failure** ⟺ `evasion_phase_collisions > 0` — drone miał
    kolizję podczas wykonywania manewru uniku (event_type='trigger' bez
    subsekwentnego 'rejoin'). Sygnalizuje, że algorytm reaktywnego
    unikania zawiódł. Kolizje powstałe z złego planowania offline NIE
    są tu liczone — patrz uzasadnienie w `reports/failure_success_methodology.md`.

    **HV degenerate** (osobna diagnostyka, *nie* failure) ⟺ `hypervolume`
    ∈ {NULL, 0} OR `front_size_last_gen` ∈ {NULL, 0}. Dla algorytmów
    SOO ze skalaryzacją weighted-sum to *tautologia konstrukcji*
    (1-pkt front w 5D), więc nieinformatywne dla porównania algorytmów.
    Dla NSGA-III informatywne — tail-risk degeneracji frontu Pareto.
    Raportowane osobno per algorytm.

    Reference: Liefooghe & Verel (2014). Failure rate to standardowy
    proxy tail-risk algorytmu, uzupełniający mean/median (które maskują
    katastrofalne runy).
    """
    if df_run.empty:
        return df_run
    df = df_run.copy()
    tpc = df.get("tracking_phase_collisions")
    epc = df.get("evasion_phase_collisions")
    fs = df.get("front_size_last_gen")
    hv = df.get("hypervolume")

    offline_fail = pd.Series(False, index=df.index)
    if tpc is not None:
        offline_fail = offline_fail | (tpc.fillna(0) > 0)
    df["is_offline_failure"] = offline_fail.astype(int)

    online_fail = pd.Series(False, index=df.index)
    if epc is not None:
        online_fail = online_fail | (epc.fillna(0) > 0)
    df["is_online_failure"] = online_fail.astype(int)
    # `online_success_rate` przedefiniowane na poziomie DataFrame jako binarne
    # **bezpieczeństwo manewru** (per-run): 1 = run bez kolizji w fazie evasion,
    # 0 = co najmniej jedna kolizja. Nadpisuje wartość z `run_metrics.online_success_rate`
    # (która jest `1 - budget_violation_rate` — niezawodność real-time, używana
    # tylko wewnątrz DB do obliczenia `online_sp1`). User-facing semantyka
    # "Online Success Rate" to fizyczne bezpieczeństwo, identyczne z §3.2.
    df["online_success_rate"] = (1 - df["is_online_failure"]).astype(float)

    # Diagnostyka MOO — raportowane osobno (głównie informatywne dla NSGA-III).
    hv_degenerate = pd.Series(False, index=df.index)
    if fs is not None:
        hv_degenerate = hv_degenerate | fs.isna() | (fs == 0)
    if hv is not None:
        hv_degenerate = hv_degenerate | hv.isna() | (hv == 0)
    df["is_hv_degenerate"] = hv_degenerate.astype(int)
    return df
