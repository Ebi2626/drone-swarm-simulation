"""ReportGenerator — generuje raport podsumowujacy eksperyment (PDF + Markdown).

Czyta dane z `analysis_output/tables/` (CSV) i `analysis_output/plots/` (PNG)
wygenerowane przez ExperimentAnalyzer, kompiluje je w:
- `analysis_output/report/experiment_report.md` — pelny raport Markdown z tabelami
  i odwolaniami do PNG (szablon jinja2),
- `analysis_output/report/experiment_report.pdf` — wielostronicowy PDF
  (matplotlib PdfPages: strony tekstowe, tabelaryczne i wykresy).

Reference: Demsar (2006) "Statistical Comparisons of Classifiers over Multiple
Data Sets", JMLR 7:1-30 — zalecana struktura raportu porownawczego.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.analysis.analyzer.plots._common import apply_style

logger = logging.getLogger(__name__)


@dataclass
class _PlotRef:
    """Referencja do pliku PNG osadzanego w raporcie.

    Args:
        title: Czytelny tytuł pod wykresem.
        absolute_path: Pełna ścieżka do PNG (na potrzeby `PdfPages.savefig`).
        relative_path: Ścieżka względem katalogu raportu (markdown links).
    """
    title: str
    absolute_path: Path
    relative_path: str


@dataclass
class ReportData:
    """Komplet danych potrzebnych do wygenerowania raportu PDF i Markdown."""
    experiment_name: str
    generation_date: str
    n_runs: int
    algorithms: list[str]
    environments: list[str]
    avoidances: list[str]
    n_datasets: int

    # Offline
    offline_summary: dict[str, pd.DataFrame] = field(default_factory=dict)
    friedman_ranks: dict[str, pd.DataFrame] = field(default_factory=dict)
    a12_effects: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Hypervolume — informational, NSGA-III only (Pareto-front quality
    # indicator nie ma sensu dla algorytmów jednokryterialnych; zob. praca
    # magisterska §3.1.3.2).
    hypervolume_nsga_only: Optional[pd.DataFrame] = None

    # Convergence — best-so-far per algorytm: stats z `iteration_metrics`
    # ostatniej generacji (per env, optimizer).
    convergence_summary: Optional[pd.DataFrame] = None

    # Timing per stage — z optimization_timings table. Pozwala porównać
    # czasochłonność `initialization`, `optimization`, `decision_and_reconstruction`
    # między algorytmami (sprawiedliwa analiza, nie tylko sumarycznie).
    timing_by_stage: Optional[pd.DataFrame] = None

    # Online
    online_summary: dict[str, pd.DataFrame] = field(default_factory=dict)
    has_online_data: bool = False
    # Friedman + A12 dla online metrics (§3.1.3.2 docs/Praca magisterska.md
    # *online phase* — analogicznie do offline, zob. `friedman_ranks` /
    # `a12_effects` powyżej).
    online_friedman_ranks: dict[str, pd.DataFrame] = field(default_factory=dict)
    online_a12_effects: dict[str, pd.DataFrame] = field(default_factory=dict)
    online_best_per_metric: Optional[pd.DataFrame] = None

    # Failure rate offline (§3.1 — physical tracking collisions). Online
    # safety raportowane wyłącznie w §2.1-2.3 jako `online_success_rate`.
    failure_offline: Optional[pd.DataFrame] = None

    # Rankings
    best_per_metric: Optional[pd.DataFrame] = None
    overall_ranking: Optional[pd.DataFrame] = None

    # Key findings (auto-generated text)
    key_findings: list[str] = field(default_factory=list)

    # Plot references
    offline_plots: list[_PlotRef] = field(default_factory=list)
    online_plots: list[_PlotRef] = field(default_factory=list)


class ReportGenerator:
    """Generates experiment summary reports in PDF and Markdown."""

    # Metryki dobrane zgodnie z `docs/Praca magisterska.md` §3.1.3.
    #
    # §3.1.3.1 — metryki oceny trajektorii fazy offline:
    #   - bezpieczeństwo (f3+f5), długość (f1), gładkość (f2+f4), spójność roju
    # §3.1.3.2 — metryki oceny algorytmów fazy offline:
    #   - wartość optymalizacji, prędkość (per-stage), krzywa optymalizacji
    #
    # `hypervolume` jest informacyjny *tylko dla NSGA-III* (jedyny algorytm
    # wielokryterialny) — raportowany w osobnej sekcji (`hypervolume_nsga_only`),
    # nie w głównej tabeli porównawczej.
    #
    # MOO indicators (IGD+, GD, spread, spacing, R2, front_size, HV_norm) są
    # nadal emitowane do CSV-ek przez `ExperimentAnalyzer`, ale w głównym
    # raporcie nie są pokazywane — per pracę: "pozostanie jedynie metryką
    # informacyjną" (stosuje się tylko do NSGA-III, brak sensu porównawczego
    # dla algorytmów jednokryterialnych).
    OFFLINE_KEY_METRICS = [
        # §3.1.3.1 — Trajectory quality
        "trajectory_safety_f3_f5",
        "trajectory_length_f1",
        "trajectory_smoothness_f2_f4",
        "swarm_cohesion_deviation",   # MAE od 5 m NN-spacing
        # §3.1.3.2 — Algorithm performance
        "final_objective",
    ]

    # Metryki używane do obliczenia overall offline ranking. Wykluczamy
    # hypervolume (NSGA-III only — pooled ranking byłby mylący) i wallclock
    # (już mamy szczegółową sekcję timing per stage).
    OVERALL_RANKING_METRICS = [
        "trajectory_safety_f3_f5",
        "trajectory_length_f1",
        "trajectory_smoothness_f2_f4",
        "swarm_cohesion_deviation",
        "final_objective",
    ]

    # Ocena fazy online — trzy klasy zgodnie z §3.1.3.3 + §3.1.3.4
    # docs/Praca magisterska.md:
    # (A) §3.1.3.3 TRAJEKTORIA fazy online: długość/czas manewru +
    #     skuteczność powrotu na nominalną trasę (plan-level metrics
    #     z `online_optimization_tasks`).
    # (B) Jakość lotu fizycznego (PyBullet, mission-wide): smoothness,
    #     energy, separacja dronów. Komplementarne — z `uav_online_metrics`.
    # (C) §3.1.3.4 ALGORYTM fazy online: performance online optimizer'a
    #     w budżecie czasowym (best_fitness, gens/evals_completed, wallclock).
    ONLINE_KEY_METRICS = [
        # Bezpieczeństwo fazy online — PRIMARY safety metric. Binarne per-run
        # (1 = run bez kolizji w fazie evasion, 0 = ≥1 kolizja). Identyczna
        # semantyka co §3.2 Wilson CI; tutaj raportowane jako Mean/Std/Median
        # w §2.1 oraz w §2.2 (Friedman) i §2.3 (A12) dla pełnego workupu
        # statystycznego. Mean = empiryczny success rate, std = sqrt(p*(1-p)).
        # Kolumna df_run nadpisywana w `_compute_failure_flag` z
        # `1 - is_online_failure`.
        "online_success_rate",
        # (A) §3.1.3.3 Trajektoria fazy online.
        # rejoin_completion_rate = PRIMARY (epizodowy mianownik, pending
        # excluded); rejoin_success_rate = surowy (pending counted).
        # rejoin_quality = TOPSIS composite (Hwang & Yoon 1981) trzech
        # wymiarów Skuteczności powrotu: pos_err, vel_err, time_to_rejoin
        # → pojedynczy skalar odległości euklidesowej od ideału (0,0,0)
        # w przestrzeni znormalizowanej medianami per środowisko.
        # Uwaga: `mean_evasion_plan_duration_s` + oryginalne 3 metryki
        # rejoin pozostają w DB dla audytu, ale nie w key metrics.
        "mean_evasion_arc_length_m",
        "rejoin_quality",
        "rejoin_completion_rate",
        "rejoin_success_rate",
        # (B) Jakość lotu fizycznego (mission-wide, komplementarne)
        "mean_smoothness_indicator",
        "mean_energy_indicator",
        "min_inter_uav_distance_m",
        # (C) §3.1.3.4 Algorytm fazy online
        # NFE jako prędkość (BBOB 2009 convention), gens informacyjnie,
        # budget_violation_rate jako niezawodność hard real-time.
        # online_sp1 = RT_succ / (1 - BVR), łączona miara prędkość×skuteczność
        # (Success Performance 1, Auger & Hansen 2005). Wybór SP1 zamiast
        # ERT (Hansen 2009 BBOB) bo failed tasks abortują z RT_unsucc=0
        # → ERT degeneruje do RT_succ.
        # Wallclock pominięty w key metrics — saturuje budżet, brak
        # discriminative info (pozostaje w DB dla audytu).
        # NOTE: real-time niezawodność (`1 - BVR`) raportujemy wyłącznie przez
        # `budget_violation_rate` (lower=lepiej). Nazwa "online_success_rate"
        # w §2 jest zarezerwowana dla collision-based safety (zob. PRIMARY).
        "mean_online_best_fitness",
        "mean_online_evaluations_completed",
        "mean_online_generations_completed",
        "budget_violation_rate",
        "online_sp1",
    ]

    # Stages w optimization_timings — kolejność wyświetlania w raporcie.
    _TIMING_STAGES: list[str] = [
        "initialization",
        "optimization",
        "decision_and_reconstruction",
        "total_optimization",
    ]

    # Plot subdirectory -> list of (file_pattern, human title).
    # {env} and {metric} are replaced when scanning.
    _OFFLINE_PLOT_SPECS: list[tuple[str, str, str]] = [
        # Trajectory quality boxplots (§3.1.3.1)
        ("boxplots", "boxplot_{env}_trajectory_safety_f3_f5.png", "Boxplot: Trajectory Safety f3+f5 ({env})"),
        ("boxplots", "boxplot_{env}_trajectory_length_f1.png", "Boxplot: Trajectory Length f1 ({env})"),
        ("boxplots", "boxplot_{env}_trajectory_smoothness_f2_f4.png", "Boxplot: Trajectory Smoothness f2+f4 ({env})"),
        ("boxplots", "boxplot_{env}_swarm_cohesion_deviation.png", "Boxplot: Swarm Cohesion Deviation ({env})"),
        # Algorithm performance boxplots (§3.1.3.2)
        ("boxplots", "boxplot_{env}_final_objective.png", "Boxplot: Final Objective ({env})"),
        # Krzywa optymalizacji (§3.1.3.2)
        ("convergence", "convergence_{env}_best_so_far.png", "Convergence: Best-so-far ({env})"),
        # CD diagrams dla głównych metryk porównawczych
        ("cd_diagrams", "cd_{env}_trajectory_safety_f3_f5.png", "CD Diagram: Trajectory Safety ({env})"),
        ("cd_diagrams", "cd_{env}_trajectory_length_f1.png", "CD Diagram: Trajectory Length ({env})"),
        ("cd_diagrams", "cd_{env}_trajectory_smoothness_f2_f4.png", "CD Diagram: Trajectory Smoothness ({env})"),
        ("cd_diagrams", "cd_{env}_swarm_cohesion_deviation.png", "CD Diagram: Swarm Cohesion ({env})"),
        ("cd_diagrams", "cd_{env}_final_objective.png", "CD Diagram: Final Objective ({env})"),
        # Ranking heatmaps
        ("rankings", "ranking_final_objective.png", "Ranking Heatmap: Final Objective"),
        ("rankings", "ranking_trajectory_safety_f3_f5.png", "Ranking Heatmap: Trajectory Safety"),
        # Offline safety: rzeczywiste kolizje fizyczne podczas tracking phase
        # (komplementarne do trajectory_safety_f3_f5 — kara planu na węzłach).
        ("bar", "bar_{env}_failure_rate_offline.png", "Offline Failure Rate — physical tracking collisions ({env})"),
    ]

    _ONLINE_PLOT_SPECS: list[tuple[str, str, str]] = [
        # Bezpieczeństwo fazy online — collision-based safety:
        # boxplot per-run binary (Mean/Std view) + CD diagram (rank-based).
        # Stara wizualizacja Wilson CI proportion pozostaje w §3.2 tabeli.
        ("boxplots", "boxplot_{env}_online_success_rate.png",
         "Boxplot: Online maneuver safety ({env}) — collision-free evasion runs"),
        ("cd_diagrams", "cd_{env}_online_success_rate.png",
         "CD Diagram: Online maneuver safety ({env}) — collision-free evasion runs"),
        # §3.1.3.3 Trajektoria fazy online — boxploty plan-level metrics.
        ("boxplots", "boxplot_{env}_mean_evasion_arc_length_m.png",
         "Boxplot: Evasion arc length ({env}) — §3.1.3.3 Długość manewru"),
        ("boxplots", "boxplot_{env}_rejoin_completion_rate.png",
         "Boxplot: Rejoin completion rate ({env}) — §3.1.3.3 Niezawodność powrotu (epizodowa)"),
        ("boxplots", "boxplot_{env}_rejoin_success_rate.png",
         "Boxplot: Rejoin success rate ({env}) — §3.1.3.3 (surowa, z pending)"),
        ("boxplots", "boxplot_{env}_rejoin_quality.png",
         "Boxplot: Rejoin quality ({env}) — §3.1.3.3 TOPSIS composite (pos, vel, time)"),
        ("cd_diagrams", "cd_{env}_rejoin_quality.png",
         "CD Diagram: Rejoin quality ({env}) — §3.1.3.3"),
        # §3.1.3.4 Algorytm fazy online — boxploty + CD.
        # NFE (mean_online_evaluations_completed) = primary throughput;
        # budget_violation_rate = safety hard real-time.
        ("boxplots", "boxplot_{env}_mean_online_best_fitness.png",
         "Boxplot: Online best_fitness ({env}) — §3.1.3.4 Wartość optymalizacji"),
        ("boxplots", "boxplot_{env}_mean_online_evaluations_completed.png",
         "Boxplot: Online NFE ({env}) — §3.1.3.4 Prędkość (Number of Function Evaluations)"),
        ("boxplots", "boxplot_{env}_budget_violation_rate.png",
         "Boxplot: Budget violation rate ({env}) — §3.1.3.4 Niezawodność real-time"),
        ("boxplots", "boxplot_{env}_online_sp1.png",
         "Boxplot: Online SP1 ({env}) — §3.1.3.4 Success Performance 1 (Auger & Hansen 2005)"),
        ("cd_diagrams", "cd_{env}_mean_online_best_fitness.png",
         "CD Diagram: Online best_fitness ({env})"),
        ("cd_diagrams", "cd_{env}_mean_online_evaluations_completed.png",
         "CD Diagram: Online NFE ({env}) — §3.1.3.4"),
        ("cd_diagrams", "cd_{env}_budget_violation_rate.png",
         "CD Diagram: Budget violation rate ({env}) — §3.1.3.4"),
        ("cd_diagrams", "cd_{env}_online_sp1.png",
         "CD Diagram: Online SP1 ({env}) — §3.1.3.4"),
        ("cd_diagrams", "cd_{env}_rejoin_completion_rate.png",
         "CD Diagram: Rejoin completion rate ({env}) — §3.1.3.3"),
        ("cd_diagrams", "cd_{env}_rejoin_success_rate.png",
         "CD Diagram: Rejoin success rate ({env}) — §3.1.3.3 (surowa)"),
        # §3.1.3.4 Krzywa optymalizacji online — mean ± std best_fitness
        # per generation per (env, algo).
        ("convergence", "online_convergence_{env}.png",
         "Online Convergence: best_fitness per generation ({env}) — §3.1.3.4"),
    ]

    def __init__(self, experiment_dir: str | Path) -> None:
        """Powiąż generator z katalogiem eksperymentu i jego artefaktami.

        Args:
            experiment_dir: Katalog eksperymentu zawierający `analysis_output/`
                z wcześniej wygenerowanymi tabelami i wykresami.
        """
        self.experiment_dir = Path(experiment_dir).resolve()
        self.tables_dir = self.experiment_dir / "analysis_output" / "tables"
        self.plots_dir = self.experiment_dir / "analysis_output" / "plots"
        self.out_dir = self.experiment_dir / "analysis_output" / "report"

    def generate(self) -> tuple[Path, Path]:
        """Wygeneruj raport w dwóch formatach (Markdown + PDF).

        Returns:
            Krotka `(md_path, pdf_path)` z bezwzględnymi ścieżkami zapisanych
            plików.

        Efekty uboczne:
            Tworzy `analysis_output/report/experiment_report.{md,pdf}` (oraz
            katalog raportu, jeśli nie istnieje).
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        data = self._collect_data()
        md_path = self._generate_markdown(data)
        pdf_path = self._generate_pdf(data)
        logger.info("Report generated: %s, %s", md_path, pdf_path)
        return md_path, pdf_path

    def _collect_data(self) -> ReportData:
        """Wczytaj tabele CSV i pliki PNG z `analysis_output/` do `ReportData`."""
        # Experiment metadata from directory name
        exp_name = self.experiment_dir.name
        envs = self._detect_values("environment")
        algs = self._detect_values("optimizer")
        avoidances = self._detect_avoidances()

        # Count runs from DB
        n_runs = self._count_runs()

        # Offline summaries
        offline_summary: dict[str, pd.DataFrame] = {}
        for m in self.OFFLINE_KEY_METRICS:
            df = self._read_csv(f"summary_{m}.csv")
            if df is not None:
                offline_summary[m] = df

        # Friedman ranks — per-env CSV concat'owane z dodatkową kolumną
        # `environment`. Pooling cross-env nie jest emitowany przez
        # `ExperimentAnalyzer` (zob. _build_tables docstring), więc czytamy
        # `{env}_friedman_<m>.csv` dla każdego env.
        friedman_ranks: dict[str, pd.DataFrame] = {}
        for m in self.OFFLINE_KEY_METRICS:
            parts: list[pd.DataFrame] = []
            for env in envs:
                df_env = self._read_csv(f"{env}_friedman_{m}.csv")
                if df_env is not None:
                    df_env = df_env.copy()
                    df_env["environment"] = env
                    parts.append(df_env)
            if parts:
                friedman_ranks[m] = pd.concat(parts, ignore_index=True)

        # A12 effect sizes — per-env
        a12_effects: dict[str, pd.DataFrame] = {}
        for m in self.OFFLINE_KEY_METRICS:
            parts = []
            for env in envs:
                df_env = self._read_csv(f"{env}_a12_{m}.csv")
                if df_env is not None:
                    df_env = df_env.copy()
                    df_env["environment"] = env
                    parts.append(df_env)
            if parts:
                a12_effects[m] = pd.concat(parts, ignore_index=True)

        # Hypervolume — informational, NSGA-III only (Pareto-front quality
        # indicator nie ma sensu dla algorytmów jednokryterialnych).
        hv_full = self._read_csv("summary_hypervolume.csv")
        hypervolume_nsga_only: Optional[pd.DataFrame] = None
        if hv_full is not None and "optimizer" in hv_full.columns:
            mask = hv_full["optimizer"].astype(str).str.lower().str.startswith("nsga")
            hv_nsga = hv_full[mask].copy()
            if not hv_nsga.empty:
                hypervolume_nsga_only = hv_nsga

        # Convergence summary — best_so_far na ostatniej iteracji per
        # (env, optimizer, seed) z `iteration_metrics`, agregat statystyk
        # przez `summary_stats`. Wartość final best-so-far powinna być spójna
        # z `final_objective` z `run_metrics`, ale dodatkowo emitujemy
        # `convergence_speed_gen` (przybliżona prędkość konwergencji).
        convergence_summary = self._compute_convergence_summary()

        # Timing per stage — z `optimization_timings`. Pozwala na uczciwe
        # porównanie czasochłonności per faza algorytmu.
        timing_by_stage = self._compute_timing_by_stage()

        # Online summaries
        online_summary: dict[str, pd.DataFrame] = {}
        has_online = False
        for m in self.ONLINE_KEY_METRICS:
            df = self._read_csv(f"online_summary_{m}.csv")
            if df is not None:
                online_summary[m] = df
                has_online = True

        # Online Friedman + A12 dla metryk §3.1.3.2 (analogicznie do offline).
        # CSV files emitowane przez ExperimentAnalyzer z prefixem
        # `online_{env}_` (zob. ExperimentAnalyzer._emit_pairwise_tests).
        online_friedman_ranks: dict[str, pd.DataFrame] = {}
        online_a12_effects: dict[str, pd.DataFrame] = {}
        for m in self.ONLINE_KEY_METRICS:
            f_parts: list[pd.DataFrame] = []
            a_parts: list[pd.DataFrame] = []
            for env in envs:
                df_f = self._read_csv(f"online_{env}_friedman_{m}.csv")
                if df_f is not None:
                    df_f = df_f.copy()
                    df_f["environment"] = env
                    f_parts.append(df_f)
                df_a = self._read_csv(f"online_{env}_a12_{m}.csv")
                if df_a is not None:
                    df_a = df_a.copy()
                    df_a["environment"] = env
                    a_parts.append(df_a)
            if f_parts:
                online_friedman_ranks[m] = pd.concat(f_parts, ignore_index=True)
            if a_parts:
                online_a12_effects[m] = pd.concat(a_parts, ignore_index=True)
        online_best_per_metric = self._compute_best_per_metric(online_friedman_ranks)

        # Failure rate offline (§3.1 — physical tracking collisions). Online
        # safety jest raportowane wyłącznie w §2.1-2.3 przez `online_success_rate`.
        failure_offline = self._read_csv("failure_rate_offline.csv")

        # Best per metric + overall ranking
        best_per_metric = self._compute_best_per_metric(friedman_ranks)
        overall_ranking = self._compute_overall_ranking(friedman_ranks)

        # n_datasets from first available Friedman result
        n_datasets = 0
        for df in friedman_ranks.values():
            if "n_datasets" in df.columns:
                n_datasets = int(df["n_datasets"].iloc[0])
                break

        # Collect plot references
        offline_plots = self._collect_plots(
            self._OFFLINE_PLOT_SPECS, envs,
        )
        online_plots = self._collect_plots(
            self._ONLINE_PLOT_SPECS, envs,
        )

        # Key findings
        key_findings = self._generate_key_findings(
            overall_ranking, best_per_metric, friedman_ranks,
            online_summary, failure_offline,
        )

        return ReportData(
            experiment_name=exp_name,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            n_runs=n_runs,
            algorithms=algs,
            environments=envs,
            avoidances=avoidances,
            n_datasets=n_datasets,
            offline_summary=offline_summary,
            friedman_ranks=friedman_ranks,
            a12_effects=a12_effects,
            hypervolume_nsga_only=hypervolume_nsga_only,
            convergence_summary=convergence_summary,
            timing_by_stage=timing_by_stage,
            online_summary=online_summary,
            has_online_data=has_online,
            failure_offline=failure_offline,
            best_per_metric=best_per_metric,
            overall_ranking=overall_ranking,
            key_findings=key_findings,
            offline_plots=offline_plots,
            online_plots=online_plots,
            online_friedman_ranks=online_friedman_ranks,
            online_a12_effects=online_a12_effects,
            online_best_per_metric=online_best_per_metric,
        )

    def _detect_values(self, col: str) -> list[str]:
        """Zwróć posortowane unikalne wartości kolumny `col` z dowolnego summary CSV."""
        for m in self.OFFLINE_KEY_METRICS:
            df = self._read_csv(f"summary_{m}.csv")
            if df is not None and col in df.columns:
                return sorted(df[col].dropna().unique().tolist())
        return []

    def _detect_avoidances(self) -> list[str]:
        """Zwróć listę algorytmów avoidance z tabeli `runs` w `analysis.db`."""
        import sqlite3

        db_path = self.experiment_dir / "analysis.db"
        if not db_path.exists():
            return []
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT avoidance_algo FROM runs "
                    "WHERE avoidance_algo IS NOT NULL ORDER BY avoidance_algo"
                ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    def _count_runs(self) -> int:
        """Zwróć liczbę wszystkich runów w `analysis.db` (`0` przy braku DB)."""
        import sqlite3

        db_path = self.experiment_dir / "analysis.db"
        if not db_path.exists():
            return 0
        try:
            with sqlite3.connect(db_path) as conn:
                row = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def _read_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Wczytaj `tables_dir/filename` jako DataFrame; `None` przy braku/błędzie."""
        path = self.tables_dir / filename
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                return None
        return None

    def _compute_best_per_metric(
        self, friedman: dict[str, pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        """Zwróć najlepszy algorytm (rank 1) per (metryka, środowisko) z Friedmana.

        Args:
            friedman: Mapa `metric → DataFrame` z kolumnami `optimizer`,
                `avg_rank`, `environment`, opcjonalnie `p_value`.

        Returns:
            DataFrame `metric, environment, best_algorithm, avg_rank, p_value`
            (jeden wiersz per kombinacja metryka × środowisko) albo `None`,
            gdy brak danych.
        """
        if not friedman:
            return None
        rows = []
        for metric, df in friedman.items():
            if "avg_rank" not in df.columns or df.empty:
                continue
            if "environment" in df.columns:
                # per-env: jeden best per środowisko
                for env, sub in df.groupby("environment"):
                    if sub.empty:
                        continue
                    best_idx = sub["avg_rank"].idxmin()
                    best = sub.loc[best_idx]
                    rows.append({
                        "metric": metric,
                        "environment": env,
                        "best_algorithm": best["optimizer"],
                        "avg_rank": best["avg_rank"],
                        "p_value": best.get("p_value", float("nan")),
                    })
            else:
                # legacy fallback (cross-env friedman_<metric>.csv — nie powinno
                # występować po refactorze ExperimentAnalyzer, ale zachowane
                # dla wstecznej kompatybilności).
                best_idx = df["avg_rank"].idxmin()
                best = df.loc[best_idx]
                rows.append({
                    "metric": metric,
                    "environment": "all",
                    "best_algorithm": best["optimizer"],
                    "avg_rank": best["avg_rank"],
                    "p_value": best.get("p_value", float("nan")),
                })
        return pd.DataFrame(rows) if rows else None

    def _compute_overall_ranking(
        self, friedman: dict[str, pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        """Zwróć ranking algorytmów uśredniony po metrykach `OVERALL_RANKING_METRICS`.

        Wyklucza `hypervolume` (NSGA-III only — pooled ranking byłby mylący)
        i `total_wallclock_offline_s` (czas mierzony osobno w sekcji timing).
        Per `docs/Praca magisterska.md` §3.1.3.1 + §3.1.3.2 — uwzględniamy
        cztery metryki oceny trajektorii oraz wartość optymalizacji.
        """
        if not friedman:
            return None
        all_ranks = []
        for metric in self.OVERALL_RANKING_METRICS:
            df = friedman.get(metric)
            if df is None or "avg_rank" not in df.columns or "optimizer" not in df.columns:
                continue
            all_ranks.append(df[["optimizer", "avg_rank"]])
        if not all_ranks:
            return None
        combined = pd.concat(all_ranks, ignore_index=True)
        overall = (
            combined.groupby("optimizer")["avg_rank"]
            .mean()
            .reset_index()
            .sort_values("avg_rank")
            .reset_index(drop=True)
        )
        return overall

    def _compute_convergence_summary(self) -> Optional[pd.DataFrame]:
        """Tabela statystyk `best_so_far` na ostatniej iteracji per (env, optimizer).

        Czyta `iteration_metrics` z `analysis.db`, wyciąga ostatnią iterację
        per run, agreguje przez `summary_stats` — daje n/mean/std/min/max/
        median/q25/q75 best_so_far per algorytm. Krzywa optymalizacji
        (§3.1.3.2 pracy) jest przedstawiona wykresami w `convergence/`;
        tutaj suplementarna tabela liczbowa.

        Returns:
            DataFrame z kolumnami `environment, optimizer, n, mean, std,
            min, max, median, q25, q75` albo `None` przy braku danych.
        """
        import sqlite3
        db_path = self.experiment_dir / "analysis.db"
        if not db_path.exists():
            return None
        query = """
        SELECT
            r.environment, r.optimizer_algo AS optimizer, r.seed,
            im.best_so_far
        FROM iteration_metrics im
        JOIN runs r ON r.run_id = im.run_id
        WHERE im.iteration = (
            SELECT MAX(iteration) FROM iteration_metrics WHERE run_id = im.run_id
        )
          AND im.best_so_far IS NOT NULL
        """
        try:
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(query, conn)
        except Exception:
            return None
        if df.empty:
            return None
        from src.analysis.analyzer.statistical_tests import summary_stats
        return summary_stats(df, metric="best_so_far",
                              group_cols=("environment", "optimizer"))

    def _compute_timing_by_stage(self) -> Optional[pd.DataFrame]:
        """Agregat czasów `optimization_timings` per (env, optimizer, stage).

        Dla każdej kombinacji (środowisko, algorytm, etap) wylicza
        n/mean/std/median z `wall_time_s`. Etapy: `initialization`,
        `optimization`, `decision_and_reconstruction`, `total_optimization`.
        Pozwala porównać czasochłonność każdej fazy w sposób uczciwy
        (różne algorytmy spędzają różny czas w różnych etapach).

        Returns:
            DataFrame `environment, optimizer, stage, n, mean_s, std_s,
            median_s, min_s, max_s` albo `None` przy braku tabeli.
        """
        import sqlite3
        db_path = self.experiment_dir / "analysis.db"
        if not db_path.exists():
            return None
        query = """
        SELECT
            r.environment,
            r.optimizer_algo AS optimizer,
            t.stage_name AS stage,
            t.wall_time_s
        FROM optimization_timings t
        JOIN runs r ON r.run_id = t.run_id
        WHERE t.wall_time_s IS NOT NULL
        """
        try:
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(query, conn)
        except Exception:
            return None
        if df.empty:
            return None
        agg = (
            df.groupby(["environment", "optimizer", "stage"], dropna=False)
              ["wall_time_s"]
              .agg(["count", "mean", "std", "median", "min", "max"])
              .reset_index()
              .rename(columns={
                  "count": "n",
                  "mean": "mean_s",
                  "std": "std_s",
                  "median": "median_s",
                  "min": "min_s",
                  "max": "max_s",
              })
        )
        # Sortuj wedle ustalonej kolejności stage'ów dla czytelności.
        stage_order = {s: i for i, s in enumerate(self._TIMING_STAGES)}
        agg["_stage_ord"] = agg["stage"].map(stage_order).fillna(99)
        agg = agg.sort_values(
            ["environment", "optimizer", "_stage_ord"]
        ).drop(columns="_stage_ord").reset_index(drop=True)
        return agg

    def _collect_plots(
        self,
        specs: list[tuple[str, str, str]],
        environments: list[str],
    ) -> list[_PlotRef]:
        """Zwróć listę `_PlotRef` z istniejących PNG dopasowanych do `specs`."""
        import os

        refs = []
        for subdir, pattern, title_pattern in specs:
            if "{env}" in pattern:
                for env in environments:
                    fname = pattern.replace("{env}", env)
                    title = title_pattern.replace("{env}", env)
                    abs_path = self.plots_dir / subdir / fname
                    if abs_path.exists():
                        rel = os.path.relpath(abs_path, self.out_dir)
                        refs.append(_PlotRef(title=title, absolute_path=abs_path, relative_path=rel))
            else:
                abs_path = self.plots_dir / subdir / pattern
                if abs_path.exists():
                    rel = os.path.relpath(abs_path, self.out_dir)
                    refs.append(_PlotRef(title=title_pattern, absolute_path=abs_path, relative_path=rel))
        return refs

    def _generate_key_findings(
        self,
        overall_ranking: Optional[pd.DataFrame],
        best_per_metric: Optional[pd.DataFrame],
        friedman_ranks: dict[str, pd.DataFrame],
        online_summary: dict[str, pd.DataFrame],
        failure_offline: Optional[pd.DataFrame],
    ) -> list[str]:
        """Wygeneruj listę zdań „key findings" na podstawie zagregowanych danych."""
        findings: list[str] = []

        # 1. Best overall offline algorithm
        if overall_ranking is not None and not overall_ranking.empty:
            best = overall_ranking.iloc[0]
            n_metrics = len(self.OVERALL_RANKING_METRICS)
            findings.append(
                f"{best['optimizer']} achieves the best overall offline "
                f"ranking (avg Friedman rank: {best['avg_rank']:.2f}) across "
                f"{n_metrics} key metrics."
            )

        # 2. Statistical significance count
        if friedman_ranks:
            n_sig = sum(
                1 for df in friedman_ranks.values()
                if "p_value" in df.columns and (df["p_value"] < 0.05).any()
            )
            findings.append(
                f"Friedman test shows statistically significant differences "
                f"(p < 0.05) in {n_sig}/{len(friedman_ranks)} offline metrics."
            )

        # 3. Offline failure hot-spots (tracking-phase collisions)
        if failure_offline is not None and "failure_rate" in failure_offline.columns:
            worst_idx = failure_offline["failure_rate"].idxmax()
            worst = failure_offline.loc[worst_idx]
            if worst["failure_rate"] > 0:
                findings.append(
                    f"Highest offline failure rate (tracking-phase collisions): "
                    f"{worst['optimizer']} in {worst['environment']} "
                    f"({worst['failure_rate']:.0%} of runs)."
                )

        # 4. Online maneuver safety leader — z `online_success_rate` summary
        # (per-run binary, Mean = success rate; collision-based safety).
        osr = online_summary.get("online_success_rate")
        if osr is not None and "mean" in osr.columns and "optimizer" in osr.columns:
            best_idx = osr["mean"].idxmax()
            best = osr.loc[best_idx]
            findings.append(
                f"Highest online maneuver safety (collision-free evasion): "
                f"{best['optimizer']} in {best.get('environment', 'all')} "
                f"({best['mean']:.0%} of runs)."
            )

        if not findings:
            findings.append("Insufficient data for automated findings.")

        return findings

    def _generate_markdown(self, data: ReportData) -> Path:
        """Wyrenderuj `report_template.md.j2` do `experiment_report.md`."""
        import jinja2

        template_path = Path(__file__).parent / "report_template.md.j2"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_path.parent)),
            undefined=jinja2.Undefined,
            keep_trailing_newline=True,
        )
        template = env.get_template(template_path.name)

        rendered = template.render(
            experiment_name=data.experiment_name,
            generation_date=data.generation_date,
            n_runs=data.n_runs,
            algorithms=data.algorithms,
            environments=data.environments,
            avoidances=data.avoidances,
            n_datasets=data.n_datasets,
            offline_key_metrics=self.OFFLINE_KEY_METRICS,
            online_key_metrics=self.ONLINE_KEY_METRICS,
            offline_summary=data.offline_summary,
            friedman_ranks=data.friedman_ranks,
            a12_effects=data.a12_effects,
            hypervolume_nsga_only=data.hypervolume_nsga_only,
            convergence_summary=data.convergence_summary,
            timing_by_stage=data.timing_by_stage,
            timing_stages=self._TIMING_STAGES,
            online_summary=data.online_summary,
            has_online_data=data.has_online_data,
            failure_offline=data.failure_offline,
            best_per_metric=data.best_per_metric,
            overall_ranking=data.overall_ranking,
            key_findings=data.key_findings,
            offline_plots=data.offline_plots,
            online_plots=data.online_plots,
            online_friedman_ranks=data.online_friedman_ranks,
            online_a12_effects=data.online_a12_effects,
            online_best_per_metric=data.online_best_per_metric,
        )

        md_path = self.out_dir / "experiment_report.md"
        md_path.write_text(rendered, encoding="utf-8")
        logger.info("Markdown report: %s", md_path)
        return md_path

    def _generate_pdf(self, data: ReportData) -> Path:
        """Skompiluj wielostronicowy PDF raportu (matplotlib `PdfPages`)."""
        apply_style()
        pdf_path = self.out_dir / "experiment_report.pdf"

        with PdfPages(str(pdf_path)) as pdf:
            # Page 1: Title
            pdf.savefig(self._pdf_title_page(data))

            # Pages: Offline summary tables
            for metric in self.OFFLINE_KEY_METRICS:
                if metric in data.offline_summary:
                    fig = self._pdf_table_page(
                        data.offline_summary[metric],
                        f"Offline Summary: {metric.replace('_', ' ').title()}",
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

            # Pages: Friedman ranking summary
            if data.best_per_metric is not None and not data.best_per_metric.empty:
                fig = self._pdf_table_page(
                    data.best_per_metric,
                    "Friedman Rankings: Best Algorithm per Metric",
                )
                pdf.savefig(fig)
                plt.close(fig)

            # Pages: Overall ranking
            if data.overall_ranking is not None and not data.overall_ranking.empty:
                fig = self._pdf_table_page(
                    data.overall_ranking,
                    "Overall Offline Ranking (avg Friedman rank)",
                )
                pdf.savefig(fig)
                plt.close(fig)

            # Pages: Hypervolume (NSGA-III only — informational)
            if data.hypervolume_nsga_only is not None and not data.hypervolume_nsga_only.empty:
                fig = self._pdf_table_page(
                    data.hypervolume_nsga_only,
                    "Hypervolume (NSGA-III only, informational)",
                )
                pdf.savefig(fig)
                plt.close(fig)

            # Pages: Convergence (best-so-far summary)
            if data.convergence_summary is not None and not data.convergence_summary.empty:
                fig = self._pdf_table_page(
                    data.convergence_summary,
                    "Convergence: Best-so-far at final iteration",
                )
                pdf.savefig(fig)
                plt.close(fig)

            # Pages: Timing per stage
            if data.timing_by_stage is not None and not data.timing_by_stage.empty:
                fig = self._pdf_table_page(
                    data.timing_by_stage,
                    "Optimization Timing per Stage (wall-clock seconds)",
                )
                pdf.savefig(fig)
                plt.close(fig)

            # Pages: Key offline plots
            for pref in data.offline_plots:
                fig = self._pdf_plot_page(pref.absolute_path, pref.title)
                if fig is not None:
                    pdf.savefig(fig)
                    plt.close(fig)

            # Pages: Offline failure (§3.1 — tracking-phase collisions)
            if data.failure_offline is not None and not data.failure_offline.empty:
                fig = self._pdf_table_page(data.failure_offline, "Offline Failure Rate")
                pdf.savefig(fig)
                plt.close(fig)

            # Pages: Online summary tables
            for metric in self.ONLINE_KEY_METRICS:
                if metric in data.online_summary:
                    fig = self._pdf_table_page(
                        data.online_summary[metric],
                        f"Online Summary: {metric.replace('_', ' ').title()}",
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

            # Pages: Key online plots
            for pref in data.online_plots:
                fig = self._pdf_plot_page(pref.absolute_path, pref.title)
                if fig is not None:
                    pdf.savefig(fig)
                    plt.close(fig)

            # Final page: Conclusions
            pdf.savefig(self._pdf_conclusions_page(data))

        logger.info("PDF report: %s", pdf_path)
        return pdf_path

    def _pdf_title_page(self, data: ReportData) -> matplotlib.figure.Figure:
        """Zbuduj stronę tytułową PDF (A4) z metadanymi eksperymentu."""
        fig = plt.figure(figsize=(8.27, 11.69))  # A4
        fig.text(
            0.5, 0.65, "Experiment Report",
            ha="center", va="center", fontsize=28,
            fontfamily="serif", fontweight="bold",
        )
        fig.text(
            0.5, 0.57, data.experiment_name,
            ha="center", va="center", fontsize=16,
            fontfamily="serif", style="italic",
        )
        fig.text(
            0.5, 0.48,
            f"Generated: {data.generation_date}",
            ha="center", va="center", fontsize=11, fontfamily="serif",
        )

        meta_lines = [
            f"Total runs: {data.n_runs}",
            f"Algorithms: {', '.join(data.algorithms)}",
            f"Environments: {', '.join(data.environments)}",
        ]
        if data.avoidances:
            meta_lines.append(f"Avoidances: {', '.join(data.avoidances)}")
        meta_lines.append(f"Friedman datasets (env x seed): {data.n_datasets}")

        fig.text(
            0.5, 0.35, "\n".join(meta_lines),
            ha="center", va="center", fontsize=10,
            fontfamily="serif", linespacing=1.8,
        )

        fig.text(
            0.5, 0.08,
            "Statistical methodology: Friedman + Nemenyi (Demsar 2006),\n"
            "Vargha-Delaney A12 effect size (Vargha & Delaney 2000),\n"
            "Wilson 95% CI for proportions (Wilson 1927; Newcombe 1998).",
            ha="center", va="center", fontsize=8,
            fontfamily="serif", color="gray",
        )
        return fig

    # Columns that are internal metadata and should not appear in report tables.
    _DROP_COLUMNS = {"low_power_warning"}

    def _pdf_table_page(
        self,
        df: pd.DataFrame,
        title: str,
    ) -> matplotlib.figure.Figure:
        """Zrenderuj `df` jako tabelę matplotlib na stronie A4 z tytułem."""
        # Drop internal columns
        display_df = df.drop(
            columns=[c for c in self._DROP_COLUMNS if c in df.columns],
        ).copy()

        # Format numeric columns to 4 decimal places
        for col in display_df.select_dtypes(include=["float64", "float32"]).columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

        # Estimate table height fraction (header + data rows)
        n_rows = len(display_df) + 1
        row_height_frac = 0.035
        table_height_frac = min(n_rows * row_height_frac, 0.70)

        # Place axes in the vertical center of the page
        ax_bottom = max(0.5 - table_height_frac / 2 - 0.02, 0.05)
        ax_height = min(table_height_frac + 0.04, 0.85)

        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0.05, ax_bottom, 0.90, ax_height])
        ax.axis("off")

        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width(col=list(range(len(display_df.columns))))

        # Style header row
        for j in range(len(display_df.columns)):
            cell = table[0, j]
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")

        # Alternate row shading
        for i in range(1, len(display_df) + 1):
            for j in range(len(display_df.columns)):
                if i % 2 == 0:
                    table[i, j].set_facecolor("#D9E2F3")

        # Title above table
        title_y = ax_bottom + ax_height + 0.03
        fig.text(
            0.5, min(title_y, 0.95), title,
            ha="center", va="bottom", fontsize=14,
            fontfamily="serif", fontweight="bold",
        )

        return fig

    def _pdf_plot_page(
        self, png_path: Path, title: str,
    ) -> Optional[matplotlib.figure.Figure]:
        """Osadź PNG `png_path` na stronie A4 z tytułem; `None` przy błędzie odczytu."""
        if not png_path.exists():
            return None
        try:
            img = plt.imread(str(png_path))
        except Exception:
            logger.warning("Cannot read plot: %s", png_path)
            return None

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.imshow(img, aspect="equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontfamily="serif", fontweight="bold", pad=10)
        fig.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.95)
        return fig

    def _pdf_conclusions_page(self, data: ReportData) -> matplotlib.figure.Figure:
        """Wyrenderuj końcową stronę raportu z rankingiem i `key_findings`."""
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(
            0.5, 0.90, "Summary and Key Findings",
            ha="center", va="center", fontsize=20,
            fontfamily="serif", fontweight="bold",
        )

        # Overall ranking
        if data.overall_ranking is not None and not data.overall_ranking.empty:
            ranking_text = "Overall Offline Ranking (avg Friedman rank):\n\n"
            for i, row in data.overall_ranking.iterrows():
                ranking_text += f"  {int(i) + 1}. {row['optimizer']}  —  {row['avg_rank']:.2f}\n"
            fig.text(
                0.10, 0.75, ranking_text,
                ha="left", va="top", fontsize=11,
                fontfamily="serif", linespacing=1.6,
            )

        # Key findings
        if data.key_findings:
            findings_text = "Key Findings:\n\n"
            for finding in data.key_findings:
                # Wrap long lines
                wrapped = _wrap_text(finding, max_chars=85)
                findings_text += f"  \u2022 {wrapped}\n"
            fig.text(
                0.10, 0.50, findings_text,
                ha="left", va="top", fontsize=10,
                fontfamily="serif", linespacing=1.6,
            )

        fig.text(
            0.5, 0.05,
            "Generated by ReportGenerator | src/analysis/analyzer/report_generator.py",
            ha="center", va="center", fontsize=8,
            fontfamily="serif", color="gray",
        )
        return fig


def _wrap_text(text: str, max_chars: int = 85) -> str:
    """Zwróć `text` zawinięty po słowach do maks. `max_chars` znaków na linię."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = "    " + word  # indent continuation
        else:
            current = current + " " + word if current else word
    if current:
        lines.append(current)
    return "\n".join(lines)
