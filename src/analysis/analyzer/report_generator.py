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
    """Reference to a plot file for report embedding."""
    title: str
    absolute_path: Path
    relative_path: str  # relative to report dir, for markdown


@dataclass
class ReportData:
    """All data needed to render both report formats."""
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
    significant_pairs: dict[str, pd.DataFrame] = field(default_factory=dict)
    a12_effects: dict[str, pd.DataFrame] = field(default_factory=dict)

    # Online
    online_summary: dict[str, pd.DataFrame] = field(default_factory=dict)
    has_online_data: bool = False

    # Failure
    failure_offline: Optional[pd.DataFrame] = None
    failure_online: Optional[pd.DataFrame] = None

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

    OFFLINE_KEY_METRICS = [
        "hypervolume",
        "igd_plus",
        "gd_final",
        "convergence_speed_gen",
        "front_size_last_gen",
    ]

    ONLINE_KEY_METRICS = [
        "collision_count",
        "evasion_event_count",
        "min_inter_uav_distance_m",
        "mean_inter_uav_distance_m",
        "mean_energy_indicator",
        "mean_smoothness_indicator",
    ]

    # Plot subdirectory -> list of (file_pattern, human title).
    # {env} and {metric} are replaced when scanning.
    _OFFLINE_PLOT_SPECS: list[tuple[str, str, str]] = [
        ("boxplots", "boxplot_{env}_hypervolume.png", "Boxplot: Hypervolume ({env})"),
        ("boxplots", "boxplot_{env}_igd_plus.png", "Boxplot: IGD+ ({env})"),
        ("boxplots", "boxplot_{env}_gd_final.png", "Boxplot: GD ({env})"),
        ("convergence", "convergence_{env}_hypervolume.png", "Convergence: Hypervolume ({env})"),
        ("convergence", "convergence_{env}_best_so_far.png", "Convergence: Best-so-far ({env})"),
        ("cd_diagrams", "cd_hypervolume.png", "CD Diagram: Hypervolume"),
        ("cd_diagrams", "cd_igd_plus.png", "CD Diagram: IGD+"),
        ("rankings", "ranking_hypervolume.png", "Ranking Heatmap: Hypervolume"),
        ("rankings", "ranking_igd_plus.png", "Ranking Heatmap: IGD+"),
    ]

    _ONLINE_PLOT_SPECS: list[tuple[str, str, str]] = [
        ("bar", "bar_{env}_success_rate.png", "Success Rate ({env})"),
        ("bar", "bar_{env}_collisions.png", "Collision Count ({env})"),
        ("bar", "bar_{env}_failure_rate_online.png", "Online Failure Rate ({env})"),
    ]

    def __init__(self, experiment_dir: str | Path) -> None:
        self.experiment_dir = Path(experiment_dir).resolve()
        self.tables_dir = self.experiment_dir / "analysis_output" / "tables"
        self.plots_dir = self.experiment_dir / "analysis_output" / "plots"
        self.out_dir = self.experiment_dir / "analysis_output" / "report"

    def generate(self) -> tuple[Path, Path]:
        """Generate both Markdown and PDF reports.

        Returns:
            (md_path, pdf_path) tuple of generated file paths.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        data = self._collect_data()
        md_path = self._generate_markdown(data)
        pdf_path = self._generate_pdf(data)
        logger.info("Report generated: %s, %s", md_path, pdf_path)
        return md_path, pdf_path

    def _collect_data(self) -> ReportData:
        """Read existing CSV tables and plot PNGs into ReportData."""
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

        # Friedman ranks
        friedman_ranks: dict[str, pd.DataFrame] = {}
        for m in self.OFFLINE_KEY_METRICS:
            df = self._read_csv(f"friedman_{m}.csv")
            if df is not None:
                friedman_ranks[m] = df

        # Wilcoxon pairwise (filter significant only: p_holm < 0.05)
        significant_pairs: dict[str, pd.DataFrame] = {}
        for m in self.OFFLINE_KEY_METRICS:
            df = self._read_csv(f"wilcoxon_{m}.csv")
            if df is not None and "p_value_holm" in df.columns:
                sig = df[df["p_value_holm"] < 0.05]
                if not sig.empty:
                    significant_pairs[m] = sig

        # A12 effect sizes
        a12_effects: dict[str, pd.DataFrame] = {}
        for m in self.OFFLINE_KEY_METRICS:
            df = self._read_csv(f"a12_{m}.csv")
            if df is not None:
                a12_effects[m] = df

        # Online summaries
        online_summary: dict[str, pd.DataFrame] = {}
        has_online = False
        for m in self.ONLINE_KEY_METRICS:
            df = self._read_csv(f"online_summary_{m}.csv")
            if df is not None:
                online_summary[m] = df
                has_online = True

        # Failure rates
        failure_offline = self._read_csv("failure_rate_offline.csv")
        failure_online = self._read_csv("failure_rate_online.csv")

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
            online_summary, failure_offline, failure_online,
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
            significant_pairs=significant_pairs,
            a12_effects=a12_effects,
            online_summary=online_summary,
            has_online_data=has_online,
            failure_offline=failure_offline,
            failure_online=failure_online,
            best_per_metric=best_per_metric,
            overall_ranking=overall_ranking,
            key_findings=key_findings,
            offline_plots=offline_plots,
            online_plots=online_plots,
        )

    def _detect_values(self, col: str) -> list[str]:
        """Detect unique values of `col` from any available summary CSV."""
        for m in self.OFFLINE_KEY_METRICS:
            df = self._read_csv(f"summary_{m}.csv")
            if df is not None and col in df.columns:
                return sorted(df[col].dropna().unique().tolist())
        return []

    def _detect_avoidances(self) -> list[str]:
        """Detect avoidance algorithms from analysis.db runs table."""
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
        """Count total runs from analysis.db."""
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
        """Extract best algorithm (rank 1) per metric from Friedman results."""
        if not friedman:
            return None
        rows = []
        for metric, df in friedman.items():
            if "avg_rank" not in df.columns or df.empty:
                continue
            best_idx = df["avg_rank"].idxmin()
            best = df.loc[best_idx]
            rows.append({
                "metric": metric,
                "best_algorithm": best["optimizer"],
                "avg_rank": best["avg_rank"],
                "p_value": best.get("p_value", float("nan")),
            })
        return pd.DataFrame(rows) if rows else None

    def _compute_overall_ranking(
        self, friedman: dict[str, pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        """Average Friedman rank across all key offline metrics."""
        if not friedman:
            return None
        all_ranks = []
        for df in friedman.values():
            if "avg_rank" not in df.columns or "optimizer" not in df.columns:
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

    def _collect_plots(
        self,
        specs: list[tuple[str, str, str]],
        environments: list[str],
    ) -> list[_PlotRef]:
        """Collect existing plot PNGs matching the spec patterns."""
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
        failure_online: Optional[pd.DataFrame],
    ) -> list[str]:
        """Auto-generate key findings from data."""
        findings: list[str] = []

        # 1. Best overall offline algorithm
        if overall_ranking is not None and not overall_ranking.empty:
            best = overall_ranking.iloc[0]
            n_metrics = len(friedman_ranks)
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

        # 3. Safest online (lowest collision count)
        if "collision_count" in online_summary:
            cc = online_summary["collision_count"]
            if "mean" in cc.columns and "optimizer" in cc.columns:
                safest_idx = cc["mean"].idxmin()
                safest = cc.loc[safest_idx]
                findings.append(
                    f"Lowest mean collision count: {safest['optimizer']} "
                    f"({safest['mean']:.2f} collisions, "
                    f"environment: {safest.get('environment', 'all')})."
                )

        # 4. Offline failure hot-spots
        if failure_offline is not None and "failure_rate" in failure_offline.columns:
            worst_idx = failure_offline["failure_rate"].idxmax()
            worst = failure_offline.loc[worst_idx]
            if worst["failure_rate"] > 0:
                findings.append(
                    f"Highest offline failure rate: {worst['optimizer']} "
                    f"in {worst['environment']} "
                    f"({worst['failure_rate']:.0%} of runs)."
                )

        # 5. Online failure hot-spots
        if failure_online is not None and "failure_rate" in failure_online.columns:
            worst_idx = failure_online["failure_rate"].idxmax()
            worst = failure_online.loc[worst_idx]
            if worst["failure_rate"] > 0:
                findings.append(
                    f"Highest online failure rate: {worst['optimizer']} "
                    f"in {worst['environment']} "
                    f"({worst['failure_rate']:.0%} of runs)."
                )

        if not findings:
            findings.append("Insufficient data for automated findings.")

        return findings

    def _generate_markdown(self, data: ReportData) -> Path:
        """Render jinja2 template to Markdown."""
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
            significant_pairs=data.significant_pairs,
            a12_effects=data.a12_effects,
            online_summary=data.online_summary,
            has_online_data=data.has_online_data,
            failure_offline=data.failure_offline,
            failure_online=data.failure_online,
            best_per_metric=data.best_per_metric,
            overall_ranking=data.overall_ranking,
            key_findings=data.key_findings,
            offline_plots=data.offline_plots,
            online_plots=data.online_plots,
        )

        md_path = self.out_dir / "experiment_report.md"
        md_path.write_text(rendered, encoding="utf-8")
        logger.info("Markdown report: %s", md_path)
        return md_path

    def _generate_pdf(self, data: ReportData) -> Path:
        """Compile multi-page PDF report using matplotlib PdfPages."""
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

            # Pages: Key offline plots
            for pref in data.offline_plots:
                fig = self._pdf_plot_page(pref.absolute_path, pref.title)
                if fig is not None:
                    pdf.savefig(fig)
                    plt.close(fig)

            # Pages: Failure rates
            for label, df in [
                ("Offline Failure Rate", data.failure_offline),
                ("Online Failure Rate", data.failure_online),
            ]:
                if df is not None and not df.empty:
                    fig = self._pdf_table_page(df, label)
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
            "Statistical methodology: Friedman + Nemenyi (Demsar 2006), "
            "Wilcoxon + Holm (Arcuri & Briand 2014),\n"
            "Vargha-Delaney A12 effect size, Bootstrap CI (10k resamples).",
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
        """Render a DataFrame as a matplotlib table on an A4 page."""
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
        """Embed a PNG plot image on an A4 page."""
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
        """Render key findings as text on the final page."""
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
    """Simple word-wrap for matplotlib text (no textwrap to avoid indent issues)."""
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
