"""Regression test: tabela `metric_definitions` została usunięta 2026-05-07.

Powód usunięcia: tabela była self-documenting metadata (kolumny
`metric_name, scope, comparable_across_algorithms, description`), ale
przez >6 miesięcy istnienia żaden populator jej nie wypełniał ani żaden
konsumer nie czytał. Semantyka metryk jest dokumentowana w
`src/analysis/ETL_TABLES.md` i komentarzach `schema.sql` przy
`iteration_metrics` / `run_metrics` — tabela była nadmiarowa.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def test_metric_definitions_table_removed(tmp_path: Path) -> None:
    from src.analysis.db.initialize_database import initialize_database

    db_path = initialize_database(tmp_path / "exp")
    conn = sqlite3.connect(db_path)
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "metric_definitions" not in tables, (
        "metric_definitions powinno być usunięte; nikt nigdy z niej nie czytał"
    )


def test_pareto_run_metrics_table_removed(tmp_path: Path) -> None:
    """Tabela `pareto_run_metrics` usunięta 2026-05-07.

    7 z 10 kolumn duplikowało `run_metrics`; pozostałe 3 (epsilon_indicator,
    reference_set_id, indicator_config_json) to YAGNI nieużywane przez
    >6 miesięcy. Single source of truth dla MOO per-run = `run_metrics`.
    """
    from src.analysis.db.initialize_database import initialize_database

    db_path = initialize_database(tmp_path / "exp")
    conn = sqlite3.connect(db_path)
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "pareto_run_metrics" not in tables
