"""Inicjalizacja schematu bazy SQLite (`analysis.db`) dla eksperymentu."""
from pathlib import Path
import sqlite3


def initialize_database(experiment_dir: str | Path) -> Path:
    """Utwórz `analysis.db` w `experiment_dir` i zaaplikuj `schema.sql`.

    Args:
        experiment_dir: Katalog eksperymentu (tworzony, gdy nie istnieje).

    Returns:
        Ścieżkę do utworzonej (lub odświeżonej) bazy danych.

    Efekty uboczne:
        Tworzy plik DB, włącza `PRAGMA foreign_keys`, wykonuje schemat
        i zapisuje wpisy `meta` (`schema_version`, `experiment_dir`,
        `schema_path`).
    """
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    db_path = experiment_dir / "analysis.db"
    schema_path = Path(__file__).resolve().parent / "schema.sql"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    schema_sql = schema_path.read_text(encoding="utf-8")

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(schema_sql)
        conn.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            [
                ("schema_version", "1"),
                ("experiment_dir", str(experiment_dir)),
                ("schema_path", str(schema_path)),
            ],
        )
        conn.commit()

    return db_path