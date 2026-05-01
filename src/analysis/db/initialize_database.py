# src/analysis/db/initialize_database.py
from pathlib import Path
import sqlite3


def initialize_database(experiment_dir: str | Path) -> Path:
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