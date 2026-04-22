import sqlite3
from pathlib import Path
from datetime import datetime

class RunRegistry:
    STATUS = {"PENDING": "PENDING", "STARTED": "STARTED",
              "COMPLETED": "COMPLETED", "FAILED": "FAILED"}

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        # check_same_thread=False wymagane przy joblib multiprocessing
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")  # kluczowe dla concurrent writes
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    optimizer   TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    avoidance   TEXT NOT NULL,
                    seed        INTEGER NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'PENDING',
                    started_at  TEXT,
                    finished_at TEXT,
                    error_msg   TEXT,
                    UNIQUE(optimizer, environment, avoidance, seed)
                )
            """)

    def populate(self, sweep_params: list[dict]):
        """Inicjalizuje 1600 wpisów – wywołać raz, przed startem."""
        with self._connect() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO runs
                    (optimizer, environment, avoidance, seed, status)
                VALUES (:optimizer, :environment, :avoidance, :seed, 'PENDING')
            """, sweep_params)

    def should_run(self, optimizer, environment, avoidance, seed) -> bool:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT status FROM runs
                WHERE optimizer=? AND environment=? AND avoidance=? AND seed=?
            """, (optimizer, environment, avoidance, seed)).fetchone()
        
        # Wznawiamy PENDING, FAILED oraz przerwane STARTED
        return row is None or row[0] in ("PENDING", "FAILED", "STARTED")

    def mark_started(self, optimizer, environment, avoidance, seed):
        # Aktualizujemy czas startu i jednocześnie czyścimy stary czas zakończenia 
        # oraz komunikaty o błędach (niezbędne przy wznawianiu po crashu)
        self._update_status(optimizer, environment, avoidance, seed,
                            "STARTED", 
                            started_at=datetime.utcnow().isoformat(),
                            finished_at=None,
                            error_msg=None)

    def mark_completed(self, optimizer, environment, avoidance, seed):
        self._update_status(optimizer, environment, avoidance, seed,
                            "COMPLETED", finished_at=datetime.utcnow().isoformat())

    def mark_failed(self, optimizer, environment, avoidance, seed, error_msg: str):
        self._update_status(optimizer, environment, avoidance, seed,
                            "FAILED", finished_at=datetime.utcnow().isoformat(),
                            error_msg=error_msg[:1000])

    def _update_status(self, optimizer, environment, avoidance, seed, status, **kwargs):
        set_clause = ", ".join(f"{k}=?" for k in kwargs)
        values = list(kwargs.values()) + [optimizer, environment, avoidance, seed]
        with self._connect() as conn:
            conn.execute(f"""
                UPDATE runs SET status=?, {set_clause}
                WHERE optimizer=? AND environment=? AND avoidance=? AND seed=?
            """, [status] + values)

    def get_summary(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM runs GROUP BY status"
            ).fetchall()
        return {r[0]: r[1] for r in rows}