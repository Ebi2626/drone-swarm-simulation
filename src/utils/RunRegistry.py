import logging
import sqlite3
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


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

        if row is None:
            # Sygnał diagnostyczny: job-key z main.py nie pasuje do żadnego
            # wpisu PENDING. Najczęstsza przyczyna — niespójność między
            # `prepare_experiment.populate(...)` a `_get_registry_job_key`,
            # albo wczytanie eksperymentu z innym `experiment_meta.id`.
            # Job się uruchomi (zachowanie wsteczne), a UPSERT w mark_started
            # utworzy wiersz — ale brak ostrzeżenia byłby przyczyną „registry
            # niespójny z parquet" (patrz plan.md, Krok 6, dawna patologia
            # exp_20260426_b9b56922_complex_test).
            logger.warning(
                f"[RunRegistry] Brak wpisu dla klucza "
                f"(optimizer={optimizer!r}, environment={environment!r}, "
                f"avoidance={avoidance!r}, seed={seed}). "
                f"Run zostanie uruchomiony i UPSERT-owany — sprawdź "
                f"prepare_experiment.populate() i `experiment_meta.id`."
            )
            return True

        # Wznawiamy PENDING, FAILED oraz przerwane STARTED
        return row[0] in ("PENDING", "FAILED", "STARTED")

    def mark_started(self, optimizer, environment, avoidance, seed):
        # Aktualizujemy czas startu i jednocześnie czyścimy stary czas zakończenia
        # oraz komunikaty o błędach (niezbędne przy wznawianiu po crashu)
        self._upsert_status(optimizer, environment, avoidance, seed,
                            "STARTED",
                            started_at=datetime.utcnow().isoformat(),
                            finished_at=None,
                            error_msg=None)

    def mark_completed(self, optimizer, environment, avoidance, seed):
        self._upsert_status(optimizer, environment, avoidance, seed,
                            "COMPLETED", finished_at=datetime.utcnow().isoformat())
        self._log_progress("COMPLETED")

    def mark_failed(self, optimizer, environment, avoidance, seed, error_msg: str):
        self._upsert_status(optimizer, environment, avoidance, seed,
                            "FAILED", finished_at=datetime.utcnow().isoformat(),
                            error_msg=error_msg[:1000])
        self._log_progress("FAILED")

    def _log_progress(self, event: str) -> None:
        """
        Loguje postęp eksperymentu po każdym mark_completed / mark_failed.
        Pojedyncze SELECT GROUP BY — narzut zaniedbywalny względem kosztu
        symulacji PyBullet. Używamy logger.info, żeby progresu nie
        gubił `>/dev/null` ani filtry warning-only.

        Format: `[RunRegistry] {event}: 7/12 done (5 success, 2 failed),
                 1 running, 4 pending`
        """
        summary = self.get_summary()
        completed = summary.get("COMPLETED", 0)
        failed = summary.get("FAILED", 0)
        started = summary.get("STARTED", 0)
        pending = summary.get("PENDING", 0)
        total = completed + failed + started + pending
        done = completed + failed

        logger.info(
            f"[RunRegistry] {event}: {done}/{total} done "
            f"({completed} success, {failed} failed), "
            f"{started} running, {pending} pending"
        )

    def _upsert_status(self, optimizer, environment, avoidance, seed, status, **fields):
        """
        UPSERT: jeśli row dla (optimizer, environment, avoidance, seed) istnieje,
        aktualizuje `status` + pola w `fields`; w przeciwnym razie wstawia
        nowy wiersz z domyślnymi NULL-ami dla pominiętych pól.

        Niezbędne gdy `populate()` nie został wywołany albo job-key z main.py
        nie pasuje do żadnego wpisu PENDING (poprzednia wersja używała UPDATE,
        który był no-op i registry zostawał pusty — patrz plan.md, Krok 6).
        """
        # INSERT zawiera 4 unique cols + status + przekazane pola.
        insert_cols = ["optimizer", "environment", "avoidance", "seed", "status"] + list(fields.keys())
        insert_vals = [optimizer, environment, avoidance, seed, status] + list(fields.values())
        placeholders = ", ".join("?" * len(insert_cols))
        cols_sql = ", ".join(insert_cols)

        # ON CONFLICT aktualizuje TYLKO status + pola explicite przekazane
        # (np. mark_completed zostawia started_at z mark_started bez zmian).
        update_cols = ["status"] + list(fields.keys())
        update_sql = ", ".join(f"{c}=excluded.{c}" for c in update_cols)

        with self._connect() as conn:
            conn.execute(f"""
                INSERT INTO runs ({cols_sql})
                VALUES ({placeholders})
                ON CONFLICT(optimizer, environment, avoidance, seed) DO UPDATE SET
                    {update_sql}
            """, insert_vals)

    def get_summary(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM runs GROUP BY status"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def _ensure_quality_columns(self) -> None:
        """Idempotentna migracja: dodaje kolumny `optimization_path`,
        `motion_observed`, `data_quality_flag` do tabeli `runs` jeśli ich nie
        ma. Pozwala wzbogacić istniejące bazy bez recreate.
        """
        with self._connect() as conn:
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
            for col, sql_type in (
                ("optimization_path", "TEXT"),
                ("motion_observed", "INTEGER"),
                ("data_quality_flag", "TEXT"),
            ):
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {sql_type}")

    def reconcile_with_parquet(self, parquet_path: str | Path) -> dict:
        """
        Wzbogaca registry o flagi jakości danych z `master_metrics.parquet`.

        Dla każdego wiersza parquet aktualizuje `optimization_path`,
        `motion_observed`, `data_quality_flag` w registry, łącząc po
        (optimizer, environment, avoidance, seed). Wypisuje WARNING dla
        wpisów parquet, których nie ma w registry — sygnał, że populate
        nie pokrył wszystkich kombinacji albo job-key z main.py rozjeżdża się
        ze schematem prepare_experiment.

        Pozwala traktować registry jako single source of truth dla raportów
        statusu (status × data_quality_flag) bez ręcznego joinowania
        parquet ⇄ sqlite.

        Args:
            parquet_path: Ścieżka do `master_metrics.parquet` zbudowanego
                przez ExperimentAggregator.

        Returns:
            Dict ze statystykami: ``{"updated": int, "missing_in_registry": int}``.
        """
        import pandas as pd  # lazy: pandas niepotrzebny dla zwykłego mark_*

        df = pd.read_parquet(parquet_path)
        required = {"optimizer", "environment", "avoidance", "seed",
                    "optimization_path", "motion_observed", "data_quality_flag"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"parquet {parquet_path} brakuje wymaganych kolumn: {missing}"
            )

        self._ensure_quality_columns()

        updated = 0
        missing_in_registry = 0
        with self._connect() as conn:
            for row in df.itertuples(index=False):
                cur = conn.execute("""
                    UPDATE runs SET
                        optimization_path = ?,
                        motion_observed = ?,
                        data_quality_flag = ?
                    WHERE optimizer=? AND environment=? AND avoidance=? AND seed=?
                """, (
                    str(row.optimization_path),
                    int(bool(row.motion_observed)),
                    str(row.data_quality_flag),
                    str(row.optimizer),
                    str(row.environment),
                    str(row.avoidance),
                    int(row.seed),
                ))
                if cur.rowcount > 0:
                    updated += 1
                else:
                    missing_in_registry += 1
                    logger.warning(
                        f"[RunRegistry] reconcile: brak w registry dla "
                        f"(optimizer={row.optimizer!r}, environment={row.environment!r}, "
                        f"avoidance={row.avoidance!r}, seed={row.seed})"
                    )

        result = {"updated": updated, "missing_in_registry": missing_in_registry}
        logger.info(f"[RunRegistry] reconcile_with_parquet: {result}")
        return result