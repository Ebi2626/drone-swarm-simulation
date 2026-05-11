"""ExperimentAggregator — orkestrator ETL.

Kolejność (krytyczna):
1. `initialize_database`  — schema.
2. `populate_database`    — ładuje CSV/h5 do tabel surowych + liczy MOO
                            quality bez referencji (spread, spacing, R2).
3. `build_reference_pareto_sets`     — merged ND-front cross-run per
                                       (env, n_obj). Wymaga wszystkich runów
                                       w DB (krok 2 musi być kompletny).
4. `backfill_moo_quality_with_reference` — re-liczy GD/IGD+ z R,
                                       UPDATE'uje iteration_metrics +
                                       run_metrics.

Reference: Riquelme, Lücken & Baran (2015) "Performance metrics in
multi-objective optimization", CLEI EJ 18(1).
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from src.analysis.db import initialize_database, populate_database
from src.analysis.db.build_reference_pareto import (
    backfill_moo_quality_with_reference,
    build_reference_pareto_sets,
)


logger = logging.getLogger(__name__)


class ExperimentAggregator:
    """Orkiestrator ETL — populuje bazę SQLite analiz dla całego eksperymentu."""

    def aggregate(self, experiment_dir: str | Path) -> Path:
        """Zbuduj kompletną bazę `analysis.db` dla `experiment_dir`.

        Uruchamia 4 fazy: inicjalizację schematu, populację surowych tabel
        i metryk MOO bez referencji, budowę cross-run reference Pareto sets
        oraz backfill GD/IGD+ względem nowo zbudowanej referencji. Drugi
        etap jest atomowy — wyjątek powoduje jawny `rollback` (bez tego
        częściowe `DELETE` zostawia pustą tabelę po commicie).

        Args:
            experiment_dir: Katalog eksperymentu zawierający per-run
                podkatalogi z CSV/h5.

        Returns:
            Ścieżkę do utworzonej (lub zaktualizowanej) bazy `analysis.db`.
        """
        experiment_dir = Path(experiment_dir).expanduser().resolve()
        db_path = initialize_database(experiment_dir)
        populate_database(experiment_dir)

        # Cross-run post-pass: reference Pareto sets + GD/IGD+ backfill.
        # Atomicity: w przypadku wyjątku robimy explicit rollback, żeby
        # częściowy `DELETE FROM reference_pareto_sets` (z `build_reference_pareto_sets`)
        # nie utknął jako pusty stan w DB. Bez `rollback` w `except`, blok
        # `with sqlite3.connect` commit'uje przy normalnym wyjściu (a my
        # wyjątek zżeramy → wyjście "normalne" → commit pustej tabeli).
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            try:
                refs = build_reference_pareto_sets(conn, experiment_dir)
                if refs:
                    backfill_moo_quality_with_reference(conn, refs)
                    logger.info(
                        f"ExperimentAggregator: backfilled GD/IGD+ dla "
                        f"{len(refs)} grup (env, n_obj)."
                    )
                else:
                    logger.info(
                        "ExperimentAggregator: brak runów z h5 do "
                        "zbudowania reference Pareto sets — pomijam GD/IGD+ backfill."
                    )
                conn.commit()
            except Exception as e:  # pragma: no cover
                conn.rollback()
                logger.error(
                    f"ExperimentAggregator: błąd budowy reference set / "
                    f"backfill (rollback wykonany): {e}",
                    exc_info=True,
                )

        return db_path
