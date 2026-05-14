"""Post-pass: TOPSIS composite `run_metrics.rejoin_quality` per run.

§3.1.3.3 docs/Praca magisterska.md — Skuteczność powrotu na nominalną
trajektorię agregowana w pojedynczy skalar przez odległość euklidesową
od idealnego punktu (0,0,0) w 3D znormalizowanej przestrzeni metryk:

  rejoin_quality_i = sqrt(
      (pos_err_at_rejoin_m_i / median_pos_env)²
    + (vel_err_at_rejoin_mps_i / median_vel_env)²
    + (time_to_rejoin_s_i / median_time_env)²
  )

  rejoin_quality_r = mean(rejoin_quality_i) for i in T_r^ok

gdzie median_*_env to medianę odpowiedniej metryki across wszystkich
udanych rejoin'ów w danym środowisku (skala referencyjna wyrównująca
jednostki m, m/s, s).

Wywoływany RAZ po pełnym przejściu populate_run_metrics (medians
wymagają cross-run agregacji per środowisko). Idempotent.

Reference: Hwang, C.-L. & Yoon, K. (1981) Multiple Attribute Decision
Making §4.2 — TOPSIS, odległość od idealnego rozwiązania w przestrzeni
znormalizowanej (cytowane też w §3.1.3.2 dla final_objective).
"""
from __future__ import annotations

import logging
import math
import sqlite3
from collections import defaultdict
from typing import Optional


logger = logging.getLogger(__name__)


_METRICS = (
    "pos_err_at_rejoin_m",
    "vel_err_at_rejoin_mps",
    "time_to_rejoin_s",
)


def populate_rejoin_quality(conn: sqlite3.Connection) -> None:
    """Post-pass UPDATE'ujący `run_metrics.rejoin_quality` we wszystkich runach.

    Idempotent — re-run regeneruje medians i nadpisuje wartości.

    Args:
        conn: Aktywne połączenie do bazy.

    Efekty uboczne:
        UPDATE `run_metrics.rejoin_quality` per row (NULL gdy brak udanych
        rejoin'ów w runie).
    """
    medians = _compute_environment_medians(conn)
    if not medians:
        logger.warning(
            "populate_rejoin_quality: brak udanych rejoin'ów w żadnym "
            "środowisku — rejoin_quality pozostanie NULL we wszystkich runach."
        )
        return

    for env, m in medians.items():
        logger.info(
            "populate_rejoin_quality: env=%r medians pos=%.3fm, vel=%.3fm/s, time=%.3fs",
            env, m[0], m[1], m[2],
        )

    _update_rejoin_quality(conn, medians)


def _compute_environment_medians(
    conn: sqlite3.Connection,
) -> dict[str, tuple[float, float, float]]:
    """Median(pos, vel, time) per environment across all successful rejoins.

    Zwraca słownik env → (median_pos, median_vel, median_time). Pomija
    środowiska bez ani jednego udanego rejoin'u.

    Zero-component guard: median ≤ 1e-9 zastąpione przez 1.0 (neutralny
    mianownik) — np. urban env może mieć median_vel=0 jeśli wszystkie
    udane rejoin'y mają zerowy błąd prędkości. Identyczna logika jak
    w populate_final_objective_aggregated.
    """
    import statistics

    query = """
        SELECT r.environment,
               t.pos_err_at_rejoin_m, t.vel_err_at_rejoin_mps,
               t.time_to_rejoin_s
        FROM online_optimization_tasks t
        JOIN runs r ON r.run_id = t.run_id
        WHERE t.outcome = 'rejoined_ok'
          AND t.pos_err_at_rejoin_m IS NOT NULL
          AND t.vel_err_at_rejoin_mps IS NOT NULL
          AND t.time_to_rejoin_s IS NOT NULL
    """
    by_env: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for env, p, v, t in conn.execute(query):
        by_env[env].append((float(p), float(v), float(t)))

    medians: dict[str, tuple[float, float, float]] = {}
    for env, triples in by_env.items():
        pos_med = float(statistics.median([x[0] for x in triples]))
        vel_med = float(statistics.median([x[1] for x in triples]))
        time_med = float(statistics.median([x[2] for x in triples]))
        # Zero-component guard (zob. populate_final_objective_aggregated).
        if pos_med <= 1e-9:
            pos_med = 1.0
        if vel_med <= 1e-9:
            vel_med = 1.0
        if time_med <= 1e-9:
            time_med = 1.0
        medians[env] = (pos_med, vel_med, time_med)

    return medians


def _update_rejoin_quality(
    conn: sqlite3.Connection,
    medians: dict[str, tuple[float, float, float]],
) -> None:
    """Per-run UPDATE rejoin_quality = mean(TOPSIS_i) for i in T_r^ok."""
    # Pobierz wszystkie rekordy z online_optimization_tasks z udanym rejoin'em
    # oraz environment z runs, w jednym SELECT.
    query = """
        SELECT r.run_id, r.environment,
               t.pos_err_at_rejoin_m, t.vel_err_at_rejoin_mps,
               t.time_to_rejoin_s
        FROM online_optimization_tasks t
        JOIN runs r ON r.run_id = t.run_id
        WHERE t.outcome = 'rejoined_ok'
          AND t.pos_err_at_rejoin_m IS NOT NULL
          AND t.vel_err_at_rejoin_mps IS NOT NULL
          AND t.time_to_rejoin_s IS NOT NULL
    """
    # Agregacja per-run.
    per_run_scores: dict[str, list[float]] = defaultdict(list)
    for run_id, env, pos, vel, t in conn.execute(query):
        m = medians.get(env)
        if m is None:
            continue
        pos_med, vel_med, time_med = m
        score = math.sqrt(
            (pos / pos_med) ** 2 + (vel / vel_med) ** 2 + (t / time_med) ** 2
        )
        per_run_scores[run_id].append(score)

    updates: list[tuple[float, str]] = []
    for run_id, scores in per_run_scores.items():
        mean_score = float(sum(scores) / len(scores))
        updates.append((mean_score, run_id))

    if updates:
        conn.executemany(
            "UPDATE run_metrics SET rejoin_quality = ? WHERE run_id = ?",
            updates,
        )

    logger.info(
        "populate_rejoin_quality: zaktualizowano rejoin_quality dla %d runów "
        "(z udanym rejoin'em); pozostałe %d runów pozostają NULL.",
        len(updates),
        conn.execute("SELECT COUNT(*) FROM run_metrics").fetchone()[0] - len(updates),
    )
