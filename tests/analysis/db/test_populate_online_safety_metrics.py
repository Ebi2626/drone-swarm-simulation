"""Testy dla `populate_online_safety_metrics` (Faza 3 plan.md, notatki.md).

Sprawdza:
- Inter-UAV distance liczona poprawnie dla 2 dronów lecących równolegle.
- Single-drone case → wszystkie inter-UAV pola NULL.
- Energy indicator wyższy dla szybszego drona.
- Smoothness indicator wyższy dla drona z accel niż dla drona constant-vel.
- `safety_violation_count` reaguje na threshold.
"""
from __future__ import annotations

import math
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.analysis.db.initialize_database import initialize_database
from src.analysis.db.populate_online_safety_metrics import (
    populate_online_safety_metrics,
)


def _make_conn() -> tuple[sqlite3.Connection, str]:
    """Pusta baza + 1 wpis runa + zwrot connection."""
    tmp = tempfile.mkdtemp()
    exp_dir = Path(tmp) / "exp"
    exp_dir.mkdir()
    db_path = initialize_database(exp_dir)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    run_id = "msffoa_forest_msffoa_seed1"
    conn.execute(
        """
        INSERT INTO runs (run_id, run_dir_name, source_path,
            optimizer_algo, avoidance_algo, environment, seed,
            algorithm_pair)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, run_id, str(exp_dir / run_id),
         "msffoa", "msffoa", "forest", 1, "msffoa + msffoa"),
    )
    return conn, run_id


def _insert_samples(
    conn: sqlite3.Connection, run_id: str,
    drone_id: int, samples: list[tuple[float, float, float, float, float, float, float]]
) -> None:
    """samples = [(t, x, y, z, vx, vy, vz), ...]."""
    rows = []
    for i, (t, x, y, z, vx, vy, vz) in enumerate(samples):
        rows.append((run_id, i, t, drone_id, x, y, z, 0, 0, 0, vx, vy, vz))
    conn.executemany(
        """
        INSERT INTO trajectory_samples
            (run_id, sample_index, sim_time, drone_id, x, y, z,
             roll, pitch, yaw, vx, vy, vz)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


class TestInterUAVDistance:
    def test_two_parallel_drones_5m_apart(self) -> None:
        conn, run_id = _make_conn()
        N = 10
        d0 = [(i * 0.1, i * 1.0, 0.0, 5.0, 1.0, 0.0, 0.0) for i in range(N)]
        d1 = [(i * 0.1, i * 1.0, 5.0, 5.0, 1.0, 0.0, 0.0) for i in range(N)]
        _insert_samples(conn, run_id, 0, d0)
        _insert_samples(conn, run_id, 1, d1)

        populate_online_safety_metrics(conn, run_id, safety_threshold_m=1.0)

        rows = conn.execute(
            """
            SELECT uav_id, min_inter_uav_distance_m, mean_inter_uav_distance_m,
                   inter_uav_safety_violation_count
            FROM uav_online_metrics WHERE run_id = ?
            ORDER BY uav_id
            """,
            (run_id,),
        ).fetchall()
        assert len(rows) == 2
        for uav_id, min_d, mean_d, viol in rows:
            assert min_d == pytest.approx(5.0, abs=1e-6)
            assert mean_d == pytest.approx(5.0, abs=1e-6)
            assert viol == 0  # 5 m > 1 m threshold

    def test_violations_with_high_threshold(self) -> None:
        conn, run_id = _make_conn()
        N = 10
        d0 = [(i * 0.1, i * 1.0, 0.0, 5.0, 1.0, 0.0, 0.0) for i in range(N)]
        d1 = [(i * 0.1, i * 1.0, 5.0, 5.0, 1.0, 0.0, 0.0) for i in range(N)]
        _insert_samples(conn, run_id, 0, d0)
        _insert_samples(conn, run_id, 1, d1)

        populate_online_safety_metrics(conn, run_id, safety_threshold_m=10.0)

        rows = conn.execute(
            "SELECT inter_uav_safety_violation_count FROM uav_online_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        assert all(r[0] == N for r in rows)  # 5 m < 10 m threshold = każda próbka łamie

    def test_single_drone_returns_none_for_inter_uav(self) -> None:
        conn, run_id = _make_conn()
        N = 5
        _insert_samples(conn, run_id, 0,
                        [(i * 0.1, i * 1.0, 0.0, 5.0, 1.0, 0.0, 0.0) for i in range(N)])
        populate_online_safety_metrics(conn, run_id)
        row = conn.execute(
            """
            SELECT min_inter_uav_distance_m, mean_inter_uav_distance_m,
                   inter_uav_safety_violation_count
            FROM uav_online_metrics WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        assert row == (None, None, None)


class TestEnergyIndicator:
    def test_faster_drone_has_higher_energy_indicator(self) -> None:
        conn, run_id = _make_conn()
        N = 10
        # Drone 0: constant 1 m/s
        d0 = [(i * 0.1, i * 0.1, 0.0, 5.0, 1.0, 0.0, 0.0) for i in range(N)]
        # Drone 1: constant 5 m/s — pokonuje 5x więcej drogi w tym samym czasie.
        d1 = [(i * 0.1, i * 0.5, 5.0, 5.0, 5.0, 0.0, 0.0) for i in range(N)]
        _insert_samples(conn, run_id, 0, d0)
        _insert_samples(conn, run_id, 1, d1)
        populate_online_safety_metrics(conn, run_id)

        rows = conn.execute(
            "SELECT uav_id, energy_indicator FROM uav_online_metrics WHERE run_id = ? ORDER BY uav_id",
            (run_id,),
        ).fetchall()
        e0, e1 = rows[0][1], rows[1][1]
        # energy = ∫v² dt / path_length_3d
        # Dla constant v: ∫v² dt = v² * T; path = v * T → energy = v.
        assert e0 == pytest.approx(1.0, abs=1e-3)
        assert e1 == pytest.approx(5.0, abs=1e-3)
        assert e1 > e0

    def test_speed_stats(self) -> None:
        conn, run_id = _make_conn()
        N = 5
        _insert_samples(conn, run_id, 0,
                        [(i * 0.1, i * 0.3, 0.0, 5.0, 3.0, 0.0, 0.0) for i in range(N)])
        populate_online_safety_metrics(conn, run_id)
        row = conn.execute(
            "SELECT mean_speed_mps, max_speed_mps FROM uav_online_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert row[0] == pytest.approx(3.0, abs=1e-6)
        assert row[1] == pytest.approx(3.0, abs=1e-6)


class TestSmoothnessIndicator:
    def test_constant_velocity_zero_smoothness(self) -> None:
        conn, run_id = _make_conn()
        N = 10
        _insert_samples(conn, run_id, 0,
                        [(i * 0.1, i * 0.5, 0.0, 5.0, 5.0, 0.0, 0.0) for i in range(N)])
        populate_online_safety_metrics(conn, run_id)
        row = conn.execute(
            "SELECT smoothness_indicator FROM uav_online_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        assert row[0] == pytest.approx(0.0, abs=1e-6)

    def test_accelerating_drone_higher_smoothness_cost(self) -> None:
        conn, run_id = _make_conn()
        N = 10
        # constant-vel drone
        d0 = [(i * 0.1, i * 0.5, 0.0, 5.0, 5.0, 0.0, 0.0) for i in range(N)]
        # accelerating: vx rośnie liniowo z 1 do 10 m/s
        d1 = []
        for i in range(N):
            t = i * 0.1
            v = 1.0 + i * 1.0
            x = 0.5 * (1.0 + v) * t  # not exact but ok for test
            d1.append((t, x, 5.0, 5.0, v, 0.0, 0.0))
        _insert_samples(conn, run_id, 0, d0)
        _insert_samples(conn, run_id, 1, d1)
        populate_online_safety_metrics(conn, run_id)

        rows = conn.execute(
            "SELECT uav_id, smoothness_indicator FROM uav_online_metrics WHERE run_id = ? ORDER BY uav_id",
            (run_id,),
        ).fetchall()
        s0, s1 = rows[0][1], rows[1][1]
        assert s0 == pytest.approx(0.0, abs=1e-6)
        assert s1 > 1.0  # accel ≈ 10 m/s² → ‖a‖² ≈ 100 → indicator ≫ 0


class TestIdempotency:
    def test_rerun_does_not_duplicate(self) -> None:
        conn, run_id = _make_conn()
        N = 5
        _insert_samples(conn, run_id, 0,
                        [(i * 0.1, i * 0.5, 0.0, 5.0, 5.0, 0.0, 0.0) for i in range(N)])
        populate_online_safety_metrics(conn, run_id)
        populate_online_safety_metrics(conn, run_id)
        n = conn.execute(
            "SELECT COUNT(*) FROM uav_online_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
        assert n == 1

    def test_no_samples_no_rows(self) -> None:
        conn, run_id = _make_conn()
        populate_online_safety_metrics(conn, run_id)
        n = conn.execute(
            "SELECT COUNT(*) FROM uav_online_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
        assert n == 0
