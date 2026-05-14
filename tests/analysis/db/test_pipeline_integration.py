"""Integration test pełnego pipeline'u (Faza 4 plan.md).

Przygotuje katalog runa z minimalnym zestawem CSV (trajectories + online +
convergence) i sprawdza że:
- `vw_run_summary` ma kolumny online metric i niezerowe wartości.
- `vw_run_online_summary` agreguje z `online_optimization_tasks` + `run_metrics`.
- `vw_algo_cross_sim_comparison` poprawnie agreguje przez algorithm × env.
"""
from __future__ import annotations

import csv
import sqlite3
import tempfile
from pathlib import Path

import pytest


def _setup_run_dir(exp_dir: Path, run_dir_name: str) -> Path:
    run_dir = exp_dir / run_dir_name
    run_dir.mkdir(parents=True)

    with (run_dir / "world_boundaries.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Axis", "Dimension", "Min_Bound", "Max_Bound", "Center"])
        w.writerow(["X", 100, 0, 100, 50])
        w.writerow(["Y", 100, 0, 100, 50])
        w.writerow(["Z", 10, 0, 10, 5])

    # 2 drony, 10 sampli, 5 m apart, constant velocity 1 m/s — pos
    # increments dx=0.1 m per dt=0.1 s, zgodne z logged vx=1.0 m/s.
    with (run_dir / "trajectories.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "drone_id", "x", "y", "z",
                    "roll", "pitch", "yaw", "vx", "vy", "vz"])
        for i in range(10):
            t = round(i * 0.1, 3)
            x = round(i * 0.1, 3)
            w.writerow([t, 0, x, 0.0, 5.0, 0, 0, 0, 1.0, 0.0, 0.0])
            w.writerow([t, 1, x, 5.0, 5.0, 0, 0, 0, 1.0, 0.0, 0.0])

    with (run_dir / "online_optimization.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id", "drone_id", "trigger_time", "algorithm", "status", "reason",
            "best_fitness", "evaluations_completed", "generations_completed",
            "wallclock_s", "time_budget_s", "chosen_axis", "plan_waypoints_json",
            "plan_total_duration_s", "plan_arc_length_m", "outcome",
            "pos_err_at_rejoin_m", "vel_err_at_rejoin_mps", "time_to_rejoin_s",
        ])
        w.writerow([
            run_dir_name, 0, 0.5, "MSFOA", "ok", "ok",
            50.0, 100, 10, 0.4, 0.5, "right", "[]",
            1.5, 5.0, "rejoined_ok", 0.1, 0.05, 1.0,
        ])

    with (run_dir / "convergence_traces.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "drone_id", "trigger_time", "algorithm",
                    "generation", "best_fitness"])
        for g, fit in enumerate([100.0, 70.0, 50.0]):
            w.writerow([run_dir_name, 0, 0.5, "MSFOA", g, fit])

    # optimization_history.h5 z 5-obj F-vector (best feasible w gen=2 ind=1).
    try:
        import h5py
        import numpy as np
        h5_dir = run_dir / "optimization_history"
        h5_dir.mkdir(parents=True, exist_ok=True)
        # 3 generacje × 4 individuals × 5 objectives.
        obj = np.full((3, 4, 5), 999.0, dtype=np.float64)
        obj[2, 1] = [11.0, 22.0, 33.0, 44.0, 55.0]   # best last-gen
        obj[2, 0] = [99.0, 99.0, 99.0, 99.0, 99.0]
        with h5py.File(h5_dir / "optimization_history.h5", "w") as h:
            h.create_dataset("objectives_matrix", data=obj)
    except ImportError:
        pass  # h5py unavailable — skip h5 fixture

    return run_dir


@pytest.fixture
def populated_db():
    from src.analysis.db.initialize_database import initialize_database
    from src.analysis.db.populate_database import populate_database

    with tempfile.TemporaryDirectory() as tmp:
        exp_dir = Path(tmp) / "exp"
        _setup_run_dir(exp_dir, "msffoa_forest_msffoa_seed1")
        _setup_run_dir(exp_dir, "msffoa_forest_msffoa_seed2")

        initialize_database(exp_dir)
        populate_database(exp_dir)

        db_path = exp_dir / "analysis.db"
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            yield conn


class TestPipelineIntegration:
    def test_run_metrics_has_online_aggregates(self, populated_db) -> None:
        rows = populated_db.execute(
            """
            SELECT run_id, min_inter_uav_distance_m, mean_inter_uav_distance_m,
                   total_inter_uav_safety_violations,
                   mean_energy_indicator, mean_smoothness_indicator
            FROM run_metrics
            ORDER BY run_id
            """
        ).fetchall()
        assert len(rows) == 2
        for r in rows:
            assert r["min_inter_uav_distance_m"] == pytest.approx(5.0, abs=1e-6)
            assert r["mean_inter_uav_distance_m"] == pytest.approx(5.0, abs=1e-6)
            assert r["total_inter_uav_safety_violations"] == 0
            assert r["mean_energy_indicator"] == pytest.approx(1.0, abs=1e-3)
            assert r["mean_smoothness_indicator"] == pytest.approx(0.0, abs=1e-3)

    def test_vw_run_summary_exposes_online(self, populated_db) -> None:
        rows = populated_db.execute(
            "SELECT min_inter_uav_distance_m, mean_energy_indicator FROM vw_run_summary"
        ).fetchall()
        assert all(r["min_inter_uav_distance_m"] == pytest.approx(5.0, abs=1e-6) for r in rows)

    def test_vw_run_online_summary_agg(self, populated_db) -> None:
        rows = populated_db.execute(
            """
            SELECT run_id, algorithm, total_evasion_triggers,
                   min_inter_uav_distance_m, mean_energy_indicator
            FROM vw_run_online_summary
            ORDER BY run_id
            """
        ).fetchall()
        assert len(rows) == 2
        for r in rows:
            assert r["algorithm"] == "MSFOA"
            assert r["total_evasion_triggers"] == 1
            assert r["min_inter_uav_distance_m"] == pytest.approx(5.0, abs=1e-6)

    def test_vw_algo_cross_sim_comparison(self, populated_db) -> None:
        rows = populated_db.execute(
            """
            SELECT environment, algorithm, runs_count,
                   total_evasion_triggers_all_runs,
                   mean_min_inter_uav_distance_m, total_inter_uav_safety_violations,
                   mean_energy_indicator
            FROM vw_algo_cross_sim_comparison
            """
        ).fetchall()
        assert len(rows) == 1
        r = rows[0]
        assert r["environment"] == "forest"
        assert r["algorithm"] == "MSFOA"
        assert r["runs_count"] == 2
        assert r["total_evasion_triggers_all_runs"] == 2
        assert r["mean_min_inter_uav_distance_m"] == pytest.approx(5.0, abs=1e-6)
        assert r["total_inter_uav_safety_violations"] == 0

    def test_run_files_registers_online_csvs(self, populated_db) -> None:
        rows = populated_db.execute(
            """
            SELECT file_role, exists_flag FROM run_files
            WHERE file_role IN ('online_optimization_csv', 'convergence_traces_csv')
            """
        ).fetchall()
        roles = {r["file_role"]: r["exists_flag"] for r in rows}
        assert roles.get("online_optimization_csv") == 1
        assert roles.get("convergence_traces_csv") == 1

    def test_evasion_trajectory_metrics_aggregated(self, populated_db) -> None:
        """§3.1.3.3: agregaty trajektorii fazy online z `online_optimization_tasks`.

        Fixture ma jeden task per run z:
            plan_arc_length_m=5.0, plan_total_duration_s=1.5,
            pos_err_at_rejoin_m=0.1, vel_err_at_rejoin_mps=0.05,
            time_to_rejoin_s=1.0, outcome='rejoined_ok'.
        AVG po jednym wpisie = wartość wpisu; rejoin_success_rate = 1/1 = 1.0.
        """
        rows = populated_db.execute(
            """
            SELECT mean_evasion_arc_length_m, median_evasion_arc_length_m,
                   mean_evasion_plan_duration_s,
                   mean_pos_err_at_rejoin_m, mean_vel_err_at_rejoin_mps,
                   mean_time_to_rejoin_s, rejoin_success_rate,
                   rejoin_completion_rate, budget_violation_rate,
                   online_success_rate, online_sp1,
                   mean_online_evaluations_completed
            FROM run_metrics
            ORDER BY run_id
            """
        ).fetchall()
        assert len(rows) == 2
        for r in rows:
            assert r["mean_evasion_arc_length_m"] == pytest.approx(5.0)
            assert r["median_evasion_arc_length_m"] == pytest.approx(5.0)
            assert r["mean_evasion_plan_duration_s"] == pytest.approx(1.5)
            assert r["mean_pos_err_at_rejoin_m"] == pytest.approx(0.1)
            assert r["mean_vel_err_at_rejoin_mps"] == pytest.approx(0.05)
            assert r["mean_time_to_rejoin_s"] == pytest.approx(1.0)
            assert r["rejoin_success_rate"] == pytest.approx(1.0)
            # Fixture ma 1 trigger z outcome=rejoined_ok (sukces, 0 fail) →
            # completion_rate = 1/(1+0) = 1.0. Brak pending w fixture →
            # completion_rate == success_rate.
            assert r["rejoin_completion_rate"] == pytest.approx(1.0)
            # Fixture ma status='ok' → 0 violations / 1 trigger = 0.0.
            assert r["budget_violation_rate"] == pytest.approx(0.0)
            # Fixture: status='ok' → success_rate = 1.0,
            # SP1 = RT_succ / p_s = NFE / 1.0 = NFE.
            # (Brak failed tasks w fixture → RT_succ = mean NFE.)
            assert r["online_success_rate"] == pytest.approx(1.0)
            assert r["online_sp1"] == pytest.approx(r["mean_online_evaluations_completed"])

    def test_offline_objectives_extracted_from_h5(self, populated_db) -> None:
        """Best feasible F-vector z h5 trafia do `run_metrics`.

        Po 2026-05-13 `final_objective` jest skalarem weighted-normalized-sum
        liczonym w post-pass `populate_final_objective_aggregated`:
            final_objective = Σ w_i · F[i] / F_ref_env[i]
        Gdy fixture ma wszystkie runy z identycznym F = [11, 22, 33, 44, 55],
        median = F, więc F/F_ref = 1.0 per komponent → final_objective = Σ w_i.
        Canonical weights `[0.05, 0.5, 0.8, 1.0, 0.25]` → sum = 2.6 (brak hydra
        configu w fixture'ach → canonical fallback).
        """
        pytest.importorskip("h5py")
        rows = populated_db.execute(
            """
            SELECT final_objective, final_objective_f1_trajectory,
                   final_objective_f2_height_angle, total_threat_cost,
                   total_turn_penalty, total_coordination_cost,
                   final_objective_weights_json
            FROM run_metrics
            ORDER BY run_id
            """
        ).fetchall()
        for r in rows:
            # final_objective = Σ canonical_weights = 2.6 dla identical-F fixture'ów.
            assert r["final_objective"] == pytest.approx(2.6)
            assert r["final_objective_f1_trajectory"] == pytest.approx(11.0)
            assert r["final_objective_f2_height_angle"] == pytest.approx(22.0)
            assert r["total_threat_cost"] == pytest.approx(33.0)
            assert r["total_turn_penalty"] == pytest.approx(44.0)
            assert r["total_coordination_cost"] == pytest.approx(55.0)
            assert r["final_objective_weights_json"] is not None
