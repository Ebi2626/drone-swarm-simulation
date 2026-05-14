"""Post-hoc kalkulator metryk online z `trajectory_samples`.

Liczy 3 grupy metryk per UAV i zapisuje do `uav_online_metrics`:

1. **Inter-UAV Collision Avoidance** — min/mean odległości tego UAVa od
   najbliższego sąsiada w roju w czasie. Liczone na wspólnej osi czasu
   (próbki tej samej `sim_time` agregowane przez `INTERSECT` po `drone_id`).

2. **Energy efficiency proxy** — `∫ ‖v‖² dt / total_path_length_3d`. Standard
   w UAV literature: F_drag ∝ v², więc ∫ v² dt to proxy total drag work.
   Reference: McAllister et al. (2017), "Quantifying energy efficiency of
   multirotor UAV trajectories."

3. **Trajectory smoothness** — `∫ ‖a‖² dt`, gdzie a liczone z różnicowania
   prędkości po próbkach. Standard w trajectory smoothness: minimum-acceleration
   cost (Hauser & Hubicki 2007). Niższa wartość = gładsza trasa.

Dane wejściowe: `trajectory_samples (run_id, sample_index, sim_time, drone_id,
x, y, z, vx, vy, vz)` — schemat zdefiniowany w `schema.sql` § 13.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

import numpy as np


# Domyślny próg bezpieczeństwa pomiędzy dronami (m).
# 4.0 m = 2× collision_radius (CF2X w pyBullet, `configs/config.yaml`):
# dwa drony stykają się collision spheres przy odległości środków = 2 × 2 m.
# Próg 1.0 m był wcześniej zawsze poniżej realnego min_inter_uav (~1.9 m)
# — skutkował zerowym `inter_uav_safety_violation_count` w analizach.
DEFAULT_INTER_UAV_SAFETY_THRESHOLD_M = 4.0


def populate_online_safety_metrics(
    conn: sqlite3.Connection,
    run_id: str,
    safety_threshold_m: float = DEFAULT_INTER_UAV_SAFETY_THRESHOLD_M,
) -> None:
    """Wylicz per-UAV metryki bezpieczeństwa, energii i gładkości; zapisz do `uav_online_metrics`.

    Idempotentna: czyści istniejące wiersze `WHERE run_id = ?` i wstawia świeże.
    Wywoływana po `populate_uav_metrics`, a przed `populate_run_metrics` (które
    agreguje wyniki do poziomu run).

    Args:
        conn: Aktywne połączenie do bazy.
        run_id: Identyfikator runa.
        safety_threshold_m: Próg minimalnej akceptowalnej odległości
            inter-UAV [m]; naruszenia liczone do `violation_count`.

    Efekty uboczne:
        Nadpisuje wiersze `uav_online_metrics` dla `run_id`.
    """
    samples = _fetch_trajectory_samples(conn, run_id)
    if not samples:
        conn.execute(
            "DELETE FROM uav_online_metrics WHERE run_id = ?", (run_id,)
        )
        return

    per_drone = _split_per_drone(samples)
    drone_ids = sorted(per_drone.keys())

    # Inter-UAV distance liczona w siatce time × drone — wymaga wspólnej osi
    # czasu. `_align_time_grid` zwraca:
    #   times: (T,) sorted unique sim_times
    #   pos[d_id]: (T, 3) z NaN gdzie próbki brak.
    times, aligned_pos = _align_time_grid(per_drone)
    inter_uav_per_drone = _compute_inter_uav_distance_per_drone(
        aligned_pos, drone_ids, safety_threshold_m
    )

    rows: list[tuple] = []
    for drone_id in drone_ids:
        d = per_drone[drone_id]
        kin = _compute_kinematic_metrics(d)

        inter = inter_uav_per_drone.get(drone_id, {
            "min_distance_m": None,
            "max_distance_m": None,
            "mean_distance_m": None,
            "violation_count": None,
        })

        extra = {
            "computation": "polyline_metrics + pairwise_min_distance",
            "safety_threshold_m": safety_threshold_m,
            "energy_indicator_unit": "m/s^2",
            "smoothness_indicator_unit": "m^2/s^3",
        }

        rows.append((
            run_id,
            int(drone_id),
            inter["min_distance_m"],
            inter["max_distance_m"],
            inter["mean_distance_m"],
            inter["violation_count"],
            float(safety_threshold_m),
            kin["energy_indicator"],
            kin["speed_squared_integral"],
            kin["mean_speed_mps"],
            kin["max_speed_mps"],
            kin["smoothness_indicator"],
            kin["accel_squared_integral"],
            kin["mean_accel_mps2"],
            kin["max_accel_mps2"],
            kin["sample_count"],
            kin["duration_s"],
            json.dumps(extra, ensure_ascii=False, sort_keys=True),
        ))

    conn.execute(
        "DELETE FROM uav_online_metrics WHERE run_id = ?", (run_id,)
    )
    conn.executemany(
        """
        INSERT INTO uav_online_metrics (
            run_id, uav_id,
            min_inter_uav_distance_m, max_inter_uav_distance_m, mean_inter_uav_distance_m,
            inter_uav_safety_violation_count, inter_uav_safety_threshold_m,
            energy_indicator, speed_squared_integral, mean_speed_mps, max_speed_mps,
            smoothness_indicator, accel_squared_integral, mean_accel_mps2, max_accel_mps2,
            sample_count, duration_s, extra_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _fetch_trajectory_samples(
    conn: sqlite3.Connection, run_id: str
) -> list[tuple]:
    cur = conn.execute(
        """
        SELECT drone_id, sim_time, x, y, z, vx, vy, vz
        FROM trajectory_samples
        WHERE run_id = ?
        ORDER BY drone_id, sample_index
        """,
        (run_id,),
    )
    return cur.fetchall()


def _split_per_drone(samples: list[tuple]) -> dict[int, dict]:
    """Pogrupuj próbki po drone_id i zwróć numpy arrays."""
    grouped: dict[int, list[tuple]] = {}
    for drone_id, sim_time, x, y, z, vx, vy, vz in samples:
        grouped.setdefault(int(drone_id), []).append(
            (float(sim_time), float(x), float(y), float(z),
             _none_to_nan(vx), _none_to_nan(vy), _none_to_nan(vz))
        )

    out: dict[int, dict] = {}
    for drone_id, rows in grouped.items():
        arr = np.asarray(rows, dtype=np.float64)
        out[drone_id] = {
            "t": arr[:, 0],
            "pos": arr[:, 1:4],
            "vel": arr[:, 4:7],
        }
    return out


def _none_to_nan(value: Optional[float]) -> float:
    return float("nan") if value is None else float(value)


def _align_time_grid(
    per_drone: dict[int, dict]
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Buduje wspólną oś czasu i pozycje z NaN gdzie próbki brak.

    Logger zapisuje próbki ze stałą `log_freq` synchronizowaną przez
    `ctrl_freq` — w praktyce wszystkie drony mają te same `sim_time`'y
    z dokładnością do `log_step_interval`. Używamy `np.unique` na wszystkich
    timestamps + per-drone `np.searchsorted` do interpolacji nearest.
    """
    all_times = np.unique(
        np.concatenate([d["t"] for d in per_drone.values()])
    )
    aligned: dict[int, np.ndarray] = {}
    for drone_id, d in per_drone.items():
        # Próbka logera ma stałe sim_time — dla każdego elementu all_times,
        # znajdź najbliższy sim_time tego drona (tolerance 1ms). Jeśli brak,
        # NaN.
        idx = np.searchsorted(d["t"], all_times)
        idx = np.clip(idx, 0, len(d["t"]) - 1)
        diffs = np.abs(d["t"][idx] - all_times)
        # Spróbuj poprzedni indeks dla "nearest" semantics.
        prev_idx = np.clip(idx - 1, 0, len(d["t"]) - 1)
        prev_diffs = np.abs(d["t"][prev_idx] - all_times)
        use_prev = prev_diffs < diffs
        nearest_idx = np.where(use_prev, prev_idx, idx)
        nearest_diff = np.where(use_prev, prev_diffs, diffs)

        pos_aligned = d["pos"][nearest_idx].copy()
        # Tolerance: log_freq=10 Hz → dt=0.1 s → tolerancję 0.05 s przyjmujemy.
        out_of_range = nearest_diff > 0.05
        pos_aligned[out_of_range] = np.nan
        aligned[drone_id] = pos_aligned
    return all_times, aligned


def _compute_inter_uav_distance_per_drone(
    aligned_pos: dict[int, np.ndarray],
    drone_ids: list[int],
    safety_threshold_m: float,
) -> dict[int, dict]:
    """Per UAV: min/mean odległości od najbliższego sąsiada w czasie.

    Dla każdego time-step t i UAV i: liczymy `min_j ‖p_i(t) - p_j(t)‖` i
    agregujemy przez `min`/`mean` po t. `violation_count` = liczba time-stepów
    gdzie ten min jest poniżej `safety_threshold_m`.

    Pojedynczy UAV w roju → brak inter-UAV pair → wszystkie pola None.
    """
    if len(drone_ids) < 2:
        return {d: {
            "min_distance_m": None,
            "max_distance_m": None,
            "mean_distance_m": None,
            "violation_count": None,
        } for d in drone_ids}

    # Stack: (D, T, 3).
    stack = np.stack(
        [aligned_pos[d] for d in drone_ids], axis=0
    )
    D, T, _ = stack.shape

    # Pairwise distances per time-step: (D, D, T).
    # diff[i, j, t, axis] = pos[i, t, axis] - pos[j, t, axis].
    diff = stack[:, None, :, :] - stack[None, :, :, :]
    pair_dist = np.linalg.norm(diff, axis=-1)  # (D, D, T)
    # Wykluczyć przekątną (i==j) — ustawiamy na +inf żeby min ignorował.
    for t in range(T):
        np.fill_diagonal(pair_dist[:, :, t], np.inf)

    out: dict[int, dict] = {}
    for i, d in enumerate(drone_ids):
        # Najbliższy sąsiad tego UAVa w każdym time-step.
        per_t_min = np.min(pair_dist[i, :, :], axis=0)  # (T,)
        # Filtr NaN (wynik gdy któryś z UAV nie ma próbki na tym kroku).
        valid = np.isfinite(per_t_min)
        if not np.any(valid):
            out[d] = {
                "min_distance_m": None,
                "max_distance_m": None,
                "mean_distance_m": None,
                "violation_count": None,
            }
            continue
        valid_dists = per_t_min[valid]
        out[d] = {
            "min_distance_m": float(np.min(valid_dists)),
            "max_distance_m": float(np.max(valid_dists)),
            "mean_distance_m": float(np.mean(valid_dists)),
            "violation_count": int(np.sum(valid_dists < safety_threshold_m)),
        }
    return out


def _compute_kinematic_metrics(per_drone: dict) -> dict:
    """Energy + smoothness indicator z trajektorii pojedynczego UAVa.

    Zwraca wszystkie pola (None gdy < 2 próbki).
    """
    t = per_drone["t"]
    pos = per_drone["pos"]
    vel_logged = per_drone["vel"]
    n = len(t)

    if n < 2:
        return _empty_kinematic()

    duration_s = float(t[-1] - t[0])
    if duration_s <= 0.0:
        return _empty_kinematic()

    # Path length 3D (z pozycji — bezpieczniej niż całkować ‖v‖dt jeśli logger
    # zaokrąglił v).
    seg_lengths = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    path_length_3d = float(np.sum(seg_lengths))

    # Velocity norm — preferujemy zalogowane vx/vy/vz, ale gdy NaN (starszy
    # logger lub crash), fallback na różnicowanie pozycji.
    if np.all(np.isfinite(vel_logged)):
        speed = np.linalg.norm(vel_logged, axis=1)  # (n,)
    else:
        # Forward diff: v_i ≈ (pos_{i+1} - pos_i) / dt_i. Ostatnia próbka
        # przyjmuje przedostatnią dla zachowania długości n.
        dt = np.diff(t)
        dt = np.where(dt > 0, dt, 1e-9)
        v_diff = np.diff(pos, axis=0) / dt[:, None]
        speed = np.linalg.norm(v_diff, axis=1)
        speed = np.concatenate([speed, speed[-1:]])

    # Trapezoidal integration of v² over time.
    dt_full = np.diff(t)
    speed_sq = speed ** 2
    speed_sq_integral = float(np.sum(0.5 * (speed_sq[:-1] + speed_sq[1:]) * dt_full))

    # Acceleration: różnicowanie speed lub vel.
    if np.all(np.isfinite(vel_logged)):
        accel_vec = np.diff(vel_logged, axis=0) / np.where(dt_full > 0, dt_full, 1e-9)[:, None]
        accel_norm = np.linalg.norm(accel_vec, axis=1)
    else:
        # Druga pochodna pozycji.
        accel_norm = np.zeros(n - 1)
        for i in range(1, n - 1):
            dt_l = max(t[i] - t[i - 1], 1e-9)
            dt_r = max(t[i + 1] - t[i], 1e-9)
            v_l = (pos[i] - pos[i - 1]) / dt_l
            v_r = (pos[i + 1] - pos[i]) / dt_r
            a = (v_r - v_l) / max((dt_l + dt_r) * 0.5, 1e-9)
            accel_norm[i] = float(np.linalg.norm(a))

    if accel_norm.size == 0:
        return _empty_kinematic()

    accel_sq = accel_norm ** 2
    # Match length to dt: integral over time of ‖a‖².
    if accel_sq.size == dt_full.size:
        accel_sq_integral = float(
            np.sum(0.5 * (accel_sq[:-1] + accel_sq[1:]) * dt_full[:-1])
            if accel_sq.size >= 2 else 0.0
        )
    else:
        accel_sq_integral = float(np.sum(accel_sq) * float(np.mean(dt_full)))

    energy_indicator = (
        speed_sq_integral / path_length_3d if path_length_3d > 1e-9 else None
    )
    smoothness_indicator = (
        accel_sq_integral / duration_s if duration_s > 1e-9 else None
    )

    return {
        "duration_s": duration_s,
        "sample_count": int(n),
        "energy_indicator": energy_indicator,
        "speed_squared_integral": speed_sq_integral,
        "mean_speed_mps": float(np.mean(speed)),
        "max_speed_mps": float(np.max(speed)),
        "smoothness_indicator": smoothness_indicator,
        "accel_squared_integral": accel_sq_integral,
        "mean_accel_mps2": float(np.mean(accel_norm)),
        "max_accel_mps2": float(np.max(accel_norm)),
    }


def _empty_kinematic() -> dict:
    return {
        "duration_s": None,
        "sample_count": None,
        "energy_indicator": None,
        "speed_squared_integral": None,
        "mean_speed_mps": None,
        "max_speed_mps": None,
        "smoothness_indicator": None,
        "accel_squared_integral": None,
        "mean_accel_mps2": None,
        "max_accel_mps2": None,
    }
