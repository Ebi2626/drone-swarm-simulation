"""
Sanity-check trajektorii zaraz po wyjściu ze strategii optymalizacji.

Wykrywa:
  - NaN / Inf w pozycjach,
  - drony, które przeleciały <1 m (per-drone path length),
  - drony, których ostatni waypoint pokrywa się z pozycją startową
    (strategia "zawróciła w miejscu" — niezależnie od jej wewnętrznej
    metryki sukcesu).

NIE rzuca wyjątków — tylko `print` z prefiksem ``⚠`` na stdout, dzięki
czemu komunikat trafia do logów workerów Hydra-Joblib *przed* startem
PyBullet. Bez tego patologie były wykrywane dopiero w ETL post-mortem
(patrz plan.md, Krok 2).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


_NEAR_ZERO_M = 1.0  # próg odróżniający „dron nie ruszył" od dryfu numerycznego


def validate_trajectories(
    trajectories: NDArray[np.float64],
    start_positions: NDArray[np.float64],
    label: str = "trajectory",
) -> dict[str, Any]:
    """Sprawdza poprawność trajektorii zwróconej przez strategię.

    Args:
        trajectories: (n_drones, n_waypoints, 3) — wynik strategii optymalizacji.
        start_positions: (n_drones, 3) — punkty startowe (do porównania
            z ostatnim waypoint'em w sekcji „dron nie ruszył").
        label: Etykieta używana w komunikatach (np. nazwa strategii).

    Returns:
        Dict z wynikami:
        ``{"finite": bool, "near_zero_drones": list[int],
        "stuck_at_start_drones": list[int], "per_drone_length_m": list[float]}``.
        Funkcja nie zwraca True/False jednoznacznego — rozdzielamy patologie,
        żeby caller (np. ETL) mógł je rozróżniać.
    """
    arr = np.asarray(trajectories, dtype=np.float64)
    starts = np.asarray(start_positions, dtype=np.float64)

    if arr.ndim != 3 or arr.shape[2] != 3:
        print(
            f"⚠ [{label}] niespodziewany kształt trajektorii: {arr.shape} "
            f"(oczekiwane (n_drones, n_waypoints, 3))"
        )
        return {
            "finite": False,
            "near_zero_drones": [],
            "stuck_at_start_drones": [],
            "per_drone_length_m": [],
        }

    n_drones, n_wp, _ = arr.shape
    finite = bool(np.isfinite(arr).all())

    # Per-drone path length
    if n_wp >= 2:
        seg = np.linalg.norm(np.diff(arr, axis=1), axis=2)  # (n_drones, n_wp-1)
        per_drone_len = seg.sum(axis=1)
    else:
        per_drone_len = np.zeros(n_drones)

    near_zero = np.where(per_drone_len < _NEAR_ZERO_M)[0].tolist()

    # „Stuck at start" — ostatni waypoint blisko punktu startowego
    end_to_start = np.linalg.norm(arr[:, -1, :] - starts, axis=1)
    stuck = np.where(end_to_start < _NEAR_ZERO_M)[0].tolist()

    if not finite:
        print(f"⚠ [{label}] zawiera NaN/Inf — drony mogą zawisnąć w PyBullet!")

    if near_zero:
        print(
            f"⚠ [{label}] drony {near_zero} mają trajektorie poniżej "
            f"{_NEAR_ZERO_M:.0f}m: per_drone_len={per_drone_len.round(2).tolist()}. "
            f"PyBullet prawdopodobnie nie ruszy ich z miejsca."
        )

    if stuck:
        print(
            f"⚠ [{label}] drony {stuck} kończą trajektorię w pobliżu pozycji "
            f"startowej (end-to-start={end_to_start.round(2).tolist()} m). "
            f"Strategia mogła zwrócić zdegenerowaną trajektorię (zawrót w miejscu)."
        )

    if finite and not near_zero and not stuck:
        print(
            f"[{label}] OK: {n_drones} dronów × {n_wp} waypointów. "
            f"Per-drone L [m]: min={per_drone_len.min():.1f}, "
            f"max={per_drone_len.max():.1f}, mean={per_drone_len.mean():.1f}"
        )

    return {
        "finite": finite,
        "near_zero_drones": near_zero,
        "stuck_at_start_drones": stuck,
        "per_drone_length_m": per_drone_len.tolist(),
    }
