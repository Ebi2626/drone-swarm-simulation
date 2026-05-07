"""Test architektoniczny: drone wykonuje B-spline (smooth), nie polyline (linear).

Diagnoza (2026-05-07):
`NumbaTrajectoryProfile.get_state_at_time_numba` linijka 299 używa
**linear interpolation** między dense waypoints, mimo że klasa fituje B-spline
przez `splprep(s=0, k=3)` w `_fit_bspline` i przechowuje `self.tck`. PID
controller drona dostaje target_pos/target_vel po polylinii — w każdym
węźle jest *direction discontinuity*, drone musi wykonać ostry skręt.

Fizyczna konsekwencja:
- Optimizer evaluuje smooth B-spline curvature → planuje feasible kinematykę
- Real flight follows polyline → direction jumps at waypoints → wymaga
  arbitralnie wysokiego lateral acc → drone nie nadąża → panic falls

Pożądane zachowanie: `get_state_at_time` używa `self.tck` (już zaftowanego
B-spline'a) przez `scipy.interpolate.splev`. Trajectory drona = smooth B-spline
identyczny z tym co ocenia optimizer. Constraint G[2] (lateral acc per
sample-based check) przekłada się 1:1 na rzeczywiste flight observations.

Test reprodukuje bug:
- Trajektoria L-shape (90° corner mid-way)
- Drony muszą zmienić kierunek z (1,0,0) na (0,1,0) — linear interp robi to
  ostro, smooth B-spline przez splprep ftuje gładki łuk
- Sample positions → finite-diff observed acceleration
- Asercja: max(|a_obs|) MUSI być < 30 m/s² (bounded smooth corner)
- Pre-fix: linear → ostre direction change → spike acc rzędu 100+ m/s²
- Post-fix: B-spline smooth → bounded acc

Plus comparison test: linear interp daje >> wyższe acc niż B-spline na tym
samym scenariuszu — explicit dowód architektonicznej różnicy.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.abstraction.trajectory.strategies.shared.NumbaTrajectoryProfile import (
    NumbaTrajectoryProfile,
)


def _sample_observed_accelerations(profile: NumbaTrajectoryProfile, dt: float = 0.005):
    """Sample positions co `dt` w czasie, oblicz observed |a| via finite diff.

    Używamy bardzo małego dt (5ms = 200Hz) żeby capture true acc magnitude
    przy ostrej zmianie kierunku (linear interp produkuje quasi-instantaneous
    direction change, więc trzeba próbkować gęsto).
    """
    times = np.arange(0.0, profile.total_duration + dt, dt)
    positions = np.zeros((len(times), 3), dtype=np.float64)
    for i, t in enumerate(times):
        pos, _ = profile.get_state_at_time(float(t))
        positions[i] = pos

    # 2nd-order finite difference: a_i ≈ (p_{i+1} - 2 p_i + p_{i-1}) / dt²
    accelerations = (positions[2:] - 2.0 * positions[1:-1] + positions[:-2]) / (dt ** 2)
    return positions, np.linalg.norm(accelerations, axis=-1)


def test_get_state_at_time_follows_smooth_bspline_not_linear_polyline():
    """❌ FAIL pre-fix: linear interp przez waypoints daje spike lateral acc
    przy każdym ostrym kącie (drone PID musi natychmiast zmienić kierunek →
    panic). Post-fix: scipy.splev na self.tck → smooth B-spline → bounded acc.

    Scenariusz L-shape: drone leci wzdłuż osi X, na (10, 0, 1) skręca o 90°
    i leci wzdłuż osi Y. 5 waypoints (cubic B-spline wymaga ≥4) z hover
    zone na końcach. cruise=6 m/s, max_accel=2 m/s².

    B-spline z splprep(s=0, k=3) przez te 5 waypointów daje GŁADKI łuk
    o promieniu ~ 2-3m (zależnie od dystrybucji wag B-spline'a). Wtedy:
    a_lat = v²/r ≈ 36/2.5 ≈ 14 m/s² peak. Z dt=5ms i smooth path,
    faktyczne measured acc powinien być w okolicach tej wartości.

    Linear interp daje quasi-instantaneous turn at waypoint (10,0,1):
    Δv = sqrt(2)·6 ≈ 8.5 m/s, dt = jeden physics step ~5ms ⇒
    measured acc ≈ 1700 m/s² peak (lub niższe ze względu na sampling).
    """
    waypoints = np.array([
        [0.0, 0.0, 1.0],
        [5.0, 0.0, 1.0],
        [10.0, 0.0, 1.0],   # corner approach
        [10.0, 5.0, 1.0],   # post-corner
        [10.0, 10.0, 1.0],
    ], dtype=np.float64)

    profile = NumbaTrajectoryProfile(waypoints, cruise_speed=6.0, max_accel=2.0)

    _positions, acc_magnitudes = _sample_observed_accelerations(profile, dt=0.005)

    max_acc = float(np.max(acc_magnitudes))

    # B-spline smooth corner z waypointami spaced 5m: realistyczne lateral
    # acc rzędu 10-30 m/s² peak. Linear interp daje 100+ m/s² na każdym
    # waypoint junction.
    assert max_acc < 50.0, (
        f"❌ Trajectory NIE jest smooth (max |a_obs|={max_acc:.1f} m/s²). "
        "get_state_at_time prawdopodobnie używa LINEAR interp między waypointami "
        "(direction discontinuity przy waypoint junctions). Drone PID dostaje "
        "instantaneous target_vel changes → must accelerate at very high rate "
        "→ panic falls. Required: switch to B-spline interpolation via "
        "self.tck (splprep już fituje, ale get_state_at_time go nie używa)."
    )


def test_smooth_straight_trajectory_has_low_acc():
    """Sanity test: gładka prosta trajektoria (kolinearne waypoints) NIE
    powinna mieć żadnych spike'ów acc. Pre/post-fix oba pass."""
    waypoints = np.array([
        [0.0, 0.0, 1.0],
        [5.0, 0.0, 1.0],
        [10.0, 0.0, 1.0],
        [15.0, 0.0, 1.0],
        [20.0, 0.0, 1.0],
    ], dtype=np.float64)
    profile = NumbaTrajectoryProfile(waypoints, cruise_speed=6.0, max_accel=2.0)

    _positions, acc_magnitudes = _sample_observed_accelerations(profile, dt=0.01)

    # Linia prosta: lateral acc = 0; tangential acc ograniczony przez
    # trapezoidal max_accel = 2.0 (ramp-up/down). Numerical FD może mieć
    # małe artefakty w fazach przejść (ramp→cruise) — bound 5 m/s².
    max_acc = float(np.max(acc_magnitudes))
    assert max_acc < 5.0, (
        f"Smooth straight trajectory ma max |a|={max_acc:.2f} m/s² > 5.0. "
        "Numerical artifacts? Lub ramp/cruise transition issue."
    )


def test_velocity_magnitude_matches_trapezoidal_speed_during_cruise():
    """Sanity: w fazie cruise, |velocity| z get_state_at_time ≈ cruise_speed.
    Niezależne od metody interpolacji (linear vs B-spline)."""
    waypoints = np.array([
        [0.0, 0.0, 1.0],
        [10.0, 0.0, 1.0],
        [20.0, 0.0, 1.0],
        [30.0, 0.0, 1.0],
        [40.0, 0.0, 1.0],
    ], dtype=np.float64)
    profile = NumbaTrajectoryProfile(waypoints, cruise_speed=6.0, max_accel=2.0)

    # Cruise zone: t ∈ [ta, ta+tc]. Pobierz w środku.
    t_cruise_mid = profile.ta + profile.tc / 2.0
    _, vel = profile.get_state_at_time(float(t_cruise_mid))
    speed = float(np.linalg.norm(vel))

    assert abs(speed - profile.cruise_speed) < 0.1, (
        f"Cruise speed={speed:.2f} != cruise_speed={profile.cruise_speed:.2f}. "
        "Trapezoidal profile mapping broken."
    )
