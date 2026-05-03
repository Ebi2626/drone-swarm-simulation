"""Testy CruiseDecelProfile (Bug #2 plan, Krok 1).

Profil ma:
  - v(0) = cruise_speed (drone wchodzi z uformowanym wektorem prędkości)
  - v(t_c) = cruise_speed (koniec fazy cruise)
  - v(t_total) = 0 (lub v_end > 0 dla bardzo krótkich tras)
  - ciągłe ramp hamowania → finite-diff |a|(t) ograniczone przez max_accel.
"""
import numpy as np
import pytest

from src.trajectory.CruiseDecelProfile import CruiseDecelProfile


def test_full_profile_decel_to_zero():
    """Długa trasa: pełna faza cruise + pełna deceleracja do v=0."""
    p = CruiseDecelProfile(total_distance=100.0, cruise_speed=8.0, max_accel=2.0)

    # Faza decel: t_d = cruise/max_accel = 4.0 s, s_d = 0.5*8*4 = 16 m.
    assert p.t_d == pytest.approx(4.0)
    assert p.s_d == pytest.approx(16.0)
    assert p.v_end == pytest.approx(0.0)
    assert p.t_c == pytest.approx((100.0 - 16.0) / 8.0)
    assert p.total_duration == pytest.approx(p.t_c + p.t_d)

    # Punkt początkowy: v = cruise_speed
    _, v0 = p.get_state(0.0)
    assert v0 == pytest.approx(8.0)

    # Punkt końcowy: v = 0
    _, v_end = p.get_state(p.total_duration)
    assert v_end == pytest.approx(0.0)

    # Środek decel: v = cruise/2
    t_mid_decel = p.t_c + 0.5 * p.t_d
    _, v_mid = p.get_state(t_mid_decel)
    assert v_mid == pytest.approx(4.0, abs=1e-6)


def test_short_route_no_full_decel():
    """Krótka trasa: nie ma miejsca na decel do 0, kończymy z v_end > 0.

    cruise=8, max_accel=2, s_d_full = 16. Trasa = 5 m < 16 m.
    v_end² = cruise² - 2*a*s = 64 - 20 = 44 → v_end ≈ 6.63 m/s.
    """
    p = CruiseDecelProfile(total_distance=5.0, cruise_speed=8.0, max_accel=2.0)
    assert p.t_c == pytest.approx(0.0)
    assert p.s_c == pytest.approx(0.0)
    assert p.v_end == pytest.approx(np.sqrt(64.0 - 20.0), abs=1e-6)
    assert p.total_duration == pytest.approx((8.0 - p.v_end) / 2.0)

    # v(0) = cruise, v(end) = v_end
    _, v0 = p.get_state(0.0)
    _, v_end = p.get_state(p.total_duration)
    assert v0 == pytest.approx(8.0)
    assert v_end == pytest.approx(p.v_end, abs=1e-6)


def test_finite_diff_accel_bounded_by_max_accel():
    """Regresja Bug #2: finite-diff |a|(t) NIE eksploduje na końcu krzywej
    (jak to robił `ConstantSpeedProfile`)."""
    p = CruiseDecelProfile(total_distance=80.0, cruise_speed=6.0, max_accel=2.0)
    t = np.linspace(0.0, p.total_duration, 200)
    speeds = np.array([p.get_state(float(ti))[1] for ti in t])

    # |dv/dt| centralne — z marginesem 1.5× na numerykę dyskretyzacji.
    dt = t[1] - t[0]
    accel = np.abs(np.diff(speeds) / dt)
    assert accel.max() <= p.max_accel * 1.5, (
        f"|a|(t) przekracza max_accel: peak={accel.max():.3f}, max_accel={p.max_accel}"
    )


def test_zero_distance_returns_zero():
    p = CruiseDecelProfile(total_distance=0.0, cruise_speed=5.0, max_accel=2.0)
    dist, speed = p.get_state(1.0)
    assert dist == 0.0
    assert speed == 0.0
    assert p.total_duration == 0.0


def test_distance_monotonic_non_decreasing():
    p = CruiseDecelProfile(total_distance=50.0, cruise_speed=7.0, max_accel=1.5)
    t = np.linspace(0.0, p.total_duration, 100)
    distances = np.array([p.get_state(float(ti))[0] for ti in t])
    diffs = np.diff(distances)
    assert (diffs >= -1e-9).all(), "dystans nie może maleć"


def test_kinematic_safe_total_accel_bounded(tmp_path=None):
    """Bug #2 Krok 5: |a_total| (lateral + longitudinal) ≤ max_accel
    na krzywej z dużą krzywizną (zigzag), gdy `decel_at_end=True`.

    Weryfikuje, że BSplineTrajectory clampuje cruise_speed wg κ_max
    i adaptuje decel rate. Bez tego clamping'u lateral acc = v²·κ wybijał
    drony z PID w trakcie symulacji (4/5 dronów upadło).
    """
    from src.trajectory.BSplineTrajectory import BSplineTrajectory

    # Aggresywny zigzag: K=0.5/m w środkowych segmentach.
    wp = np.array([
        [0.0, 0.0, 3.0],
        [2.0, 2.0, 3.0],
        [4.0, -2.0, 3.0],
        [6.0, 2.0, 3.0],
        [8.0, 0.0, 3.0],
    ])
    cruise = 6.0
    max_accel = 2.0
    traj = BSplineTrajectory(
        waypoints=wp, cruise_speed=cruise, max_accel=max_accel,
        constant_speed=True, decel_at_end=True,
    )

    # Cruise musi zostać sklamrowane bo κ_max jest duża.
    assert traj.kinematic_clamp["applied_cruise"] < cruise, (
        f"clamp nie zadziałał: applied={traj.kinematic_clamp['applied_cruise']}"
    )
    assert traj.kinematic_clamp["applied_decel"] <= max_accel + 1e-6

    # Próbkujemy pos i vel; liczymy a_total = sqrt(longitudinal² + lateral²).
    n = 300
    t = np.linspace(0.0, traj.total_duration, n)
    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))
    for i, ti in enumerate(t):
        p, v = traj.get_state_at_time(float(ti))
        positions[i] = p
        velocities[i] = v

    dt = t[1] - t[0]
    accel = np.diff(velocities, axis=0) / dt
    accel_norm = np.linalg.norm(accel, axis=1)

    # Marża 1.3× na numerykę (dyskretyzacja t i splev).
    assert accel_norm.max() <= max_accel * 1.3, (
        f"|a_total|.max() = {accel_norm.max():.3f} przekracza max_accel={max_accel} (×1.3 marża)"
    )


def test_api_compat_with_trapezoidal():
    """Profil eksponuje pola wymagane przez `BSplineTrajectory.get_state_at_time`."""
    p = CruiseDecelProfile(total_distance=20.0, cruise_speed=4.0, max_accel=1.0)
    for attr in ("t_a", "s_a", "t_c", "s_c", "t_d", "v_peak",
                 "total_duration", "max_accel", "cruise_speed", "total_distance"):
        assert hasattr(p, attr), f"brak atrybutu {attr}"
