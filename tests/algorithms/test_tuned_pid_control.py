"""TDD failing tests dla `TunedDSLPIDControl` + lookahead + anti-tilt
(Kamień 2026-05-07: PID death-spiral mitigation).

Diagnoza: original DSLPIDControl (gym_pybullet_drones) podczas sharp lateral
maneuver wpada w death-spiral — drone tilts → vertical thrust component drops
< gravity → altitude loss → PID requests more thrust → tilts more → falls.

Tests cover three independent fixes:
- A. **Tuned gains** (P/I reduced, D doubled, integral clamp tighter)
- B. **Lookahead saturation** (cap pos_e seen by PID — w SwarmFlightController)
- C. **Anti-tilt** (clip target_thrust direction at max physical tilt =
    arccos(1/thrust2weight) ≈ 63.6° dla CF2X T2W=2.25)

Każdy fix testowany w izolacji żeby łatwo zlokalizować regression.
"""
from __future__ import annotations

import numpy as np
import pytest


# CF2X URDF: thrust2weight=2.25 → max_tilt = arccos(1/2.25) ≈ 1.111 rad ≈ 63.61°
CF2X_THRUST2WEIGHT = 2.25
EXPECTED_MAX_TILT_RAD = float(np.arccos(1.0 / CF2X_THRUST2WEIGHT))


# ============================================================================
# A. Tuned gains
# ============================================================================


class TestTunedGains:
    def test_default_gains_match_proposal(self):
        """❌ FAIL pre-impl: TunedDSLPIDControl nie istnieje. Po implementacji
        domyślne gains odpowiadają proposalowi (P/3 reduced ~25%, I reduced
        60%, D ~2x)."""
        from gym_pybullet_drones.utils.enums import DroneModel
        from src.algorithms.TunedDSLPIDControl import TunedDSLPIDControl

        ctrl = TunedDSLPIDControl(drone_model=DroneModel.CF2X)
        np.testing.assert_allclose(ctrl.P_COEFF_FOR, [0.3, 0.3, 0.9])
        np.testing.assert_allclose(ctrl.I_COEFF_FOR, [0.02, 0.02, 0.02])
        np.testing.assert_allclose(ctrl.D_COEFF_FOR, [0.4, 0.4, 0.8])

    def test_integral_clamp_tightened(self):
        """Integral clamp ±0.5 (xy) i ±0.05 (z) — vs original ±2.0 / ±0.15.
        Prevents wind-up during sharp maneuvers.
        """
        from gym_pybullet_drones.utils.enums import DroneModel
        from src.algorithms.TunedDSLPIDControl import TunedDSLPIDControl

        ctrl = TunedDSLPIDControl(drone_model=DroneModel.CF2X)
        assert ctrl.integral_clamp_xy == pytest.approx(0.5)
        assert ctrl.integral_clamp_z == pytest.approx(0.05)

    def test_custom_gains_override_defaults(self):
        """Gains konfiguralne via constructor — można dalej tunować."""
        from gym_pybullet_drones.utils.enums import DroneModel
        from src.algorithms.TunedDSLPIDControl import TunedDSLPIDControl

        ctrl = TunedDSLPIDControl(
            drone_model=DroneModel.CF2X,
            p_coeff_for=[1.0, 1.0, 2.0],
            i_coeff_for=[0.1, 0.1, 0.1],
            d_coeff_for=[0.5, 0.5, 0.5],
        )
        np.testing.assert_allclose(ctrl.P_COEFF_FOR, [1.0, 1.0, 2.0])
        np.testing.assert_allclose(ctrl.I_COEFF_FOR, [0.1, 0.1, 0.1])
        np.testing.assert_allclose(ctrl.D_COEFF_FOR, [0.5, 0.5, 0.5])

    def test_integral_clamp_actually_applied_during_control(self):
        """Sanity: po wielu krokach z dużym pos_e, integral_pos_e nie
        przekracza tighter clamp.
        """
        from gym_pybullet_drones.utils.enums import DroneModel
        from src.algorithms.TunedDSLPIDControl import TunedDSLPIDControl

        ctrl = TunedDSLPIDControl(drone_model=DroneModel.CF2X)
        cur_pos = np.zeros(3)
        cur_quat = np.array([0, 0, 0, 1])  # identity
        cur_vel = np.zeros(3)
        target_pos = np.array([100.0, 100.0, 100.0])  # huge pos_e
        target_vel = np.zeros(3)

        # Iteruj kilkadziesiąt kroków — integral wind-up
        for _ in range(100):
            ctrl._dslPIDPositionControl(
                control_timestep=1.0 / 240.0,
                cur_pos=cur_pos, cur_quat=cur_quat, cur_vel=cur_vel,
                target_pos=target_pos, target_rpy=np.zeros(3), target_vel=target_vel,
            )

        # Integral xy clamp = 0.5
        assert abs(ctrl.integral_pos_e[0]) <= 0.5 + 1e-6
        assert abs(ctrl.integral_pos_e[1]) <= 0.5 + 1e-6
        # Integral z clamp = 0.05
        assert abs(ctrl.integral_pos_e[2]) <= 0.05 + 1e-6


# ============================================================================
# C. Anti-tilt saturation
# ============================================================================


class TestAntiTilt:
    def test_max_tilt_computed_from_urdf(self):
        """Max tilt = arccos(1/thrust2weight) liczony z URDF.
        Dla CF2X T2W≈2.24, ≈63.5°. Po refaktorze 2026-05-07 (T2W z URDF)
        nie ma module-level CF2X_MAX_TILT_RAD — wartość obliczana per-instance.
        """
        from gym_pybullet_drones.utils.enums import DroneModel
        from src.algorithms.TunedDSLPIDControl import (
            TunedDSLPIDControl, compute_max_tilt_rad_from_urdf,
        )

        ctrl = TunedDSLPIDControl(drone_model=DroneModel.CF2X)
        # CF2X URDF daje T2W≈2.2425 (tolerancja 0.5° względem hardcoded 2.25).
        assert ctrl.max_tilt_rad == pytest.approx(EXPECTED_MAX_TILT_RAD, abs=0.01)
        # Sanity: ~63.6° w stopniach
        assert np.degrees(ctrl.max_tilt_rad) == pytest.approx(63.6, abs=0.5)

        # Helper function jawnie z parametrami z URDF (DSLPIDControl fields).
        max_tilt = compute_max_tilt_rad_from_urdf(
            kf=float(ctrl.KF),
            max_pwm=float(ctrl.MAX_PWM),
            pwm2rpm_const=float(ctrl.PWM2RPM_CONST),
            pwm2rpm_scale=float(ctrl.PWM2RPM_SCALE),
            weight=float(ctrl.GRAVITY),
        )
        assert max_tilt == pytest.approx(ctrl.max_tilt_rad, abs=1e-6)

    def test_anti_tilt_clips_extreme_horizontal_demand(self):
        """target_thrust z dużym horizontal component (drone needs huge
        lateral acc) → po anti-tilt clipped żeby tilt = max_tilt.
        Verify: cos(θ_after) ≈ cos(max_tilt)."""
        from src.algorithms.TunedDSLPIDControl import apply_anti_tilt_clip

        # Scenario: drone potrzebuje 100 m/s² horizontal, 9.81 m/s² vertical
        # (just gravity). Bez clipa tilt = arctan(100/9.81) = 84.4° → drone falls.
        target_thrust = np.array([100.0, 0.0, 9.81])
        clipped = apply_anti_tilt_clip(target_thrust, EXPECTED_MAX_TILT_RAD)

        # Po clip cos(tilt) = clipped[2] / |clipped|
        norm = float(np.linalg.norm(clipped))
        cos_tilt = float(clipped[2] / norm)
        assert cos_tilt == pytest.approx(np.cos(EXPECTED_MAX_TILT_RAD), abs=1e-6), (
            f"Po clip cos_tilt={cos_tilt} != cos(max_tilt)={np.cos(EXPECTED_MAX_TILT_RAD)}. "
            "Anti-tilt nie aplikuje fizycznego limit."
        )
        # Vertical component zachowany
        assert clipped[2] == pytest.approx(target_thrust[2])

    def test_anti_tilt_passthrough_when_below_max(self):
        """Gdy commanded tilt < max_tilt, nic nie zmieniamy."""
        from src.algorithms.TunedDSLPIDControl import apply_anti_tilt_clip

        # Mały tilt: arctan(2/9.81) ≈ 11.5° << 63.6°
        target_thrust = np.array([2.0, 0.0, 9.81])
        clipped = apply_anti_tilt_clip(target_thrust, EXPECTED_MAX_TILT_RAD)
        np.testing.assert_allclose(clipped, target_thrust)

    def test_anti_tilt_preserves_lateral_direction(self):
        """Po clip, kierunek lateral (XY) zachowany (tylko magnitude scaled)."""
        from src.algorithms.TunedDSLPIDControl import apply_anti_tilt_clip

        target_thrust = np.array([30.0, 40.0, 9.81])  # |xy|=50, large tilt
        clipped = apply_anti_tilt_clip(target_thrust, EXPECTED_MAX_TILT_RAD)
        # Direction (x:y ratio) zachowany
        ratio_in = target_thrust[0] / target_thrust[1]
        ratio_out = clipped[0] / clipped[1]
        assert ratio_in == pytest.approx(ratio_out, rel=1e-6)


# ============================================================================
# B. Lookahead saturation w SwarmFlightController
# ============================================================================


class TestLookaheadSaturation:
    def test_clip_target_pos_when_too_far(self):
        """Drone at (0,0,1), target at (10,0,1), lookahead=1m.
        Clipped target → (1,0,1)."""
        from src.algorithms.SwarmFlightController import clip_target_lookahead

        cur_pos = np.array([0.0, 0.0, 1.0])
        target_pos = np.array([10.0, 0.0, 1.0])
        clipped = clip_target_lookahead(cur_pos, target_pos, max_lookahead_m=1.0)
        np.testing.assert_allclose(clipped, [1.0, 0.0, 1.0], atol=1e-6)

    def test_pass_through_when_within_lookahead(self):
        """target distance < lookahead → no clipping."""
        from src.algorithms.SwarmFlightController import clip_target_lookahead

        cur_pos = np.array([0.0, 0.0, 1.0])
        target_pos = np.array([0.5, 0.5, 1.0])  # distance ≈ 0.71 < 1.0
        clipped = clip_target_lookahead(cur_pos, target_pos, max_lookahead_m=1.0)
        np.testing.assert_allclose(clipped, target_pos)

    def test_zero_distance_safe(self):
        """Drone exactly at target — bez błędów numerycznych."""
        from src.algorithms.SwarmFlightController import clip_target_lookahead

        pos = np.array([5.0, 5.0, 5.0])
        clipped = clip_target_lookahead(pos, pos, max_lookahead_m=1.0)
        np.testing.assert_allclose(clipped, pos)

    def test_3d_clipping(self):
        """Lookahead pracuje w 3D — distance euclidean."""
        from src.algorithms.SwarmFlightController import clip_target_lookahead

        cur_pos = np.array([0.0, 0.0, 0.0])
        target_pos = np.array([3.0, 4.0, 0.0])  # distance = 5
        clipped = clip_target_lookahead(cur_pos, target_pos, max_lookahead_m=2.5)
        # → kierunek zachowany, magnitude = 2.5
        assert np.linalg.norm(clipped - cur_pos) == pytest.approx(2.5, abs=1e-6)
        # Direction same
        np.testing.assert_allclose(clipped / 2.5, target_pos / 5.0, atol=1e-6)


# ============================================================================
# A. Yaw damping fix (Kamień 2026-05-07, follow-up po drone 2 panic)
# ============================================================================


class TestYawDamping:
    """Drone 2 panic w forest+ssa run miał yaw spin (0→75° w 1s) tuż przed
    upadkiem. Default DSLPIDControl D_COEFF_TOR=[20000, 20000, 12000] —
    yaw damping 40% niższy niż roll/pitch. Match yaw damping do RP wartości
    (12000→20000) → stabilniejszy yaw control podczas extreme maneuvers.
    """

    def test_tuned_yaw_d_gain_matches_rp(self):
        """❌ FAIL pre-impl: yaw D_COEFF_TOR = 20000 (match roll/pitch),
        nie default 12000."""
        from gym_pybullet_drones.utils.enums import DroneModel
        from src.algorithms.TunedDSLPIDControl import TunedDSLPIDControl

        ctrl = TunedDSLPIDControl(drone_model=DroneModel.CF2X)
        # Roll/pitch damping default 20000, yaw teraz też 20000
        np.testing.assert_allclose(
            ctrl.D_COEFF_TOR, [20000.0, 20000.0, 20000.0]
        )

    def test_torque_gains_configurable(self):
        """Torque gains również override-owalne via constructor."""
        from gym_pybullet_drones.utils.enums import DroneModel
        from src.algorithms.TunedDSLPIDControl import TunedDSLPIDControl

        ctrl = TunedDSLPIDControl(
            drone_model=DroneModel.CF2X,
            d_coeff_tor=[15000.0, 15000.0, 25000.0],
        )
        np.testing.assert_allclose(
            ctrl.D_COEFF_TOR, [15000.0, 15000.0, 25000.0]
        )


# ============================================================================
# C. Tighter lookahead (1.0m → 0.5m default)
# ============================================================================


class TestTighterLookahead:
    """Drone 2 panic showed PID overshoot mimo Tuned PID. Reducing default
    lookahead 1.0→0.5m daje PID *connesless* input — drone widzi target
    0.5m ahead, pos_e w PID jest mały, mniej extreme thrust commands."""

    def test_default_lookahead_is_half_meter(self):
        """❌ FAIL pre-impl: DEFAULT_TARGET_LOOKAHEAD_M = 0.5
        (zamiast 1.0 z initial implementation)."""
        from src.algorithms.SwarmFlightController import DEFAULT_TARGET_LOOKAHEAD_M

        assert DEFAULT_TARGET_LOOKAHEAD_M == pytest.approx(0.5)

    def test_clip_at_default_when_target_2m_away(self):
        """Drone 2m od targetu → clip do 0.5m (default), nie 1m."""
        from src.algorithms.SwarmFlightController import (
            DEFAULT_TARGET_LOOKAHEAD_M, clip_target_lookahead,
        )
        cur_pos = np.zeros(3)
        target_pos = np.array([2.0, 0.0, 0.0])
        clipped = clip_target_lookahead(cur_pos, target_pos, DEFAULT_TARGET_LOOKAHEAD_M)
        np.testing.assert_allclose(clipped, [0.5, 0.0, 0.0], atol=1e-6)
