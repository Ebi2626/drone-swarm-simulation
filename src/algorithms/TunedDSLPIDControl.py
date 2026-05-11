"""Tuned DSL PID control + anti-tilt safety dla CF2X drone.

Default `DSLPIDControl` podczas sharp lateral maneuver może wpaść w
**death-spiral**:
1. Trajectory wymaga lateral acc → drone tilts (kąt θ od pionu)
2. Vertical thrust component = `T·cos(θ)` spada
3. Gdy `T·cos(θ) < m·g`, drone traci altitude
4. PID z (gain 1.25 — agresywne) komenduje więcej thrustu
5. Drone tilts MORE (extra horizontal pos_e push)
6. RPM saturation (commanded > URDF max=15000) → real thrust niższy
7. Drone fall

Three-layer fix:
A. **Tuned gains** — P/I reduced (less aggressive thrust commands), D doubled
   (more damping). Integral clamps tighter (prevent wind-up).
B. **Lookahead saturation** (w `SwarmFlightController.compute_actions`) —
   cap pos_e seen by PID by clipping target_pos to within max_lookahead.
   Drone "łagodnie" dogania zamiast skoku.
C. **Anti-tilt clip** (here, `apply_anti_tilt_clip`) — saturate target_thrust
   direction tak żeby tilt ≤ `max_tilt = arccos(1/thrust2weight)`. To jest
   FIZYCZNY limit przy max thrust gdzie drone JESZCZE może utrzymać
   altitude (`T·cos(max_tilt) = mg`). Powyżej: niemożliwe.

Reference:
- Mellinger & Kumar (2011) "Minimum snap trajectory generation and control
  for quadrotors", ICRA — geometric controller w SE(3) z explicit thrust
  saturation. My static (cube of thrust direction) jest uproszczeniem.
"""
from __future__ import annotations

import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel


# Max physical tilt liczony dynamicznie z URDF (Mellinger & Kumar 2011 §III.B):
# `T·cos(θ_max) = m·g` ⇒ `cos(θ_max) = 1/T2W`, gdzie `T2W = MAX_THRUST_TOTAL / (m·g)`.
# Wcześniej hardcoded `2.25` dla CF2X — fail gdy user zmieni model na CF2P
# lub większy drone z innym thrust2weight.


def compute_max_tilt_rad_from_urdf(
    kf: float,
    max_pwm: float,
    pwm2rpm_const: float,
    pwm2rpm_scale: float,
    weight: float,
) -> float:
    """Oblicza maksymalny tilt z parametrów PID controllera (load-z-URDF).

    Wzór: `T2W = 4 · KF · MAX_RPM² / (m·g)` gdzie
    `MAX_RPM = pwm2rpm_const + pwm2rpm_scale · MAX_PWM` (per BaseControl).

    Args:
        kf: Współczynnik thrust per rotor (N·s²/rev²). Z `DSLPIDControl.KF`.
        max_pwm: Maksymalna wartość PWM. Z `DSLPIDControl.MAX_PWM`.
        pwm2rpm_const, pwm2rpm_scale: Stałe konwersji PWM→RPM.
            Z `DSLPIDControl.PWM2RPM_*`.
        weight: Ciężar drone'a `m·g` (N). Z `DSLPIDControl.GRAVITY`
            (UWAGA: pole nazywa się `GRAVITY`, ale przechowuje weight).

    Returns:
        max tilt w radianach. Dla T2W ≤ 1 (drone nie zdoła unieść własnego
        ciężaru) zwraca 0.0 — anti-tilt nie pozwoli na żaden tilt.
    """
    if weight <= 0.0:
        return 0.0
    max_rpm = pwm2rpm_const + pwm2rpm_scale * max_pwm
    max_thrust_total = 4.0 * kf * max_rpm * max_rpm
    if max_thrust_total <= weight:
        return 0.0
    t2w = max_thrust_total / weight
    return float(np.arccos(1.0 / t2w))


def apply_anti_tilt_clip(
    target_thrust: np.ndarray,
    max_tilt_rad: float,
) -> np.ndarray:
    """C-fix: saturate `target_thrust` direction at max_tilt.

    Gdy commanded `target_thrust` ma tilt (kąt od world-Z) > max_tilt, scale
    horizontal component down zachowując vertical. Wynik: ten sam vertical
    thrust budget (altitude maintained), ale lateral thrust limited do
    `v_thrust × tan(max_tilt)`. Drone akceptuje lateral lag zamiast spadać.

    Args:
        target_thrust: shape (3,), world-frame thrust vector
            (z-component dodatnio = vertical up).
        max_tilt_rad: maksymalny kąt od pionu (w radianach).

    Returns:
        clipped target_thrust shape (3,). Vertical component zachowany.
    """
    thrust_norm = float(np.linalg.norm(target_thrust))
    if thrust_norm < 1e-9:
        return target_thrust.copy()

    v_component = float(target_thrust[2])
    if v_component <= 0:
        # Drone target points downward — anti-tilt nie aplikuje
        # (PID komenduje thrust w dół, prawdopodobnie scenariusz lądowania)
        return target_thrust.copy()

    cos_tilt = v_component / thrust_norm
    cos_max = float(np.cos(max_tilt_rad))
    if cos_tilt >= cos_max:
        # Tilt już ≤ max — passthrough
        return target_thrust.copy()

    # Tilt > max: ograniczamy horizontal magnitude do v · tan(max_tilt).
    # Zachowujemy kierunek horizontal i pełen vertical component.
    h_max = v_component * float(np.tan(max_tilt_rad))
    h_xy = target_thrust[:2]
    h_xy_norm = float(np.linalg.norm(h_xy))
    if h_xy_norm <= h_max or h_xy_norm < 1e-9:
        return target_thrust.copy()

    scale = h_max / h_xy_norm
    out = target_thrust.copy()
    out[:2] = h_xy * scale
    return out


class TunedDSLPIDControl(DSLPIDControl):
    """PID control z optymalizowanymi gains + anti-tilt safety.

    Dziedziczy DSLPIDControl, override:
    - `__init__`: tuned default gains (configurable)
    - `_dslPIDPositionControl`: tighter integral clamp + anti-tilt clip
      przed obliczeniem `target_z_ax` (commanded drone orientation).

    Gains rationale (porównanie do default DSLPIDControl):
        P_FOR: [0.4, 0.4, 1.25] → [0.3, 0.3, 0.9]  (-25% / -28%)
        I_FOR: [0.05, 0.05, 0.05] → [0.02, 0.02, 0.02]  (-60%)
        D_FOR: [0.2, 0.2, 0.5] → [0.4, 0.4, 0.8]  (+100% / +60%)
        ∫_xy clamp: ±2.0 → ±0.5  (-75%)
        ∫_z  clamp: ±0.15 → ±0.05  (-67%)

    P/I reduced + integral tighter → prevent overaggressive thrust commands
    (drone realizuje mniej extreme attitude).
    D doubled → damping przeciwdziała oscylacjom.
    """

    DEFAULT_P_COEFF_FOR = np.array([0.3, 0.3, 0.9])
    DEFAULT_I_COEFF_FOR = np.array([0.02, 0.02, 0.02])
    DEFAULT_D_COEFF_FOR = np.array([0.4, 0.4, 0.8])
    DEFAULT_INTEGRAL_CLAMP_XY = 0.5
    DEFAULT_INTEGRAL_CLAMP_Z = 0.05

    # Yaw damping match do roll/pitch: default DSLPIDControl
    # `D_COEFF_TOR=[20000, 20000, 12000]` (yaw 40% niższy niż RP) pozwala
    # drone'owi wpaść w yaw spin (0→75° w 1s) podczas extreme lateral
    # maneuver tuż przed fall. Tuned matchuje yaw do RP → stable yaw control.
    # P/I_COEFF_TOR defaults zostawiamy (działają dobrze dla roll/pitch).
    DEFAULT_D_COEFF_TOR = np.array([20000.0, 20000.0, 20000.0])

    def __init__(
        self,
        drone_model: DroneModel,
        g: float = 9.8,
        p_coeff_for=None,
        i_coeff_for=None,
        d_coeff_for=None,
        d_coeff_tor=None,
        integral_clamp_xy: float | None = None,
        integral_clamp_z: float | None = None,
        max_tilt_rad: float | None = None,
    ):
        """Skonfiguruj nadpisane gainy PID i parametry anti-tilt.

        Args:
            drone_model: Model drona (`DroneModel.CF2X` itd.).
            g: Przyspieszenie ziemskie [m/s²].
            p_coeff_for, i_coeff_for, d_coeff_for: `(3,)` gainy P/I/D dla
                `[x, y, z]`. `None` ⇒ wartości `DEFAULT_*`.
            d_coeff_tor: `(3,)` gainy D dla momentu obrotowego (RPY); `None`
                ⇒ `DEFAULT_D_COEFF_TOR` z dopasowanym yawem.
            integral_clamp_xy, integral_clamp_z: Symetryczne progi
                clampingu integratora w XY i Z; `None` ⇒ wartości domyślne.
            max_tilt_rad: Twardy limit kąta tilt; `None` ⇒ obliczone
                z parametrów URDF przez `compute_max_tilt_rad_from_urdf`.
        """
        super().__init__(drone_model=drone_model, g=g)
        self.P_COEFF_FOR = np.array(
            p_coeff_for if p_coeff_for is not None else self.DEFAULT_P_COEFF_FOR,
            dtype=np.float64,
        )
        self.I_COEFF_FOR = np.array(
            i_coeff_for if i_coeff_for is not None else self.DEFAULT_I_COEFF_FOR,
            dtype=np.float64,
        )
        self.D_COEFF_FOR = np.array(
            d_coeff_for if d_coeff_for is not None else self.DEFAULT_D_COEFF_FOR,
            dtype=np.float64,
        )
        self.D_COEFF_TOR = np.array(
            d_coeff_tor if d_coeff_tor is not None else self.DEFAULT_D_COEFF_TOR,
            dtype=np.float64,
        )
        self.integral_clamp_xy = float(
            integral_clamp_xy if integral_clamp_xy is not None
            else self.DEFAULT_INTEGRAL_CLAMP_XY
        )
        self.integral_clamp_z = float(
            integral_clamp_z if integral_clamp_z is not None
            else self.DEFAULT_INTEGRAL_CLAMP_Z
        )
        # max_tilt_rad: dynamiczny default z URDF (po super().__init__ —
        # KF/MAX_PWM/PWM2RPM_*/GRAVITY dostępne). Nie używamy hardcoded
        # `CF2X_THRUST2WEIGHT` — wspiera dowolny model drone'a.
        self.max_tilt_rad = float(
            max_tilt_rad if max_tilt_rad is not None
            else compute_max_tilt_rad_from_urdf(
                kf=float(self.KF),
                max_pwm=float(self.MAX_PWM),
                pwm2rpm_const=float(self.PWM2RPM_CONST),
                pwm2rpm_scale=float(self.PWM2RPM_SCALE),
                weight=float(self.GRAVITY),
            )
        )

    def _dslPIDPositionControl(
        self,
        control_timestep,
        cur_pos,
        cur_quat,
        cur_vel,
        target_pos,
        target_rpy,
        target_vel,
    ):
        """Nadpisany PID pozycji z węższym integralnym clamp i saturacją tilt.

        Identyczna logika jak `DSLPIDControl._dslPIDPositionControl`, z dwoma
        zmianami:
        1. Integral clamp pochodzi z `self.integral_clamp_xy/_z`
           (zamiast hardcoded `±2 / ±0.15`).
        2. `apply_anti_tilt_clip` ogranicza `target_thrust` przed obliczeniem
           `target_z_ax`.

        Args:
            control_timestep: Krok czasowy regulatora [s].
            cur_pos, cur_vel: `(3,)` aktualna pozycja/prędkość drona [m, m/s].
            cur_quat: `(4,)` quaternion `(x, y, z, w)` aktualnej orientacji.
            target_pos, target_vel: `(3,)` zadana pozycja/prędkość.
            target_rpy: `(3,)` zadane kąty Euler (RPY); używany jest jedynie yaw.

        Returns:
            Krotkę `(thrust, target_euler, pos_e)`:
              - `thrust` (`float`) — komendowane PWM.
              - `target_euler` `(3,)` — komendowane RPY w `XYZ` [rad].
              - `pos_e` `(3,)` — uchyb pozycji `target_pos − cur_pos` [m].
        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel

        # A-fix: integral wind-up + tighter clamp
        self.integral_pos_e = self.integral_pos_e + pos_e * control_timestep
        self.integral_pos_e[:2] = np.clip(
            self.integral_pos_e[:2],
            -self.integral_clamp_xy, self.integral_clamp_xy,
        )
        self.integral_pos_e[2] = float(np.clip(
            self.integral_pos_e[2],
            -self.integral_clamp_z, self.integral_clamp_z,
        ))

        # PID target thrust (z gravity feed-forward)
        target_thrust = (
            np.multiply(self.P_COEFF_FOR, pos_e)
            + np.multiply(self.I_COEFF_FOR, self.integral_pos_e)
            + np.multiply(self.D_COEFF_FOR, vel_e)
            + np.array([0.0, 0.0, self.GRAVITY])
        )

        # C-fix: anti-tilt — clip target_thrust direction at max_tilt
        target_thrust = apply_anti_tilt_clip(target_thrust, self.max_tilt_rad)

        scalar_thrust = max(0.0, float(np.dot(target_thrust, cur_rotation[:, 2])))
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

        thrust_norm = float(np.linalg.norm(target_thrust))
        if thrust_norm < 1e-9:
            target_z_ax = np.array([0.0, 0.0, 1.0])
        else:
            target_z_ax = target_thrust / thrust_norm
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0.0])
        target_y_ax = np.cross(target_z_ax, target_x_c)
        ty_norm = float(np.linalg.norm(target_y_ax))
        if ty_norm > 1e-9:
            target_y_ax = target_y_ax / ty_norm
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = np.vstack([target_x_ax, target_y_ax, target_z_ax]).transpose()

        target_euler = Rotation.from_matrix(target_rotation).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print(
                "[WARN] TunedDSLPIDControl: target_euler outside [-pi, pi] "
                f"at ctrl iter {self.control_counter}"
            )
        return thrust, target_euler, pos_e
