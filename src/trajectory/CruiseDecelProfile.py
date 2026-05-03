import numpy as np


class CruiseDecelProfile:
    """Profil prędkości: stała (cruise) + faza hamowania na końcu trajektorii.

    Zaprojektowany pod krótkie manewry uniku, gdzie:
      - dron WCHODZI w trajektorię z już uformowanym wektorem prędkości
        (≈ cruise_speed), więc fala startu z v=0 jak w `TrapezoidalProfile` to
        regression — patrz docstring `BSplineSmoother` (Faza 8.1, axis=right).
      - dron WYCHODZI w punkcie rejoin z prędkością → 0 stopniowo (zamiast
        nieciągłego skoku v(cruise)→0 jak w `ConstantSpeedProfile`), żeby
        finite-diff |a|(t) na końcu krzywej nie eksplodował (Bug #2 z plan.md).

    Profil:
      v(0)            = cruise_speed
      v(t_c)          = cruise_speed                (koniec fazy cruise)
      v(t_c + t_d)    = 0                           (koniec fazy decel)
      total_duration  = t_c + t_d
      t_d             = cruise_speed / max_accel     (czas hamowania)
      s_d             = 0.5 · cruise · t_d           (dystans hamowania)

    Edge case (krótka trasa, `total_distance < s_d_full`): nie ma miejsca na
    pełne hamowanie do zera, więc kończymy z `v_end = sqrt(v² - 2 a s_total) > 0`.
    To akceptowalne — ConstantSpeed przy tym samym dystansie zrobiłby
    instantaneous step, a my mamy ramp do `v_end`. Drobny step `v_end → 0`
    w `BSplineTrajectory.get_state_at_time` na samym brzegu i tak zostanie
    wymaskowany przez `MODE_REJOIN_BLEND` (mostek BLEND mieszający komendy
    PID przez ~0.6 s, [SwarmFlightController.MODE_REJOIN_BLEND]).

    API kompatybilne z `TrapezoidalProfile`/`ConstantSpeedProfile`:
    eksponuje `t_a`, `s_a`, `t_c`, `s_c`, `t_d`, `v_peak`, `total_duration`,
    `cruise_speed`, `max_accel`, `total_distance` — co pozwala
    `BSplineTrajectory` używać tego profilu bez ingerencji w `get_state_at_time`.
    """

    def __init__(self, total_distance: float, cruise_speed: float, max_accel: float):
        self.total_distance = max(float(total_distance), 0.0)
        self.cruise_speed = max(float(cruise_speed), 1e-6)
        self.max_accel = max(float(max_accel), 1e-6)

        # Brak fazy przyspieszania — wchodzimy z v = cruise.
        self.t_a = 0.0
        self.s_a = 0.0
        self.v_peak = self.cruise_speed

        if self.total_distance <= 1e-6:
            self.t_c = 0.0
            self.s_c = 0.0
            self.t_d = 0.0
            self.s_d = 0.0
            self.v_end = self.cruise_speed
            self.total_duration = 0.0
            return

        s_d_full = 0.5 * self.cruise_speed * (self.cruise_speed / self.max_accel)
        if self.total_distance >= s_d_full:
            # Pełna faza cruise + pełna faza decel do v=0.
            self.s_d = s_d_full
            self.t_d = self.cruise_speed / self.max_accel
            self.s_c = self.total_distance - s_d_full
            self.t_c = self.s_c / self.cruise_speed
            self.v_end = 0.0
        else:
            # Trasa za krótka na pełne hamowanie — kończymy z v_end > 0.
            # v_end² = v² - 2 a s ⇒ v_end = sqrt(...).
            self.s_d = self.total_distance
            self.s_c = 0.0
            self.t_c = 0.0
            v_end_sq = self.cruise_speed ** 2 - 2.0 * self.max_accel * self.total_distance
            self.v_end = float(np.sqrt(max(0.0, v_end_sq)))
            self.t_d = (self.cruise_speed - self.v_end) / self.max_accel

        self.total_duration = self.t_c + self.t_d

    def get_state(self, t: float) -> tuple[float, float]:
        if self.total_distance <= 1e-6:
            return 0.0, 0.0

        t_clipped = float(np.clip(t, 0.0, self.total_duration))

        if t_clipped <= self.t_c:
            # Faza cruise — stała prędkość.
            dist = self.cruise_speed * t_clipped
            speed = self.cruise_speed
        else:
            # Faza decel — równomierne hamowanie cruise → v_end.
            dt = t_clipped - self.t_c
            speed = self.cruise_speed - self.max_accel * dt
            dist = self.s_c + self.cruise_speed * dt - 0.5 * self.max_accel * dt * dt

        speed = max(self.v_end, speed)
        dist = min(self.total_distance, max(0.0, dist))
        return float(dist), float(speed)
