import numpy as np


class CruiseDecelProfile:
    """Profil prędkości: cruise + faza hamowania na końcu trajektorii.

    Zaprojektowany pod krótkie manewry uniku, gdzie dron wchodzi w trajektorię
    z już uformowanym wektorem prędkości ≈ cruise_speed (start z v=0 jak w
    `TrapezoidalProfile` byłby regresją), a wychodzi w punkcie rejoin z
    prędkością → 0 stopniowo (zamiast nieciągłego skoku v(cruise)→0 jak
    w `ConstantSpeedProfile`), żeby finite-diff |a|(t) na końcu krzywej
    nie eksplodował.

    Profil:
        v(0)            = cruise_speed
        v(t_c)          = cruise_speed         (koniec fazy cruise)
        v(t_c + t_d)    = 0                    (koniec fazy decel)
        total_duration  = t_c + t_d
        t_d             = cruise_speed / max_accel
        s_d             = 0.5 · cruise · t_d

    Edge case (krótka trasa, `total_distance < s_d_full`): brak miejsca na
    pełne hamowanie do zera, kończymy z `v_end = sqrt(v² - 2·a·s_total) > 0`.
    Drobny step `v_end → 0` na brzegu jest wymaskowany przez REJOIN_BLEND
    w `SwarmFlightController` (BLEND mostkuje komendy PID przez ~0.6 s).

    API kompatybilne z `TrapezoidalProfile` / `ConstantSpeedProfile`.
    """

    def __init__(self, total_distance: float, cruise_speed: float, max_accel: float):
        self.total_distance = max(float(total_distance), 0.0)
        self.cruise_speed = max(float(cruise_speed), 1e-6)
        self.max_accel = max(float(max_accel), 1e-6)

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
            self.s_d = s_d_full
            self.t_d = self.cruise_speed / self.max_accel
            self.s_c = self.total_distance - s_d_full
            self.t_c = self.s_c / self.cruise_speed
            self.v_end = 0.0
        else:
            # Trasa za krótka na pełne hamowanie. v_end² = v² - 2·a·s.
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
            dist = self.cruise_speed * t_clipped
            speed = self.cruise_speed
        else:
            dt = t_clipped - self.t_c
            speed = self.cruise_speed - self.max_accel * dt
            dist = self.s_c + self.cruise_speed * dt - 0.5 * self.max_accel * dt * dt

        speed = max(self.v_end, speed)
        dist = min(self.total_distance, max(0.0, dist))
        return float(dist), float(speed)
