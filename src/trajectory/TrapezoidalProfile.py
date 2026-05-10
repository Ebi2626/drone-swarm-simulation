import numpy as np

class TrapezoidalProfile:
    def __init__(self, total_distance: float, cruise_speed: float, max_accel: float):
        self.total_distance = total_distance
        self.cruise_speed = cruise_speed
        self.max_accel = max_accel

        self.t_a = 0.0
        self.s_a = 0.0
        self.t_c = 0.0
        self.s_c = 0.0
        self.t_d = 0.0
        self.v_peak = 0.0
        self.total_duration = 0.0

        self._compute_profile()

    def _compute_profile(self) -> None:
        if self.total_distance <= 1e-6:
            return

        t_a_nominal = self.cruise_speed / self.max_accel
        s_a_nominal = 0.5 * self.max_accel * (t_a_nominal ** 2)

        if 2 * s_a_nominal > self.total_distance:
            # Trasa za krótka na cruise — profil trójkątny.
            self.s_a = self.total_distance / 2.0
            self.v_peak = np.sqrt(self.max_accel * self.total_distance)
            self.t_a = self.v_peak / self.max_accel

            self.s_c = 0.0
            self.t_c = 0.0
        else:
            self.s_a = s_a_nominal
            self.v_peak = self.cruise_speed
            self.t_a = t_a_nominal

            self.s_c = self.total_distance - 2 * self.s_a
            self.t_c = self.s_c / self.cruise_speed

        self.t_d = self.t_a
        self.total_duration = 2 * self.t_a + self.t_c

    def get_state(self, t: float) -> tuple[float, float]:
        t = np.clip(t, 0.0, self.total_duration)

        if self.total_distance <= 1e-6:
            return 0.0, 0.0

        if t <= self.t_a:
            current_speed = self.max_accel * t
            current_distance = 0.5 * self.max_accel * (t ** 2)
        elif t <= self.t_a + self.t_c:
            dt = t - self.t_a
            current_speed = self.v_peak
            current_distance = self.s_a + self.v_peak * dt
        else:
            dt = t - (self.t_a + self.t_c)
            current_speed = self.v_peak - self.max_accel * dt
            # s = v_0·t - ½·a·t² (ruch jednostajnie opóźniony)
            current_distance = self.s_a + self.s_c + (self.v_peak * dt) - (0.5 * self.max_accel * (dt ** 2))

        current_speed = max(0.0, current_speed)
        current_distance = min(self.total_distance, current_distance)

        return current_distance, current_speed