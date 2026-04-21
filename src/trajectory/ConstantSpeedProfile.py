import numpy as np


class ConstantSpeedProfile:
    """
    Profil prędkości o stałej wartości przez całą trajektorię.

    Brak faz przyspieszania i hamowania — dron porusza się z prędkością `cruise_speed`
    od pierwszego do ostatniego kroku. Stosowany dla lokalnych manewrów uniku, gdzie
    dron wchodzi w trajektorię z już uformowanym wektorem prędkości i nie powinien
    hamować do zera tylko po to, by znów rozpędzić się na kilku metrach.

    API kompatybilne z `TrapezoidalProfile` — eksponuje te same pola
    (`t_a`, `s_a`, `t_c`, `s_c`, `t_d`, `v_peak`, `total_duration`, `max_accel`),
    co pozwala `BSplineTrajectory` używać obu profilów zamiennie.
    """

    def __init__(self, total_distance: float, speed: float):
        self.total_distance = max(float(total_distance), 0.0)
        self.cruise_speed = max(float(speed), 1e-6)

        self.max_accel = 0.0
        self.v_peak = self.cruise_speed

        # Brak fazy przyspieszania / deceleracji — całość to jedna faza cruise.
        self.t_a = 0.0
        self.s_a = 0.0
        self.t_d = 0.0
        self.s_c = self.total_distance
        self.t_c = self.total_distance / self.cruise_speed if self.total_distance > 1e-6 else 0.0
        self.total_duration = self.t_c

    def get_state(self, t: float) -> tuple[float, float]:
        if self.total_distance <= 1e-6:
            return 0.0, 0.0

        t_clipped = float(np.clip(t, 0.0, self.total_duration))
        dist = min(self.total_distance, self.cruise_speed * t_clipped)
        # Prędkość niech będzie niezerowa aż do osiągnięcia końca trajektorii,
        # by kontroler miał stabilny target_vel przez cały manewr.
        if dist >= self.total_distance - 1e-9:
            speed = 0.0
        else:
            speed = self.cruise_speed
        return dist, speed
