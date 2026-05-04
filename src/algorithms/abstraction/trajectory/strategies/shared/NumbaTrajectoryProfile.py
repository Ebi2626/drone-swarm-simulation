import numpy as np
from scipy.interpolate import splprep

from src.algorithms.abstraction.trajectory.strategies.shared.bspline_utils import calculate_trapezoidal_profile, get_state_at_time_numba

class NumbaTrajectoryProfile:
    def __init__(self, waypoints: np.ndarray, cruise_speed: float, max_accel: float):
        self.waypoints = np.asarray(waypoints, dtype=np.float64)

        # Szybka kalkulacja z Numby
        diffs = np.diff(self.waypoints, axis=0)
        self.distances = np.linalg.norm(diffs, axis=1)
        self.cumulative_distances = np.insert(np.cumsum(self.distances), 0, 0.0)
        self.total_distance = self.cumulative_distances[-1]
        self.max_accel = max_accel
        # Konsumenci avoidance (EvasionContextBuilder, BSplineSmoother, BSplineYZGenes,
        # ) wymagają referencyjnej prędkości przelotowej dla profilu uniku.
        # Po refaktorze BSplineTrajectory→NumbaTrajectoryProfile zniknął zagnieżdżony
        # `.profile.cruise_speed` — eksponujemy bezpośrednio.
        self.cruise_speed = cruise_speed

        # B-spline coef (tck) i arc_length — wymagane przez avoidance (splev na
        # `base_spline.tck` daje tangent/pozycję ciągłą). NumbaTrajectoryProfile sam w sobie
        # interpoluje liniowo między waypointami, ale waypointy pochodzą z gęstego
        # próbkowania B-spline po offline optimization (dense_samples=200 default), więc
        # ponowne dopasowanie krzywej `splprep(s=0, k=3)` jest tożsame z pierwotną krzywą.
        # `arc_length` aliasujemy do total_distance (przy 200 próbkach różnica < 0.1%).
        self.arc_length = float(self.total_distance)
        self.tck, self.u_params = self._fit_bspline()

        # Wyliczenie profilu w Numbie
        self.ta, self.tc, self.td, self.sa, self.sc, self.v_peak, self.total_duration = calculate_trapezoidal_profile(
            self.total_distance, cruise_speed, max_accel
        )

    def _fit_bspline(self):
        # splprep(k=3) wymaga >= 4 unikalnych punktów. Dla zdegenerowanych przypadków
        # (waypoints powtórzone, np. dron stoi w miejscu) zwracamy None — avoidance
        # wykrywa to przez `arc_length <= 1e-6` i wtedy nie woła splev.
        n = self.waypoints.shape[0]
        if n < 4 or self.total_distance < 1e-6:
            return None, None
        tck, u_params = splprep(
            [self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2]],
            s=0,
            k=3,
        )
        return tck, u_params

    def get_state_at_time(self, t: float):
        # Mostek pomiędzy Pythonem zorientowanym obiektowo a kompilowanym backendem Numba:
        return get_state_at_time_numba(
            self.waypoints, self.distances, self.cumulative_distances, 
            t, self.ta, self.tc, self.td, self.sa, self.sc, self.v_peak, self.max_accel
        )