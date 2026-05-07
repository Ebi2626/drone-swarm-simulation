import numpy as np
from scipy.interpolate import splev, splprep

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
        """Returns (position, velocity) at given time `t` along the trajectory.

        Refactor 2026-05-07 (Kamień: B-spline tracking):
        Wcześniej delegowane do `get_state_at_time_numba` które używało
        **linear interpolation** między waypointami (linijka 299) — to
        produkowało direction discontinuities przy waypoint junctions
        (instantaneous turn = lateral acc spikes 100x+ m/s²). Drone PID
        nie potrafi takiego acc → panic falls.

        Teraz używa `scipy.interpolate.splev` na `self.tck` (B-spline
        zafitowany w `_fit_bspline`). Path geometryczny = identyczny smooth
        B-spline którym ocenia optimizer w `compute_max_observed_acceleration`.
        Trapezoidal profile mapuje t → s (arc length) → u (spline parameter)
        przez interpolację (cumulative_distances ↔ u_params).

        Fallback: gdy `self.tck is None` (degenerate path < 4 waypoints lub
        zero distance), używa starego linear-interp helper'a (numba).

        Test: `tests/trajectory/test_numba_trajectory_bspline_following.py
        ::test_get_state_at_time_follows_smooth_bspline_not_linear_polyline`.
        """
        # Trapezoidal profile: t → arc length s + speed
        if t <= 0.0:
            return self.waypoints[0].copy(), np.zeros(3, dtype=np.float64)
        if t >= self.total_duration:
            return self.waypoints[-1].copy(), np.zeros(3, dtype=np.float64)

        if t < self.ta:
            current_dist = 0.5 * self.max_accel * t * t
            current_speed = self.max_accel * t
        elif t < self.ta + self.tc:
            t_cruise = t - self.ta
            current_dist = self.sa + self.v_peak * t_cruise
            current_speed = self.v_peak
        else:
            t_dec = t - self.ta - self.tc
            current_dist = (
                self.sa + self.sc + self.v_peak * t_dec
                - 0.5 * self.max_accel * t_dec * t_dec
            )
            current_speed = self.v_peak - self.max_accel * t_dec

        # Fallback: degenerate path (< 4 waypoints lub zero arc length)
        # gdzie B-spline nie daje się zafittować — pozostaje linear interp.
        if self.tck is None:
            return get_state_at_time_numba(
                self.waypoints, self.distances, self.cumulative_distances,
                t, self.ta, self.tc, self.td, self.sa, self.sc,
                self.v_peak, self.max_accel,
            )

        # Map arc length → spline parameter u przez monotonicznę interpolację.
        # `splprep(s=0, k=3)` zwraca u_params skorelowane z waypointami;
        # `cumulative_distances[i]` to arc length przy waypoint[i].
        u = float(np.interp(current_dist, self.cumulative_distances, self.u_params))

        # Position from B-spline: splev returns list of arrays per dimension
        pos_components = splev(u, self.tck)
        pos = np.array(
            [float(pos_components[0]), float(pos_components[1]), float(pos_components[2])],
            dtype=np.float64,
        )

        # Tangent direction (unit) from spline 1st derivative
        tangent_components = splev(u, self.tck, der=1)
        tangent = np.array(
            [float(tangent_components[0]), float(tangent_components[1]), float(tangent_components[2])],
            dtype=np.float64,
        )
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm > 1e-9:
            tangent = tangent / tangent_norm

        velocity = tangent * current_speed
        return pos, velocity