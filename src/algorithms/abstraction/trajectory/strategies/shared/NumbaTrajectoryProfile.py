"""Profil trajektorii: trapezoidalny rozkład prędkości na gładkim B-spline.

Wczytuje zestaw waypointów, fituje na nich kubiczny B-spline (`scipy.splprep`)
i wystawia `get_state_at_time(t) -> (pos, vel)` używane przez kontroler PID
oraz oracle predykcji w `SwarmFlightController`.
"""
import numpy as np
from scipy.interpolate import splev, splprep

from src.algorithms.abstraction.trajectory.strategies.shared.bspline_utils import calculate_trapezoidal_profile, get_state_at_time_numba


class NumbaTrajectoryProfile:
    """Trapezoidalny profil prędkości na B-spline z waypointów.

    Pola publiczne:
        waypoints, distances, cumulative_distances, total_distance,
        arc_length, cruise_speed, max_accel, tck, u_params,
        ta, tc, td, sa, sc, v_peak, total_duration
        (długości faz przyspieszania/cruise/hamowania, dystanse fazowe,
        peak speed, łączny czas przejazdu).
    """

    def __init__(self, waypoints: np.ndarray, cruise_speed: float, max_accel: float):
        """Zafituj B-spline na waypointach i wylicz parametry profilu trapezoidalnego.

        Args:
            waypoints: `(W, 3)` punkty trasy `[x, y, z]` w kolejności
                chronologicznej.
            cruise_speed: Docelowa prędkość przelotowa [m/s].
            max_accel: Maksymalne przyspieszenie/hamowanie [m/s²].
        """
        self.waypoints = np.asarray(waypoints, dtype=np.float64)

        diffs = np.diff(self.waypoints, axis=0)
        self.distances = np.linalg.norm(diffs, axis=1)
        self.cumulative_distances = np.insert(np.cumsum(self.distances), 0, 0.0)
        self.total_distance = self.cumulative_distances[-1]
        self.max_accel = max_accel
        # Konsumenci avoidance (EvasionContextBuilder, BSplineSmoother) wymagają
        # referencyjnej prędkości przelotowej — eksponujemy bezpośrednio.
        self.cruise_speed = cruise_speed

        # arc_length aliasujemy do total_distance: waypointy pochodzą z gęstego
        # próbkowania B-spline (dense_samples=200), różnica < 0.1%.
        self.arc_length = float(self.total_distance)
        self.tck, self.u_params = self._fit_bspline()

        self.ta, self.tc, self.td, self.sa, self.sc, self.v_peak, self.total_duration = calculate_trapezoidal_profile(
            self.total_distance, cruise_speed, max_accel
        )

    def _fit_bspline(self):
        """Zafituj kubiczny B-spline (`splprep(k=3)`) na waypointach.

        Returns:
            Para `(tck, u_params)` z `scipy.interpolate.splprep`, lub
            `(None, None)` dla zdegenerowanej trasy (< 4 waypointów albo
            zerowa długość).
        """
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
        """Zwróć stan trajektorii (pozycja, prędkość) w chwili `t` od startu.

        Profil trapezoidalny mapuje `t → s` (długość łuku), a następnie
        `s → u` (parametr splajnu) przez interpolację
        `cumulative_distances ↔ u_params`. Tym samym splajnem ocenia
        `compute_max_observed_acceleration` w fazie offline — geometria jest
        identyczna. Liniowa interpolacja między waypointami powodowałaby
        nieciągłości kierunku przy złączach (skoki lateralnego przyspieszenia
        100× powyżej max_accel), które PID drona kończą paniczną utratą lotu.

        Args:
            t: Chwila czasowa od startu trajektorii [s]; `t ≤ 0` zwraca
                pozycję początkową, `t ≥ total_duration` — końcową; oba
                z zerową prędkością.

        Returns:
            Para `(pos, vel)`: `pos` `(3,)` w metrach, `vel` `(3,)` w m/s.

        Fallback: gdy `self.tck is None` (trasa < 4 waypointów albo
        zerowa długość), wywołuje pomocnik numba z interpolacją liniową.
        """
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

        pos_components = splev(u, self.tck)
        pos = np.array(
            [float(pos_components[0]), float(pos_components[1]), float(pos_components[2])],
            dtype=np.float64,
        )

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