import numpy as np

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
        
        # Wyliczenie profilu w Numbie
        self.ta, self.tc, self.td, self.sa, self.sc, self.v_peak, self.total_duration = calculate_trapezoidal_profile(
            self.total_distance, cruise_speed, max_accel
        )
        
    def get_state_at_time(self, t: float):
        # Mostek pomiędzy Pythonem zorientowanym obiektowo a kompilowanym backendem Numba:
        return get_state_at_time_numba(
            self.waypoints, self.distances, self.cumulative_distances, 
            t, self.ta, self.tc, self.td, self.sa, self.sc, self.v_peak, self.max_accel
        )