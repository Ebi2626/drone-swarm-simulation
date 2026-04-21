import numpy as np
from scipy.interpolate import splev
from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import KinematicState, ThreatAlert
from src.trajectory.BSplineTrajectory import BSplineTrajectory
from src.algorithms.avoidance.BaseAvoidance import (
    EvasionContext
)

class EvasionContextBuilder:
    """
    Preprocesor akademicki dla algorytmów unikania kolizji.
    Odpowiada za transformację surowego stanu drona i wykrytego zagrożenia 
    w sformalizowany problem optymalizacyjny z precyzyjnie zdefiniowaną 
    przestrzenią rozwiązań (Search Space).
    """
    def __init__(self, t_min=2.0, t_max=4.0, rejoin_arc_m=8.0, 
                 floor_margin=1.0, ceiling_margin=1.0, lateral_margin=4.0):
        self.t_min = t_min                    # Minimalny czas uniku
        self.t_max = t_max                    # Horyzont predykcji
        self.rejoin_arc_m = rejoin_arc_m      # Baza powrotu (rejoin)
        self.floor_margin = floor_margin
        self.ceiling_margin = ceiling_margin
        self.lateral_margin = lateral_margin

    def build(self,
              drone_id: int,
              current_time: float,
              drone_state: KinematicState,
              threat: ThreatAlert,
              base_spline: BSplineTrajectory,
              base_arc_progress: float,
              env_bounds: tuple[np.ndarray, np.ndarray]) -> EvasionContext:
        
        # 1. Granice operacyjne środowiska
        world_min, world_max = env_bounds
        floor_z = float(world_min[2]) + self.floor_margin
        ceiling_z = float(world_max[2]) - self.ceiling_margin
        
        # 2. Predykcja punktu powrotu (Rejoin Point) na podstawie dynamiki
        current_speed = float(np.linalg.norm(drone_state.velocity))
        cruise = float(getattr(base_spline.profile, "cruise_speed", 8.0))
        ref_speed = max(current_speed, cruise * 0.5)
        
        target_arc = min(
            base_spline.arc_length,
            base_arc_progress + max(self.rejoin_arc_m, current_speed * self.t_min)
        )
        rejoin_point = self._sample_base_at_arc(base_spline, target_arc)
        rejoin_point[2] = float(np.clip(rejoin_point[2], floor_z, ceiling_z))
        
        # 3. Kinematyka kierunkowa do konstrukcji bazy poszukiwań
        forward_3d = self._compute_forward_direction(drone_state.velocity, base_spline, base_arc_progress)
        forward_xy = np.array([forward_3d[0], forward_3d[1], 0.0])
        fnorm = float(np.linalg.norm(forward_xy))
        forward_xy = forward_xy / fnorm if fnorm > 1e-6 else np.array([1.0, 0.0, 0.0])
        lateral_xy = np.array([-forward_xy[1], forward_xy[0], 0.0])
        
        # 4. Generowanie przestrzeni poszukiwań uwzględniającej wektor prędkości przeszkody
        search_min, search_max = self._build_dynamic_search_space(
            drone_state.position, rejoin_point, threat,
            forward_xy, lateral_xy, ref_speed,
            floor_z, ceiling_z, world_min, world_max
        )
        
        return EvasionContext(
            drone_id=drone_id,
            current_time=current_time,
            drone_state=drone_state,
            threat=threat,
            base_spline=base_spline,
            rejoin_point=rejoin_point,
            rejoin_base_arc=target_arc,
            world_bounds=env_bounds,
            search_space_min=search_min,
            search_space_max=search_max
        )

    def _build_dynamic_search_space(self, current_pos, rejoin_point, threat: ThreatAlert,
                                    forward_xy, lateral_xy, ref_speed,
                                    floor_z, ceiling_z, world_min, world_max):
        """
        Zastępuje statyczny Bounding Box (A*) dynamiczną definicją Search Space.
        Inkorporacja predykcji przyszłej pozycji przeszkody w horyzoncie t_max.
        """
        forward_margin = ref_speed * self.t_max
        
        # Fundamentalna modyfikacja VO: Ekstrapolacja wektora prędkości przeszkody [file:3]
        obs_pos = threat.obstacle_state.position
        obs_future_pos = obs_pos + (threat.obstacle_state.velocity * self.t_max)
        
        key_pts = np.vstack([current_pos, rejoin_point, obs_pos, obs_future_pos])
        
        # Marginesy dla manewru wymijania
        extra_forward = current_pos + forward_xy * forward_margin
        extra_lateral_p = current_pos + lateral_xy * self.lateral_margin
        extra_lateral_n = current_pos - lateral_xy * self.lateral_margin
        key_pts = np.vstack([key_pts, extra_forward, extra_lateral_p, extra_lateral_n])
        
        bbox_min = key_pts.min(axis=0)
        bbox_max = key_pts.max(axis=0)
        
        # Dodanie marginesów na bufor bezpieczeństwa i promień drona
        pad = threat.obstacle_state.radius + 1.0
        bbox_min -= pad
        bbox_max += pad
        
        # Aplikacja twardych granic kinematycznych i środowiskowych
        bbox_min[2] = max(bbox_min[2], floor_z)
        bbox_max[2] = min(bbox_max[2], ceiling_z)
        
        bbox_min[:2] = np.maximum(bbox_min[:2], world_min[:2] + pad)
        bbox_max[:2] = np.minimum(bbox_max[:2], world_max[:2] - pad)
        
        return bbox_min, bbox_max

    @staticmethod
    def _sample_base_at_arc(spline: BSplineTrajectory, arc: float) -> np.ndarray:
        if spline.arc_length <= 1e-6:
            u = 1.0
        else:
            u = float(np.clip(arc / spline.arc_length, 0.0, 1.0))
        return np.array(splev(u, spline.tck), dtype=np.float64)

    @staticmethod
    def _compute_forward_direction(current_vel, base_spline, base_arc_progress) -> np.ndarray:
        speed = float(np.linalg.norm(current_vel))
        if speed > 0.5:
            return current_vel / speed
            
        if base_spline.arc_length > 1e-6:
            u = float(np.clip(base_arc_progress / base_spline.arc_length, 0.0, 1.0))
        else:
            u = 0.0
            
        tangent = np.array(splev(u, base_spline.tck, der=1), dtype=np.float64)
        tnorm = float(np.linalg.norm(tangent))
        if tnorm > 1e-6:
            return tangent / tnorm
            
        return np.array([1.0, 0.0, 0.0])