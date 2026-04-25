from dataclasses import dataclass
import numpy as np
from typing import Optional

from src.trajectory.BSplineTrajectory import BSplineTrajectory

@dataclass(slots=True)
class KinematicState:
    position: np.ndarray  
    velocity: np.ndarray  
    radius: float

@dataclass(slots=True)
class ThreatAlert:
    obstacle_state: KinematicState
    distance: float
    time_to_collision: float
    relative_velocity: np.ndarray

@dataclass(slots=True)
class EvasionContext:
    drone_id: int
    current_time: float
    drone_state: KinematicState
    threat: ThreatAlert
    base_spline: BSplineTrajectory  # <--- Upewnij się, że to pole tu jest!
    rejoin_point: np.ndarray
    rejoin_base_arc: float
    world_bounds: tuple[np.ndarray, np.ndarray]
    search_space_min: np.ndarray
    search_space_max: np.ndarray

    def evaluate_collision_risk(self, candidate_pos: np.ndarray, time_offset: float) -> float:
        future_obs_pos = self.threat.obstacle_state.position + (self.threat.obstacle_state.velocity * time_offset)
        return float(np.linalg.norm(candidate_pos - future_obs_pos))
    
class ThreatAnalyzer:
    """Moduł analityczny oddzielony od kontrolera lotu."""
    def __init__(self, trigger_ttc: float = 1.5, trigger_dist: float = 6.0, critical_dist: float = 25.0):
        self.trigger_ttc = trigger_ttc
        self.trigger_dist = trigger_dist
        self.critical_dist = critical_dist

    def analyze(self, hits: list, drone_state: KinematicState) -> Optional[ThreatAlert]:
        if not hits: return None
        
        dangerous_hits = []
        for hit in hits:
            if hit.distance > self.critical_dist: continue
            
            rel_vel = drone_state.velocity - hit.velocity
            dir_to_hit = (hit.hit_position - drone_state.position) / hit.distance
            closing_speed = float(np.dot(rel_vel, dir_to_hit))
            
            if closing_speed > 0.1:
                ttc = hit.distance / closing_speed
                if ttc < self.trigger_ttc or hit.distance < self.trigger_dist:
                    dangerous_hits.append((hit, ttc, rel_vel))
                    
        if not dangerous_hits: return None
        
        # Wybierz najbardziej krytyczne zagrożenie (najmniejsze TTC)
        critical_hit, ttc, rel_vel = min(dangerous_hits, key=lambda x: x[1])
        
        return ThreatAlert(
            obstacle_state=KinematicState(position=critical_hit.hit_position, velocity=critical_hit.velocity, radius=0.5),
            distance=critical_hit.distance,
            time_to_collision=ttc,
            relative_velocity=rel_vel
        )