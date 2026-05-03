from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from numba import njit

from src.trajectory.BSplineTrajectory import BSplineTrajectory

# --------------------------------------------------------------------------- #
# KERNELS NUMBA (Prekompilowane jądra matematyczne)                           #
# --------------------------------------------------------------------------- #

@njit(cache=True, fastmath=True)
def jit_evaluate_collision_risk(cx: float, cy: float, cz: float,
                                ox: float, oy: float, oz: float,
                                ovx: float, ovy: float, ovz: float,
                                time_offset: float) -> float:
    # Błyskawiczna dystans metryki Euklidesowej bez narzutu np.linalg.norm
    fx = ox + ovx * time_offset
    fy = oy + ovy * time_offset
    fz = oz + ovz * time_offset
    return ((cx - fx)**2 + (cy - fy)**2 + (cz - fz)**2) ** 0.5


@njit(cache=True, fastmath=True)
def jit_analyze_hits(distances: np.ndarray,
                     hit_positions: np.ndarray,
                     hit_velocities: np.ndarray,
                     drone_pos: np.ndarray,
                     drone_vel: np.ndarray,
                     critical_dist: float,
                     trigger_ttc: float,
                     trigger_dist: float):
    
    best_idx = -1
    best_ttc = np.inf
    best_rvx = 0.0
    best_rvy = 0.0
    best_rvz = 0.0

    n = len(distances)
    for i in range(n):
        dist = distances[i]
        if dist > critical_dist or dist < 1e-12:
            continue

        hx = hit_positions[i, 0]
        hy = hit_positions[i, 1]
        hz = hit_positions[i, 2]

        hvx = hit_velocities[i, 0]
        hvy = hit_velocities[i, 1]
        hvz = hit_velocities[i, 2]

        rvx = drone_vel[0] - hvx
        rvy = drone_vel[1] - hvy
        rvz = drone_vel[2] - hvz

        dx = hx - drone_pos[0]
        dy = hy - drone_pos[1]
        dz = hz - drone_pos[2]

        # Skalarny iloczyn skalarny / dystans
        closing_speed = (rvx * dx + rvy * dy + rvz * dz) / dist

        if closing_speed > 0.1:
            ttc = dist / closing_speed
            if ttc < trigger_ttc or dist < trigger_dist:
                if ttc < best_ttc:
                    best_ttc = ttc
                    best_idx = i
                    best_rvx = rvx
                    best_rvy = rvy
                    best_rvz = rvz

    return best_idx, best_ttc, best_rvx, best_rvy, best_rvz

# --------------------------------------------------------------------------- #
# KLASY DOMENOWE                                                              #
# --------------------------------------------------------------------------- #

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
    # Sequential cooperative planning (2026-05-01): jeśli zagrożenie to znany
    # dron z planowaną trajektorią (base lub evasion spline), predyktor może
    # wykorzystać dokładną ścieżkę zamiast liniowej ekstrapolacji prędkości.
    # `trajectory` MUSI mieć metodę `get_state_at_time(t) -> (pos, vel)` —
    # `BSplineTrajectory` i `NumbaTrajectoryProfile` spełniają kontrakt.
    # `trajectory_start_offset` to bieżący czas-na-splajnie celu (offset od
    # początku jego trajektorii). Predykcja punktu w czasie t to:
    # `trajectory.get_state_at_time(trajectory_start_offset + t)`.
    trajectory: Optional[object] = None
    trajectory_start_offset: float = 0.0


@dataclass(slots=True)
class EvasionContext:
    drone_id: int
    current_time: float
    drone_state: KinematicState
    threat: ThreatAlert
    base_spline: BSplineTrajectory
    rejoin_point: np.ndarray
    rejoin_base_arc: float
    world_bounds: tuple[np.ndarray, np.ndarray]
    search_space_min: np.ndarray
    search_space_max: np.ndarray
    # Sticky axis: oś wybrana przez *poprzedni* plan (jeśli trwa unik).
    # Planner preferuje tę oś, o ile nadal ma wystarczającą przestrzeń,
    # eliminując flip-flopping up↔right↔left w korytarzu z wieloma przeszkodami.
    preferred_axis_hint: Optional[str] = None
    # Multi-threat awareness (regression fix 2026-05-01 #3): secondary threats
    # to inne drony / obiekty w zasięgu które NIE są primary threat ale ich
    # przewidziana pozycja musi być uwzględniona w fitness c_safety. Bez tego
    # optymalizator omijał drone B i wpadał w drone C (34/80 kolizji w eksp.
    # użytkownika 16 runs). Lista MOŻE BYĆ pusta (back-compat) — wtedy fitness
    # uwzględnia tylko primary threat.
    secondary_threats: list[ThreatAlert] = field(default_factory=list)

    def evaluate_collision_risk(self, candidate_pos: np.ndarray, time_offset: float) -> float:
        op = self.threat.obstacle_state.position
        ov = self.threat.obstacle_state.velocity
        
        return jit_evaluate_collision_risk(
            float(candidate_pos[0]), float(candidate_pos[1]), float(candidate_pos[2]),
            float(op[0]), float(op[1]), float(op[2]),
            float(ov[0]), float(ov[1]), float(ov[2]),
            float(time_offset)
        )
    
class ThreatAnalyzer:
    """Moduł analityczny oddzielony od kontrolera lotu."""
    
    def __init__(self, trigger_ttc: float = 1.5, trigger_dist: float = 6.0, critical_dist: float = 25.0):
        self.trigger_ttc = trigger_ttc
        self.trigger_dist = trigger_dist
        self.critical_dist = critical_dist

    def analyze(self, hits: list, drone_state: KinematicState) -> Optional[ThreatAlert]:
        if not hits: 
            return None
        
        # Ekstrakcja danych obiektowych do ciągłych tablic C/NumPy.
        # Takie liniowe przepisanie jest wielokrotnie szybsze od 
        # wywoływania metod numpy i tworzenia obiektów wewnątrz pętli.
        n = len(hits)
        distances = np.empty(n, dtype=np.float64)
        positions = np.empty((n, 3), dtype=np.float64)
        velocities = np.empty((n, 3), dtype=np.float64)
        
        for i, hit in enumerate(hits):
            distances[i] = hit.distance
            positions[i] = hit.hit_position
            velocities[i] = hit.velocity
            
        best_idx, best_ttc, rvx, rvy, rvz = jit_analyze_hits(
            distances,
            positions,
            velocities,
            np.asarray(drone_state.position, dtype=np.float64),
            np.asarray(drone_state.velocity, dtype=np.float64),
            self.critical_dist,
            self.trigger_ttc,
            self.trigger_dist
        )
        
        if best_idx == -1:
            return None
            
        critical_hit = hits[best_idx]
        
        return ThreatAlert(
            obstacle_state=KinematicState(
                position=critical_hit.hit_position,
                velocity=critical_hit.velocity,
                radius=float(getattr(critical_hit, "obstacle_radius", 0.5)),
            ),
            distance=float(critical_hit.distance),
            time_to_collision=best_ttc,
            relative_velocity=np.array([rvx, rvy, rvz], dtype=np.float64)
        )