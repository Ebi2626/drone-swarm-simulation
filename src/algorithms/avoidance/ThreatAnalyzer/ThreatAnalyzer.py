"""Analiza zagrożeń kolizyjnych z trafień LiDAR-u + dataklasy stanu uniku.

`ThreatAnalyzer.analyze` przerabia listę trafień LiDAR-u w `Optional[ThreatAlert]`
identyfikujący najgroźniejsze zagrożenie wg progów TTC/dystansu. `EvasionContext`
to kontener stanu wstrzykiwany do każdej strategii uniku.
"""
from dataclasses import dataclass, field
import numpy as np
from typing import Optional
from numba import njit

from src.trajectory.BSplineTrajectory import BSplineTrajectory


@njit(cache=True, fastmath=True)
def jit_evaluate_collision_risk(cx: float, cy: float, cz: float,
                                ox: float, oy: float, oz: float,
                                ovx: float, ovy: float, ovz: float,
                                time_offset: float) -> float:
    """Dystans euklidesowy 3D od kandydata `(cx, cy, cz)` do przewidzianej pozycji
    przeszkody `(ox, oy, oz) + (ovx, ovy, ovz) · time_offset`.
    """
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
    """Wybierz najgroźniejsze trafienie LiDAR-u (najmniejszy TTC poniżej progów).

    Args:
        distances: `(N,)` dystansy do trafień [m].
        hit_positions: `(N, 3)` pozycje trafień.
        hit_velocities: `(N, 3)` prędkości obiektów uderzonych.
        drone_pos: `(3,)` aktualna pozycja drona.
        drone_vel: `(3,)` aktualna prędkość drona.
        critical_dist: Filtr — pomijamy trafienia powyżej tej odległości [m].
        trigger_ttc: Próg wyzwolenia po TTC [s].
        trigger_dist: Próg wyzwolenia po dystansie [m].

    Returns:
        Krotka `(best_idx, best_ttc, rel_vx, rel_vy, rel_vz)`. `best_idx == -1`,
        gdy żadne trafienie nie spełnia progów.
    """
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


@dataclass(slots=True)
class KinematicState:
    """Stan kinematyczny ciała sferycznego.

    Pola:
        position: `(3,)` pozycja [m].
        velocity: `(3,)` prędkość [m/s].
        radius: Promień ciała [m] — używany do liczenia bufora kolizji.
    """
    position: np.ndarray
    velocity: np.ndarray
    radius: float


@dataclass(slots=True)
class ThreatAlert:
    """Pojedyncze zagrożenie wykryte przez `ThreatAnalyzer.analyze`.

    Pola:
        obstacle_state: Stan kinematyczny przeszkody (pozycja, prędkość, promień).
        distance: Bieżący dystans drona od przeszkody [m].
        time_to_collision: TTC z `jit_analyze_hits` [s].
        relative_velocity: `(3,)` `v_drone − v_obstacle` [m/s].
        trajectory: Opcjonalna referencja do splajnu przeszkody (gdy znamy
            jej dokładną trasę — patrz cooperative planning); musi mieć
            metodę `get_state_at_time(t) → (pos, vel)`.
        trajectory_start_offset: Bieżący czas przeszkody na splajnie [s];
            predykcja w chwili `t` to
            `trajectory.get_state_at_time(start_offset + t)`.
    """
    obstacle_state: KinematicState
    distance: float
    time_to_collision: float
    relative_velocity: np.ndarray
    # Sequential cooperative planning: gdy zagrożenie to znany dron z
    # planowaną trajektorią (base lub evasion spline), predyktor używa
    # dokładnej ścieżki zamiast liniowej ekstrapolacji prędkości.
    # Kontrakt `trajectory`: metoda `get_state_at_time(t) -> (pos, vel)` —
    # spełniają ją `BSplineTrajectory` i `NumbaTrajectoryProfile`.
    # `trajectory_start_offset` = bieżący czas-na-splajnie celu; predykcja
    # punktu w czasie t to `trajectory.get_state_at_time(start_offset + t)`.
    trajectory: Optional[object] = None
    trajectory_start_offset: float = 0.0


@dataclass(slots=True)
class EvasionContext:
    """Pełny kontekst pojedynczego trigger'a uniku, wstrzykiwany do strategii.

    Pola:
        drone_id: Indeks drona w głównym roju.
        current_time: Bieżący czas symulacji [s].
        drone_state: Stan kinematyczny drona.
        threat: Najgroźniejsze wykryte zagrożenie.
        base_spline: Bazowa trajektoria offline drona — używana do wyznaczenia
            punktu powrotu i predykcji własnego ruchu po uniku.
        rejoin_point: `(3,)` punkt powrotu na trasę bazową [m].
        rejoin_base_arc: Długość łuku bazowej trasy w punkcie powrotu [m].
        world_bounds: Para `(world_min, world_max)` `(3,)` — granice świata.
        search_space_min, search_space_max: `(3,)` adaptacyjne granice
            przestrzeni poszukiwań planera (zob. `EvasionContextBuilder`).
        preferred_axis_hint: Sticky-axis — oś z poprzedniego planu, jeśli
            unik trwa; przeciwdziała oscylacji wyboru osi w korytarzu z
            wieloma przeszkodami.
        secondary_threats: Inne drony / obiekty w zasięgu (multi-threat
            awareness) — ich przewidywana pozycja musi wejść do `c_safety`.
            Pusta lista ⇒ planer uwzględnia tylko `threat`.
    """
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
    # Multi-threat awareness: inne drony / obiekty w zasięgu które NIE są
    # primary threat ale ich przewidziana pozycja musi wejść do fitness
    # `c_safety`. Bez tego optymalizator omijał drone B i wpadał w drone C
    # (regresja kolizji obserwowana w wieloagentowych runach urban).
    # Pusta lista (default) ⇒ fitness uwzględnia tylko primary threat.
    secondary_threats: list[ThreatAlert] = field(default_factory=list)

    def evaluate_collision_risk(self, candidate_pos: np.ndarray, time_offset: float) -> float:
        """Dystans kandydata `(3,)` do przewidzianej pozycji `threat` w `now + time_offset` [m]."""
        op = self.threat.obstacle_state.position
        ov = self.threat.obstacle_state.velocity
        
        return jit_evaluate_collision_risk(
            float(candidate_pos[0]), float(candidate_pos[1]), float(candidate_pos[2]),
            float(op[0]), float(op[1]), float(op[2]),
            float(ov[0]), float(ov[1]), float(ov[2]),
            float(time_offset)
        )
    
class ThreatAnalyzer:
    """Identyfikator najgroźniejszego trafienia LiDAR-u, oddzielony od pętli sterowania."""

    def __init__(self, trigger_ttc: float = 1.5, trigger_dist: float = 6.0, critical_dist: float = 25.0):
        """Skonfiguruj progi wyzwolenia uniku.

        Args:
            trigger_ttc: Próg TTC powyżej którego trafienie jest ignorowane [s].
            trigger_dist: Próg dystansu do wyzwolenia uniku [m].
            critical_dist: Filtr wstępny — trafienia dalej niż `critical_dist`
                w ogóle nie wchodzą do analizy [m].
        """
        self.trigger_ttc = trigger_ttc
        self.trigger_dist = trigger_dist
        self.critical_dist = critical_dist

    def analyze(self, hits: list, drone_state: KinematicState) -> Optional[ThreatAlert]:
        """Wybierz najgroźniejsze trafienie z listy `hits` lub `None`.

        Args:
            hits: Lista trafień LiDAR-u (z polami `distance`, `hit_position`,
                `velocity`, opcjonalnie `obstacle_radius`).
            drone_state: Aktualny stan kinematyczny drona.

        Returns:
            `ThreatAlert` z najmniejszym TTC poniżej progów albo `None`,
            gdy żadne trafienie nie spełnia warunków.
        """
        if not hits:
            return None

        # Linearne przepisanie do ciągłych tablic C/NumPy jest wielokrotnie
        # szybsze niż wywoływanie metod numpy w pętli na obiektach Pythona.
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