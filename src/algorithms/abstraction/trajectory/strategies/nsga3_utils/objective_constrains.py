""""
Objectives and Constraints Evaluator.
Moduł odpowiedzialny za ocenę jakości rozwiązań (Fitness) oraz weryfikację ograniczeń (Constraints).
Implementuje precyzyjną detekcję kolizji dla obiektów typu Cylinder i Box w środowisku zurbanizowanym.
Oraz wymusza poprawność geometryczną trajektorii typu Polyline.

Kluczowe funkcjonalności:
1. Wielokryterialna ocena (Długość, Ryzyko, Energia).
2. Obsługa typów przeszkód: CYLINDER (rzut 2D + Z) i BOX (AABB).
3. Segmentowa weryfikacja ciągłości ruchu (Segment-based checks).
4. Ograniczenia geometryczne dla łamanej (Uniformity, Smoothness).
"""

from typing import List, Dict, Optional, Any
import numpy as np
from numpy.typing import NDArray

# Importy helperów
from .core_math import get_xp, to_device
from src.environments.abstraction.generate_obstacles import ObstaclesData

# --- Helpery Geometryczne (Wektoryzowane) ---

def _dist_segment_to_cylinder(
    seg_start: NDArray, 
    seg_end: NDArray, 
    obs_center: NDArray, 
    obs_radius: NDArray, 
    obs_height: NDArray
) -> NDArray:
    """
    Oblicza naruszenie strefy cylindra przez odcinek.
    """
    xp = get_xp(seg_start)
    
    # Rozpakowanie współrzędnych (Broadcasting: Segments vs Obstacles)
    s_start = seg_start[:, None, :]
    s_end = seg_end[:, None, :]
    o_center = obs_center[None, :, :]
    
    # 1. Analiza w płaszczyźnie XY (indeksy 0, 1)
    ab_xy = s_end[..., :2] - s_start[..., :2]
    ap_xy = o_center[..., :2] - s_start[..., :2]
    
    # Projekcja punktu na odcinek (t) w 2D
    dot_ap_ab = xp.sum(ap_xy * ab_xy, axis=-1)
    dot_ab_ab = xp.sum(ab_xy * ab_xy, axis=-1)
    
    t = xp.where(dot_ab_ab > 1e-8, dot_ap_ab / dot_ab_ab, 0.0)
    t_clamped = xp.clip(t, 0.0, 1.0)
    
    closest_xy = s_start[..., :2] + ab_xy * t_clamped[..., None]
    
    diff_xy = closest_xy - o_center[..., :2]
    dist_sq_xy = xp.sum(diff_xy**2, axis=-1)
    
    # Warunek 1: Czy jesteśmy wewnątrz promienia?
    r_sq = obs_radius[None, :]**2
    violation_xy = xp.maximum(0.0, r_sq - dist_sq_xy)
    
    # 2. Analiza wysokości Z
    z_start = s_start[..., 2]
    z_end = s_end[..., 2]
    z_closest = z_start + t_clamped * (z_end - z_start)
    
    obs_z_base = o_center[..., 2]
    obs_z_top = obs_z_base + obs_height[None, :]
    
    in_z_range = (z_closest >= obs_z_base) & (z_closest <= obs_z_top)
    
    return xp.where(in_z_range, violation_xy, 0.0)

def _dist_segment_to_box(
    seg_start: NDArray, 
    seg_end: NDArray, 
    obs_center: NDArray, 
    obs_dims: NDArray
) -> NDArray:
    """
    Uproszczona detekcja kolizji Odcinek vs AABB.
    """
    xp = get_xp(seg_start)
    radius = xp.maximum(obs_dims[:, 0], obs_dims[:, 1]) / 2.0
    height = obs_dims[:, 2]
    return _dist_segment_to_cylinder(seg_start, seg_end, obs_center, radius, height)


# --- 1. Funkcje Celu (Objectives) ---

def calc_path_length(trajectories: NDArray[np.float64]) -> NDArray[np.float64]:
    """Oblicza sumaryczną długość euklidesową trasy."""
    xp = get_xp(trajectories)
    diffs = trajectories[:, :, 1:, :] - trajectories[:, :, :-1, :]
    seg_lengths = xp.sqrt(xp.sum(diffs**2, axis=-1))
    return xp.sum(xp.sum(seg_lengths, axis=-1), axis=-1)


def calc_elevation_changes(trajectories: NDArray[np.float64]) -> NDArray[np.float64]:
    """Oblicza sumę zmian wysokości."""
    xp = get_xp(trajectories)
    z_coords = trajectories[..., 2]
    z_diffs = xp.abs(z_coords[:, :, 1:] - z_coords[:, :, :-1])
    return xp.sum(xp.sum(z_diffs, axis=-1), axis=-1)


def calc_collision_risk_segments(
    trajectories: NDArray[np.float64], 
    obstacles_list: List[ObstaclesData], 
    safety_margin: float = 1.0
) -> NDArray[np.float64]:
    """
    Oblicza stopień naruszenia przestrzeni przeszkód.


    Args:
        trajectories: Tensor (Pop, Drones, Waypoints, 3).
        obstacles_list: Lista batchy przeszkód.
        safety_margin: Dodatkowy bufor bezpieczeństwa dodawany do promienia przeszkody.
        
    Returns:
        risk_score: Tensor (Pop, ) - suma kwadratów głębokości penetracji przeszkód.
    """
    xp = get_xp(trajectories)
    pop_size, n_drones, n_steps, _ = trajectories.shape
    
    seg_starts = trajectories[:, :, :-1, :].reshape(-1, 3)
    seg_ends = trajectories[:, :, 1:, :].reshape(-1, 3)
    
    total_violation = xp.zeros(pop_size)
    
    for obs_batch in obstacles_list:
        data = to_device(obs_batch.data, xp)
        count = obs_batch.count
        shape = obs_batch.shape_type
        
        if count == 0:
            continue
            
        active_obs = data[:count]
        centers = active_obs[:, :3]
        d1 = active_obs[:, 3]
        d2 = active_obs[:, 4]
        d3 = active_obs[:, 5]
        
        batch_violation = None
        
        if shape == 'CYLINDER':
            radii = d1 + safety_margin
            heights = d2
            batch_violation = _dist_segment_to_cylinder(
                seg_starts, seg_ends, centers, radii, heights
            )
            
        elif shape == 'BOX':
            max_side = xp.maximum(d1, d2)
            radii = (max_side / 2.0 * 1.414) + safety_margin 
            heights = d3
            batch_violation = _dist_segment_to_cylinder(
                seg_starts, seg_ends, centers, radii, heights
            )
        
        if batch_violation is not None:
            seg_risk = xp.sum(batch_violation, axis=1)
            risk_per_pop = seg_risk.reshape(pop_size, -1)
            total_violation += xp.sum(risk_per_pop, axis=1)

    return total_violation

# --- 2. Funkcje Ograniczeń (Constraints) ---

def constr_battery_limit(
    trajectories: NDArray[np.float64], 
    start_pos: NDArray[np.float64], 
    target_pos: NDArray[np.float64], 
    max_ratio: float = 5.0
) -> NDArray[np.float64]:
    """Ograniczenie budżetu energetycznego."""
    xp = get_xp(trajectories)
    diffs = trajectories[:, :, 1:, :] - trajectories[:, :, :-1, :]
    actual_lengths = xp.sum(xp.sqrt(xp.sum(diffs**2, axis=-1)), axis=-1)
    
    ref_vec = target_pos - start_pos
    ref_dist = xp.sqrt(xp.sum(ref_vec**2, axis=-1))
    
    max_allowed = ref_dist[None, :] * max_ratio
    cv_per_drone = xp.maximum(0.0, actual_lengths - max_allowed)
    return xp.sum(cv_per_drone, axis=1)


def constr_inter_agent_separation_segments(
    trajectories: NDArray[np.float64], 
    min_dist: float = 1.5,
    ignore_ratio: float = 0.1
) -> NDArray[np.float64]:
    """Weryfikacja separacji między dronami."""
    xp = get_xp(trajectories)
    pop_size, n_drones, n_steps, _ = trajectories.shape
    
    if ignore_ratio >= 0.5:
        ignore_ratio = 0.45
    start_idx = int(n_steps * ignore_ratio)
    end_idx = int(n_steps * (1.0 - ignore_ratio))
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = n_steps
    
    total_cv = xp.zeros(pop_size)
    eye = xp.eye(n_drones, dtype=bool)
    
    for t in range(start_idx, end_idx):
        pos_t = trajectories[:, :, t, :]
        diff = pos_t[:, :, None, :] - pos_t[:, None, :, :]
        dist_sq = xp.sum(diff**2, axis=-1)
        dist = xp.sqrt(dist_sq)
        
        dist_no_self = dist + (eye * 1e6) 
        penetration = xp.maximum(0.0, min_dist - dist_no_self)
        
        step_cv = xp.sum(xp.sum(penetration, axis=-1), axis=-1) / 2.0
        total_cv += step_cv
        
    return total_cv


def constr_segment_uniformity(
    trajectories: NDArray[np.float64], 
    tolerance_std: float = 2.0
) -> NDArray[np.float64]:
    """
    [NOWA] Ograniczenie wymuszające równomierne rozmieszczenie waypointów.
    
    Liczy odchylenie standardowe długości segmentów. Jeśli punkty są "zlepione"
    w jednym miejscu a inne daleko - odchylenie będzie duże.
    
    Args:
        tolerance_std: Dopuszczalne odchylenie standardowe (w metrach).
        
    Returns:
        CV: Wartość naruszenia (std_dev - tolerance).
    """
    xp = get_xp(trajectories)
    
    # 1. Wektory segmentów
    diffs = trajectories[:, :, 1:, :] - trajectories[:, :, :-1, :]
    
    # 2. Długości segmentów (Pop, Drones, Waypoints-1)
    seg_lengths = xp.sqrt(xp.sum(diffs**2, axis=-1))
    
    # 3. Odchylenie standardowe długości segmentów per dron (Pop, Drones)
    std_lengths = xp.std(seg_lengths, axis=-1)
    
    # 4. Naruszenie
    cv = xp.maximum(0.0, std_lengths - tolerance_std)
    
    # 5. Suma po dronach
    return xp.sum(cv, axis=1)


def constr_path_smoothness(
    trajectories: NDArray[np.float64],
    max_turn_factor: float = 10.0
) -> NDArray[np.float64]:
    """
    [NOWA] Ograniczenie gładkości trasy (Laplacian Smoothing).
    Penalizuje "zygzaki" poprzez badanie wektora drugiej różnicy (krzywizny lokalnej).
    P(i-1) -> P(i) -> P(i+1). 
    Wektor: P(i+1) - 2*P(i) + P(i-1) powinien być mały dla gładkiej linii.
    
    Returns:
        CV: Suma kwadratów "jerku" przekraczająca próg.
    """
    xp = get_xp(trajectories)
    
    # Druga różnica dyskretna (aproksymacja przyspieszenia/krzywizny)
    # Kształt: (Pop, Drones, Waypoints-2, 3)
    second_diff = (trajectories[:, :, 2:, :] 
                   - 2 * trajectories[:, :, 1:-1, :] 
                   + trajectories[:, :, :-2, :])
    
    # Kwadrat normy (siła "szarpnięcia")
    jerk_sq = xp.sum(second_diff**2, axis=-1)
    
    # Suma po całej trasie
    total_jerk = xp.sum(jerk_sq, axis=-1)
    
    # Normalizacja (zależna od liczby punktów)
    # Próg jest heurystyczny - jeśli jerk jest ogromny, trasa jest chaotyczna
    cv = xp.maximum(0.0, total_jerk - max_turn_factor)
    
    return xp.sum(cv, axis=1)

# --- 3. Wrapper ---

class VectorizedEvaluator:
    """Klasa łącząca logikę oceny dla frameworku pymoo."""
    
    def __init__(
        self, 
        obstacles: List[ObstaclesData], 
        start_pos: NDArray, 
        target_pos: NDArray,
        params: Optional[Dict[str, Any]] = None
    ):
        self.obstacles = obstacles
        self.start = start_pos
        self.target = target_pos
        self.params = params or {}
        
    def evaluate(self, trajectories: NDArray, out: Dict[str, Any]) -> None:
        """
        Główna metoda ewaluacji.
        Zapisuje wyniki do słownika 'out' pod kluczami 'F' (Cele) i 'G' (Ograniczenia).
        """
        xp = get_xp(trajectories)
        
        # Pobieranie parametrów z konfiguracji
        p_min_dist = self.params.get("min_dist", 1.5)
        p_ignore_ratio = self.params.get("ignore_ratio", 0.1)
        p_safety_margin = self.params.get("safety_margin", 1.0)
        p_smoothness_factor = self.params.get("max_jerk", 50.0) # Próg gładkości
        p_uniformity_tol = self.params.get("uniformity_std", 5.0) # Próg równomierności
        
        # 1. Cele (Objectives)
        f1 = calc_path_length(trajectories)
        f2 = calc_collision_risk_segments(trajectories, self.obstacles, safety_margin=p_safety_margin)
        f3 = calc_elevation_changes(trajectories)
        
        # 2. Ograniczenia (Constraints)
        g1 = constr_battery_limit(trajectories, 
                                  to_device(self.start, xp), 
                                  to_device(self.target, xp))
                                  
        g2 = constr_inter_agent_separation_segments(
            trajectories, 
            min_dist=p_min_dist, 
            ignore_ratio=p_ignore_ratio
        )
        
        g3 = xp.maximum(0.0, f2 - 0.1) # Kolizja z przeszkodami jako twardy constraint
        
        # --- NOWE OGRANICZENIA GEOMETRYCZNE ---
        # Wymuszenie równych odstępów (kluczowe dla "rozciągnięcia" trasy)
        g4 = constr_segment_uniformity(trajectories, tolerance_std=p_uniformity_tol)
        
        # Wymuszenie gładkości (by nie tworzył chaotycznych kłębków w miejscu)
        g5 = constr_path_smoothness(trajectories, max_turn_factor=p_smoothness_factor)
        
        out["F"] = xp.column_stack([f1, f2, f3])
        # Dodajemy g4 i g5 do wektora ograniczeń
        out["G"] = xp.column_stack([g1, g2, g3, g4, g5])
