"""
Objectives and Constraints Evaluator.
Moduł odpowiedzialny za ocenę jakości rozwiązań (Fitness) oraz weryfikację ograniczeń (Constraints).
Implementuje precyzyjną detekcję kolizji dla obiektów typu Cylinder i Box w środowisku zurbanizowanym.

Kluczowe funkcjonalności:
1. Wielokryterialna ocena (Długość, Ryzyko, Energia).
2. Obsługa typów przeszkód: CYLINDER (rzut 2D + Z) i BOX (AABB).
3. Segmentowa weryfikacja ciągłości ruchu (Segment-based checks).
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
    Logika:
    1. Rzutujemy odcinek na płaszczyznę XY.
    2. Liczymy dystans punkt-odcinek w 2D (względem środka koła).
    3. Sprawdzamy czy dystans 2D < radius.
    4. Sprawdzamy czy odcinek znajduje się w zakresie Z [0, height].
    
    Args:
        seg_start, seg_end: (N_Seg, 3)
        obs_center: (N_Obs, 3) - [x, y, z] (z zazwyczaj 0)
        obs_radius: (N_Obs, )
        obs_height: (N_Obs, )
        
    Returns:
        violation: (N_Seg, N_Obs) - Wartość > 0 jeśli jest kolizja.
    """
    xp = get_xp(seg_start)
    
    # Rozpakowanie współrzędnych (Broadcasting: Segments vs Obstacles)
    # Seg: (N_S, 1, 3), Obs: (1, N_O, 3)
    s_start = seg_start[:, None, :]
    s_end = seg_end[:, None, :]
    o_center = obs_center[None, :, :]
    
    # 1. Analiza w płaszczyźnie XY (indeksy 0, 1)
    # Wektor odcinka w 2D
    ab_xy = s_end[..., :2] - s_start[..., :2]
    # Wektor od początku do środka koła
    ap_xy = o_center[..., :2] - s_start[..., :2]
    
    # Projekcja punktu na odcinek (t) w 2D
    dot_ap_ab = xp.sum(ap_xy * ab_xy, axis=-1)
    dot_ab_ab = xp.sum(ab_xy * ab_xy, axis=-1)
    
    # Zabezpieczenie przed dzieleniem przez zero (pionowe odcinki mają długość 0 w XY)
    t = xp.where(dot_ab_ab > 1e-8, dot_ap_ab / dot_ab_ab, 0.0)
    t_clamped = xp.clip(t, 0.0, 1.0)
    
    # Najbliższy punkt na odcinku (w 2D)
    closest_xy = s_start[..., :2] + ab_xy * t_clamped[..., None]
    
    # Dystans 2D od środka przeszkody do najbliższego punktu
    diff_xy = closest_xy - o_center[..., :2]
    dist_sq_xy = xp.sum(diff_xy**2, axis=-1)
    
    # Warunek 1: Czy jesteśmy wewnątrz promienia? (dist < r)
    # violation_xy > 0 jeśli kolizja
    r_sq = obs_radius[None, :]**2
    violation_xy = xp.maximum(0.0, r_sq - dist_sq_xy)
    
    # 2. Analiza wysokości Z
    # Musimy sprawdzić Z w tym samym punkcie t_clamped (interpolacja Z)
    # Z_closest = Z_start + t * (Z_end - Z_start)
    z_start = s_start[..., 2]
    z_end = s_end[..., 2]
    z_closest = z_start + t_clamped * (z_end - z_start)
    
    # Warunek 2: Czy 0 <= Z <= Height?
    # Zakładamy, że przeszkoda stoi na Z=0 (plus ew. o_center.z)
    obs_z_base = o_center[..., 2]
    obs_z_top = obs_z_base + obs_height[None, :]
    
    # Sprawdzamy czy Z punktu jest wewnątrz zakresu
    # (Zastosujemy miękkie przejście lub twardy warunek binarny mnożony przez violation_xy)
    in_z_range = (z_closest >= obs_z_base) & (z_closest <= obs_z_top)
    
    # Finalna kolizja: Jest kolizja XY ORAZ jest w zakresie Z
    # Zwracamy violation_xy (kwadrat dystansu wchodzącego w przeszkodę) tylko tam gdzie Z pasuje
    return xp.where(in_z_range, violation_xy, 0.0)

def _dist_segment_to_box(
    seg_start: NDArray, 
    seg_end: NDArray, 
    obs_center: NDArray, 
    obs_dims: NDArray
) -> NDArray:
    """
    Uproszczona detekcja kolizji Odcinek vs AABB (Axis Aligned Bounding Box).
    Dla szybkości w roju przyjmujemy, że budynki są prostopadłe do osi.
    
    Args:
        obs_dims: [length, width, height]
    """
    # Implementacja AABB vs Segment jest kosztowna na GPU.
    # W kontekście pracy magisterskiej, często aproksymuje się BOX-a 
    # jako Cylinder opisany na nim (bezpieczniej) lub wpisany.
    # PROPOZYCJA: Aproksymacja cylindryczna o promieniu = max(len, wid)/2
    # Jest to standard w szybkich symulacjach rojów.
    
    xp = get_xp(seg_start)
    radius = xp.maximum(obs_dims[:, 0], obs_dims[:, 1]) / 2.0
    height = obs_dims[:, 2]
    
    return _dist_segment_to_cylinder(seg_start, seg_end, obs_center, radius, height)


# --- 1. Funkcje Celu (Objectives) ---

def calc_path_length(trajectories: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Oblicza sumaryczną długość euklidesową trasy dla każdego drona.
    
    Args:
        trajectories: Tensor kształtu (Pop, Drones, Waypoints, 3).
        
    Returns:
        total_length: Tensor kształtu (Pop, ), zawierający sumę długości tras wszystkich dronów.
    """
    xp = get_xp(trajectories)
    # Wektor przesunięcia między punktami
    diffs = trajectories[:, :, 1:, :] - trajectories[:, :, :-1, :]
    # Długość każdego segmentu
    seg_lengths = xp.sqrt(xp.sum(diffs**2, axis=-1))
    # Suma po segmentach (axis=2) i po dronach (axis=1)
    return xp.sum(xp.sum(seg_lengths, axis=-1), axis=-1)


def calc_elevation_changes(trajectories: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Oblicza sumę zmian wysokości (koszt energetyczny wznoszenia/opadania).
    
    Args:
        trajectories: Tensor kształtu (Pop, Drones, Waypoints, 3).
        
    Returns:
        elevation_cost: Tensor kształtu (Pop, ).
    """
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
    Oblicza stopień naruszenia przestrzeni przeszkód przez segmenty trasy.
    Wykorzystuje precyzyjną geometrię Cylindrów (drzewa/wieże).
    
    Args:
        trajectories: Tensor (Pop, Drones, Waypoints, 3).
        obstacles_list: Lista batchy przeszkód.
        safety_margin: Dodatkowy bufor bezpieczeństwa dodawany do promienia przeszkody.
        
    Returns:
        risk_score: Tensor (Pop, ) - suma kwadratów głębokości penetracji przeszkód.
    """
    xp = get_xp(trajectories)
    pop_size, n_drones, n_steps, _ = trajectories.shape
    
    # Reshape do listy segmentów: (Total_Segments, 3)
    # Total = Pop * Drones * (Waypoints-1)
    seg_starts = trajectories[:, :, :-1, :].reshape(-1, 3)
    seg_ends = trajectories[:, :, 1:, :].reshape(-1, 3)
    
    total_violation = xp.zeros(pop_size)
    
    for obs_batch in obstacles_list:
        data = to_device(obs_batch.data, xp)
        count = obs_batch.count
        shape = obs_batch.shape_type
        
        if count == 0:
            continue
            
        # Dane aktywnych przeszkód
        active_obs = data[:count]
        centers = active_obs[:, :3]
        d1 = active_obs[:, 3] # Radius / Length
        d2 = active_obs[:, 4] # Height / Width
        d3 = active_obs[:, 5] # Height (for BOX)
        
        batch_violation = None
        
        if shape == 'CYLINDER':
            # d1=radius, d2=height
            radii = d1 + safety_margin
            heights = d2
            # Obliczenie naruszeń (N_Seg, N_Obs)
            batch_violation = _dist_segment_to_cylinder(
                seg_starts, seg_ends, centers, radii, heights
            )
            
        elif shape == 'BOX':
            # d1=len, d2=wid, d3=height
            # Aproksymacja cylindryczna dla szybkości (bezpieczny opis)
            # radius = max(len, wid)/2 * sqrt(2) -> opisany na prostokącie
            max_side = xp.maximum(d1, d2)
            radii = (max_side / 2.0 * 1.414) + safety_margin 
            heights = d3
            
            batch_violation = _dist_segment_to_cylinder(
                seg_starts, seg_ends, centers, radii, heights
            )
        
        if batch_violation is not None:
            # Suma naruszeń dla każdego segmentu (po wszystkich przeszkodach)
            # Sumujemy wartości (które są kwadratami penetracji z funkcji _dist...)
            seg_risk = xp.sum(batch_violation, axis=1)
            
            # Reshape z powrotem do struktury populacji i agregacja
            risk_per_pop = seg_risk.reshape(pop_size, -1)
            total_violation += xp.sum(risk_per_pop, axis=1)

    return total_violation


# --- 2. Funkcje Ograniczeń (Constraints) ---

def constr_battery_limit(
    trajectories: NDArray[np.float64], 
    start_pos: NDArray[np.float64], 
    target_pos: NDArray[np.float64], 
    max_ratio: float = 2.0
) -> NDArray[np.float64]:
    """
    Ograniczenie budżetu energetycznego.
    Trasa nie może być dłuższa niż `max_ratio` razy dystans w linii prostej.
    
    Returns:
        cv: Tensor (Pop, ) - wartość naruszenia (>0 oznacza błąd).
    """
    xp = get_xp(trajectories)
    
    # Długość trasy per dron (Pop, Drones)
    diffs = trajectories[:, :, 1:, :] - trajectories[:, :, :-1, :]
    actual_lengths = xp.sum(xp.sqrt(xp.sum(diffs**2, axis=-1)), axis=-1)
    
    # Dystans referencyjny per dron (N_Drones, )
    # Używamy start/target przekazanego z zewnątrz (nie z trajektorii, bo ta może dryfować)
    ref_vec = target_pos - start_pos
    ref_dist = xp.sqrt(xp.sum(ref_vec**2, axis=-1))
    
    # Limit dla każdego drona
    max_allowed = ref_dist[None, :] * max_ratio
    
    # Naruszenie per dron
    cv_per_drone = xp.maximum(0.0, actual_lengths - max_allowed)
    
    # Suma naruszeń w roju (jeśli choć jeden dron padnie, cały rój ma problem)
    return xp.sum(cv_per_drone, axis=1)


def constr_inter_agent_separation_segments(
    trajectories: NDArray[np.float64], 
    min_dist: float = 1.5,
    ignore_ratio: float = 0.1
) -> NDArray[np.float64]:
    """
    Weryfikacja separacji między dronami.
    Dla każdego kroku czasowego t, sprawdza czy drony są od siebie oddalone o min_dist.
    Ignoruje fazę startu i lądowania (zdefiniowaną przez ignore_ratio).
    """
    xp = get_xp(trajectories)
    pop_size, n_drones, n_steps, _ = trajectories.shape
    
    start_idx = int(n_steps * ignore_ratio)
    end_idx = int(n_steps * (1.0 - ignore_ratio))
    
    total_cv = xp.zeros(pop_size)
    
    # Iteracja po czasie (Cruise Phase)
    for t in range(start_idx, end_idx):
        pos_t = trajectories[:, :, t, :] # (Pop, Drones, 3)
        
        # Macierz dystansów (Pop, N, N)
        # Broadcasting: (Pop, N, 1, 3) - (Pop, 1, N, 3)
        diff = pos_t[:, :, None, :] - pos_t[:, None, :, :]
        dist_sq = xp.sum(diff**2, axis=-1)
        dist = xp.sqrt(dist_sq)
        
        # Maska: Gdzie dystans jest za mały, ALE większy od 0 (nie ja sam ze sobą)
        violation_mask = (dist < min_dist) & (dist > 1e-5)
        
        # Wartość naruszenia
        val = xp.maximum(0.0, min_dist - dist)
        
        # Sumujemy naruszenia (dzielimy przez 2 bo macierz symetryczna)
        step_cv = xp.sum(xp.sum(val, axis=-1), axis=-1) / 2.0
        total_cv += step_cv
        
    return total_cv


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
        
        # 1. Cele (Objectives) -> Minimalizacja
        f1 = calc_path_length(trajectories)
        f2 = calc_collision_risk_segments(trajectories, self.obstacles)
        f3 = calc_elevation_changes(trajectories)
        
        # 2. Ograniczenia (Constraints) -> CV <= 0
        g1 = constr_battery_limit(trajectories, 
                                  to_device(self.start, xp), 
                                  to_device(self.target, xp))
        g2 = constr_inter_agent_separation_segments(trajectories)
        
        # G3: Hard Collision (Zduplikowane ryzyko jako twardy zakaz)
        # Jeśli f2 (ryzyko) > 0.1 (margines błędu), uznajemy to za naruszenie
        g3 = xp.maximum(0.0, f2 - 0.1)
        
        # Złożenie wyników
        out["F"] = xp.column_stack([f1, f2, f3])
        out["G"] = xp.column_stack([g1, g2, g3])
