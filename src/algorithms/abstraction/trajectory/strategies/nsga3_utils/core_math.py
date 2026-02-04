"""
Core Math Engine for NSGA-III Drone Swarm Strategy.
Ten moduł odpowiada za ciężkie obliczenia numeryczne.
Jest zaprojektowany tak, aby działać zarówno na CPU (NumPy) jak i GPU (CuPy).

Kluczowe funkcjonalności:
1. Dynamiczny wybór backendu (CPU/GPU).
2. Generowanie macierzy bazowej B-Spline (Basis Matrix).
3. Tensorowa transformacja punktów kontrolnych na trajektorie.
4. Wektoryzowane obliczanie odległości Punkt-Odcinek (dla kolizji).
"""

import numpy as np
from typing import Any

# --- 1. Dynamiczny Backend (CPU/GPU) ---

# Próba importu CuPy (biblioteka kompatybilna z NumPy działająca na NVIDIA CUDA)
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

def get_xp(array: Any):
    """
    Zwraca odpowiedni moduł (numpy lub cupy) na podstawie typu tablicy wejściowej.
    Dzięki temu funkcje są 'agnostyczne' sprzętowo.
    """
    if HAS_CUPY and isinstance(array, cp.ndarray):
        return cp
    return np

def to_device(array: Any, target_xp) -> Any:
    """
    Przenosi tablicę na docelowe urządzenie (CPU -> GPU lub GPU -> CPU).
    """
    if target_xp == np:
        # Chcemy NumPy (CPU)
        if HAS_CUPY and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)
    elif target_xp == cp:
        # Chcemy CuPy (GPU)
        if HAS_CUPY:
            return cp.asarray(array)
        # Fallback jeśli CuPy niedostępne, ale wymuszono (nie powinno się zdarzyć przy poprawnej logice)
        return np.asarray(array)
    return array

# --- 2. Matematyka B-Spline (Macierz Bazowa) ---

def precompute_bspline_matrix(n_control: int, n_waypoints: int, degree: int = 3) -> np.ndarray:
    """
    Generuje stałą macierz transformacji B-Spline (M).
    
    Matematyka:
    Trajektoria T = P * M^T
    gdzie:
    - P to punkty kontrolne
    - M to wartości funkcji bazowych B-Spline dla każdego kroku czasowego t.
    
    Args:
        n_control: Liczba punktów kontrolnych.
        n_waypoints: Liczba punktów w wynikowej trajektorii.
        degree: Stopień krzywej (domyślnie 3 - Cubic B-Spline).
        
    Returns:
        Macierz o wymiarach (n_waypoints, n_control).
        Zwracana zawsze jako NumPy (liczone raz na CPU, potem transferowane).
    """
    # 1. Tworzenie wektora węzłów (Knot Vector)
    # Musi być 'clamped' (powtórzony na końcach), aby krzywa zaczynała/kończyła się na skrajnych punktach kontrolnych.
    # Ilość węzłów = n_control + degree + 1
    # Struktura: [0, 0, 0, 0, ..., 1, 1, 1, 1]
    
    n_knots = n_control + degree + 1
    knots = np.zeros(n_knots)
    
    # Wypełnienie środka
    # np.linspace(0, 1, n_knots - 2 * degree)
    inner_knots = np.linspace(0, 1, n_knots - 2 * degree)
    
    # Sklejanie: [0]*degree + [inner] + [1]*degree
    # Ale dla pewności robimy to ręcznie dla 'clamped uniform'
    for i in range(degree):
        knots[i] = 0.0
        knots[-(i+1)] = 1.0
    
    knots[degree:-degree] = inner_knots
    
    # 2. Obliczenie wartości funkcji bazowych (Algorytm Cox-de Boor)
    # Chcemy macierz M[t_idx, control_idx] = wartość wpływu
    
    t_values = np.linspace(0, 1, n_waypoints)
    basis_matrix = np.zeros((n_waypoints, n_control))
    
    for t_idx, t in enumerate(t_values):
        # Dla t=1 (koniec) obsługa brzegowa
        if t == 1.0:
            basis_matrix[t_idx, -1] = 1.0
            continue
            
        # Znajdź 'span' (przedział węzłów), w którym znajduje się t
        # (Uproszczona implementacja de Boora dla generacji bazy)
        for i in range(n_control):
            # N_{i,0}(t) = 1 jeśli knots[i] <= t < knots[i+1], else 0
            # Wyższych rzędów liczymy rekurencyjnie (tu iteracyjnie)
            
            # Tablica tymczasowa na wartości N dla danego t
            # Potrzebujemy bufora na 'degree+1' niezerowych funkcji bazowych
            pass 
        
        # --- Implementacja numeryczna De Boora dla Bazy ---
        # Zamiast pisać pełny algorytm, użyjemy właściwości, że N_{i,p}
        # zależy tylko od węzłów.
        # Dla wydajności i pewności implementacji, użyjemy prostej ewaluacji:
        
        for i in range(n_control):
             basis_matrix[t_idx, i] = _bspline_basis(i, degree, t, knots)
             
    return basis_matrix

def _bspline_basis(i, p, t, knots):
    """Funkcja pomocnicza Cox-de Boor (rekurencyjna)."""
    if p == 0:
        return 1.0 if knots[i] <= t < knots[i+1] else 0.0
    
    denom1 = knots[i+p] - knots[i]
    term1 = 0.0
    if denom1 > 0:
        term1 = ((t - knots[i]) / denom1) * _bspline_basis(i, p-1, t, knots)
        
    denom2 = knots[i+p+1] - knots[i+1]
    term2 = 0.0
    if denom2 > 0:
        term2 = ((knots[i+p+1] - t) / denom2) * _bspline_basis(i+1, p-1, t, knots)
        
    return term1 + term2

# --- 3. Generowanie Trajektorii (Tensorowe) ---

def trajectory_from_genotype(pop_genotype, basis_matrix):
    """
    Przekształca punkty kontrolne w gęstą trajektorię.
    Operacja: T = P x M.T
    
    Args:
        pop_genotype: Tensor (Pop_Size, N_Drones, N_Controls, 3)
        basis_matrix: Macierz (N_Waypoints, N_Controls)
        
    Returns:
        Trajektorie: Tensor (Pop_Size, N_Drones, N_Waypoints, 3)
    """
    xp = get_xp(pop_genotype)
    
    # Upewnienie się, że basis_matrix jest na tym samym urządzeniu
    M = to_device(basis_matrix, xp)
    
    # pop_genotype: [P, D, C, 3]
    # M: [W, C] -> transponujemy do [C, W]
    # Chcemy zakontraktować wymiar C (Controls)
    
    # Metoda einsum (najbardziej czytelna dla tensorów)
    # 'pdcz, wc -> pdwz'
    # p=pop, d=drone, c=control, z=coord(3)
    # w=waypoint
    
    trajectories = xp.einsum('pdcz, wc -> pdwz', pop_genotype, M)
    
    return trajectories

# --- 4. Detekcja Kolizji Odcinkowej (Point-to-Segment) ---

def vectorized_segment_distance(segment_starts, segment_ends, points):
    """
    Oblicza minimalną odległość od zbioru punktów (przeszkód) do zbioru odcinków.
    W pełni zrównoleglona.
    
    Args:
        segment_starts: (..., 3) - Początki odcinków
        segment_ends:   (..., 3) - Końce odcinków
        points:         (..., 3) - Punkty (środki przeszkód)
        
    Note:
        Funkcja zakłada, że wymiary wejściowe są kompatybilne do broadcastingu.
        Zazwyczaj caller spłaszcza dane do (N_Segments, 3) i (M_Points, 3),
        a następnie robi reshape do (N, 1, 3) i (1, M, 3) przed wywołaniem.
        
    Returns:
        Dystanse: (..., ...) - Macierz odległości
    """
    xp = get_xp(segment_starts)
    
    # Wektory odcinków AB
    ab = segment_ends - segment_starts # Wektor kierunkowy odcinka
    
    # Wektory od początku odcinka do punktu P (AP)
    ap = points - segment_starts
    
    # Obliczenie rzutu punktu na prostą (parametr t)
    # t = dot(AP, AB) / dot(AB, AB)
    
    # Iloczyn skalarny AP * AB (ostatnia oś to xyz=3)
    dot_ap_ab = xp.sum(ap * ab, axis=-1)
    
    # Długość kwadratowa odcinka AB
    dot_ab_ab = xp.sum(ab * ab, axis=-1)
    
    # Zabezpieczenie przed dzieleniem przez zero (dla odcinków o długości 0)
    # Jeśli odcinek ma długość 0, t=0
    t = xp.where(dot_ab_ab > 1e-8, dot_ap_ab / dot_ab_ab, 0.0)
    
    # Clamp t do przedziału [0, 1], aby rzut pozostał na odcinku (a nie na przedłużeniu prostej)
    t_clamped = xp.clip(t, 0.0, 1.0)
    
    # Najbliższy punkt na odcinku: C = A + t_clamped * AB
    # Musimy dodać wymiar, aby broadcasting zadziałał z powrotem na współrzędne
    closest_point = segment_starts + ab * t_clamped[..., None]
    
    # Dystans euklidesowy między Punktem P a Najbliższym Punktem C
    diff = points - closest_point
    dist = xp.sqrt(xp.sum(diff * diff, axis=-1))
    
    return dist
