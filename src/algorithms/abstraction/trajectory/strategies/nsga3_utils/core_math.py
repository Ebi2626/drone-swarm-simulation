""""
Core Math Engine for NSGA-III Drone Swarm Strategy.
Ten moduł odpowiada za ciężkie obliczenia numeryczne.
Jest zaprojektowany tak, aby działać zarówno na CPU (NumPy) jak i GPU (CuPy).


Kluczowe funkcjonalności:
1. Dynamiczny wybór backendu (CPU/GPU).
2. Interpolacja trajektorii (Polyline Resampling).
3. Wektoryzowane obliczanie odległości Punkt-Odcinek (dla kolizji).
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
        # Fallback jeśli CuPy niedostępne
        return np.asarray(array)
    return array


# --- 2. Interpolacja Trajektorii (Resampling) ---


def resample_polyline_batch(
    waypoints: Any, 
    num_samples: int = 100
) -> Any:
    """
    Interpoluje liniowo punkty trasy (waypoints) do zadanej liczby gęstych punktów.
    Działa dla wsadu (Batch): (Pop, Drones, Waypoints, 3) -> (Pop, Drones, Num_Samples, 3).
    Funkcja jest agnostyczna backendowo (działa na numpy i cupy).
    """
    xp = get_xp(waypoints)
    pop_size, n_drones, n_in, dims = waypoints.shape
    
    # Przygotowanie wyjścia
    flat_waypoints = waypoints.reshape(-1, n_in, dims) # (Batch, N_In, 3)
    flat_out = xp.empty((pop_size * n_drones, num_samples, dims), dtype=waypoints.dtype)
    
    t_in = xp.linspace(0, 1, n_in)
    t_out = xp.linspace(0, 1, num_samples)
    
    # CuPy i NumPy mają nieco inne API do interpolacji
    # NumPy: interp(x, xp, fp) (działa na 1D)
    # CuPy: interp(x, xp, fp) (też 1D)
    
    # Niestety brak pełnej wektoryzacji batchowej w interp, trzeba pętlą (ale szybką na GPU)
    # Dla bardzo dużych batchy można by napisać kernel CUDA, ale tu pętla wystarczy.
    
    if xp == np:
        # Wersja CPU (NumPy)
        for i in range(pop_size * n_drones):
            w = flat_waypoints[i]
            for d in range(dims):
                flat_out[i, :, d] = xp.interp(t_out, t_in, w[:, d])
    else:
        # Wersja GPU (CuPy)
        # CuPy interp wymaga, by xp i fp były na GPU
        for i in range(pop_size * n_drones):
            w = flat_waypoints[i]
            for d in range(dims):
                flat_out[i, :, d] = xp.interp(t_out, t_in, w[:, d])
            
    return flat_out.reshape(pop_size, n_drones, num_samples, dims)


# --- 3. Detekcja Kolizji Odcinkowej (Point-to-Segment) ---


def vectorized_segment_distance(segment_starts, segment_ends, points):
    """
    Oblicza minimalną odległość od zbioru punktów (przeszkód) do zbioru odcinków.
    W pełni zrównoleglona.
    
    Args:
        segment_starts: (..., 3) - Początki odcinków
        segment_ends:   (..., 3) - Końce odcinków
        points:         (..., 3) - Punkty (środki przeszkód)
        
    Returns:
        Dystanse: (..., ...) - Macierz odległości
    """
    xp = get_xp(segment_starts)
    
    # Wektory odcinków AB
    ab = segment_ends - segment_starts
    
    # Wektory od początku odcinka do punktu P (AP)
    ap = points - segment_starts
    
    # Obliczenie rzutu punktu na prostą (parametr t)
    dot_ap_ab = xp.sum(ap * ab, axis=-1)
    dot_ab_ab = xp.sum(ab * ab, axis=-1)
    
    # Zabezpieczenie przed dzieleniem przez zero
    t = xp.where(dot_ab_ab > 1e-8, dot_ap_ab / dot_ab_ab, 0.0)
    
    # Clamp t do przedziału [0, 1] (odcinek, nie prosta)
    t_clamped = xp.clip(t, 0.0, 1.0)
    
    # Najbliższy punkt na odcinku
    closest_point = segment_starts + ab * t_clamped[..., None]
    
    # Dystans euklidesowy
    diff = points - closest_point
    dist = xp.sqrt(xp.sum(diff * diff, axis=-1))
    
    return dist
