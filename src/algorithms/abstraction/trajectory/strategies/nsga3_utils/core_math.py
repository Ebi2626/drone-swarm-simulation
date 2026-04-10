""""
Core Math Engine for NSGA-III Drone Swarm Strategy.
Ten moduł odpowiada za ciężkie obliczenia numeryczne.
Jest zaprojektowany tak, aby działać zarówno na CPU (NumPy) jak i GPU (CuPy).


Kluczowe funkcjonalności:
1. Dynamiczny wybór backendu (CPU/GPU).
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
