import pytest
import numpy as np

from src.algorithms.abstraction.trajectory.strategies.nsga3_utils.decision_maker import (
    normalize_matrix,
    filter_feasible,
    EqualWeightsDecision,
    SafetyPriorityDecision,
    KneePointDecision
)

# ==========================================
# TESTY FUNKCJI POMOCNICZYCH
# ==========================================

def test_normalize_matrix_standard():
    """
    Intencja: Poprawne skalowanie wartości każdej kolumny do zakresu [0, 1].
    """
    # 3 rozwiązania, 2 cele
    matrix = np.array([
        [10.0, 50.0],
        [20.0, 10.0],
        [30.0, 30.0]
    ])
    
    expected = np.array([
        [0.0, 1.0],
        [0.5, 0.0],
        [1.0, 0.5]
    ])
    
    result = normalize_matrix(matrix)
    np.testing.assert_array_almost_equal(result, expected)

def test_normalize_matrix_zero_division_protection():
    """
    Edge case: Wszystkie wartości w danej kolumnie są identyczne (max == min).
    Funkcja powinna uchronić się przed dzieleniem przez zero i zwrócić 0.0.
    """
    matrix = np.array([
        [5.0, 10.0],
        [5.0, 20.0],
        [5.0, 30.0]
    ])
    
    result = normalize_matrix(matrix)
    
    expected = np.array([
        [0.0, 0.0],
        [0.0, 0.5],
        [0.0, 1.0]
    ])
    np.testing.assert_array_almost_equal(result, expected)

def test_filter_feasible():
    """
    Intencja: Funkcja powinna odrzucać wiersze, w których suma ograniczeń (G) > 0.
    """
    F = np.array([[1], [2], [3], [4]])
    G = np.array([
        [0.0, 0.0],   # Feasible
        [0.5, 0.0],   # Infeasible
        [0.0, 0.0],   # Feasible
        [1.0, 1.0]    # Infeasible
    ])
    
    F_feas, G_feas, orig_indices = filter_feasible(F, G)
    
    assert len(F_feas) == 2
    np.testing.assert_array_equal(orig_indices, [0, 2])
    np.testing.assert_array_equal(F_feas, [[1], [3]])

def test_filter_feasible_fallback():
    """
    Edge case: Żadne rozwiązanie nie spełnia ograniczeń (wszystkie mają G > 0).
    Zgodnie z implementacją, w akcie desperacji zwracamy cały front.
    """
    F = np.array([[1], [2]])
    G = np.array([[1.0], [2.0]])
    
    F_feas, _, orig_indices = filter_feasible(F, G)
    
    assert len(F_feas) == 2
    np.testing.assert_array_equal(orig_indices, [0, 1])

# ==========================================
# TESTY STRATEGII DECYZYJNYCH
# ==========================================

@pytest.fixture
def sample_front():
    """
    Trójkryterialna macierz celów:
    F1 = Długość trasy (indeks 0)
    F2 = Zużycie energii / Parametr pomocniczy (indeks 1)
    F3 = Ryzyko kolizji (indeks 2)
    """
    F = np.array([
        [10.0, 5.0, 0.8],  # 0: Bardzo krótka, ale ekstremalnie niebezpieczna
        [20.0, 5.0, 0.5],  # 1: Kompromis (Knee point)
        [40.0, 5.0, 0.0],  # 2: Długa, ale w 100% bezpieczna
        [50.0, 5.0, 0.0],  # 3: Zbyt długa i bezpieczna (zdominowana w F1 przez 2)
        [15.0, 5.0, 0.2]   # 4: Feasible w teorii, ale dodamy constraint violation by ją odrzucić
    ])
    G = np.array([
        [0.0], [0.0], [0.0], [0.0], [1.0] 
    ])
    return F, G

def test_equal_weights_decision(sample_front):
    """
    Intencja: Strategia wybiera najlepszą średnią po normalizacji.
    """
    F, G = sample_front
    strategy = EqualWeightsDecision()
    best_idx = strategy.select_best(F, G)
    
    # Punkt 2 (40, 5.0, 0.0) ma najlepszą łączną notę po zrównaniu wag (ze względu na dystans i ryzyko)
    assert best_idx == 2  

def test_safety_priority_decision(sample_front):
    """
    Intencja: Strategia filtruje trasy bezryzykowne, a potem bierze najkrótszą z nich.
    """
    F, G = sample_front
    strategy = SafetyPriorityDecision()
    
    # Bezpieczne (F3 <= 1e-3) to indeks 2 (40, 5.0, 0.0) i indeks 3 (50, 5.0, 0.0).
    # Z nich najkrótszy (min F1) to indeks 2 (F1=40).
    best_idx = strategy.select_best(F, G)
    
    assert best_idx == 2

def test_safety_priority_fallback():
    """
    Edge case: Brak jakichkolwiek bezpiecznych tras (wszędzie F3 > 1e-3).
    Strategia powinna wybrać to rozwiązanie, które ma NAJMNIEJSZE ryzyko absolutne.
    """
    F = np.array([
        [10.0, 5.0, 0.9],
        [20.0, 5.0, 0.5],
        [30.0, 5.0, 0.4]
    ])
    G = np.zeros((3, 1))
    
    strategy = SafetyPriorityDecision()
    best_idx = strategy.select_best(F, G)
    
    # Najmniejsze ryzyko (F3=0.4) jest na indeksie 2
    assert best_idx == 2

def test_knee_point_decision():
    """
    Intencja: Strategia Knee Point wybiera punkt o minimalnym dystansie do Utopii.
    Zaprojektujmy ostry łuk (kompromis).
    """
    F = np.array([
        [0.0, 50.0, 100.0],   # Ekstremum A
        [10.0, 50.0, 10.0],   # Wyraźny Knee Point (świetny kompromis)
        [100.0, 50.0, 0.0]    # Ekstremum B
    ])
    G = np.zeros((3, 1))
    
    strategy = KneePointDecision()
    best_idx = strategy.select_best(F, G)
    
    # Indeks 1 (10.0, 50.0, 10.0) to najlepszy kompromis i najbliżej punktu utopii
    assert best_idx == 1