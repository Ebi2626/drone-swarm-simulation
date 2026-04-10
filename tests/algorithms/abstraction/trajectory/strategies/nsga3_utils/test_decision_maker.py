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
    
    # Oczekujemy:
    # Kolumna 1: min=10, max=30, range=20. Wartości znormalizowane: [0.0, 0.5, 1.0]
    # Kolumna 2: min=10, max=50, range=40. Wartości znormalizowane: [1.0, 0.0, 0.5]
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
    
    # Pierwsza kolumna powinna być wyzerowana, druga znormalizowana normalnie
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
    F1 = Długość trasy, F2 = Ryzyko kolizji
    Rozwiązania:
    0: (10, 0.8) - Bardzo krótka, ale ekstremalnie niebezpieczna
    1: (20, 0.5) - Kompromis (Knee point)
    2: (40, 0.0) - Długa, ale w 100% bezpieczna
    3: (50, 0.0) - Zbyt długa i bezpieczna (zdominowana w F1 przez 2)
    4: (15, 0.2) - Feasible w teorii, ale dodamy constraint violation by ją odrzucić
    """
    F = np.array([
        [10.0, 0.8],
        [20.0, 0.5],
        [40.0, 0.0],
        [50.0, 0.0],
        [15.0, 0.2]
    ])
    G = np.array([
        [0.0], [0.0], [0.0], [0.0], [1.0] # Indeks 4 odpada przez G
    ])
    return F, G

def test_equal_weights_decision(sample_front):
    """
    Intencja: Strategia wybiera najlepszą średnią po normalizacji.
    """
    F, G = sample_front
    strategy = EqualWeightsDecision()
    
    # Analiza dla poprawnych indeksów [0, 1, 2, 3]:
    # F = [[10, 0.8], [20, 0.5], [40, 0.0], [50, 0.0]]
    # Normalizacja [0, 1]:
    # F1_norm: [0.0, 0.25, 0.75, 1.0]
    # F2_norm: [1.0, 0.625, 0.0, 0.0]
    # Sumy:
    # idx 0: 0.0 + 1.0 = 1.0
    # idx 1: 0.25 + 0.625 = 0.875  <-- NAJMNIEJSZA SUMA
    # idx 2: 0.75 + 0.0 = 0.75     <-- CZEKAJ, 0.75 to najmniejsza suma!
    
    best_idx = strategy.select_best(F, G)
    assert best_idx == 2  # Punkt (40, 0.0) ma najlepszą łączną notę po zrównaniu wag.

def test_safety_priority_decision(sample_front):
    """
    Intencja: Strategia filtruje trasy bezryzykowne, a potem bierze najkrótszą z nich.
    """
    F, G = sample_front
    strategy = SafetyPriorityDecision()
    
    # Bezpieczne (F2 <= 1e-3) to indeks 2 (40, 0.0) i indeks 3 (50, 0.0).
    # Z nich najkrótszy (min F1) to indeks 2 (F1=40).
    best_idx = strategy.select_best(F, G)
    
    assert best_idx == 2

def test_safety_priority_fallback():
    """
    Edge case: Brak jakichkolwiek bezpiecznych tras (wszędzie F2 > 1e-3).
    Strategia powinna wybrać to rozwiązanie, które ma NAJMNIEJSZE ryzyko absolutne.
    """
    F = np.array([
        [10.0, 0.9],
        [20.0, 0.5],
        [30.0, 0.4]
    ])
    G = np.zeros((3, 1))
    
    strategy = SafetyPriorityDecision()
    best_idx = strategy.select_best(F, G)
    
    # Najmniejsze ryzyko (F2=0.4) jest na indeksie 2
    assert best_idx == 2

def test_knee_point_decision():
    """
    Intencja: Strategia Knee Point wybiera punkt o minimalnym dystansie do Utopii.
    Zaprojektujmy ostry łuk (kompromis).
    """
    F = np.array([
        [0.0, 100.0],  # Ekstremum A
        [10.0, 10.0],  # Wyraźny Knee Point (świetny kompromis)
        [100.0, 0.0]   # Ekstremum B
    ])
    G = np.zeros((3, 1))
    
    strategy = KneePointDecision()
    best_idx = strategy.select_best(F, G)
    
    # Znormalizowane F wyniesie w przybliżeniu: [[0, 1], [0.1, 0.1], [1, 0]]
    # Dystanse Euklidesowe od (0,0): 1.0, sqrt(0.02) ~ 0.14, 1.0.
    # Indeks 1 wygrywa z ogromną przewagą.
    assert best_idx == 1