"""
Decision Maker Module.
Moduł odpowiedzialny za wybór jednego, ostatecznego rozwiązania ze zbioru rozwiązań Pareto-optymalnych.
Implementuje techniki MCDM (Multi-Criteria Decision Making) do normalizacji i rankingu wyników.

Kluczowe funkcjonalności:
1. Protokół DecisionStrategyProtocol dla wymiennych strategii.
2. Automatyczna normalizacja wyników (Min-Max Scaling).
3. Implementacje strategii: Równe Wagi, Priorytet Bezpieczeństwa, Punkt Kolana (Knee Point).
"""

from typing import Protocol, Tuple
import numpy as np
from numpy.typing import NDArray

# --- Protokół Decydenta ---

class DecisionStrategyProtocol(Protocol):
    """
    Interfejs dla algorytmu wyboru najlepszego rozwiązania z Frontu Pareto.
    """
    def select_best(self, pareto_front_F: NDArray[np.float64], pareto_front_G: NDArray[np.float64]) -> int:
        """
        Wybiera indeks najlepszego rozwiązania.
        
        Args:
            pareto_front_F: Macierz celów (N_Solutions, N_Objectives).
            pareto_front_G: Macierz naruszeń ograniczeń (N_Solutions, N_Constraints).
                           (Zwykle w NSGA-III finalny front ma G=0, ale warto sprawdzać).
                           
        Returns:
            int: Indeks wiersza w pareto_front_F odpowiadający wybranemu rozwiązaniu.
        """
        ...

# --- Funkcje Pomocnicze ---

def normalize_matrix(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Normalizuje kolumny macierzy do zakresu [0, 1] (Min-Max Scaling).
    Niezbędne przy porównywaniu celów o różnych jednostkach (np. metry vs ryzyko).
    """
    min_vals = np.min(matrix, axis=0)
    max_vals = np.max(matrix, axis=0)
    
    # Zabezpieczenie przed dzieleniem przez zero (gdy max == min)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0 
    
    return (matrix - min_vals) / range_vals

def filter_feasible(F: NDArray, G: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Filtruje rozwiązania, odrzucając te, które naruszają ograniczenia (CV > 0).
    Zwraca: (F_feasible, G_feasible, original_indices)
    """
    # Suma naruszeń dla każdego rozwiązania
    cv_sum = np.sum(G, axis=1)
    
    # Tolerancja numeryczna dla zera
    feasible_mask = cv_sum <= 1e-6
    
    indices = np.where(feasible_mask)[0]
    
    if len(indices) == 0:
        # Jeśli nie ma żadnych poprawnych rozwiązań, zwracamy wszystkie
        # (Wybierzemy "najmniejsze zło")
        return F, G, np.arange(len(F))
        
    return F[indices], G[indices], indices

# --- Implementacje Strategii ---

class EqualWeightsDecision:
    """
    Strategia zbalansowana.
    Normalizuje cele i wybiera rozwiązanie o najmniejszej średniej wartości.
    Traktuje długość trasy, ryzyko i energię na równi.
    """
    def select_best(self, pareto_front_F: NDArray, pareto_front_G: NDArray) -> int:
        # 1. Filtruj tylko dopuszczalne (Feasible)
        F_feas, _, orig_indices = filter_feasible(pareto_front_F, pareto_front_G)
        
        # 2. Normalizacja [0, 1]
        F_norm = normalize_matrix(F_feas)
        
        # 3. Suma ważona (wagi równe = 1.0)
        # Można tu dodać wektor wag, np. weights = np.array([0.5, 0.3, 0.2])
        scores = np.sum(F_norm, axis=1)
        
        # 4. Wybór minimum (bo minimalizujemy cele)
        best_idx_local = np.argmin(scores)
        
        # Zwracamy indeks z oryginalnej tablicy
        return orig_indices[best_idx_local]


class SafetyPriorityDecision:
    """
    Strategia "Safety First".
    Hierarchiczny wybór:
    1. Odrzuć wszystko, co ma jakiekolwiek ryzyko kolizji (F2 > 0).
    2. Z pozostałych wybierz najkrótszą trasę (F1).
    
    Jeśli nie ma rozwiązań bezryzykownych, minimalizuje ryzyko (F2).
    """
    def select_best(self, pareto_front_F: NDArray, pareto_front_G: NDArray) -> int:
        F_feas, _, orig_indices = filter_feasible(pareto_front_F, pareto_front_G)
        
        # F2 to Risk Score (kolumna indeks 1)
        risk_scores = F_feas[:, 1]
        
        # Szukamy rozwiązań z "zerowym" ryzykiem (z marginesem)
        safe_mask = risk_scores <= 1e-3
        safe_indices = np.where(safe_mask)[0]
        
        if len(safe_indices) > 0:
            # Mamy bezpieczne trasy -> wybieramy najkrótszą z nich
            # F1 to Path Length (kolumna indeks 0)
            subset_F1 = F_feas[safe_indices, 0]
            best_safe_idx = np.argmin(subset_F1)
            
            # Mapowanie indeksów: subset -> safe -> feasible -> original
            return orig_indices[safe_indices[best_safe_idx]]
        else:
            # Nie ma w pełni bezpiecznych -> minimalizujemy ryzyko za wszelką cenę
            best_risk_idx = np.argmin(risk_scores)
            return orig_indices[best_risk_idx]


class KneePointDecision:
    """
    Strategia Punktu Kolana (Knee Point).
    Szuka rozwiązania, które oferuje najlepszy kompromis (trade-off).
    Punkt kolana to punkt na froncie Pareto najbardziej oddalony od prostej łączącej skrajne punkty.
    Najbardziej opłacalny zysk jednego celu kosztem drugiego.
    """
    def select_best(self, pareto_front_F: NDArray, pareto_front_G: NDArray) -> int:
        F_feas, _, orig_indices = filter_feasible(pareto_front_F, pareto_front_G)
        
        # Normalizacja jest kluczowa dla Knee Point
        F_norm = normalize_matrix(F_feas)
        
        # Znajdź punkty idealne (Utopia Point) i anty-idealne (Nadir Point) w znormalizowanej przestrzeni
        # Dla minimalizacji Utopia = [0, 0, 0], Nadir = [1, 1, 1]
        # Knee point to często punkt najbliższy Utopii w sensie odległości Euklidesowej
        # (To jest jedna z metod, najprostsza i skuteczna)
        
        dist_to_utopia = np.sqrt(np.sum(F_norm**2, axis=1))
        
        best_idx_local = np.argmin(dist_to_utopia)
        
        return orig_indices[best_idx_local]
