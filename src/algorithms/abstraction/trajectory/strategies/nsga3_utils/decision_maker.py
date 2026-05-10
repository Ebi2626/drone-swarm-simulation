"""
Wybór pojedynczego rozwiązania z frontu Pareto NSGA-III dla potrzeb fazy
wykonawczej (PyBullet). Implementacje MCDM (Hwang & Yoon 1981): Equal Weights
(suma ważona po Min-Max scaling), Safety Priority (lex-order: feasibility →
threat=0 → min path length) i Knee Point (najbliższy utopia point w
znormalizowanej przestrzeni — heurystyka Branke et al. 2004).
"""

from typing import Protocol, Tuple
import numpy as np
from numpy.typing import NDArray


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


def normalize_matrix(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Min-Max Scaling po kolumnach. Niezbędne dla porównywalności celów
    o różnych jednostkach (np. f1 [m] vs f3 [unitless threat])."""
    min_vals = np.min(matrix, axis=0)
    max_vals = np.max(matrix, axis=0)

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0

    return (matrix - min_vals) / range_vals


def filter_feasible(F: NDArray, G: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """Zwraca (F_feasible, G_feasible, original_indices) gdzie feasibility =
    sum(G) ≤ 1e-6. Gdy brak feasible, zwraca cały front (najmniejsze zło)."""
    cv_sum = np.sum(G, axis=1)
    feasible_mask = cv_sum <= 1e-6
    indices = np.where(feasible_mask)[0]

    if len(indices) == 0:
        return F, G, np.arange(len(F))

    return F[indices], G[indices], indices


class EqualWeightsDecision:
    """Suma znormalizowanych celów z równymi wagami → argmin."""
    def select_best(self, pareto_front_F: NDArray, pareto_front_G: NDArray) -> int:
        F_feas, _, orig_indices = filter_feasible(pareto_front_F, pareto_front_G)
        F_norm = normalize_matrix(F_feas)
        scores = np.sum(F_norm, axis=1)
        best_idx_local = np.argmin(scores)
        return orig_indices[best_idx_local]


class SafetyPriorityDecision:
    """Lex-order: feasibility → f3=threat ≤ 1e-3 → min f1 (path length).

    Fallback: gdy żadne rozwiązanie nie ma threat ≤ 1e-3, minimalizuje samo
    f3 (ratujemy „najmniej niebezpieczne" rozwiązanie).
    """
    def select_best(self, pareto_front_F: NDArray, pareto_front_G: NDArray) -> int:
        F_feas, _, orig_indices = filter_feasible(pareto_front_F, pareto_front_G)

        if len(F_feas) == 0:
            raise ValueError("Brak dopuszczalnych rozwiązań (wszystkie naruszają ograniczenia G).")

        # f3 = threat cost (kolumna 2 zgodnie z VectorizedEvaluator).
        risk_scores = F_feas[:, 2]

        safe_mask = risk_scores <= 1e-3
        safe_indices = np.where(safe_mask)[0]

        if len(safe_indices) > 0:
            # f1 = trajectory cost (kolumna 0).
            subset_F1 = F_feas[safe_indices, 0]
            best_safe_idx = np.argmin(subset_F1)
            return orig_indices[safe_indices[best_safe_idx]]
        else:
            best_risk_idx = np.argmin(risk_scores)
            return orig_indices[best_risk_idx]


class KneePointDecision:
    """Knee point ≈ punkt frontu Pareto najbliższy utopia point [0,…,0]
    w przestrzeni Min-Max-znormalizowanej (Branke et al. 2004 — heurystyka
    angle-based zastąpiona euklidesową, prostsza i wystarczająca dla 5D).
    """
    def select_best(self, pareto_front_F: NDArray, pareto_front_G: NDArray) -> int:
        F_feas, _, orig_indices = filter_feasible(pareto_front_F, pareto_front_G)
        F_norm = normalize_matrix(F_feas)
        dist_to_utopia = np.sqrt(np.sum(F_norm**2, axis=1))
        best_idx_local = np.argmin(dist_to_utopia)
        return orig_indices[best_idx_local]
