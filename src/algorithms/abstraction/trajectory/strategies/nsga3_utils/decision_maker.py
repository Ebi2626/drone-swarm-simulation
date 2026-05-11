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
    """Kontrakt strategii MCDM wybierającej jedno rozwiązanie z frontu Pareto."""

    def select_best(self, pareto_front_F: NDArray[np.float64], pareto_front_G: NDArray[np.float64]) -> int:
        """Wybierz indeks rozwiązania uznanego za najlepsze.

        Args:
            pareto_front_F: `(N_Solutions, N_Objectives)` wartości obiektywów.
            pareto_front_G: `(N_Solutions, N_Constraints)` naruszenia
                ograniczeń (w NSGA-III finalny front ma zwykle `G ≈ 0`,
                ale strategia powinna i tak filtrować przez `filter_feasible`).

        Returns:
            Indeks wiersza w `pareto_front_F` wskazujący wybrane rozwiązanie.
        """
        ...


def normalize_matrix(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wykonaj Min-Max scaling kolumnami, mapując każdy obiektyw na `[0, 1]`.

    Niezbędne dla porównywalności obiektywów o różnych jednostkach
    (np. f1 [m] vs f3 [bezwymiarowy koszt zagrożenia]).

    Args:
        matrix: `(N, M)` wejściowa macierz obiektywów.

    Returns:
        `(N, M)` macierz znormalizowana — kolumna o zerowym zakresie
        (`max == min`) jest zwracana jako same zera.
    """
    min_vals = np.min(matrix, axis=0)
    max_vals = np.max(matrix, axis=0)

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0

    return (matrix - min_vals) / range_vals


def filter_feasible(F: NDArray, G: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """Wybierz wykonalne wiersze (`Σ G ≤ 1e-6`) z frontu Pareto.

    Args:
        F: `(N, M)` macierz obiektywów.
        G: `(N, K)` macierz naruszeń ograniczeń.

    Returns:
        Krotka `(F_feasible, G_feasible, original_indices)`:
        - `F_feasible`, `G_feasible` — przefiltrowane wiersze.
        - `original_indices` — indeksy w pierwotnym `F`/`G`.
        Gdy żaden wiersz nie jest wykonalny, zwraca cały front
        (zasada „najmniejsze zło" — caller wybierze coś z infeasibles).
    """
    cv_sum = np.sum(G, axis=1)
    feasible_mask = cv_sum <= 1e-6
    indices = np.where(feasible_mask)[0]

    if len(indices) == 0:
        return F, G, np.arange(len(F))

    return F[indices], G[indices], indices


class EqualWeightsDecision:
    """Suma znormalizowanych obiektywów z równymi wagami; wybór `argmin`."""

    def select_best(self, pareto_front_F: NDArray, pareto_front_G: NDArray) -> int:
        """Patrz `DecisionStrategyProtocol.select_best`."""
        F_feas, _, orig_indices = filter_feasible(pareto_front_F, pareto_front_G)
        F_norm = normalize_matrix(F_feas)
        scores = np.sum(F_norm, axis=1)
        best_idx_local = np.argmin(scores)
        return orig_indices[best_idx_local]


class SafetyPriorityDecision:
    """Porządek leksykograficzny: feasibility → `f3 = threat ≤ 1e-3` → `min f1`.

    Gdy żadne rozwiązanie nie ma `threat ≤ 1e-3`, minimalizuje samo `f3`
    (zachowanie „najmniej niebezpiecznego" rozwiązania).
    """

    def select_best(self, pareto_front_F: NDArray, pareto_front_G: NDArray) -> int:
        """Patrz `DecisionStrategyProtocol.select_best`.

        Raises:
            ValueError: Gdy żadne rozwiązanie nie jest wykonalne.
        """
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
    """Punkt frontu Pareto najbliższy utopii `[0, …, 0]` w przestrzeni Min-Max.

    Branke et al. (2004) — uproszczona wersja euklidesowa zamiast heurystyki
    angle-based; wystarczająca dla 5-wymiarowej przestrzeni obiektywów.
    """

    def select_best(self, pareto_front_F: NDArray, pareto_front_G: NDArray) -> int:
        """Patrz `DecisionStrategyProtocol.select_best`."""
        F_feas, _, orig_indices = filter_feasible(pareto_front_F, pareto_front_G)
        F_norm = normalize_matrix(F_feas)
        dist_to_utopia = np.sqrt(np.sum(F_norm**2, axis=1))
        best_idx_local = np.argmin(dist_to_utopia)
        return orig_indices[best_idx_local]
