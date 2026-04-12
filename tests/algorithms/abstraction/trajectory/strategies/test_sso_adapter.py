import numpy as np
from typing import Dict, Any
from src.algorithms.abstraction.trajectory.strategies.soo_adapter import TrajectorySOOAdapter

class MockVectorizedEvaluator:
    """
    Fałszywy ewaluator (Mock) symulujący zachowanie VectorizedEvaluatora.
    Śledzi kształt otrzymanego batcha i zwraca spreparowane dane F i G.
    """
    def __init__(self):
        self.last_batch_shape = None
        self.call_count = 0

    def evaluate(self, trajectories: np.ndarray, out: Dict[str, Any]) -> None:
        self.call_count += 1
        self.last_batch_shape = trajectories.shape
        pop_size = trajectories.shape[0]

        # Wywołanie inicjalizacyjne (obliczanie F_ref dla linii prostej)
        if pop_size == 1:
            # Zwracamy bazowe wartości celów dla trajektorii referencyjnej
            out["F"] = np.array([[10.0, 20.0, 30.0]])
            # Dla referencji nie ma naruszeń ograniczeń
            out["G"] = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
            
        # Wywołanie docelowe w trakcie optymalizacji (batch)
        elif pop_size == 2:
            # Spreparowane F (Pop_size, 3)
            out["F"] = np.array([
                [10.0, 20.0, 30.0],  # Osobnik #0: Zwykła linia prosta (identyczna jak F_ref)
                [1.0,  2.0,  3.0]    # Osobnik #1: Super małe F (pozornie genialna ścieżka)
            ])
            
            # Spreparowane G (Pop_size, 5)
            out["G"] = np.array([
                [0.0, 0.0,  0.0, 0.0, 0.0],  # Osobnik #0: 0 naruszeń
                [0.0, 8.0, -5.0, 0.0, 0.0]   # Osobnik #1: Olbrzymia kara na indeksie 1 i wartość ujemna na 2
            ])
        else:
            raise ValueError(f"Nieoczekiwany pop_size w mocku: {pop_size}")


def test_trajectory_soo_adapter():
    """Główny test weryfikujący logikę SOO Adaptera i Złote Zasady."""
    
    # 1. Konfiguracja i dane początkowe
    mock_evaluator = MockVectorizedEvaluator()
    n_drones = 3
    n_inner = 4
    n_output = 20
    
    start_pos = np.zeros((n_drones, 3))
    target_pos = np.ones((n_drones, 3)) * 100.0
    weights = np.array([1.0, 1.0, 1.0])  # Równe wagi ułatwią asercje
    penalty_weight = 1000.0
    
    # 2. Inicjalizacja Adaptera (tu już następuje pierwsze zapytanie do Mocka o F_ref)
    adapter = TrajectorySOOAdapter(
        evaluator=mock_evaluator,  # type: ignore 
        start_positions=start_pos,
        target_positions=target_pos,
        n_drones=n_drones,
        n_inner=n_inner,
        n_output_samples=n_output,
        weights=weights,
        penalty_weight=penalty_weight
    )
    
    # Sprawdzenie czy poprawnie zapisał F_ref
    np.testing.assert_array_equal(adapter._f_ref, np.array([10.0, 20.0, 30.0]))
    assert mock_evaluator.call_count == 1
    
    # 3. Przygotowanie wejścia (Pop_size = 2 osobników)
    pop_size = 2
    dummy_inner_waypoints = np.zeros((pop_size, n_drones, n_inner, 3))
    
    # 4. Wykonanie Adaptera - agregacja
    fitness_values = adapter(dummy_inner_waypoints)
    
    # -------------------------------------------------------------------------
    # WERYFIKACJA ASERCJI Z KONTRAKTU
    # -------------------------------------------------------------------------
    
    # ASERCJA 1: Adapter wysyła dane do ewaluatora w jednym batchu.
    # Kształt wejściowy do ewaluatora to (Pop_size, N_drones, N_output, 3)
    expected_batch_shape = (pop_size, n_drones, n_output, 3)
    assert mock_evaluator.last_batch_shape == expected_batch_shape, \
        "Adapter nie wysłał danych w poprawnym wektoryzowanym batchu!"

    # ASERCJA 3: Osobnik #0 z trajektorią równą linii prostej ma znormalizowane F.
    # F wejściowe było [10.0, 20.0, 30.0], F_ref to [10.0, 20.0, 30.0].
    # Znormalizowane F powinno wynosić dokładnie [1.0, 1.0, 1.0].
    # Po pomnożeniu przez wagi [1, 1, 1] suma wynosi 3.0. Brak kar (G=0).
    expected_fitness_0 = 1.0 + 1.0 + 1.0 + 0.0
    np.testing.assert_almost_equal(fitness_values[0], expected_fitness_0), \
        "Błąd normalizacji (Złota Zasada #1). Linia prosta powinna dać sumę znormalizowanych wag rzędu ~3.0."

    # ASERCJA 2: Fitness dla osobnika #1 drastycznie rośnie z powodu kary (Weakest-Link Penalty).
    # F_norm = [1/10, 2/20, 3/30] = [0.1, 0.1, 0.1] -> ważona suma to 0.3.
    # Kary: G to [0.0, 8.0, -5.0, 0.0, 0.0]. 
    #   - Ujemne wartości (-5.0) są ignorowane przez np.maximum(0, G).
    #   - Bierzemy najgorszą karę na podstawie np.max(), czyli 8.0 (nie sumujemy wzdłuż naruszeń!).
    #   - Całkowita kara = 8.0 * penalty_weight(1000) = 8000.0
    # Oczekiwany fitness = 0.3 + 8000.0 = 8000.3
    expected_fitness_1 = 0.3 + 8000.0
    np.testing.assert_almost_equal(fitness_values[1], expected_fitness_1), \
        "Błąd naliczania kary Weakest-Link (Złota Zasada #2)."
        
    assert fitness_values[1] > fitness_values[0] * 1000, \
        "Kara nie zdominowała oceny! Osobnik łamiący zasady musi drastycznie stracić na jakości."

def test_f_ref_zero_division_protection():
    """Corner-case: Test upewniający się, że w przypadku gdy ref_f wynosi 0, nie dojdzie do ZeroDivisionError."""
    
    class ZeroEvaluator(MockVectorizedEvaluator):
        def evaluate(self, trajectories, out):
            # Wymuszamy 0 w referencji
            out["F"] = np.array([[0.0, 0.0, 0.0]]) 
            out["G"] = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])

    adapter = TrajectorySOOAdapter(
        evaluator=ZeroEvaluator(), # type: ignore
        start_positions=np.zeros((1, 3)),
        target_positions=np.ones((1, 3)),
        n_drones=1, n_inner=2, n_output_samples=10,
        weights=np.array([1.0, 1.0, 1.0])
    )
    
    # Wartość np.maximum(f_ref, 1.0) powinna uratować sytuację i podbić wartości do 1.0
    np.testing.assert_array_equal(adapter._f_ref, np.array([1.0, 1.0, 1.0]))