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
            out["F"] = np.array([[10.0, 20.0, 30.0]])
            out["G"] = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
            
        # Wywołanie docelowe w trakcie optymalizacji (batch)
        elif pop_size == 2:
            out["F"] = np.array([
                [10.0, 20.0, 30.0], 
                [1.0,  2.0,  3.0]   
            ])
            out["G"] = np.array([
                [0.0, 0.0,  0.0, 0.0, 0.0],  
                [0.0, 8.0, -5.0, 0.0, 0.0]   
            ])
        else:
            raise ValueError(f"Nieoczekiwany pop_size w mocku: {pop_size}")


def test_trajectory_soo_adapter():
    """Główny test weryfikujący logikę SOO Adaptera i Złote Zasady."""
    
    mock_evaluator = MockVectorizedEvaluator()
    n_drones = 3
    n_inner = 4
    
    start_pos = np.zeros((n_drones, 3))
    target_pos = np.ones((n_drones, 3)) * 100.0
    weights = np.array([1.0, 1.0, 1.0]) 
    penalty_weight = 1000.0
    
    adapter = TrajectorySOOAdapter(
        evaluator=mock_evaluator,  # type: ignore 
        start_positions=start_pos,
        target_positions=target_pos,
        n_drones=n_drones,
        n_inner=n_inner,
        weights=weights,
        penalty_weight=penalty_weight
    )
    
    np.testing.assert_array_equal(adapter._f_ref, np.array([10.0, 20.0, 30.0]))
    assert mock_evaluator.call_count == 1
    
    pop_size = 2
    dummy_inner_waypoints = np.zeros((pop_size, n_drones, n_inner, 3))
    
    fitness_values = adapter(dummy_inner_waypoints)
    
    # -------------------------------------------------------------------------
    # WERYFIKACJA ASERCJI Z KONTRAKTU
    # -------------------------------------------------------------------------
    
    # ASERCJA 1: Adapter wysyła dane do ewaluatora w jednym batchu rzadkich węzłów.
    # N_waypoints to punkty startowe (1) + wewnętrzne (n_inner) + docelowe (1).
    expected_batch_shape = (pop_size, n_drones, n_inner + 2, 3)
    assert mock_evaluator.last_batch_shape == expected_batch_shape, \
        "Adapter nie wysłał danych w poprawnym wektoryzowanym batchu!"

    # ASERCJA 3: Osobnik #0 z trajektorią równą linii prostej ma znormalizowane F.
    expected_fitness_0 = 1.0 + 1.0 + 1.0 + 0.0
    np.testing.assert_almost_equal(fitness_values[0], expected_fitness_0), \
        "Błąd normalizacji (Złota Zasada #1)."

    # ASERCJA 2: Fitness dla osobnika #1 drastycznie rośnie z powodu kary.
    expected_fitness_1 = 0.3 + 8000.0
    np.testing.assert_almost_equal(fitness_values[1], expected_fitness_1), \
        "Błąd naliczania kary Weakest-Link (Złota Zasada #2)."
        
    assert fitness_values[1] > fitness_values[0] * 1000, \
        "Kara nie zdominowała oceny! Osobnik łamiący zasady musi stracić na jakości."


def test_f_ref_zero_division_protection():
    """Corner-case: Test upewniający się, że w przypadku gdy ref_f wynosi 0, nie dojdzie do ZeroDivisionError."""
    
    class ZeroEvaluator(MockVectorizedEvaluator):
        def evaluate(self, trajectories, out):
            out["F"] = np.array([[0.0, 0.0, 0.0]]) 
            out["G"] = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])

    adapter = TrajectorySOOAdapter(
        evaluator=ZeroEvaluator(), # type: ignore
        start_positions=np.zeros((1, 3)),
        target_positions=np.ones((1, 3)),
        n_drones=1, 
        n_inner=2,
        weights=np.array([1.0, 1.0, 1.0])
    )
    
    # Wartość np.maximum(f_ref, 1e-6) powinna uratować sytuację i podbić zera do minimalnego marginesu błędu
    np.testing.assert_array_equal(adapter._f_ref, np.array([1e-6, 1e-6, 1e-6]))