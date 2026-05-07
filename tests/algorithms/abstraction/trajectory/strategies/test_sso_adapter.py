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

    # ASERCJA 3: Osobnik #0 (feasible, G≤0) ma fitness = obj_values
    # (znormalizowane F · weights). Hard gating nie modyfikuje feasible.
    expected_fitness_0 = 1.0 + 1.0 + 1.0 + 0.0
    np.testing.assert_almost_equal(fitness_values[0], expected_fitness_0), \
        "Błąd normalizacji (Złota Zasada #1)."

    # ASERCJA 2: Fitness dla osobnika #1 (infeasible, G[1]=8 > 0) =
    # HARD_INFEASIBLE_BASE + penalty_weight × total_violation.
    # G=[0,8,-5,0,0] → max(0,G)=[0,8,0,0,0] → total_violation=8.
    # Big-M kontract (Złota Zasada #2): infeasible ZAWSZE >= 1e6.
    from src.algorithms.abstraction.trajectory.strategies.soo_adapter import (
        HARD_INFEASIBLE_BASE,
    )
    expected_fitness_1 = HARD_INFEASIBLE_BASE + penalty_weight * 8.0
    np.testing.assert_almost_equal(fitness_values[1], expected_fitness_1), \
        "Błąd Big-M hard gating (Złota Zasada #2 — feasibility-first)."

    assert fitness_values[1] > fitness_values[0] * 1000, \
        "Big-M nie zdominował! Infeasible musi mieć fitness ≫ feasible."


def test_f_ref_zero_division_protection():
    """Corner-case: Test upewniający się, że w przypadku gdy ref_f wynosi 0,
    nie dojdzie do ZeroDivisionError ANI do wzmocnienia F_norm o 1e6×.

    Po refaktorze 2026-05-07: zamiast cap'a `max(f_ref, 1e-6)` używamy
    neutralnego mianownika `1.0` dla zerowych komponentów. Powód: cap=1e-6
    łamał Big-M ordering — F_norm = F/1e-6 mogło przekroczyć HARD_INFEASIBLE_BASE.
    """
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

    # Guard ustawia 1.0 dla wszystkich zerowych komponentów (neutralny mianownik).
    np.testing.assert_array_equal(adapter._f_ref, np.array([1.0, 1.0, 1.0]))
    # Lista zero-ref components dla ETL/diagnostyki.
    assert adapter._zero_ref_components == [0, 1, 2]


# ============================================================================
# FEASIBILITY-FIRST CONTRACT (2026-05-07)
# ============================================================================
# Diagnoza: SSA/OOA/MSFFOA przez TrajectorySOOAdapter traktują G jako SOFT
# penalty: fitness = obj + penalty_weight × max(0, max_G). Przy małej violation
# (np. kinematic_penalty=0.5) i dużej różnicy obj między solutions, infeasible
# z niską obj WYGRYWA nad feasible z wyższą obj. Optimizer zwraca infeasible
# trajectory, drone wykonuje, panic falls.
#
# NSGA-III ma natywnie feasibility-first: infeasible NIGDY nie domninuje
# feasible niezależnie od obj. SOO adapter musi to mieć też.
# ============================================================================


class FeasibilityFirstEvaluator:
    """Mock dający kontrolowane (F, G) dla testów feasibility ordering."""

    def __init__(self, F_init, G_init, F_pop, G_pop):
        self.F_init = np.asarray(F_init, dtype=np.float64)
        self.G_init = np.asarray(G_init, dtype=np.float64)
        self.F_pop = np.asarray(F_pop, dtype=np.float64)
        self.G_pop = np.asarray(G_pop, dtype=np.float64)

    def evaluate(self, trajectories, out):
        pop_size = trajectories.shape[0]
        if pop_size == 1:
            out["F"] = self.F_init
            out["G"] = self.G_init
        else:
            out["F"] = self.F_pop
            out["G"] = self.G_pop


def test_soo_adapter_must_prefer_feasible_over_low_obj_infeasible():
    """❌ FAIL pre-fix: TrajectorySOOAdapter używa soft penalty
    (`obj + weight×max_G`) — infeasible solution z bardzo niską obj wygrywa
    nad feasible z wysoką obj. To pozwala optimizer'owi zwracać kinematycznie
    niemożliwe trajektorie → drone wykonuje → panic falls (user 2026-05-07).

    Required fix: hard infeasibility gating (Big-M). Każde infeasible
    (G[k]>0 dla dowolnego k) → fitness ≥ HARD_INFEASIBLE_BASE. Feasible
    ZAWSZE preferowane przed infeasible — niezależnie od różnic obj.
    """
    # Setup: F_ref = [1,1,1,1,1] (po init z F_init=[1,...])
    # Solution A: feasible, ALE wysokie obj (suma 100×5 weights × 1 = 500)
    # Solution B: infeasible (kinematic_penalty=0.5), ALE bardzo niskie obj (suma 5)
    #
    # Pre-fix soft penalty (penalty_weight=100):
    #   fitness_A = 500 + 0 = 500
    #   fitness_B = 5 + 50 = 55  ← infeasible WYGRYWA (niższe = lepsze)
    #
    # Post-fix Big-M:
    #   fitness_A = 500 (feasible, czyste obj)
    #   fitness_B = 1e6 + 0.5×100 = 1000050 (infeasible, Big-M)
    #   feasible WYGRYWA jak powinno
    evaluator = FeasibilityFirstEvaluator(
        F_init=[[1.0, 1.0, 1.0, 1.0, 1.0]],
        G_init=[[0.0, 0.0, 0.0]],
        F_pop=[
            [100.0, 100.0, 100.0, 100.0, 100.0],  # A: feasible, wysokie obj
            [1.0, 1.0, 1.0, 1.0, 1.0],            # B: niskie obj ALE...
        ],
        G_pop=[
            [0.0, 0.0, 0.0],   # A: feasible
            [0.0, 0.0, 0.5],   # B: kinematic_penalty=0.5 — naruszenie
        ],
    )

    adapter = TrajectorySOOAdapter(
        evaluator=evaluator,  # type: ignore[arg-type]
        start_positions=np.zeros((1, 3)),
        target_positions=np.ones((1, 3)),
        n_drones=1,
        n_inner=2,
        weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        penalty_weight=100.0,
    )

    fitness = adapter(np.zeros((2, 1, 2, 3)))

    # GŁÓWNA ASERCJA: feasible (idx=0) MUSI mieć NIŻSZE fitness
    # niż infeasible (idx=1) — niezależnie od różnicy obj_values.
    assert fitness[0] < fitness[1], (
        f"❌ Feasibility-first contract NARUSZONY. "
        f"Feasible (idx=0) fitness={fitness[0]:.2f}, "
        f"infeasible (idx=1) fitness={fitness[1]:.2f}. "
        "Soft penalty pozwala infeasible-z-niską-obj wygrać. "
        "Optimizer zwróci kinematycznie niemożliwą trajektorię → drone "
        "wykonuje → panic fall. Required: Big-M hard gating w soo_adapter.py."
    )


def test_soo_adapter_orders_feasible_solutions_by_obj():
    """Sanity: feasible solutions order'owane po obj (lower=better) — Big-M
    nie powinien wpływać na ordering wewnątrz feasible bucket."""
    evaluator = FeasibilityFirstEvaluator(
        F_init=[[1.0, 1.0, 1.0]],
        G_init=[[0.0, 0.0, 0.0]],
        F_pop=[
            [10.0, 10.0, 10.0],  # A: medium obj
            [1.0, 1.0, 1.0],     # B: low obj (best)
        ],
        G_pop=[
            [0.0, 0.0, 0.0],     # both feasible
            [0.0, 0.0, 0.0],
        ],
    )
    adapter = TrajectorySOOAdapter(
        evaluator=evaluator,  # type: ignore[arg-type]
        start_positions=np.zeros((1, 3)),
        target_positions=np.ones((1, 3)),
        n_drones=1, n_inner=2,
        weights=np.array([1.0, 1.0, 1.0]),
        penalty_weight=100.0,
    )
    fitness = adapter(np.zeros((2, 1, 2, 3)))
    assert fitness[1] < fitness[0], (
        f"Feasible ordering broken: A={fitness[0]}, B={fitness[1]}, B should win."
    )


def test_soo_adapter_orders_infeasible_solutions_by_violation():
    """Sanity: infeasible solutions order'owane po total violation
    (mniejsza violation = lepsze) — wszystkie >> max feasible fitness."""
    evaluator = FeasibilityFirstEvaluator(
        F_init=[[1.0, 1.0, 1.0]],
        G_init=[[0.0, 0.0, 0.0]],
        F_pop=[
            [1.0, 1.0, 1.0],   # A: low obj, but big violation
            [10.0, 10.0, 10.0],  # B: high obj, small violation
        ],
        G_pop=[
            [0.0, 0.0, 5.0],    # A: violation=5
            [0.0, 0.0, 0.1],    # B: violation=0.1 (smaller)
        ],
    )
    adapter = TrajectorySOOAdapter(
        evaluator=evaluator,  # type: ignore[arg-type]
        start_positions=np.zeros((1, 3)),
        target_positions=np.ones((1, 3)),
        n_drones=1, n_inner=2,
        weights=np.array([1.0, 1.0, 1.0]),
        penalty_weight=100.0,
    )
    fitness = adapter(np.zeros((2, 1, 2, 3)))
    # B (smaller violation) powinno wygrać nad A (large violation),
    # mimo wyższych obj_values.
    assert fitness[1] < fitness[0], (
        f"Infeasible ordering broken: A (vio=5) fitness={fitness[0]}, "
        f"B (vio=0.1) fitness={fitness[1]}. B should win (smaller violation)."
    )


# ============================================================================
# REGRESSION: zerowe komponenty F_ref nie łamią Big-M ordering
# ============================================================================
# `_compute_reference_scales` używa `np.where(f_ref <= 1e-9, 1.0, f_ref)`
# zamiast cap'a `max(f_ref, 1e-6)`. Konsekwencja: dla naturalnie zerowego
# komponentu (np. f3 threat dla korytarza bez przeszkód), normalizacja
# pozostaje proporcjonalna 1:1 (F_norm[k] = F[k]) zamiast wzmacniać przez 1e6.
# Feasibility-first contract zachowany: feasible obj ~ O(1..100) << Big-M = 1e6.
# ============================================================================

import pytest  # noqa: E402


def test_soo_adapter_big_m_robust_to_zero_f_ref_component():
    """Feasibility-first contract MUSI być zachowany niezależnie od skali F_ref.

    Setup: straight-line ma f3=0 (brak threat na trasie). F_ref[2] = 1.0 (guard).
    Solution A: feasible z `f3 = 2.0` → `F_norm[2] = 2.0` → fitness ~ 8.0 (przy weight=0.8).
    Solution B: infeasible (violation=0.5) → fitness ≈ HARD_INFEASIBLE_BASE = 1e6.

    Oczekiwanie: A (feasible) MUSI mieć fitness < B (infeasible).
    """
    from src.algorithms.abstraction.trajectory.strategies.soo_adapter import (
        HARD_INFEASIBLE_BASE,
    )

    # F_init[2] = 0 → guard ustawia _f_ref[2] = 1.0 (neutralny mianownik).
    evaluator = FeasibilityFirstEvaluator(
        F_init=[[1.0, 1.0, 0.0, 1.0, 1.0]],   # f3 = 0 dla straight-line
        G_init=[[0.0, 0.0, 0.0]],
        F_pop=[
            [1.0, 1.0, 2.0, 1.0, 1.0],  # A: feasible, ale f3=2.0
            [1.0, 1.0, 0.0, 1.0, 1.0],  # B: infeasible
        ],
        G_pop=[
            [0.0, 0.0, 0.0],   # A: feasible
            [0.0, 0.0, 0.5],   # B: infeasible (kinematic_penalty)
        ],
    )
    adapter = TrajectorySOOAdapter(
        evaluator=evaluator,  # type: ignore[arg-type]
        start_positions=np.zeros((1, 3)),
        target_positions=np.ones((1, 3)),
        n_drones=1, n_inner=2,
        weights=np.array([1.0, 1.0, 0.8, 1.0, 1.0]),  # weight[2]=0.8 (jak w configs)
        penalty_weight=100.0,
    )
    fitness = adapter(np.zeros((2, 1, 2, 3)))

    # Feasibility-first contract: feasible (idx=0) musi mieć fitness < infeasible (idx=1).
    # Bez bug'a: fitness[0] = sum(weights × F_norm) = 1+1+0.8·2+1+1 = 5.6.
    # fitness[1] = HARD_INFEASIBLE_BASE + 100·0.5 = 1.000050e6.
    assert fitness[0] < HARD_INFEASIBLE_BASE, (
        f"Feasible fitness={fitness[0]:.2e} przekroczył HARD_INFEASIBLE_BASE "
        f"({HARD_INFEASIBLE_BASE}) — guard f_ref nie zadziałał."
    )
    assert fitness[0] < fitness[1], (
        f"Feasibility-first ordering złamany: feasible={fitness[0]:.2e}, "
        f"infeasible={fitness[1]:.2e}."
    )