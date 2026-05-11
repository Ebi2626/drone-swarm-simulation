"""Adapter SOO dla `VectorizedEvaluator`: skalaryzacja MOO → fitness skalarny.

Most między wielokryterialnym `VectorizedEvaluator` (F: obiektywy,
G: ograniczenia) a skalarnym fitnessem wymaganym przez metaheurystyki
single-objective (PSO, DE, warianty FOA, mealpy SSA/OOA).

Dwie reguły niezmienne:
1. **Normalizacja obiektywów** — F jest dzielone przez `F_ref` (referencyjna
   linia prosta) przed ważeniem, więc każdy obiektyw operuje w bezwymiarowej
   skali ~1.0.
2. **Hard Feasibility-First (Big-M)** — rozwiązania niewykonalne
   (jakiekolwiek `G[k] > 0`) dostają fitness ≥ `HARD_INFEASIBLE_BASE` (1e6)
   proporcjonalny do sumy naruszeń; rozwiązania wykonalne dostają samo
   `obj_values`. Skutek: wykonalne ZAWSZE dominują niewykonalne, niezależnie
   od różnic w obiektywach. Odwzorowanie feasibility-first dominance z
   NSGA-III (Deb 2000 §V.A) w wariancie SOO.

Soft penalty (`np.max(0, G) * weight`) pozwalał wcześniej, by niewykonalne
rozwiązanie z niskim obj wygrywało nad wykonalnym z wyższym obj — optymalizator
zwracał kinematycznie niewykonalne trajektorie i drony spadały.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from src.algorithms.abstraction.trajectory.objective_constrains import VectorizedEvaluator


# Big-M dla feasibility-first ordering (Golden Rule #2). Większe niż
# jakakolwiek plausible obj_value po normalizacji (F_norm ~ O(1) per obj,
# suma ≤ ~100 dla typowych weights), więc każde infeasible ma fitness
# bezpiecznie > każde feasible.
HARD_INFEASIBLE_BASE: float = 1e6


class TrajectorySOOAdapter:
    """Skalaryzator wyjść `VectorizedEvaluator` do pojedynczego fitnessu.

    Instancja jest wywoływalna (`__call__`) i może być bezpośrednio
    przekazana jako `fitness_function` do dowolnego optymalizatora SOO
    operującego na tensorach inner-waypointów.

    Przebieg pojedynczego wywołania:

    1. Przyjmuje inner waypointy `(Pop, N_drones, Inner, 3)`.
    2. Dokleja pozycje startowe i docelowe (rekonstrukcja pełnego wieloboku).
    3. Pyta `VectorizedEvaluator` → `F (Pop, M)`, `G (Pop, K)`.
    4. Normalizuje `F` przez `F_ref` (reguła #1).
    5. Agreguje `G` przez Big-M feasibility-first (reguła #2):
       `total_violation = Σ max(0, G[k])` — kara rośnie liniowo z każdym
       naruszeniem (porządkowanie wewnątrz kubełka niewykonalnych); każde
       niewykonalne (jakiekolwiek `G[k] > 0`) trafia do kubełka
       `≥ HARD_INFEASIBLE_BASE`.
    6. Zwraca `(Pop,)` fitness skalarny (mniej = lepiej).

    Args:
        evaluator: Skonfigurowany `VectorizedEvaluator`.
        start_positions: `(N_drones, 3)` pozycje startowe [m].
        target_positions: `(N_drones, 3)` pozycje docelowe [m].
        n_drones: Liczba dronów w roju.
        n_inner: Liczba wewnętrznych węzłów kontrolnych na drona.
        weights: `(M=5,)` wagi dla 5 obiektywów `VectorizedEvaluator`:
            `[w_f1_trajectory, w_f2_height_angle, w_f3_threat,
            w_f4_turn, w_f5_coordination]`.
        penalty_weight: Mnożnik wielkości naruszenia w kubełku Big-M
            (rozdziela rozwiązania niewykonalne wewnątrz kubełka).
        history_writer: Opcjonalny logger zapisu danych per generacja.
    """

    def __init__(
        self,
        evaluator: VectorizedEvaluator,
        start_positions: NDArray[np.float64],
        target_positions: NDArray[np.float64],
        n_drones: int,
        n_inner: int,
        weights: NDArray[np.float64],
        penalty_weight: float = 100.0,
        history_writer: Optional[Any] = None,
    ) -> None:
        self.evaluator = evaluator
        self.n_drones = n_drones
        self.n_inner = n_inner
        self.weights = np.asarray(weights, dtype=np.float64)
        self.penalty_weight = penalty_weight
        self.history_writer = history_writer

        # Broadcast-ready endpoint shapes: (1, N_drones, 1, 3)
        self._starts_bc = np.asarray(start_positions, dtype=np.float64)[np.newaxis, :, np.newaxis, :]
        self._targets_bc = np.asarray(target_positions, dtype=np.float64)[np.newaxis, :, np.newaxis, :]

        # Golden Rule #1: compute F_ref from a straight-line trajectory
        self._f_ref = self._compute_reference_scales()
        
        # State tracking for logger/optimizer access
        self.last_objectives: NDArray[np.float64] | None = None
        self.last_constraints: NDArray[np.float64] | None = None
        self._debug_printed: bool = False

    def _compute_reference_scales(self) -> NDArray[np.float64]:
        """Wylicz `F_ref` na bazie referencyjnej trajektorii prostej (linia start→target).

        Strategia bez sztucznego cap'a:
        - Dla `f_ref[k] > 1e-9`: używamy faktycznej wartości referencyjnej —
          normalizacja proporcjonalna do skali obiektywu.
        - Dla `f_ref[k] ≤ 1e-9` (komponent zerowy na linii prostej, np. f3
          threat w korytarzu bez przeszkód): `1.0` jako neutralny mianownik.
          Skutek: `F_norm[k] = F[k]` zachowuje feasibility-first ordering —
          wykonalne `F[k] = O(1..10)` daje fitness `O(1..100)`, zawsze poniżej
          `HARD_INFEASIBLE_BASE = 1e6`.

        Cap `max(f_ref, 1e-6)` byłby pułapką: dla `f_ref[k] = 0` normalizacja
        przez `1e-6` dałaby `F_norm[k] = 1e6 · F[k]`, pozwalając wykonalnemu
        rozwiązaniu wyjść powyżej Big-M i złamać regułę #2 (test xfail:
        `test_soo_adapter_big_m_robust_to_zero_f_ref_component`).

        Returns:
            `(M,)` referencyjne wartości obiektywów, zawsze `> 0`.
        """
        # Generate evenly spaced intermediate points along the straight line
        t_vals = np.linspace(0, 1, self.n_inner + 2)[1:-1]
        t = t_vals.reshape(1, 1, self.n_inner, 1)

        # Interpolation via broadcasting
        inner_ref = self._starts_bc + t * (self._targets_bc - self._starts_bc)

        # Assemble sparse points
        sparse_ref = np.concatenate(
            [self._starts_bc, inner_ref, self._targets_bc], axis=2
        )

        out_ref: Dict[str, Any] = {}
        self.evaluator.evaluate(sparse_ref, out_ref)

        f_ref = np.asarray(out_ref["F"][0], dtype=np.float64)

        # Guard zerowych referencji — zwracamy `1.0` (neutralny mianownik)
        # zamiast `1e-6`, żeby nie wzmacniać F_norm. Próg `1e-9` chroni
        # przed numerycznym szumem (faktyczne zera są rzadkie ale możliwe
        # dla f3=threat na czystym korytarzu).
        zero_ref_mask = f_ref <= 1e-9
        if np.any(zero_ref_mask):
            self._zero_ref_components = np.where(zero_ref_mask)[0].tolist()
        else:
            self._zero_ref_components = []
        return np.where(zero_ref_mask, 1.0, f_ref)

    def __call__(self, inner_waypoints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Oceń całą populację inner-waypointów i zwróć fitness skalarny.

        Args:
            inner_waypoints: `(Pop_size, N_drones, N_inner, 3)` — same węzły
                wewnętrzne (bez `start` i `target`, doklejane w środku).

        Returns:
            `(Pop_size,)` wartości fitnessu skalarnego (mniej = lepiej).
            Wykonalne dostają wartości `O(1..100)`, niewykonalne `≥ 1e6`.
        """
        pop_size = inner_waypoints.shape[0]

        # Use np.broadcast_to to create memory-efficient views without copying
        starts = np.broadcast_to(self._starts_bc, (pop_size, self.n_drones, 1, 3))
        targets = np.broadcast_to(self._targets_bc, (pop_size, self.n_drones, 1, 3))

        # Reconstruct the full polyline (discrete waypoints)
        trajectories = np.concatenate([starts, inner_waypoints, targets], axis=2)

        # Query VectorizedEvaluator
        out: Dict[str, Any] = {}
        self.evaluator.evaluate(trajectories, out)

        F = out["F"]  # Objectives: (Pop_size, M)
        G = out["G"]  # Constraints: (Pop_size, K)

        self.last_objectives = F
        self.last_constraints = G

        # Golden Rule #1: Normalize objectives by reference scales
        # F shapes: (Pop_size, M) / (M,) -> Broadcasts naturally
        F_norm = F / self._f_ref

        # Weighted objective sum (vectorized dot product)
        # F_norm @ weights mathematically executes \sum_{i=1}^{M} w_i * F_{norm, i}
        obj_values = F_norm @ self.weights  # Result shape: (Pop_size,)

        # Golden Rule #2: Hard Feasibility-First gating (Big-M).
        # pymoo standard: G <= 0 feasible, G > 0 violation.
        # Total violation = sum of positive parts (Big-M ordering INSIDE
        # infeasible bucket: smaller violation → smaller fitness in bucket).
        violations_clipped = np.maximum(0.0, G)
        total_violation = np.sum(violations_clipped, axis=1)  # (Pop_size,)
        infeasible_mask = total_violation > 0.0

        # Big-M base — większe niż jakakolwiek plausible obj_value (z F_norm
        # typowo O(1-10), suma weighted obj ≤ ~100). 1e6 to bezpieczny margin.
        # Każde infeasible: fitness ≥ HARD_INFEASIBLE_BASE → feasible (≤ ~100)
        # ZAWSZE wygrywa.
        fitness = np.where(
            infeasible_mask,
            HARD_INFEASIBLE_BASE + self.penalty_weight * total_violation,
            obj_values,
        )

        # Optional: Print debug info only on the first pass
        if not self._debug_printed:
            print(f"\n[DEBUG SOO] Obj Norm (First Mem): {F_norm[0]} | Raw G: {G[0]}")
            print(
                f"[DEBUG SOO] Fitness: {fitness[0]:.4f} | "
                f"Feasible={not bool(infeasible_mask[0])}"
            )
            self._debug_printed = True

        return fitness
