"""IFitnessEvaluator dla optymalizatorów ewolucyjnych (Faza 2).

Skalarna ważona suma 4 składowych zgodnie ze specyfikacją:
    cost = w_safety · C_safety + w_energy · C_energy
         + w_jerk · C_jerk + w_symmetry · C_symmetry

Składowe:
  - C_safety   : kara za zbliżenie do przewidywanej pozycji przeszkody.
                 Quadratic hinge poniżej `safe_clearance` ponad fizycznym
                 promieniem przeszkody. Reference: Mehdi et al. 2017.
  - C_energy   : ∫ |∂²p/∂u²|² du — proxy dla energii sterowania.
                 (Kontroler PID minimalizuje przyspieszenie → koreluje z poborem mocy.)
  - C_jerk     : max_u |∂³p/∂u³| — peak control effort, kara za gwałtowne manewry.
  - C_symmetry : odchylenie od osi `preferred_axis_hint` (sticky-axis Fiorini-Shiller),
                 0 jeśli hint nieaktywny. Trzyma roje w spójnej decyzji unik-axis,
                 zapobiega flip-floppingowi.

Próbkowanie: `n_samples` punktów uniform w u∈[0,1]. Wszystkie składowe są
liczone na tej samej siatce — koszt amortyzowany.

Kontrakt wag: WSZYSTKIE wagi muszą być ≥ 0. Suma nie musi być znormalizowana
(skalowanie jest implicite w wartościach C_*) — jeśli to potrzebne, użytkownik
może je ręcznie wyważyć w yamlu.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splev

from src.algorithms.avoidance.interfaces import IFitnessEvaluator, IObstaclePredictor
from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import EvasionContext

if TYPE_CHECKING:
    from src.trajectory.BSplineTrajectory import BSplineTrajectory


logger = logging.getLogger(__name__)


class WeightedSumFitness(IFitnessEvaluator):
    """Skalarna ważona suma — `IFitnessEvaluator.evaluate` dla optymalizatorów
    ewolucyjnych. AStar tej klasy NIE używa (`axis_score` zaimplementowane
    minimalnie dla zgodności kontraktu z `IPathOptimizer._pick_preferred_axis`).
    """

    def __init__(
        self,
        w_safety: float = 50.0,
        w_energy: float = 0.5,
        w_jerk: float = 0.1,
        w_symmetry: float = 1.0,
        safe_clearance_m: float = 1.5,
        n_samples: int = 32,
        # Parametry AxisBiasFitness-style — używane TYLKO przez `axis_score`
        # (gdyby evolutionary fitness był wymieniony jako axis-picker). W
        # praktyce rzadko aktywowane — domyślne wartości jak w AxisBiasFitness.
        prefer_axis_order: list[str] | tuple[str, ...] = ("right", "left", "up", "down"),
        bias_preferred: float = 1.0,
        bias_perpendicular: float = 1.4,
        bias_oppose: float = 2.5,
    ) -> None:
        if min(w_safety, w_energy, w_jerk, w_symmetry) < 0:
            raise ValueError("Wagi WeightedSumFitness muszą być nieujemne.")
        self.w_safety = float(w_safety)
        self.w_energy = float(w_energy)
        self.w_jerk = float(w_jerk)
        self.w_symmetry = float(w_symmetry)
        self.safe_clearance = float(safe_clearance_m)
        self.n_samples = int(n_samples)

        self.prefer_axis_order = list(prefer_axis_order)
        self.bias_preferred = float(bias_preferred)
        self.bias_perpendicular = float(bias_perpendicular)
        self.bias_oppose = float(bias_oppose)
        n = len(self.prefer_axis_order)
        self._order_scores: dict[str, float] = {
            a: (n - i) / (10.0 * n) for i, a in enumerate(self.prefer_axis_order)
        }

    # -------------------------- Evolutionary --------------------------------- #

    def evaluate(
        self,
        candidate: "BSplineTrajectory | None",
        context: EvasionContext,
        predictor: IObstaclePredictor,
    ) -> float:
        """Skalar fitness; LOWER = better. Kontrakt jak w `IFitnessEvaluator`.

        :param candidate: BSpline z `IPathRepresentation.decode_genes` lub
            `None` gdy dekodowanie zawiodło — wtedy zwracamy ekstremalną karę.
        :param context: pełny `EvasionContext`.
        :param predictor: `IObstaclePredictor` do estymacji pozycji przeszkody w czasie.
        """
        components = self.evaluate_components(candidate, context, predictor)
        # Ekstremalna kara dla niezdekodowalnych — `evaluate_components` zwraca
        # wektor sentinelowy (1e9, 1e9, 1e9, 1e9), więc skalar też jest ekstremalny.
        return float(
            self.w_safety * components[0]
            + self.w_energy * components[1]
            + self.w_jerk * components[2]
            + self.w_symmetry * components[3]
        )

    def evaluate_components(
        self,
        candidate: "BSplineTrajectory | None",
        context: EvasionContext,
        predictor: IObstaclePredictor,
    ) -> NDArray[np.float64]:
        """Wektor [c_safety, c_energy, c_jerk, c_symmetry] (przed wagami).

        Zaprojektowane dla `NSGA3OnlineOptimizer` (multi-objective NSGA-III) —
        zwraca surowe składowe by pymoo mógł budować Pareto-front. SOO
        (`MealpyOptimizer`, `MSFFOAOnlineOptimizer`) wołają `evaluate` które
        wewnętrznie składa to przez wagi.

        :return: shape (4,). Sentinel `[1e9]*4` gdy `candidate is None` lub
                 ścieżka degeneracyjna (numerical error w splev).
        """
        SENTINEL = np.full(4, 1e9, dtype=np.float64)
        if candidate is None:
            return SENTINEL

        try:
            u = np.linspace(0.0, 1.0, self.n_samples)
            pos = np.asarray(splev(u, candidate.tck), dtype=np.float64).T  # (N, 3)
            d2 = np.asarray(splev(u, candidate.tck, der=2), dtype=np.float64).T
            d3 = np.asarray(splev(u, candidate.tck, der=3), dtype=np.float64).T

            duration = float(candidate.total_duration)
            t_samples = u * duration

            # Multi-threat c_safety: primary threat + secondary_threats (other drones).
            # Bez secondary_threats (back-compat) liczymy tylko primary.
            all_threats = [context.threat] + list(getattr(context, "secondary_threats", []))
            c_safety = 0.0
            for threat in all_threats:
                obs_positions = np.empty((self.n_samples, 3), dtype=np.float64)
                for i, t in enumerate(t_samples):
                    state = predictor.predict_state(threat, float(t))
                    obs_positions[i] = state.position

                obs_r = float(threat.obstacle_state.radius)
                min_dist = obs_r + self.safe_clearance
                d = np.linalg.norm(pos - obs_positions, axis=1)
                violation = np.maximum(0.0, min_dist - d)
                c_safety += float(np.sum(violation ** 2))

            c_energy = float(np.mean(np.sum(d2 ** 2, axis=1)))
            c_jerk = float(np.max(np.linalg.norm(d3, axis=1)))
            c_symmetry = self._symmetry_cost(pos, context)

            return np.array([c_safety, c_energy, c_jerk, c_symmetry], dtype=np.float64)

        except Exception as e:
            logger.warning(
                f"WeightedSumFitness.evaluate_components: błąd numeryczny "
                f"d{context.drone_id}: {e}"
            )
            return SENTINEL

    def _symmetry_cost(
        self,
        pos: NDArray[np.float64],  # (N, 3)
        context: EvasionContext,
    ) -> float:
        """Kara za odchylenie od osi `preferred_axis_hint` (sticky-axis).

        Jeśli `preferred_axis_hint` jest `None` — koszt = 0 (optymalizator
        ma swobodę). Jeśli ustawiony — penalizujemy projekcję spline'u na
        oś PRZECIWNĄ do hintu (np. hint=right → penalty na spline_y < start_y).
        """
        hint = context.preferred_axis_hint
        if hint is None:
            return 0.0

        start = np.asarray(context.drone_state.position, dtype=np.float64)
        # Centrum manewru — punkty względem startu.
        rel = pos - start[None, :]  # (N, 3)

        # Definicje osi w lokalnym frame:
        # - "up" / "down": Z-axis
        # - "right" / "left": kierunek prostopadły do v_drone w XY (zgodny z AStar)
        v = np.asarray(context.drone_state.velocity, dtype=np.float64)
        v_xy = np.array([v[0], v[1], 0.0])
        v_xy_norm = float(np.linalg.norm(v_xy))
        if v_xy_norm > 1e-6:
            forward = v_xy / v_xy_norm
            lateral = np.array([-forward[1], forward[0], 0.0])  # +90° rotation in XY
        else:
            lateral = np.array([0.0, 1.0, 0.0])

        if hint in ("up", "down"):
            sign = 1.0 if hint == "up" else -1.0
            # Projekcja na Z; karzemy projekcję o przeciwnym znaku do hintu.
            proj = rel[:, 2] * sign  # >0 zgodne z hintem, <0 przeciwne
        elif hint in ("right", "left"):
            sign = 1.0 if hint == "right" else -1.0
            proj = (rel @ lateral) * sign
        else:
            return 0.0

        # Kara = suma kwadratów części „przeciwnej" do hintu.
        wrong_side = np.maximum(0.0, -proj)
        return float(np.sum(wrong_side ** 2))

    # -------------------------- A* axis_score (cooperative kontrakt) -------- #

    def order_score(self, axis_name: str) -> float:
        return self._order_scores.get(axis_name, 0.0)

    def axis_score(
        self,
        axis_name: str,
        axis_dir: NDArray[np.float64],
        space: float,
        obs_vel_hat: NDArray[np.float64] | None,
        order_score: float,
    ) -> float:
        """Cooperative axis-score (analogicznie jak `AxisBiasFitness`) — używane
        WYŁĄCZNIE jeśli `WeightedSumFitness` byłby wpięty jako axis-picker do
        `AStarOptimizer` (mało prawdopodobne; AStar w yaml-ach Fazy 2 zwykle
        ma wpięty `AxisBiasFitness`). Zachowane dla type-safety kontraktu.
        """
        axis_hat = axis_dir / (np.linalg.norm(axis_dir) + 1e-9)
        if obs_vel_hat is not None:
            anti = max(0.0, -float(np.dot(axis_hat, obs_vel_hat)))
            if anti >= 0.15:
                return space * 1.0 * anti + order_score
        return order_score + 1e-3 * space
