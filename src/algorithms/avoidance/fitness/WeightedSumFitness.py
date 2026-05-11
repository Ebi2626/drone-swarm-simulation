"""IFitnessEvaluator dla optymalizatorów ewolucyjnych w fazie online avoidance.

Skalarna ważona suma 4 składowych:
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
    ewolucyjnych. `axis_score` zaimplementowane minimalnie dla zgodności
    kontraktu z `IPathOptimizer._pick_preferred_axis`.
    """

    def __init__(
        self,
        w_safety: float = 50.0,
        w_energy: float = 0.5,
        w_jerk: float = 0.1,
        w_symmetry: float = 1.0,
        safe_clearance_m: float = 1.5,
        n_samples: int = 32,
        prefer_axis_order: list[str] | tuple[str, ...] = ("right", "left", "up", "down"),
        bias_preferred: float = 1.0,
        bias_perpendicular: float = 1.4,
        bias_oppose: float = 2.5,
    ) -> None:
        """Skonfiguruj wagi 4 składowych i parametry sticky-axis.

        Args:
            w_safety, w_energy, w_jerk, w_symmetry: Nieujemne wagi
                ważonej sumy. Skala C_* jest niezrównoważona — domyślne
                wartości oddają empirycznie dobrane proporcje.
            safe_clearance_m: Minimalny dystans od przeszkody [m] —
                naruszenie generuje karę kwadratową.
            n_samples: Liczba próbek po B-spline (jednolita siatka
                `u ∈ [0, 1]`).
            prefer_axis_order, bias_preferred, bias_perpendicular,
            bias_oppose: Parametry trybu `axis_score` (rzadko używane —
                ten fitness przede wszystkim ocenia pełne spliny).

        Raises:
            ValueError: Gdy którakolwiek `w_*` jest ujemna.
        """
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

    def evaluate(
        self,
        candidate: "BSplineTrajectory | None",
        context: EvasionContext,
        predictor: IObstaclePredictor,
    ) -> float:
        """Zwróć skalarny fitness ważoną sumą 4 składowych (mniej = lepiej).

        Args:
            candidate: B-spline z `IPathRepresentation.decode_genes` lub
                `None` (zdekodowanie zawiodło) — wtedy zwracana jest
                ekstremalna kara `≈ 4e9 × max(w_*)`.
            context: Pełny `EvasionContext`.
            predictor: Estymator pozycji przeszkody w funkcji czasu.

        Returns:
            Skalarna wartość kosztu — `0` ⇒ idealna trasa, sentinel ~`1e9`
            sygnalizuje zdegenerowanego kandydata.
        """
        components = self.evaluate_components(candidate, context, predictor)
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
        """Zwróć surowy `(4,)` wektor `[c_safety, c_energy, c_jerk, c_symmetry]` przed wagami.

        Używane przez `NSGA3OnlineOptimizer` do budowy frontu Pareto;
        warianty SOO (`MealpyOptimizer`, `MSFFOAOnlineOptimizer`) wołają
        `evaluate`, który ten wektor agreguje wagami.

        Args:
            candidate: B-spline kandydata albo `None` przy zdegenerowanym
                kandydacie.
            context: Pełny `EvasionContext`.
            predictor: Estymator pozycji przeszkody.

        Returns:
            `(4,)` składowe kosztu; sentinel `[1e9, 1e9, 1e9, 1e9]` przy
            braku kandydata lub błędach numerycznych w `splev`.
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

            # Multi-threat c_safety: primary threat + secondary_threats (inne
            # drony). Pusta lista secondary ⇒ liczymy tylko primary.
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
        """Kara za odchylenie kandydata od osi `preferred_axis_hint` (sticky-axis).

        Args:
            pos: `(N, 3)` próbki splajnu po `splev`.
            context: `EvasionContext`; gdy `preferred_axis_hint is None`,
                koszt = 0 (planer ma swobodę).

        Returns:
            Σ kwadratów projekcji przeciwnej do hintu — `0` przy braku hintu
            albo gdy spline w pełni leży po stronie zgodnej z hintem.
        """
        hint = context.preferred_axis_hint
        if hint is None:
            return 0.0

        start = np.asarray(context.drone_state.position, dtype=np.float64)
        rel = pos - start[None, :]

        # Lokalne osie: up/down = Z; right/left = perpendicular do v_drone w XY.
        v = np.asarray(context.drone_state.velocity, dtype=np.float64)
        v_xy = np.array([v[0], v[1], 0.0])
        v_xy_norm = float(np.linalg.norm(v_xy))
        if v_xy_norm > 1e-6:
            forward = v_xy / v_xy_norm
            lateral = np.array([-forward[1], forward[0], 0.0])  # +90° CCW = "right"
        else:
            lateral = np.array([0.0, 1.0, 0.0])

        if hint in ("up", "down"):
            sign = 1.0 if hint == "up" else -1.0
            proj = rel[:, 2] * sign
        elif hint in ("right", "left"):
            sign = 1.0 if hint == "right" else -1.0
            proj = (rel @ lateral) * sign
        else:
            return 0.0

        # Kara = Σ|projekcji przeciwnej do hintu|² (positive proj = zgodne z hintem).
        wrong_side = np.maximum(0.0, -proj)
        return float(np.sum(wrong_side ** 2))

    def order_score(self, axis_name: str) -> float:
        """Tie-break score `0…0.1` dla `axis_name` zgodnie z `prefer_axis_order`."""
        return self._order_scores.get(axis_name, 0.0)

    def axis_score(
        self,
        axis_name: str,
        axis_dir: NDArray[np.float64],
        space: float,
        obs_vel_hat: NDArray[np.float64] | None,
        order_score: float,
    ) -> float:
        """Score osi (analog `AxisBiasFitness`) — używany rzadko, dla zgodności kontraktu.

        Pełny opis kontraktu: `IFitnessEvaluator.axis_score`.
        """
        axis_hat = axis_dir / (np.linalg.norm(axis_dir) + 1e-9)
        if obs_vel_hat is not None:
            anti = max(0.0, -float(np.dot(axis_hat, obs_vel_hat)))
            if anti >= 0.15:
                return space * 1.0 * anti + order_score
        return order_score + 1e-3 * space
