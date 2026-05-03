from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.algorithms.avoidance.interfaces import IFitnessEvaluator


class AxisBiasFitness(IFitnessEvaluator):
    """Heurystyka wyboru osi uniku — używana wyłącznie przez `AStarOptimizer`.

    Heurystyka wyboru osi uniku (right/left/up/down) na podstawie:
      - dostępnej przestrzeni wzdłuż osi (`space`),
      - anty-prędkościowego bias z progiem szumu (`anti_threshold`),
      - tie-breaku po `prefer_axis_order`.

    Sticky-axis (utrzymanie osi z poprzedniego planu) jest realizowany
    przez `AStarOptimizer._pick_preferred_axis` — to fitness wybierany
    jest TYLKO gdy nie ma poprzedniego hintu lub jest niewykonalny.

    `evaluate(spline, …)` celowo zostawione jako `NotImplementedError` z
    klasy bazowej — `AStarOptimizer` nie potrzebuje skalarnego fitness pełnej
    trajektorii (jego heurystyka kosztu jest wbudowana w bias gridowy w
    `UAV3DGridSearch.distance_between`). Ewolucyjne optymalizatory (Faza 2)
    dostaną `WeightedSumFitness` jako osobną klasę z pełną implementacją
    `evaluate()`.
    """

    def __init__(
        self,
        prefer_axis_order: list[str] | tuple[str, ...] = ("right", "left", "up", "down"),
        bias_preferred: float = 1.0,
        bias_perpendicular: float = 1.4,
        bias_oppose: float = 2.5,
        axis_anti_obsvel_gain: float = 1.0,
        anti_threshold: float = 0.15,
    ) -> None:
        self.prefer_axis_order = list(prefer_axis_order)
        self.bias_preferred = float(bias_preferred)
        self.bias_perpendicular = float(bias_perpendicular)
        self.bias_oppose = float(bias_oppose)
        self.axis_anti_obsvel_gain = float(axis_anti_obsvel_gain)
        # Próg szumu w składowej anty-obs_vel — patrz regresja Fazy 6 (incydent
        # „dron 0 → ziemia" z 2026-04-22, test:
        # `test_pick_preferred_axis_noise_level_obs_vz_does_not_override_prefer_axis_order`).
        self.anti_threshold = float(anti_threshold)

        # Tie-break score per oś (0…0.1) — preferowana oś z `prefer_axis_order`
        # dostaje minimalnie wyższy score by przeważyć w razie remisu.
        n = len(self.prefer_axis_order)
        self._order_scores: dict[str, float] = {
            a: (n - i) / (10.0 * n) for i, a in enumerate(self.prefer_axis_order)
        }

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
        axis_hat = axis_dir / (np.linalg.norm(axis_dir) + 1e-9)
        if obs_vel_hat is not None:
            anti = max(0.0, -float(np.dot(axis_hat, obs_vel_hat)))
            if anti >= self.anti_threshold:
                return space * self.axis_anti_obsvel_gain * anti + order_score
        # Brak istotnego anty-bias-u → tie-break + drobny dodatek od dostępnej
        # przestrzeni (faworyzuje osie z większą rezerwą).
        return order_score + 1e-3 * space
