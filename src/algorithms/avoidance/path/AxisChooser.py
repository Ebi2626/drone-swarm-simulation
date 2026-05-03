"""AxisChooser — deterministyczna heurystyka wyboru osi uniku.

Refactor 2026-05-02 (Single-Arc Deflection): hierarchical decomposition problemu.
Pytanie "którędy uciec?" (dyskretne: right/left/up/down) jest oddzielone od
"jak daleko?" (ciągłe: magnitude). Dyskretną decyzję podejmujemy analitycznie
przed search'em — nie ma sensu eksplorować jej przez evolutionary algos.

Score per oś:
    score = w_clearance * clearance_norm
          + w_anti_threat * max(0, anti_threat_dot)
          + w_secondary_blocking * sec_blocking_factor

Gdzie:
- `clearance` [m] — dystans od `drone_state.position` do `search_space_min/max`
  w kierunku osi. Większa clearance = więcej miejsca na manewr.
- `anti_threat ∈ [-1, 1]` = `-dot(axis_unit, threat.velocity_unit)` — preferuje
  osie odsuwające drone od kierunku ruchu zagrożenia (Fiorini-Shiller VO,
  klasyka collision avoidance).
- `sec_blocking_factor ∈ [0, 1]` = 1 jeśli żaden secondary threat nie leży w
  cone'ie ±`secondary_block_cone_deg` wokół axis_unit w obrębie
  `secondary_block_range_m`. Inaczej `0.3` (oś częściowo zablokowana).

Sticky-axis (Fiorini-Shiller anti-flip-flop): jeśli `context.preferred_axis_hint`
to viable axis (score(hint) >= sticky_threshold * score(best)), zwróć hint.
Eliminuje flip-flopping up↔right↔left w trakcie wieloetapowego uniku.

Reference: Fiorini & Shiller (1998), "Motion Planning in Dynamic Environments
Using Velocity Obstacles", Int. J. Robotics Research 17(7).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import EvasionContext


logger = logging.getLogger(__name__)


_AXES_NAMES: tuple[str, str, str, str] = ("right", "left", "up", "down")


class AxisChooser:
    """Deterministic O(1) wyborca osi uniku.

    Konstruktor parametryczny przez Hydra `_target_`. Brak runtime state — tylko
    wagi i progi. Stateless, thread-safe.
    """

    def __init__(
        self,
        w_clearance: float = 1.0,
        w_anti_threat: float = 1.0,
        w_secondary_blocking: float = 0.5,
        sticky_hint_threshold: float = 0.7,
        secondary_block_cone_deg: float = 20.0,
        secondary_block_range_m: float = 10.0,
    ) -> None:
        if min(w_clearance, w_anti_threat, w_secondary_blocking) < 0:
            raise ValueError("Wagi AxisChooser muszą być nieujemne.")
        if not 0.0 <= sticky_hint_threshold <= 1.0:
            raise ValueError("sticky_hint_threshold musi być w [0, 1].")
        if not 0.0 < secondary_block_cone_deg < 90.0:
            raise ValueError("secondary_block_cone_deg musi być w (0, 90).")
        if secondary_block_range_m < 0.0:
            raise ValueError("secondary_block_range_m musi być nieujemne.")

        self.w_clearance = float(w_clearance)
        self.w_anti_threat = float(w_anti_threat)
        self.w_secondary_blocking = float(w_secondary_blocking)
        self.sticky_hint_threshold = float(sticky_hint_threshold)
        self.secondary_block_cone_cos = float(
            np.cos(np.deg2rad(secondary_block_cone_deg))
        )
        self.secondary_block_range_m = float(secondary_block_range_m)

    def pick(
        self,
        context: "EvasionContext",
    ) -> tuple[str, NDArray[np.float64]]:
        """Wybierz oś uniku.

        :return: `(axis_name, axis_unit_vector_3d)`. `axis_unit_vector_3d` jest
            znormalizowany do długości 1.0 (zawsze).
        """
        # Compute forward (drone heading XY) i lateral_xy (perpendicular).
        v = np.asarray(context.drone_state.velocity, dtype=np.float64)
        v_xy = np.array([v[0], v[1], 0.0], dtype=np.float64)
        v_xy_norm = float(np.linalg.norm(v_xy))
        if v_xy_norm > 1e-6:
            forward_xy = v_xy / v_xy_norm
        else:
            forward_xy = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        # +90° rotation in XY plane (matches sign convention of dawnego A*).
        lateral_xy = np.array(
            [-forward_xy[1], forward_xy[0], 0.0], dtype=np.float64
        )

        axis_units: dict[str, NDArray[np.float64]] = {
            "right": lateral_xy.copy(),
            "left": -lateral_xy.copy(),
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
            "down": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        }

        # Threat velocity unit (None jeśli zagrożenie stoi/quasi-stoi).
        t_vel = np.asarray(
            context.threat.obstacle_state.velocity, dtype=np.float64
        )
        t_vel_norm = float(np.linalg.norm(t_vel))
        threat_vel_unit: NDArray[np.float64] | None = (
            t_vel / t_vel_norm if t_vel_norm > 1e-3 else None
        )

        drone_pos = np.asarray(context.drone_state.position, dtype=np.float64)
        bbox_min = np.asarray(context.search_space_min, dtype=np.float64)
        bbox_max = np.asarray(context.search_space_max, dtype=np.float64)
        secondary_threats = list(getattr(context, "secondary_threats", []))

        # Score every axis.
        clearances: dict[str, float] = {}
        scores_raw: dict[str, dict[str, float]] = {}
        for name in _AXES_NAMES:
            axis = axis_units[name]
            clearance = self._clearance_along_axis(
                drone_pos, axis, bbox_min, bbox_max
            )
            anti_threat = (
                -float(np.dot(axis, threat_vel_unit))
                if threat_vel_unit is not None
                else 0.0
            )
            sec_factor = self._secondary_blocking_factor(
                drone_pos, axis, secondary_threats
            )
            clearances[name] = clearance
            scores_raw[name] = {
                "clearance": clearance,
                "anti_threat": anti_threat,
                "sec_factor": sec_factor,
            }

        clearance_max = max(clearances.values()) or 1.0
        scores: dict[str, float] = {}
        for name in _AXES_NAMES:
            r = scores_raw[name]
            scores[name] = (
                self.w_clearance * (r["clearance"] / clearance_max)
                + self.w_anti_threat * max(0.0, r["anti_threat"])
                + self.w_secondary_blocking * r["sec_factor"]
            )

        best_name = max(scores.keys(), key=lambda n: scores[n])
        best_score = scores[best_name]

        # Sticky hint (anti-flip-flop).
        hint = getattr(context, "preferred_axis_hint", None)
        if (
            hint in scores
            and best_score > 1e-9
            and scores[hint] >= self.sticky_hint_threshold * best_score
        ):
            return hint, axis_units[hint]

        return best_name, axis_units[best_name]

    @staticmethod
    def _clearance_along_axis(
        drone_pos: NDArray[np.float64],
        axis_unit: NDArray[np.float64],
        bbox_min: NDArray[np.float64],
        bbox_max: NDArray[np.float64],
    ) -> float:
        """Dystans od `drone_pos` do brzegu bbox w kierunku `axis_unit` [m].

        Liczymy `t_max` taki że `drone_pos + t_max * axis_unit` jest jeszcze
        w bbox. Dla każdej osi i = 0,1,2 (X, Y, Z):
            jeśli axis[i] > 0: t_i = (bbox_max[i] - drone[i]) / axis[i]
            jeśli axis[i] < 0: t_i = (bbox_min[i] - drone[i]) / axis[i]
            jeśli axis[i] = 0: nieograniczony w tym wymiarze.
        Wynik = min(t_i across i).
        """
        t_max = float("inf")
        for i in range(3):
            a = float(axis_unit[i])
            if abs(a) < 1e-9:
                continue
            if a > 0.0:
                t_i = (float(bbox_max[i]) - float(drone_pos[i])) / a
            else:
                t_i = (float(bbox_min[i]) - float(drone_pos[i])) / a
            if t_i < t_max:
                t_max = t_i
        return max(0.0, t_max if t_max != float("inf") else 0.0)

    def _secondary_blocking_factor(
        self,
        drone_pos: NDArray[np.float64],
        axis_unit: NDArray[np.float64],
        secondary_threats: list,
    ) -> float:
        """1.0 jeśli żaden secondary threat nie blokuje osi, 0.3 jeśli tak.

        Threat blokuje gdy:
        - jest w obrębie `secondary_block_range_m` od `drone_pos`,
        - vector od drone_pos do threat ma cos angle do axis_unit
          >= self.secondary_block_cone_cos (czyli threat leży w cone'ie ±θ wokół osi).
        """
        for st in secondary_threats:
            obs_pos = np.asarray(
                st.obstacle_state.position, dtype=np.float64
            )
            rel = obs_pos - drone_pos
            d = float(np.linalg.norm(rel))
            if d > self.secondary_block_range_m or d < 1e-6:
                continue
            cos_angle = float(np.dot(rel / d, axis_unit))
            if cos_angle >= self.secondary_block_cone_cos:
                return 0.3
        return 1.0
