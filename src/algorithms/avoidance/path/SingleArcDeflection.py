"""SingleArcDeflection — minimal-DOF path representation dla evolutionary avoidance.

Refactor 2026-05-02 (zastępuje `BSplineYZGenes`). Geometryczne wymuszenie
single-hump shape: drone wybiera JEDNĄ oś (przez `AxisChooser`), JEDEN punkt
peak deflection (gen `magnitude`), JEDNĄ pozycję wzdłuż trasy (gen `peak_position`).
Zigzag, multi-bend, kombinacje Y+Z są **niemożliwe geometrycznie**.

Search space: gene_dim = 2 (zamiast 10 z BSplineYZGenes):
    magnitude     ∈ [magnitude_min_m, magnitude_max_m]   [m]
    peak_position ∈ [peak_position_min, peak_position_max] [u along start→rejoin]

Mapowanie genów → spline (5 waypoints):
    1. axis_name, axis_unit = AxisChooser.pick(context)
    2. start  = context.drone_state.position
    3. rejoin = context.rejoin_point
    4. peak   = lerp(start, rejoin, peak_position) + magnitude × axis_unit
    5. q1     = lerp(start, peak, 0.5)   ← pomocnicze
    6. q3     = lerp(peak, rejoin, 0.5)  ← pomocnicze
    7. waypoints = [start, q1, peak, q3, rejoin]   (5 pkt — wystarczy dla cubic spline)
    8. floor/ceiling clamp na peak.z
    9. BSplineTrajectory(constant_speed=True, decel_at_end=True)
   10. min_applied_cruise_ratio filter (regression fix 2026-05-01)

Reference: similar prioritized-axis decomposition w klasycznych collision
avoidance papers (Fiorini-Shiller VO 1998), z analytical trajectory shape
zamiast full optimization (Mehdi et al. 2017 "Reactive Avoidance for UAVs").
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.algorithms.avoidance.interfaces import IPathRepresentation
from src.trajectory.BSplineTrajectory import BSplineTrajectory

if TYPE_CHECKING:
    from src.algorithms.avoidance.path.AxisChooser import AxisChooser
    from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import EvasionContext


logger = logging.getLogger(__name__)


class SingleArcDeflection(IPathRepresentation):
    """IPathRepresentation z 2D search space (magnitude, peak_position).

    Konstruktor parametryczny przez Hydra. `axis_chooser` wstrzykiwany przez
    `_target_` jako sub-instancja.
    """

    def __init__(
        self,
        axis_chooser: "AxisChooser",
        magnitude_min_m: float = 0.8,
        magnitude_max_m: float = 4.0,
        peak_position_min: float = 0.3,
        peak_position_max: float = 0.7,
        evasion_speed_multiplier: float = 0.85,
        floor_safe_margin_m: float = 1.0,
        ceiling_safe_margin_m: float = 1.0,
        min_applied_cruise_ratio: float = 0.4,
    ) -> None:
        if magnitude_min_m <= 0 or magnitude_max_m <= magnitude_min_m:
            raise ValueError(
                f"magnitude bounds invalid: min={magnitude_min_m}, max={magnitude_max_m}"
            )
        if not 0.0 < peak_position_min < peak_position_max < 1.0:
            raise ValueError(
                f"peak_position bounds invalid: [{peak_position_min}, {peak_position_max}]"
            )
        if floor_safe_margin_m < 0 or ceiling_safe_margin_m < 0:
            raise ValueError("floor/ceiling_safe_margin_m muszą być nieujemne.")
        if not 0.0 <= min_applied_cruise_ratio <= 1.0:
            raise ValueError(
                f"min_applied_cruise_ratio musi być w [0, 1]; got {min_applied_cruise_ratio}"
            )
        self.axis_chooser = axis_chooser
        self.magnitude_min = float(magnitude_min_m)
        self.magnitude_max = float(magnitude_max_m)
        self.peak_position_min = float(peak_position_min)
        self.peak_position_max = float(peak_position_max)
        self.speed_multiplier = float(evasion_speed_multiplier)
        self.floor_safe_margin = float(floor_safe_margin_m)
        self.ceiling_safe_margin = float(ceiling_safe_margin_m)
        self.min_applied_cruise_ratio = float(min_applied_cruise_ratio)

    # ---------------------- Evolutionary contract ---------------------- #

    def gene_dim(self, context: "EvasionContext") -> int:
        return 2

    def gene_bounds(
        self,
        context: "EvasionContext",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lb = np.array(
            [self.magnitude_min, self.peak_position_min], dtype=np.float64
        )
        ub = np.array(
            [self.magnitude_max, self.peak_position_max], dtype=np.float64
        )
        return lb, ub

    def decode_genes(
        self,
        genes: NDArray[np.float64],
        context: "EvasionContext",
    ) -> BSplineTrajectory | None:
        if len(genes) != 2:
            logger.error(
                f"SingleArcDeflection.decode_genes: expected 2 genes, got {len(genes)}"
            )
            return None

        magnitude = float(np.clip(genes[0], self.magnitude_min, self.magnitude_max))
        peak_position = float(
            np.clip(genes[1], self.peak_position_min, self.peak_position_max)
        )

        # Wybór osi przez deterministic AxisChooser.
        _, axis_unit = self.axis_chooser.pick(context)

        start = np.asarray(context.drone_state.position, dtype=np.float64)
        rejoin = np.asarray(context.rejoin_point, dtype=np.float64)

        peak = start + peak_position * (rejoin - start) + magnitude * axis_unit

        # Floor/ceiling Z clamp z bezpiecznym buforem.
        world_min, world_max = context.world_bounds
        z_floor = float(world_min[2]) + self.floor_safe_margin
        z_ceiling = float(world_max[2]) - self.ceiling_safe_margin
        peak[2] = float(np.clip(peak[2], z_floor, z_ceiling))

        # 5 waypoints: [start, q1, peak, q3, rejoin].
        q1 = 0.5 * (start + peak)
        q3 = 0.5 * (peak + rejoin)
        waypoints = np.vstack([start, q1, peak, q3, rejoin])

        return self._build_spline(waypoints, context)

    # ------------------- A*-style contract (legacy) ------------------- #

    def waypoints_to_spline(
        self,
        waypoints: NDArray[np.float64],
        context: "EvasionContext",
        *,
        axis_name: str | None = None,
    ) -> BSplineTrajectory | None:
        """Thin BSpline builder dla `GenericOptimizingAvoidance` po
        `optimize() → OptimizationResult.waypoints`. Używa tej samej budowy +
        clamp + filter jak `decode_genes`, ale bezpośrednio z surowych waypts.
        """
        if waypoints is None or len(waypoints) < 4:
            logger.warning(
                f"SingleArcDeflection.waypoints_to_spline: za krótka sekwencja "
                f"(len={0 if waypoints is None else len(waypoints)})"
            )
            return None
        return self._build_spline(np.asarray(waypoints, dtype=np.float64), context)

    # --------------------------- helpers ------------------------------ #

    def _build_spline(
        self,
        waypoints: NDArray[np.float64],
        context: "EvasionContext",
    ) -> BSplineTrajectory | None:
        current_speed = float(np.linalg.norm(context.drone_state.velocity))
        cruise = float(getattr(context.base_spline, "cruise_speed", 8.0))
        evasion_cruise = max(current_speed, cruise * self.speed_multiplier)
        max_accel = float(getattr(context.base_spline, "max_accel", 2.0))

        try:
            spline = BSplineTrajectory(
                waypoints=waypoints,
                cruise_speed=evasion_cruise,
                max_accel=max_accel,
                constant_speed=True,
                decel_at_end=True,
            )
        except Exception as e:
            logger.warning(
                f"SingleArcDeflection._build_spline: BSpline build fail "
                f"d{context.drone_id}: {e}"
            )
            return None

        # Filter kinematycznej wykonalności (regression fix 2026-05-01 #2).
        clamp = getattr(spline, "kinematic_clamp", None)
        if clamp is not None:
            req = float(clamp.get("requested_cruise", evasion_cruise))
            app = float(clamp.get("applied_cruise", evasion_cruise))
            if req > 1e-6 and (app / req) < self.min_applied_cruise_ratio:
                logger.debug(
                    f"SingleArcDeflection: d{context.drone_id} REJECTED severe "
                    f"clamp ratio {app/req:.3f} < {self.min_applied_cruise_ratio:.3f}"
                )
                return None

        return spline
