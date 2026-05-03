from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splev

from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import EvasionContext
from src.algorithms.avoidance.interfaces import IPathRepresentation
from src.algorithms.avoidance.path._jit_kernels import (
    jit_douglas_peucker,
    jit_insert_tangent_leads,
    jit_resample_uniform,
)
from src.trajectory.BSplineTrajectory import BSplineTrajectory


logger = logging.getLogger(__name__)


class BSplineSmoother(IPathRepresentation):
    """Konwersja surowych waypointów (z A* lub ewolucyjnych) na wykonalny BSpline.

    Pipeline:
      1. Douglas-Peucker — redukcja redundantnych punktów (`epsilon_m`).
      2. Resampling do `[min_waypoints, max_waypoints]` jeśli skrajny przypadek.
      3. Klamp Z do `[world_min_z, world_max_z]`.
      4. Dla osi lateralnych (right/left) liniowa interpolacja Z między skrajami
         — zapobiega dziedziczeniu Z-oscylacji z gridu A* przy `constant_speed=True`
         (regresja Fazy 8.1 — incydent „drone 2 z=6.5→0.12 m w 1 s podczas
         axis=right" w eksperymencie 21-24-21, test:
         `test_lateral_evasion_forces_horizontal_z_profile_in_smoother`).
      5. Wstawienie lead-in / lead-out (styczne do v_drone na starcie i do
         tangensu base_spline na rejoin) — gładki powrót do trasy bazowej.
      6. Pętla `lead_mult ∈ {1.0, 2.0, 3.0}`: budowa BSplinu, pomiar
         odchyłki tangensu na rejoin. Wybierany wariant z najlepszym `cos θ`.
      7. Odrzucenie planu jeśli `cos θ < cos(reject_tangent_angle_deg)`.

    Parametry sterujące dziedziczone z `configs/avoidance/astar.yaml`,
    domyślne wartości zachowują dotychczasowe zachowanie produkcyjne.
    """

    def __init__(
        self,
        douglas_peucker_epsilon_m: float = 0.6,
        min_waypoints: int = 4,
        max_waypoints: int = 8,
        evasion_speed_multiplier: float = 0.85,
        reject_tangent_angle_deg: float = 25.0,
        lead_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0),
    ) -> None:
        self.dp_epsilon = float(douglas_peucker_epsilon_m)
        self.min_waypoints = int(min_waypoints)
        self.max_waypoints = int(max_waypoints)
        self.speed_multiplier = float(evasion_speed_multiplier)
        self.reject_tangent_angle_deg = float(reject_tangent_angle_deg)
        self.lead_multipliers = tuple(float(m) for m in lead_multipliers)

    def waypoints_to_spline(
        self,
        waypoints: NDArray[np.float64],
        context: EvasionContext,
        *,
        axis_name: str | None = None,
    ) -> BSplineTrajectory | None:
        if waypoints is None or len(waypoints) < 2:
            logger.warning(
                f"BSplineSmoother: waypoints zdegenerowane "
                f"(len={0 if waypoints is None else len(waypoints)}) — zwracam None"
            )
            return None

        current_pos = np.asarray(context.drone_state.position, dtype=np.float64)
        rejoin_point = np.asarray(context.rejoin_point, dtype=np.float64)
        current_vel = np.asarray(context.drone_state.velocity, dtype=np.float64)
        current_speed = float(np.linalg.norm(current_vel))

        world_min, world_max = context.world_bounds
        floor_z = float(world_min[2])
        ceiling_z = float(world_max[2])

        obs_pos = np.asarray(context.threat.obstacle_state.position, dtype=np.float64)
        obs_radius = float(context.threat.obstacle_state.radius)

        cruise = float(getattr(context.base_spline, "cruise_speed", 8.0))
        ref_speed = max(current_speed, cruise * 0.5)
        evasion_cruise = max(current_speed, cruise * self.speed_multiplier)
        max_accel = float(getattr(context.base_spline, "max_accel", 2.0))

        # 1) Douglas-Peucker → redukcja gęstości waypointów z grid A*.
        smoothed = jit_douglas_peucker(np.asarray(waypoints, dtype=np.float64), self.dp_epsilon)
        smoothed[0] = current_pos
        smoothed[-1] = rejoin_point

        # 2) Egzekwowanie przedziału [min_waypoints, max_waypoints].
        if len(smoothed) < self.min_waypoints:
            smoothed = jit_resample_uniform(
                np.asarray(waypoints, dtype=np.float64), self.min_waypoints
            )
            smoothed[0] = current_pos
            smoothed[-1] = rejoin_point
        if len(smoothed) > self.max_waypoints:
            smoothed = jit_resample_uniform(smoothed, self.max_waypoints)
            smoothed[0] = current_pos
            smoothed[-1] = rejoin_point

        forward_3d = self._compute_forward_direction(
            current_vel, context.base_spline, context.rejoin_base_arc
        )
        base_tangent_at_rejoin = self._base_tangent_at_arc(
            context.base_spline, context.rejoin_base_arc
        )
        cos_threshold = float(np.cos(np.deg2rad(self.reject_tangent_angle_deg)))

        evasion_spline: BSplineTrajectory | None = None
        final_cos = -1.0

        # 6) Pętla lead_mult — wybierz wariant z najlepszą zgodnością tangensów.
        for lead_mult in self.lead_multipliers:
            wp_try = jit_insert_tangent_leads(
                smoothed,
                current_pos,
                forward_3d,
                rejoin_point,
                base_tangent_at_rejoin,
                ref_speed,
                obs_pos,
                obs_radius,
                lead_mult,
            )
            wp_try[:, 2] = np.clip(wp_try[:, 2], floor_z, ceiling_z)

            # 4) Z-linearizacja dla osi lateralnych (right/left) — patrz docstring.
            if axis_name in ("right", "left") and len(wp_try) >= 2:
                n = len(wp_try)
                z_start = float(wp_try[0, 2])
                z_end = float(wp_try[-1, 2])
                alphas = np.linspace(0.0, 1.0, n)
                wp_try[:, 2] = (1.0 - alphas) * z_start + alphas * z_end

            try:
                spline_try = BSplineTrajectory(
                    waypoints=wp_try,
                    cruise_speed=evasion_cruise,
                    max_accel=max_accel,
                    constant_speed=True,
                    decel_at_end=True,
                )
            except Exception as e:
                logger.warning(
                    f"BSplineSmoother: BSpline build fail "
                    f"d{context.drone_id} (lead_mult={lead_mult:.1f}): {e}"
                )
                continue

            tangent_end = np.asarray(splev(1.0, spline_try.tck, der=1), dtype=np.float64)
            tnorm = float(np.linalg.norm(tangent_end))
            if tnorm < 1e-6:
                continue
            tangent_end /= tnorm
            cos_theta = float(np.dot(tangent_end, base_tangent_at_rejoin))

            if evasion_spline is None or cos_theta > final_cos:
                evasion_spline = spline_try
                final_cos = cos_theta
            if cos_theta >= cos_threshold:
                break  # wystarczająco dobry — przerywamy

        if evasion_spline is None:
            logger.warning(
                f"BSplineSmoother: d{context.drone_id} — żaden wariant lead_mult "
                f"nie zbudował poprawnego splinu"
            )
            return None

        # 7) Odrzucenie po przekroczeniu limitu odchyłki tangensu na rejoin.
        if final_cos < cos_threshold:
            angle_deg = float(np.degrees(np.arccos(np.clip(final_cos, -1.0, 1.0))))
            logger.warning(
                f"BSplineSmoother: d{context.drone_id} — plan odrzucony, "
                f"odchyłka tangensu={angle_deg:.1f}° > {self.reject_tangent_angle_deg:.1f}°"
            )
            return None

        return evasion_spline

    @staticmethod
    def _compute_forward_direction(
        current_vel: NDArray[np.float64],
        base_spline: BSplineTrajectory,
        base_arc_progress: float,
    ) -> NDArray[np.float64]:
        speed = float(np.linalg.norm(current_vel))
        if speed > 0.5:
            return current_vel / speed
        if base_spline.arc_length > 1e-6:
            u = float(np.clip(base_arc_progress / base_spline.arc_length, 0.0, 1.0))
        else:
            u = 0.0
        tangent = np.asarray(splev(u, base_spline.tck, der=1), dtype=np.float64)
        tnorm = float(np.linalg.norm(tangent))
        if tnorm > 1e-6:
            return tangent / tnorm
        return np.array([1.0, 0.0, 0.0])

    @staticmethod
    def _base_tangent_at_arc(spline: BSplineTrajectory, arc: float) -> NDArray[np.float64]:
        if spline.arc_length <= 1e-6:
            return np.array([1.0, 0.0, 0.0])
        u = float(np.clip(arc / spline.arc_length, 0.0, 1.0))
        tangent = np.asarray(splev(u, spline.tck, der=1), dtype=np.float64)
        tnorm = float(np.linalg.norm(tangent))
        if tnorm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return tangent / tnorm
