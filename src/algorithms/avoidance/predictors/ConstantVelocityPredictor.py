from __future__ import annotations

import numpy as np

from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import (
    KinematicState,
    ThreatAlert,
)
from src.algorithms.avoidance.interfaces import IObstaclePredictor


class ConstantVelocityPredictor(IObstaclePredictor):
    """Liniowy model predykcji: `x(t) = x₀ + v₀·t`, prędkość niezmienna.

    Najprostsza możliwa baseline'a. Działa zaskakująco dobrze przy head-on
    encounters w horyzoncie ≤ 2 s, bo cywilne drony i quadrotor-class targets
    mają bezwładność dominującą nad sterowaniem w tej skali czasu (np. Mehdi
    et al. 2017, "Reactive Avoidance for UAVs"). Modele Kalmana / IMM mają
    sens dopiero przy dłuższych horyzontach (>3 s) lub bardziej manewrującej
    celi — Faza 2+.

    Brak parametrów konfiguracyjnych — dlatego pusty `__init__`. Hydra zwraca
    instancję przez `_target_` bez argumentów.
    """

    def predict_state(self, threat: ThreatAlert, t_offset: float) -> KinematicState:
        # Sequential cooperative planning (2026-05-01): jeśli `threat` ma
        # przypisaną `trajectory` (np. evasion spline planowanego wcześniej
        # drona w tym samym ticku), używamy jej dokładnie zamiast liniowej
        # ekstrapolacji. To jest kluczowe dla cooperative avoidance — bez tego
        # drone B (planowany później) traktuje drone A jako "lecący prosto"
        # mimo że A wykonuje już skręt uniku.
        traj = threat.trajectory
        if traj is not None and hasattr(traj, "get_state_at_time"):
            t_local = threat.trajectory_start_offset + float(t_offset)
            duration = float(getattr(traj, "total_duration", 0.0))
            if 0.0 <= t_local <= duration:
                try:
                    pos_arr, vel_arr = traj.get_state_at_time(t_local)
                    return KinematicState(
                        position=np.asarray(pos_arr, dtype=np.float64),
                        velocity=np.asarray(vel_arr, dtype=np.float64),
                        radius=threat.obstacle_state.radius,
                    )
                except Exception:
                    # Fallthrough na linear extrapolation gdy spline coś popsuje.
                    pass
            # t_local poza zakresem splajnu (przed startem lub po końcu) →
            # extrapolation z ostatniego znanego stanu.

        pos = threat.obstacle_state.position
        vel = threat.obstacle_state.velocity
        future_pos = pos + vel * float(t_offset)
        # Prędkość zachowana (constant-velocity assumption); promień stały.
        return KinematicState(
            position=future_pos,
            velocity=vel.copy(),
            radius=threat.obstacle_state.radius,
        )

    def time_to_collision(
        self,
        drone_state: KinematicState,
        threat: ThreatAlert,
    ) -> float:
        """TTC z relatywnej kinematyki w jednej osi closing-speed.

        Definicja zgodna z `ThreatAnalyzer.jit_analyze_hits`:
            closing_speed = ((v_drone − v_obs) · r̂) gdzie r̂ = (p_obs − p_drone)/|·|
            TTC = |p_obs − p_drone| / closing_speed gdy closing > 0, inaczej +inf

        Identyczna definicja jest tu replikowana (a nie importowana z
        ThreatAnalyzer) celowo: planner online'owy może być wołany z
        zewnętrznym `ThreatAlert`, którego TTC zostało wyliczone wcześniej
        — ten predictor zwraca SWÓJ pomiar niezależny, użyteczny gdy
        predictor jest wymieniony na inny model bez zmiany ThreatAnalyzera.
        """
        rel_pos = threat.obstacle_state.position - drone_state.position
        dist = float(np.linalg.norm(rel_pos))
        if dist < 1e-6:
            return 0.0
        rel_vel = drone_state.velocity - threat.obstacle_state.velocity
        closing = float(np.dot(rel_vel, rel_pos)) / dist
        if closing <= 0.1:
            return float("inf")
        return dist / closing
