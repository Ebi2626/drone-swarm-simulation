import numpy as np
from typing import Dict, Any, Optional

from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.obstacles.ObstacleShape import ObstacleShape
from src.algorithms.abstraction.trajectory.strategies.shared.bspline_utils import (
    calculate_dynamic_max_node_distance,
    evaluate_bspline_trajectory_sync
)

class VectorizedEvaluator:
    def __init__(
        self,
        obstacles: Optional[ObstaclesData],
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        n_inner_points: int,
        params: dict
    ):
        self.params = params
        self.start_pos = start_pos    # shape: (NDrones, 3)
        self.target_pos = target_pos  # shape: (NDrones, 3)
        self.n_inner_points = n_inner_points
        self.evaluation_number = 0

        # --- 1. KINEMATYKA: Obliczanie dynamicznych limitów dystansu ---
        k_factor = float(self.params.get("k_factor", 2.0))
        abs_min = float(self.params.get("absolute_min_node_dist", 5.0))

        self.max_node_distance = calculate_dynamic_max_node_distance(
            start_pos=self.start_pos,
            target_pos=self.target_pos,
            n_inner_points=self.n_inner_points,
            k_factor=k_factor,
            absolute_min=abs_min
        )

        print(f"[Evaluator] Dynamiczny limit dystansu węzłów: {self.max_node_distance:.2f}m (K={k_factor})")

        # --- 2. INICJALIZACJA PRZESZKÓD ---
        safety_margin = float(self.params.get("obstacle_safety_margin", 0.5))
        self.obstacles_xy = np.zeros((0, 2), dtype=np.float64)
        self.obstacle_radii = np.zeros((0,), dtype=np.float64)

        if obstacles is None:
            print("[Evaluator] Brak przeszkód – pominięto inicjalizację detekcji kolizji.")
            return

        data: np.ndarray = obstacles.data
        shape_type: ObstacleShape = obstacles.shape_type

        if data.ndim != 2 or data.shape[1] < 6:
            raise ValueError(f"ObstaclesData.data musi mieć kształt (N, 6), otrzymano: {data.shape}")

        if data.shape[0] == 0:
            print("[Evaluator] ObstaclesData jest pusta (0 przeszkód).")
            return

        self.obstacles_xy = data[:, :2].astype(np.float64, copy=True)

        if shape_type == ObstacleShape.CYLINDER:
            radii = data[:, 3].astype(np.float64)
            self.obstacle_radii = radii + safety_margin

        elif shape_type == ObstacleShape.BOX:
            half_lx = data[:, 3].astype(np.float64) / 2.0
            half_wy = data[:, 4].astype(np.float64) / 2.0
            circumscribed_radius = np.sqrt(half_lx**2 + half_wy**2)
            self.obstacle_radii = circumscribed_radius + safety_margin
        else:
            raise ValueError(f"Nieznany typ przeszkody: {shape_type}. Oczekiwano CYLINDER lub BOX.")

        print(f"[Evaluator] Załadowano {len(self.obstacles_xy)} przeszkód (typ: {shape_type}).")

    def evaluate(self, control_points: np.ndarray, out: Dict[str, Any]):
        """
        Ewaluacja roju na podstawie węzłów kontrolnych B-Spline.
        Kształt control_points: (PopSize, NDrones, NControl, 3)
        """
        self.evaluation_number += 1
        min_drone_dist = float(self.params.get("min_drone_distance", 2.0))

        # --- 1. EWALUACJA KOLIZJI I DŁUGOŚCI (NUMBA) ---
        # UWAGA: Funkcja evaluate_bspline_trajectory_sync musi zostać zaktualizowana 
        # o usunięcie kodu z_violations i powinna teraz zwracać tylko 3 wartości!
        obs_collisions, lengths, swarm_collisions = evaluate_bspline_trajectory_sync(
            control_points,
            self.obstacles_xy,
            self.obstacle_radii,
            min_drone_dist
        )

        swarm_collision_penalty = np.sum(obs_collisions, axis=1) 
        swarm_total_length = np.sum(lengths, axis=1)          

        # --- 2. EWALUACJA KINEMATYKI (PYTHON) ---
        diff1 = np.diff(control_points, axis=2)
        diff2 = np.diff(diff1, axis=2)
        swarm_smoothness = np.sum(diff2**2, axis=(1, 2, 3))

        dist_violations = np.maximum(0.0, np.linalg.norm(diff1, axis=-1) - self.max_node_distance)
        accel_violations = np.maximum(0.0, np.linalg.norm(diff2, axis=-1) - float(self.params.get("max_accel_limit", 5.0)))
        swarm_kinematic_penalty = np.sum(dist_violations, axis=(1, 2)) + np.sum(accel_violations, axis=(1, 2))

        # --- 3. PRZYPISANIE DO WYJŚCIA NSGA-III ---
        out["F"] = np.column_stack([
            swarm_total_length,
            swarm_smoothness,
            swarm_collisions
        ])

        out["G"] = np.column_stack([
            swarm_collision_penalty,
            swarm_collisions - 0.01,
            swarm_kinematic_penalty
        ])