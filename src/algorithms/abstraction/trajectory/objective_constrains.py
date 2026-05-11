"""Wektorowy ewaluator wielokryterialny dla planowania trajektorii roju.

`VectorizedEvaluator.evaluate` liczy dla całej populacji jednocześnie:
- 5 obiektywów (`F`): f1 trajectory cost, f2 height/angle, f3 threat,
  f4 turn cost, f5 coordination — wszystkie minimalizowane.
- 3 ograniczenia (`G`, konwencja pymoo `G ≤ 0` = wykonalne):
  obstacle collisions, swarm collisions, kinematic penalty
  (długość segmentu + acceleration limit).

Konsumowany przez wszystkie strategie planowania offline (NSGA-III, MSFFOA,
OOA, SSA) oraz `populate_offline_objectives` przy ETL.
"""
import numpy as np
from typing import Dict, Any, Optional

from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.obstacles.ObstacleShape import ObstacleShape
from src.algorithms.abstraction.trajectory.strategies.shared.bspline_utils import (
    calculate_dynamic_max_node_distance,
    compute_max_observed_acceleration,
    evaluate_bspline_trajectory_sync,
)


# Tolerancja feasibility dla swarm-collisions: G[1] = swarm − tol ≤ 0
# ⇒ swarm ≤ tol. Konieczne, bo Numba detekcja sumuje wszystkie penetration
# depths i drobne błędy numeryczne (próbkowanie B-spline) dają non-zero
# nawet dla rozwiązań nieparujących.
SWARM_COLLISION_TOLERANCE: float = 0.01


class VectorizedEvaluator:
    """Liczy 5 obiektywów i 3 ograniczenia dla całej populacji rozwiązań.

    Trzymany pomiędzy generacjami; wystawia `last_objectives` /
    `last_constraints` dla strategii SOO oraz akumulowany licznik NFE
    (`individuals_evaluated`).
    """

    def __init__(
        self,
        obstacles: Optional[ObstaclesData],
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        n_inner_points: int,
        params: dict
    ):
        """Wczytaj geometrię świata i przygotuj cache stałych ewaluacji.

        Args:
            obstacles: Geometria przeszkód statycznych (cylindry/boksy);
                `None` lub puste ⇒ brak przeszkód, f3 i obstacle G zwracają 0.
            start_pos: `(N, 3)` pozycje startowe [m].
            target_pos: `(N, 3)` pozycje docelowe [m].
            n_inner_points: Liczba wewnętrznych węzłów kontrolnych B-spline'a
                (bez endpointów `start`/`target`).
            params: Słownik parametrów algorytmu — odczytywane klucze:
                `k_factor`, `absolute_min_node_dist`,
                `obstacle_safety_margin`, `preferred_height`,
                `min_drone_distance`, `coordination_penalty_factor`,
                `cruise_speed`, `max_accel`.
        """
        self.params = params
        self.start_pos = start_pos    # shape: (NDrones, 3)
        self.target_pos = target_pos  # shape: (NDrones, 3)
        self.n_inner_points = n_inner_points
        # `evaluation_number` = liczba wywołań `evaluate(...)` (calls)
        # `individuals_evaluated` = liczba ewaluowanych osobników (NFE — Number
        #   of Function Evaluations, standard w meta-heuristics literature,
        #   Hansen et al. 2009 BBOB). Monotoniczny i porównywalny cross-algorytm.
        self.evaluation_number = 0
        self.individuals_evaluated = 0

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

        safety_margin = float(self.params.get("obstacle_safety_margin", 0.5))
        self.obstacles_xy = np.zeros((0, 2), dtype=np.float64)
        self.obstacle_radii = np.zeros((0,), dtype=np.float64)

        if obstacles is not None and obstacles.data.shape[0] > 0:
            data = obstacles.data
            self.obstacles_xy = data[:, :2].astype(np.float64, copy=True)
            if obstacles.shape_type == ObstacleShape.CYLINDER:
                self.obstacle_radii = data[:, 3].astype(np.float64) + safety_margin
            elif obstacles.shape_type == ObstacleShape.BOX:
                # BOX → cylinder przybliżenie przez promień opisanego okręgu
                # (półprzekątna). Świadomie konserwatywne: drone w narożnikach
                # BOX-u jest „wewnątrz" promienia opisanego choć poza ścianami,
                # więc f3_threat & obstacle_collisions w urban są lekko
                # zawyżone (większy bufor bezpieczeństwa). Alternatywa
                # point-in-rectangle SDF byłaby ~3× wolniejsza w pętli
                # vectorized — niewarte dla safety_margin ≥ 1.0m, który
                # dominuje konserwatyzm.
                half_lx = data[:, 3].astype(np.float64) / 2.0
                half_wy = data[:, 4].astype(np.float64) / 2.0
                self.obstacle_radii = np.sqrt(half_lx**2 + half_wy**2) + safety_margin

        self.ideal_vectors = self.target_pos[:, :2] - self.start_pos[:, :2]
        self.ideal_lengths = np.linalg.norm(self.ideal_vectors, axis=1)
        # Guard przeciwko `start ≡ target` (zdegenerowana misja) — bez tego
        # `_f1_trajectory_cost` produkowałby NaN/Inf przez `v_norm = vec/0`,
        # co propaguje się do całego F[0] i niszczy ranking populacji.
        self._ideal_lengths_safe = np.maximum(self.ideal_lengths, 1e-9)

    def _f1_trajectory_cost(self, control_points: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """f₁: długość trasy + suma odchyleń XY od idealnej prostej `start → target`.

        Args:
            control_points: `(PopSize, NDrones, NControl, 3)` węzły kontrolne.
            lengths: `(PopSize, NDrones)` długości łuku 3D B-spline'a
                policzone wcześniej w `evaluate_bspline_trajectory_sync`.

        Returns:
            `(PopSize,)` skumulowany koszt po wszystkich dronach roju.
        """
        f_length = np.sum(lengths, axis=1) # (PopSize,)
        
        # Obliczenie dystansu punktów do prostej start-meta
        xy_pts = control_points[..., :2] # (PopSize, NDrones, NControl, 2)
        starts = self.start_pos[:, :2]   # (NDrones, 2)
        
        # Wzór na odległość punktu od prostej kierunkowej
        v_norm = self.ideal_vectors / self._ideal_lengths_safe[:, np.newaxis]
        v_norm = v_norm[np.newaxis, :, np.newaxis, :]
        starts_expanded = starts[np.newaxis, :, np.newaxis, :]
        
        # wektor od startu do aktualnego punktu
        diff = xy_pts - starts_expanded
        
        # rzut prostopadły i dystans
        cross_prod = diff[..., 0]*v_norm[..., 1] - diff[..., 1]*v_norm[..., 0]
        f_shape = np.sum(np.abs(cross_prod), axis=(1, 2))
        
        return f_length + f_shape

    def _f2_height_angle_cost(self, control_points: np.ndarray) -> np.ndarray:
        """f₂: kara za odchylenie od `preferred_height` + sumaryczny kąt wznoszenia.

        Args:
            control_points: `(PopSize, NDrones, NControl, 3)` węzły kontrolne.

        Returns:
            `(PopSize,)` koszt sumaryczny po wszystkich dronach.
        """
        h_pref = float(self.params.get("preferred_height", 15.0))
        z_pts = control_points[..., 2]
        
        # 1. f_height: kara za odstępstwo od H_pref
        f_height = np.sum(np.abs(z_pts - h_pref), axis=(1, 2))
        
        # 2. f_angle: strome wznoszenie
        diffs = np.diff(control_points, axis=2)
        dz = np.abs(diffs[..., 2])
        dxy = np.linalg.norm(diffs[..., :2], axis=-1)
        dxy[dxy == 0] = 1e-9
        f_angle = np.sum(np.arctan(dz / dxy), axis=(1, 2))
        
        return f_height + f_angle

    def _f3_threat_cost(self, control_points: np.ndarray) -> np.ndarray:
        """f₃: penetracja stref bezpieczeństwa wokół przeszkód (XY, dystans euklidesowy).

        Args:
            control_points: `(PopSize, NDrones, NControl, 3)` węzły kontrolne.

        Returns:
            `(PopSize,)` skumulowana suma `max(0, radius − dist)` po
            wszystkich parach (drone, waypoint, obstacle). `0` gdy brak
            przeszkód w środowisku.
        """
        if len(self.obstacles_xy) == 0:
            return np.zeros(control_points.shape[0])
            
        xy_pts = control_points[..., :2] # (PopSize, NDrones, NControl, 2)
        
        # Broadcasting: diff kształt (PopSize, NDrones, NControl, NObstacles, 2)
        diff = xy_pts[:, :, :, np.newaxis, :] - self.obstacles_xy[np.newaxis, np.newaxis, np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        
        # Zagrożenie: jeśli dystans < promień przeszkody, to sumujemy wgłębienie
        radii = self.obstacle_radii[np.newaxis, np.newaxis, np.newaxis, :]
        threats = np.maximum(0, radii - dists)
        return np.sum(threats, axis=(1, 2, 3))

    def _f4_turn_cost(self, control_points: np.ndarray) -> np.ndarray:
        """f₄: suma kwadratów kątów między kolejnymi segmentami w płaszczyźnie XY.

        Args:
            control_points: `(PopSize, NDrones, NControl, 3)` węzły kontrolne.

        Returns:
            `(PopSize,)` Σ θ² po wszystkich załamaniach — penalizuje
            ostre zakręty.
        """
        xy_points = control_points[..., :2]
        vectors = np.diff(xy_points, axis=2)
        
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms[norms == 0] = 1e-9
        unit_vectors = vectors / norms
        
        dot_products = np.sum(unit_vectors[:, :, :-1, :] * unit_vectors[:, :, 1:, :], axis=-1)
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angles = np.arccos(dot_products)
        return np.sum(angles**2, axis=(1, 2))

    def _f5_coordination_cost(self, control_points: np.ndarray, safe_dist: float, penalty: float) -> np.ndarray:
        """f₅: wykładnicza kara za zbliżenie się par dronów na dystans < `safe_dist`.

        Każda para `(i, j)` liczona JEDEN raz dzięki masce górnotrójkątnej —
        bez tego suma po `NDrones × NDrones` zliczyłaby `(i, j)` i `(j, i)`,
        zawyżając karę dwukrotnie.

        Args:
            control_points: `(PopSize, NDrones, NControl, 3)` węzły kontrolne.
            safe_dist: Minimalny bezpieczny dystans między dronami [m].
            penalty: Mnożnik kary (`coordination_penalty_factor` z params).

        Returns:
            `(PopSize,)` skumulowana kara po wszystkich parach i waypointach.
        """
        PopSize, NDrones, NControl, _ = control_points.shape
        diff = control_points[:, :, np.newaxis, :, :] - control_points[:, np.newaxis, :, :, :]
        distances = np.linalg.norm(diff, axis=-1)

        a_ij = (distances < safe_dist).astype(float)
        # Maska górnotrójkątna (i < j) — (i,i) na diagonali = 0 z definicji,
        # (i>j) wyłączone żeby nie liczyć par dwukrotnie.
        upper_mask = np.triu(
            np.ones((NDrones, NDrones), dtype=np.float64), k=1,
        )[np.newaxis, :, :, np.newaxis]
        a_ij = a_ij * upper_mask

        c_ij = penalty * np.exp(safe_dist - distances)
        return np.sum(a_ij * c_ij, axis=(1, 2, 3))

    def evaluate(self, control_points: np.ndarray, out: Dict[str, Any]):
        """Policz `F` (5 obiektywów) i `G` (3 ograniczenia) dla całej populacji.

        Wypełnia `out` zgodnie z konwencją pymoo (`out["F"]`, `out["G"]`)
        i akumuluje `individuals_evaluated` (NFE) — używane przez ETL i
        adaptery SOO.

        Args:
            control_points: `(PopSize, NDrones, NControl, 3)` węzły kontrolne
                pełnego wieloboku kontrolnego (z dołączonymi `start` i `target`).
            out: Słownik wynikowy modyfikowany w miejscu — po wywołaniu
                zawiera klucze `"F"` o kształcie `(PopSize, 5)` (kolumny
                `f1..f5`) i `"G"` `(PopSize, 3)` (`obstacle_collisions`,
                `swarm_collisions − tolerance`, `kinematic_penalty`).
        """
        self.evaluation_number += 1
        self.individuals_evaluated += int(control_points.shape[0])
        min_drone_dist = float(self.params.get("min_drone_distance", 2.0))
        penalty_factor = float(self.params.get("coordination_penalty_factor", 1.0))

        obs_collisions, lengths, swarm_collisions_hard = evaluate_bspline_trajectory_sync(
            control_points,
            self.obstacles_xy,
            self.obstacle_radii,
            min_drone_dist
        )

        f1 = self._f1_trajectory_cost(control_points, lengths)
        f2 = self._f2_height_angle_cost(control_points)
        f3 = self._f3_threat_cost(control_points)
        f4 = self._f4_turn_cost(control_points)
        f5 = self._f5_coordination_cost(control_points, min_drone_dist, penalty_factor)

        # Distance violation: segment B-spline'a nie może być zbyt długi
        # (ogranicza maksymalny rozrzut control points → ogranicza krzywiznę).
        diff1 = np.diff(control_points, axis=2)
        dist_violations = np.maximum(0.0, np.linalg.norm(diff1, axis=-1) - self.max_node_distance)

        # Acceleration violation: twardy constraint na fizyczną acceleration.
        # Naiwna druga różnica `||diff2||` mierzy geometryczną krzywiznę (m),
        # nie fizyczną m/s². Optimizer mógł znajdować B-spline'y z
        # `||diff2|| ≤ limit` ale `|a_lat| = v_cruise²·||diff2||/||diff1||² ≫
        # max_accel` — drone PID nasycał silniki, tracił attitude i spadał.
        # `compute_max_observed_acceleration` zwraca poprawną fizyczną wartość.
        cruise_speed = float(self.params.get("cruise_speed", 6.0))
        max_accel = float(self.params.get("max_accel", 2.0))
        max_obs_acc = compute_max_observed_acceleration(control_points, cruise_speed)
        accel_violations = np.maximum(0.0, max_obs_acc - max_accel)

        kinematic_penalty = (
            np.sum(dist_violations, axis=(1, 2)) + np.sum(accel_violations, axis=1)
        )

        out["F"] = np.column_stack([f1, f2, f3, f4, f5])

        out["G"] = np.column_stack([
            np.sum(obs_collisions, axis=1),
            swarm_collisions_hard - SWARM_COLLISION_TOLERANCE,
            kinematic_penalty
        ])