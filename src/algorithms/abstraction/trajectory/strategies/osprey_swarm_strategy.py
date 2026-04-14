"""
Osprey Optimization Algorithm (OOA) Swarm Strategy.
Jednokryterialna optymalizacja trajektorii roju dronow z uzyciem algorytmu OOA z biblioteki mealpy.

Adaptuje wielokryterialne cele (F) i ograniczenia (G) z VectorizedEvaluator
do postaci jednokryterialnej poprzez sume wazona z karami za naruszenia ograniczen.

Przed wazeniem cele sa normalizowane wzgledem trajektorii referencyjnej (linia prosta),
co eliminuje problem roznych rzedow wielkosci pomiedzy F1, F2 i F3.

Fitness = w1*(F1/F1_ref) + w2*(F2/F2_ref) + w3*(F3/F3_ref) + penalty_weight * sum(G)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray
from hydra.core.hydra_config import HydraConfig

from mealpy import FloatVar
from mealpy import Problem as MealpyProblem
from mealpy.swarm_based.OOA import OriginalOOA

from src.algorithms.abstraction.trajectory.objective_constrains import (
    VectorizedEvaluator,
)
from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.abstraction.generate_world_boundaries import WorldData

if TYPE_CHECKING:
    from src.utils.optimization_history_writer import OptimizationHistoryWriter


# ---------------------------------------------------------------------------
# Polyline resampling (lokalna kopia – nie modyfikujemy istniejacych plikow)
# ---------------------------------------------------------------------------

def _resample_polyline(
    waypoints: NDArray[np.float64],
    num_samples: int = 100,
) -> NDArray[np.float64]:
    """Interpoluje liniowo punkty trasy do zadanej liczby gestych punktow.

    Args:
        waypoints: (Pop, Drones, N_In, 3) rzadkie punkty kontrolne.
        num_samples: N_Out – liczba punktow wyjsciowych.

    Returns:
        (Pop, Drones, N_Out, 3) gesta trajektoria.
    """
    pop_size, n_drones, n_in, dims = waypoints.shape
    flat = waypoints.reshape(pop_size * n_drones, n_in, dims)

    u = np.linspace(0, n_in - 1, num_samples)
    idx = np.clip(np.floor(u).astype(int), 0, n_in - 2)
    alpha = (u - idx).reshape(1, num_samples, 1)

    p0 = flat[:, idx, :]
    p1 = flat[:, idx + 1, :]
    result = p0 + alpha * (p1 - p0)

    return result.reshape(pop_size, n_drones, num_samples, dims)


# ---------------------------------------------------------------------------
# Adapter problemu: mealpy.Problem z normalizacja celow
# ---------------------------------------------------------------------------

class OspreyProblemAdapter(MealpyProblem):
    """Problem mealpy agregujacy wielokryterialna ocene do jednej wartosci fitness.

    Dziedziczy z ``mealpy.Problem`` – mealpy wywoluje ``obj_func(x)`` per-osobnik.
    Biblioteka nie oferuje natywnej ewaluacji batchowej calej populacji;
    w zamian ``osprey_swarm_strategy`` uruchamia solver w trybie ``mode="thread"``
    z konfigurowalna liczba workrow (``n_workers``), co pozwala VectorizedEvaluator
    dzialac rownolegle na wielu watkach (numpy zwalnia GIL).

    Normalizacja celow:
        Przy inicjalizacji obliczana jest trajektoria referencyjna (linia prosta
        Start -> Target). Wartosci F_ref sluza jako mianowniki normalizacji,
        dzieki czemu wagi ``w1, w2, w3`` operuja na bezwymiarowej skali ~1.0
        i sa porownywalne niezaleznie od rozmiaru swiata.
    """

    def __init__(
        self,
        bounds: FloatVar,
        evaluator: VectorizedEvaluator,
        start_pos: NDArray[np.float64],
        target_pos: NDArray[np.float64],
        n_drones: int,
        n_inner: int,
        n_output_samples: int,
        weights: Dict[str, float],
        penalty_weight: float,
        history_writer: OptimizationHistoryWriter | None = None,
        expected_pop_size: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(bounds=bounds, minmax="min", **kwargs)

        self.evaluator = evaluator
        self.n_drones = n_drones
        self.n_inner = n_inner
        self.n_output_samples = n_output_samples

        self.w1 = weights.get("w_path_length", 1.0)
        self.w2 = weights.get("w_collision_risk", 5.0)
        self.w3 = weights.get("w_elevation", 0.5)
        self.penalty_weight = penalty_weight

        # Logowanie historii optymalizacji
        self._history_writer = history_writer
        self._expected_pop_size = expected_pop_size
        self._gen_buffer_x: list[np.ndarray] = []
        self._gen_buffer_f: list[np.ndarray] = []

        # Pre-compute broadcast shapes (1, Drones, 1, 3)
        self._starts_bc = start_pos[np.newaxis, :, np.newaxis, :]
        self._targets_bc = target_pos[np.newaxis, :, np.newaxis, :]

        # --- Normalizacja: F_ref z trajektorii prostoliniowej ---
        self._f_scale = self._compute_reference_scales()

    # ------------------------------------------------------------------

    def _compute_reference_scales(self) -> NDArray[np.float64]:
        """Oblicza wartosci F dla trajektorii prostoliniowej (Start->Target).

        Uzywane jako mianowniki normalizacji.  Dla F2 (collision risk) wartosc
        referencyjna bywa zerowa (brak kolizji na prostej), wiec stosujemy
        dolny klamp ``epsilon`` rowny 1.0 – zapobiega dzieleniu przez zero
        i jednoczesnie utrzymuje F2 w 'surowej' skali, gdy nie ma referencji.
        """
        t_vals = np.linspace(0, 1, self.n_inner + 2)[1:-1]
        t = t_vals.reshape(1, 1, self.n_inner, 1)
        inner_ref = self._starts_bc + t * (self._targets_bc - self._starts_bc)
        sparse_ref = np.concatenate(
            [self._starts_bc, inner_ref, self._targets_bc], axis=2
        )
        traj_ref = _resample_polyline(sparse_ref, self.n_output_samples)

        out_ref: Dict[str, Any] = {}
        self.evaluator.evaluate(traj_ref, out_ref)
        f_ref = out_ref["F"][0]  # (3,)

        # Klamp: F1 i F3 powinny byc > 0 dla nietrywialnej trasy,
        # F2 moze byc 0 (brak kolizji) – uzywamy 1.0 jako skali bazowej.
        return np.maximum(f_ref, 1.0)

    # ------------------------------------------------------------------

    def obj_func(self, x: np.ndarray) -> float:
        """Oblicza znormalizowany fitness dla pojedynczego wektora decyzyjnego.

        Kroki:
            1. Dekodowanie x -> (1, Drones, Inner, 3)
            2. Budowa sparse polyline: Start -> Inner -> Target
            3. Resampling do gestej trajektorii
            4. Ewaluacja przez VectorizedEvaluator -> F(1,3), G(1,5)
            5. Normalizacja F wzgledem F_ref
            6. Agregacja: weighted sum + constraint penalty
        """
        inner = x.reshape(1, self.n_drones, self.n_inner, 3)
        sparse = np.concatenate([self._starts_bc, inner, self._targets_bc], axis=2)
        trajectories = _resample_polyline(sparse, self.n_output_samples)

        out: Dict[str, Any] = {}
        self.evaluator.evaluate(trajectories, out)

        F = out["F"][0]  # (3,)
        G = out["G"][0]  # (5,)

        # Normalizacja celow wzgledem skali referencyjnej
        F_norm = F / self._f_scale

        obj_value = self.w1 * F_norm[0] + self.w2 * F_norm[1] + self.w3 * F_norm[2]
        constraint_penalty = self.penalty_weight * float(np.sum(np.maximum(0.0, G)))

        if self._history_writer is not None and self._expected_pop_size > 0:
            self._gen_buffer_x.append(x.copy())
            self._gen_buffer_f.append(F.copy())
            if len(self._gen_buffer_x) >= self._expected_pop_size:
                self._history_writer.put_generation_data({
                    "objectives_matrix": np.stack(self._gen_buffer_f),
                    "decisions_matrix": np.stack(self._gen_buffer_x),
                })
                self._gen_buffer_x.clear()
                self._gen_buffer_f.clear()

        return float(obj_value + constraint_penalty)

    # ------------------------------------------------------------------

    def evaluate_batch(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        """Ewaluacja batchowa calej populacji przez VectorizedEvaluator.

        Metoda wykorzystuje wektoryzowany charakter evaluatora, przetwarzajac
        cala macierz populacji (pop_size, n_var) w jednym ujeciu – bez petli.

        Uwaga: mealpy nie wywoluje tej metody automatycznie; jest ona dostepna
        do uzycia w niestandardowych petlach optymalizacyjnych lub benchmarkach.

        Args:
            population: (pop_size, n_var) macierz zmiennych decyzyjnych.

        Returns:
            (pop_size,) wektor wartosci fitness.
        """
        pop_size = population.shape[0]
        inner = population.reshape(pop_size, self.n_drones, self.n_inner, 3)

        starts = np.broadcast_to(self._starts_bc, (pop_size, self.n_drones, 1, 3)).copy()
        targets = np.broadcast_to(self._targets_bc, (pop_size, self.n_drones, 1, 3)).copy()

        sparse = np.concatenate([starts, inner, targets], axis=2)
        trajectories = _resample_polyline(sparse, self.n_output_samples)

        out: Dict[str, Any] = {}
        self.evaluator.evaluate(trajectories, out)

        F = out["F"]  # (pop_size, 3)
        G = out["G"]  # (pop_size, 5)

        F_norm = F / self._f_scale[np.newaxis, :]
        obj_values = self.w1 * F_norm[:, 0] + self.w2 * F_norm[:, 1] + self.w3 * F_norm[:, 2]
        penalties = self.penalty_weight * np.max(np.maximum(0.0, G), axis=1) * 10.0

        return obj_values + penalties


# ---------------------------------------------------------------------------
# Heurystyczna inicjalizacja populacji
# ---------------------------------------------------------------------------

def _generate_starting_solutions(
    start_pos: NDArray[np.float64],
    target_pos: NDArray[np.float64],
    n_drones: int,
    n_inner: int,
    pop_size: int,
    world_data: WorldData,
    obstacles_data: Optional[ObstaclesData],
    xl: np.ndarray,
    xu: np.ndarray,
) -> np.ndarray:
    """Generuje populacje poczatkowa wokol linii prostej Start->Target."""

    t_vals = np.linspace(0, 1, n_inner + 2)[1:-1]
    t = t_vals.reshape(1, 1, n_inner, 1)
    s = start_pos[np.newaxis, :, np.newaxis, :]
    e = target_pos[np.newaxis, :, np.newaxis, :]

    base_points = s + t * (e - s)  # (1, Drones, Inner, 3)
    X = np.tile(base_points, (pop_size, 1, 1, 1))

    # Szum XY zalezny od rozmiarow przeszkod
    if (
        obstacles_data is not None
        and obstacles_data.data is not None
        and len(obstacles_data.data) > 0
    ):

        world_size_x = float(world_data.max_bounds[0] - world_data.min_bounds[0])
        world_size_y = float(world_data.max_bounds[1] - world_data.min_bounds[1])
        
        # Pozwalamy im losowo odchylić się nawet o 15% wymiarów świata na starcie
        noise_scale_x = world_size_x * 0.15
        noise_scale_y = world_size_y * 0.05 # Na osi Y odchylamy mniej, bo to oś postępu
        
        noise_xy = np.random.normal(0, 1.0, (pop_size, n_drones, n_inner, 2))
        noise_xy[..., 0] *= noise_scale_x
        noise_xy[..., 1] *= noise_scale_y
        
        X[..., :2] += noise_xy

    # Losowa wysokosc w bezpiecznym korytarzu
    min_safe_z = 0.5
    max_flight_z = float(world_data.max_bounds[2])
    X[..., 2] = np.random.uniform(min_safe_z, max_flight_z, (pop_size, n_drones, n_inner))

    X_flat = X.reshape(pop_size, -1)
    return np.clip(X_flat, xl, xu)


# ---------------------------------------------------------------------------
# Glowna funkcja strategii (implementuje TrajectoryStrategyProtocol)
# ---------------------------------------------------------------------------

def osprey_swarm_strategy(
    *,
    start_positions: NDArray[np.float64],
    target_positions: NDArray[np.float64],
    obstacles_data: Union[Any, List[Any]],
    world_data: WorldData,
    number_of_waypoints: int,
    drone_swarm_size: int,
    algorithm_params: Optional[Dict[str, Any]] = None,
) -> NDArray[np.float64]:
    """Strategia generowania trajektorii roju dronow za pomoca Osprey Optimization Algorithm.

    Uzywana jako zamiennik (drop-in replacement) dla nsga3_swarm_strategy.
    Roznice architektoniczne:
        - OOA jest algorytmem jednokryterialnym (SOO), wiec cele i ograniczenia
          sa agregowane do pojedynczej wartosci fitness z normalizacja wzgledem
          trajektorii referencyjnej.
        - Implementacja bazuje na bibliotece mealpy (OriginalOOA).
        - mealpy wymusza ewaluacje per-osobnik (obj_func). Parametry ``mode``
          i ``n_workers`` sa przekazywane do solvera; OriginalOOA w mealpy 3.x
          wymusza tryb "single", ale inne algorytmy mealpy (np. PSO)
          moga korzystac z wielowatkowosci przez ten sam interfejs.

    Args:
        start_positions: (N, 3) pozycje startowe dronow.
        target_positions: (N, 3) pozycje docelowe dronow.
        obstacles_data: przeszkody w srodowisku (pojedynczy obiekt lub lista).
        world_data: granice swiata.
        number_of_waypoints: liczba punktow gestej trajektorii wyjsciowej.
        drone_swarm_size: liczba dronow w roju.
        algorithm_params: parametry konfiguracyjne algorytmu OOA.

    Returns:
        NDArray o ksztalcie (N_drones, N_waypoints, 3).
    """
    params = algorithm_params or {}

    # --- Parametry OOA ---
    pop_size: int = params.get("pop_size", 100)
    n_epochs: int = params.get("n_gen", 500)
    n_inner: int = params.get("n_inner_waypoints", max(5, int(number_of_waypoints * 0.1)))

    weights = {
        "w_path_length": params.get("w_path_length", 1.0),
        "w_collision_risk": params.get("w_collision_risk", 5.0),
        "w_elevation": params.get("w_elevation", 0.5),
    }
    penalty_weight: float = params.get("penalty_weight", 100.0)

    # --- Multithreading ---
    n_workers: int = params.get("n_workers", 4)
    mode: str = "thread" if n_workers > 1 else "swarm"

    # --- Normalizacja listy przeszkod ---
    obs_list: List[Any] = obstacles_data if isinstance(obstacles_data, list) else [obstacles_data]

    # --- Granice zmiennych decyzyjnych ---
    margin = 50.0
    MIN_Z_ALTITUDE = 0.5

    xl_point = np.array(world_data.min_bounds, dtype=float) - margin
    xu_point = np.array(world_data.max_bounds, dtype=float) + margin
    xl_point[2] = max(MIN_Z_ALTITUDE, float(world_data.min_bounds[2]))

    n_var = drone_swarm_size * n_inner * 3
    xl = np.tile(xl_point, drone_swarm_size * n_inner)
    xu = np.tile(xu_point, drone_swarm_size * n_inner)

    from src.utils.optimization_history_writer import OptimizationHistoryWriter

    hydra_output_dir = HydraConfig.get().runtime.output_dir
    writer = OptimizationHistoryWriter(
        output_dir=os.path.join(hydra_output_dir, "optimization_history")
    )

    print(
        f"[OOA] Start. Pop: {pop_size}, Epochs: {n_epochs}, "
        f"Inner Pts: {n_inner}, Vars: {n_var}, Mode: {mode}, Workers: {n_workers}"
    )

    # --- Evaluator ---
    evaluator = VectorizedEvaluator(
        obstacles=obs_list,
        start_pos=start_positions,
        target_pos=target_positions,
        params=params,
    )

    # --- Problem (mealpy.Problem subclass) ---
    problem = OspreyProblemAdapter(
        bounds=FloatVar(lb=xl, ub=xu),
        evaluator=evaluator,
        start_pos=start_positions,
        target_pos=target_positions,
        n_drones=drone_swarm_size,
        n_inner=n_inner,
        n_output_samples=number_of_waypoints,
        weights=weights,
        penalty_weight=penalty_weight,
        history_writer=writer,
        expected_pop_size=pop_size,
        log_to="console",
    )

    # --- Populacja poczatkowa (heurystyczna) ---
    first_obs = None
    if isinstance(obstacles_data, list) and len(obstacles_data) > 0:
        first_obs = obstacles_data[0]
    elif not isinstance(obstacles_data, list):
        first_obs = obstacles_data

    starting_solutions = _generate_starting_solutions(
        start_pos=start_positions,
        target_pos=target_positions,
        n_drones=drone_swarm_size,
        n_inner=n_inner,
        pop_size=pop_size,
        world_data=world_data,
        obstacles_data=first_obs,
        xl=xl,
        xu=xu,
    )

    # --- Uruchomienie OOA ---
    model = OriginalOOA(epoch=n_epochs, pop_size=pop_size)

    try:
        best_agent = model.solve(
            problem=problem,
            mode=mode,
            n_workers=n_workers,
            starting_solutions=starting_solutions,
            seed=params.get("seed", 1),
        )

        best_x = best_agent.solution
        best_fitness = best_agent.target.fitness
        print(f"[OOA] Zakonczono. Najlepszy fitness: {best_fitness:.4f}")
        print(f"[OOA] Skale normalizacji F_ref: {problem._f_scale}")

        # --- Rekonstrukcja trajektorii ---
        inner = best_x.reshape(1, drone_swarm_size, n_inner, 3)
        s = start_positions[np.newaxis, :, np.newaxis, :]
        t = target_positions[np.newaxis, :, np.newaxis, :]
        sparse = np.concatenate([s, inner, t], axis=2)
        final_traj = _resample_polyline(sparse, num_samples=number_of_waypoints)

        return final_traj[0]  # (N_drones, N_waypoints, 3)

    except Exception as e:
        print(f"[OOA] Blad optymalizacji: {e}. Zwracam linie prosta.")
    finally:
        writer.close()

    # --- Fallback: linia prosta ---
    print("[OOA] Fallback: generowanie linii prostej.")
    t_line = np.linspace(0, 1, number_of_waypoints)
    out = np.empty((drone_swarm_size, number_of_waypoints, 3))
    min_safe_alt = params.get("min_safe_altitude", 1.0)

    for d in range(drone_swarm_size):
        for axis in range(2):
            out[d, :, axis] = np.interp(
                t_line, [0, 1], [start_positions[d, axis], target_positions[d, axis]]
            )
        z_start = max(float(start_positions[d, 2]), min_safe_alt)
        z_target = max(float(target_positions[d, 2]), min_safe_alt)
        out[d, :, 2] = np.interp(t_line, [0, 1], [z_start, z_target])

    return out
