import math
import os
import random
import numpy as np
from typing import Any, Dict, Optional, List, Union
from numpy.typing import NDArray
from contextlib import nullcontext

# Zakładam obecność funkcji resample_polyline_batch z Twojego pliku referencyjnego
from src.algorithms.abstraction.trajectory.objective_constrains import VectorizedEvaluator
from src.algorithms.abstraction.trajectory.strategies.nsga3_swarm_strategy import resample_polyline_batch
from src.algorithms.abstraction.trajectory.strategies.timing_utils import TimingCollector
from src.utils.optimization_history_writer import OptimizationHistoryWriter
from hydra.core.hydra_config import HydraConfig


def evaluate_ssa_population(
    X_flat: NDArray, 
    n_drones: int, 
    n_inner: int, 
    n_out: int,
    starts: NDArray, 
    targets: NDArray, 
    evaluator: Any,
    penalty_weight: float,
    weights: NDArray
) -> tuple[NDArray, NDArray]:
    
    pop_size = X_flat.shape[0]
    
    inner_waypoints = X_flat.reshape(pop_size, n_drones, n_inner, 3)
    starts_bc = np.tile(starts[None, :, None, :], (pop_size, 1, 1, 1))
    targets_bc = np.tile(targets[None, :, None, :], (pop_size, 1, 1, 1))
    sparse_trajectory = np.concatenate([starts_bc, inner_waypoints, targets_bc], axis=2)
    
    trajectories = resample_polyline_batch(sparse_trajectory, num_samples=n_out)
    
    out = {}
    evaluator.evaluate(trajectories, out)
    
    F = out.get("F", np.zeros((pop_size, 3)))  
    G = out.get("G", np.zeros((pop_size, 1)))  
    
    # ---------------------------------------------------------
    # PEŁNA WEKTORYZACJA: Całkowicie usunięta pętla for!
    # ---------------------------------------------------------
    # 1. Iloczyn macierzowy celów i wag: (Pop, 3) x (3,) -> (Pop,)
    obj_score = np.dot(F, weights) 
    
    # 2. Agregacja wektorowa kar dla każdego osobnika z osobna:
    constraint_violation = np.sum(np.maximum(0, G), axis=1) 
    
    # 3. Wektor sumarycznego przystosowania (fitness)
    fitness = obj_score + penalty_weight * constraint_violation
        
    # Zwracamy wektor fitness i pełną macierz celów do logowania
    return fitness, F


def sparrow_search_algorithm_classic(
    X_init: NDArray,
    bounds_min: NDArray,
    bounds_max: NDArray,
    iter_max: int,
    n_drones: int,
    n_inner: int,
    n_out: int,
    starts: NDArray,
    targets: NDArray,
    evaluator: Any,
    st: float,
    pd_ratio: float,
    sd_ratio: float,
    penalty_weight: float,
    weights: NDArray,
    _measure: Any = None
) -> NDArray:
    """
    Klasyczna implementacja modelu matematycznego SSA.
    Parametry hiperheurystyki są teraz wstrzykiwane jawnie przez argumenty.
    
    Zintegrowano inicjalizację mechanizmu logowania wewnątrz metody wraz
    z bezpiecznym zamykaniem strumieni dyskowych.
    """
    if _measure is None:
        _measure = lambda *a, **kw: nullcontext()

    # 1. Lokalna inicjalizacja asynchronicznego loggera na podstawie konfiguracji Hydra
    try:
        from src.utils.optimization_history_writer import OptimizationHistoryWriter
        output_dir = HydraConfig.get().runtime.output_dir
        log_dir = os.path.join(output_dir, "optimization_history")
        writer = OptimizationHistoryWriter(output_dir=log_dir)
    except Exception as e:
        print(f"[SSA Classic] Ostrzeżenie: Nie udało się zainicjalizować loggera. Brak zapisu historii. Błąd: {e}")
        writer = None

    try:
        n, d = X_init.shape
        X = np.copy(X_init)
        
        # Wyliczanie stałych rozmiarów podgrup stada na podstawie zadanych proporcji
        PD = max(1, int(pd_ratio * n))
        SD = max(1, int(sd_ratio * n))
        
        global_best_X = np.copy(X[0])
        global_best_f = float('inf')

        print(f"[SSA Classic] Rozpoczęto optymalizację: Populacja={n}, Wymiary={d}, Iter={iter_max}, PD={PD}, SD={SD}")
        
        # Wstępna ewaluacja populacji bazowej - powrót macierzy F_pop niezbędnej do logowania
        fitness, F_pop = evaluate_ssa_population(
            X, n_drones, n_inner, n_out, starts, targets, evaluator, penalty_weight, weights
        )

        with _measure("generation_loop"):
            for t in range(iter_max):
                sort_idx = np.argsort(fitness)
                X = X[sort_idx]
                fitness = fitness[sort_idx]
                F_pop = F_pop[sort_idx]  # Synchronizacja macierzy celów
                
                if fitness[0] < global_best_f:
                    global_best_f = fitness[0]
                    global_best_X = np.copy(X[0])
                    
                X_best = np.copy(X[0])
                X_worst = np.copy(X[-1])
                f_g = fitness[0]
                f_w = fitness[-1]
                
                X_new = np.copy(X)
                R_2 = random.random()
                
                # ETAP 1: Producenci
                for i in range(PD):
                    for j in range(d):
                        if R_2 < st:
                            alpha = random.uniform(0.0001, 1.0)
                            X_new[i, j] = X[i, j] * math.exp(-(i + 1) / (alpha * iter_max))
                        else:
                            Q = random.gauss(0, 1)
                            X_new[i, j] = X[i, j] + Q
                            
                X_P = np.copy(X_new[0])

                # ETAP 2: Wyzyskiwacze
                for i in range(PD, n):
                    for j in range(d):
                        if i > n / 2:
                            Q = random.gauss(0, 1)
                            X_new[i, j] = Q * math.exp((X_worst[j] - X[i, j]) / ((i + 1) ** 2))
                        else:
                            A_j = 1 if random.random() > 0.5 else -1
                            A_plus_j = A_j / d 
                            X_new[i, j] = X_P[j] + abs(X[i, j] - X_P[j]) * A_plus_j

                # ETAP 3: Zwiadowcy
                danger_indices = random.sample(range(n), SD)
                for i in danger_indices:
                    for j in range(d):
                        if fitness[i] > f_g:
                            beta = random.gauss(0, 1)
                            X_new[i, j] = X_best[j] + beta * abs(X[i, j] - X_best[j])
                        else:
                            K_val = random.uniform(-1, 1)
                            epsilon = 1e-8
                            step_size = abs(X[i, j] - X_worst[j]) / (fitness[i] - f_w + epsilon)
                            X_new[i, j] = X[i, j] + K_val * step_size

                # Ograniczenia przestrzeni
                for i in range(n):
                    for j in range(d):
                        if X_new[i, j] < bounds_min[j]:
                            X_new[i, j] = bounds_min[j]
                        elif X_new[i, j] > bounds_max[j]:
                            X_new[i, j] = bounds_max[j]
                            
                # Ewaluacja nowo wygenerowanych pozycji (X_new)
                fitness_new, F_new = evaluate_ssa_population(
                    X_new, n_drones, n_inner, n_out, starts, targets, evaluator, penalty_weight, weights
                )
                
                # Krok selekcji: Aktualizacja tylko lepszych rozwiązań (Greedy Acceptance)
                for i in range(n):
                    if fitness_new[i] < fitness[i]:
                        X[i] = np.copy(X_new[i])
                        fitness[i] = fitness_new[i]
                        F_pop[i] = np.copy(F_new[i])  # Indywidualna aktualizacja profilu celów
                        
                        if fitness[i] < global_best_f:
                            global_best_f = fitness[i]
                            global_best_X = np.copy(X[i])

                # 2. Rejestracja stanu na końcu iteracji - bez blokowania głównego wątku
                if writer is not None:
                    writer.put_generation_data({
                        "objectives_matrix": F_pop.copy(),
                        "decisions_matrix": X.copy()
                    })
                
                if (t + 1) % 100 == 0 or t == 0:
                    print(f"Iteracja {t+1}/{iter_max}, Najlepszy koszt (Fitness): {global_best_f:.4f}")

        return global_best_X

    finally:
        # 3. Zabezpieczenie na wypadek przerwanej symulacji (np. KeyboardInterrupt)
        # Nakazuje asynchronicznemu daemonowi opróżnienie buforów IO
        if 'writer' in locals() and writer is not None:
            writer.close()


def sparrow_search_algorithm_vectorized(
    X_init: NDArray,
    bounds_min: NDArray,
    bounds_max: NDArray,
    iter_max: int,
    n_drones: int,
    n_inner: int,
    n_out: int,
    starts: NDArray,
    targets: NDArray,
    evaluator: Any,
    st: float,
    pd_ratio: float,
    sd_ratio: float,
    penalty_weight: float,
    weights: NDArray,
    _measure: Any = None
) -> NDArray:
    """
    Zwektoryzowana implementacja modelu matematycznego SSA.
    Wykorzystuje operacje macierzowe biblioteki NumPy w celu eliminacji pętli `for` 
    i przyspieszenia symulacji dla wielowymiarowych wielowirnikowców.
    Zintegrowano wektoryzowane logowanie populacji.
    """
    if _measure is None:
        _measure = lambda *a, **kw: nullcontext()

    n, d = X_init.shape
    X = np.copy(X_init)
    
    # Wyliczanie stałych rozmiarów podgrup stada
    PD = max(1, int(pd_ratio * n))
    SD = max(1, int(sd_ratio * n))
    
    global_best_X = np.copy(X[0])
    global_best_f = float('inf')

    print(f"[SSA Vectorized] Rozpoczęto optymalizację macierzową: Pop={n}, Wym={d}, Iter={iter_max}")

    # Wstępna ewaluacja macierzowa całej populacji bazowej
    # ZMIANA: Zwracana jest również wektoryzowana macierz celów (F_pop) dla logera
    fitness, F_pop = evaluate_ssa_population(
        X, n_drones, n_inner, n_out, starts, targets, evaluator, penalty_weight, weights
    )

    output_dir = HydraConfig.get().runtime.output_dir
    writer = OptimizationHistoryWriter(output_dir=os.path.join(output_dir, "optimization_history"))

    # Pre-kalkulacja do odwracania macierzy uprzywilejowanych Wyzyskiwaczy
    # Pozwala na szybkie wyznaczanie A_plus w głównej pętli bez liczenia tego w każdej iteracji
    with _measure("generation_loop"):
        for t in range(iter_max):
            # 1. Sortowanie macierzy (po wierszach)
            sort_idx = np.argsort(fitness)
            X = X[sort_idx]
            fitness = fitness[sort_idx]
            F_pop = F_pop[sort_idx]  # <--- ZMIANA: Sortowanie macierzy celów razem z populacją
            
            # Aktualizacja globalnego optimum
            if fitness[0] < global_best_f:
                global_best_f = fitness[0]
                global_best_X = np.copy(X[0])
                
            X_best = np.copy(X[0])
            X_worst = np.copy(X[-1])
            f_g = fitness[0]
            f_w = fitness[-1]
            
            X_new = np.copy(X)
            R_2 = np.random.rand()
            
            # -------------------------------------------------------------
            # ETAP 1: Producenci (od indeksu 0 do PD)
            # -------------------------------------------------------------
            if R_2 < st:
                # Wektoryzowane zwężanie korytarza przeszukiwań (exploration mode)
                # Wymiar alphas: (PD, 1), broadcasting przemnoży odpowiednio wszystkie wymiary 'd'
                Q_explore = np.random.randn(PD, d)
                X_new[:PD] = X[:PD] + Q_explore * np.abs(X[:PD] - global_best_X)        
            else:
                # Wektoryzowana ucieczka przed drapieżnikiem (exploitation mode)
                Q_mat = np.random.randn(PD, d)
                X_new[:PD] = X[:PD] + Q_mat
                
            X_P = np.copy(X_new[0])

            # -------------------------------------------------------------
            # ETAP 2: Wyzyskiwacze (od indeksu PD do n)
            # -------------------------------------------------------------
            # Część pierwsza: uprzywilejowani (od PD do mid_index)
            mid_index = int(n / 2) + 1  

            if mid_index > PD:
                num_privileged = mid_index - PD
                
                # Wersja macierzowa wyliczenia A_plus dla wszystkich uprzywilejowanych naraz
                A_mat = np.random.choice([-1, 1], size=(num_privileged, d))
                # Przybliżenie wektorowe zgodnie z założeniem modelu akademickiego: A^+_j = A_j / d
                A_plus_mat = A_mat / d 
                
                X_new[PD:mid_index] = X_P + np.abs(X[PD:mid_index] - X_P) * A_plus_mat

            # Część druga: głodujący (od mid_index do n)
            num_starving = n - mid_index
            if num_starving > 0:
                Q_starving = np.random.randn(num_starving, d)
                row_idx_starving = np.arange(mid_index + 1, n + 1).reshape(num_starving, 1)
                # Potęgowanie wektorowe dla równania wykładniczego
                X_new[mid_index:] = Q_starving * np.exp((X_worst - X[mid_index:]) / (row_idx_starving ** 2))

            # -------------------------------------------------------------
            # ETAP 3: Zwiadowcy (grupa narażona na ataki drapieżników)
            # -------------------------------------------------------------
            # Wybór unikalnych indeksów symulujących losowe rozłożenie drapieżników
            danger_indices = np.random.choice(n, SD, replace=False)
            
            f_danger = fitness[danger_indices]
            better_mask = f_danger > f_g
            worse_mask = ~better_mask
            
            idx_better = danger_indices[better_mask]
            idx_worse = danger_indices[worse_mask]
            
            if len(idx_better) > 0:
                # Wróble uciekające ze skraju stada bezpośrednio w bezpieczne centrum (X_best)
                beta_mat = np.random.randn(len(idx_better), d)
                X_new[idx_better] = X_best + beta_mat * np.abs(X[idx_better] - X_best)
                
            if len(idx_worse) > 0:
                # Wróble w centrum rozpraszające się nawzajem losowo (unikanie kolizji)
                K_mat = np.random.uniform(-1, 1, size=(len(idx_worse), d))
                epsilon = 1e-8
                
                # reshape(-1, 1) dla bezpiecznego broadcastingu przy dzieleniu (Wymiar: Nx1)
                f_diff = (fitness[idx_worse] - f_w + epsilon).reshape(-1, 1)
                step_sizes = np.abs(X[idx_worse] - X_worst) / f_diff
                
                X_new[idx_worse] = X[idx_worse] + K_mat * step_sizes

            # -------------------------------------------------------------
            # Ograniczenia i Greedy Acceptance (Krok Selekcji)
            # -------------------------------------------------------------
            # Wektoryzowane przycinanie zmiennych przestrzennych do granic mapy
            X_new = np.clip(X_new, bounds_min, bounds_max)
            
            # Macierzowa ewaluacja nowo wygenerowanych propozycji lotu
            # ZMIANA: Zwraca również macierz celów nowej populacji
            fitness_new, F_new = evaluate_ssa_population(
                X_new, n_drones, n_inner, n_out, starts, targets, evaluator, penalty_weight, weights
            )
            
            # Wektoryzowany Greedy Acceptance: przyjmujemy genotyp drona tylko wtedy gdy f_new < f_old
            # Tworzenie wektora logicznego [True/False] o rozmiarze (n,)
            improved_mask = fitness_new < fitness
            
            # Zaktualizowanie tylko tych wierszy (osobników), dla których maska to True
            X[improved_mask] = X_new[improved_mask]
            fitness[improved_mask] = fitness_new[improved_mask]
            F_pop[improved_mask] = F_new[improved_mask]  # <--- ZMIANA: Aktualizacja macierzy celów przez szybką maskę wbudowaną w NumPy
            
            # --- ZMIANA: Asynchroniczne logowanie stanu generacji do pliku HDF5/NPZ ---
            if writer is not None:
                writer.put_generation_data({
                    "objectives_matrix": F_pop.copy(),
                    "decisions_matrix": X.copy()
                })
            # --------------------------------------------------------------------------

            # Macierzowa aktualizacja globalnego optimum po przypisaniu
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < global_best_f:
                global_best_f = fitness[current_best_idx]
                global_best_X = np.copy(X[current_best_idx])
                
            if (t + 1) % 100 == 0 or t == 0:
                print(f"Iteracja {t+1}/{iter_max}, Najlepszy koszt (Fitness): {global_best_f:.4f}")

    return global_best_X


# --- Główny Orchestrator (Punkt Wejścia dla Hydry) ---

def ssa_swarm_strategy(
    *, 
    start_positions: NDArray[np.float64],
    target_positions: NDArray[np.float64],
    obstacles_data: Union[Any, List[Any]], 
    world_data: Any,
    number_of_waypoints: int, 
    drone_swarm_size: int, 
    algorithm_params: Optional[Dict[str, Any]] = None,
    timing: Optional["TimingCollector"] = None
) -> NDArray[np.float64]:
    
    local_timing = False
    if timing is None:
        try:
            from src.algorithms.abstraction.trajectory.strategies.timing_utils import TimingCollector
            timing = TimingCollector("SSA_Swarm")
            local_timing = True
        except ImportError:
            timing = None

    _measure = timing.measure if timing is not None else lambda *a, **kw: nullcontext()

    try:
        with _measure("total_optimization"):
            with _measure("initialization"):
                params = algorithm_params or {}
                
                # 1. Rozpakowanie ogólnych parametrów optymalizacji
                pop_size = params.get("pop_size", 100)
                n_gen = params.get("n_gen", 100)
                n_inner = params.get("n_inner_waypoints", max(5, int(number_of_waypoints * 0.1)))
                
                # 2. Rozpakowanie hiperparametrów biologicznych modelu SSA
                st = params.get("st", 0.8)
                pd_ratio = params.get("pd_ratio", 0.2)
                sd_ratio = params.get("sd_ratio", 0.15)
                
                # 3. Rozpakowanie wag do skalaryzacji (Penalty + cele: dystans, gładkość, równomierność)
                penalty_weight = params.get("penalty_weight", 10000.0)
                weights = np.array([
                    params.get("weight_distance", 1.0),
                    params.get("weight_smoothness", 1.0),
                    params.get("weight_uniformity", 1.0)
                ], dtype=np.float64)
                
                if isinstance(obstacles_data, list):
                    obs_list = obstacles_data
                else:
                    obs_list = [obstacles_data]
                    
                print(f"[SSA Strategy] Inicjalizacja. Rój: {drone_swarm_size}, Inner: {n_inner}, ST={st}, Kary={penalty_weight}")

                margin = 50.0 
                xl_one = np.array(world_data.min_bounds, dtype=float) - margin
                xu_one = np.array(world_data.max_bounds, dtype=float) + margin
                
                MIN_Z_ALTITUDE = 0.5 
                xl_one[2] = max(MIN_Z_ALTITUDE, world_data.min_bounds[2])
                max_flight_z = min(world_data.max_bounds[2], xu_one[2] - 3.0)
                xu_one[2] = max_flight_z
                
                d = drone_swarm_size * n_inner * 3
                bounds_min = np.tile(xl_one, drone_swarm_size * n_inner)
                bounds_max = np.tile(xu_one, drone_swarm_size * n_inner)
                
                # Heurystyczna inicjalizacja populacji wzdłuż wektora Start -> Cel
                X_init = np.zeros((pop_size, drone_swarm_size, n_inner, 3))
                t_vals = np.linspace(0, 1, n_inner + 2)[1:-1]
                t_vals = t_vals.reshape(1, 1, n_inner, 1)
                
                s = start_positions[None, :, None, :]
                e = target_positions[None, :, None, :]
                base_points = s + t_vals * (e - s)
                
                X_init = np.tile(base_points, (pop_size, 1, 1, 1))
                
                # Dodanie szumu xy
                noise_xy = np.random.normal(0, 10.0, (pop_size, drone_swarm_size, n_inner, 2))
                X_init[..., :2] += noise_xy
                
                # Dodanie bezpiecznego korytarza dla osi Z
                random_z = np.random.uniform(MIN_Z_ALTITUDE, max_flight_z, (pop_size, drone_swarm_size, n_inner))
                X_init[..., 2] = random_z
                
                X_init = X_init.reshape(pop_size, d)
                X_init = np.clip(X_init, bounds_min, bounds_max)

                evaluator = VectorizedEvaluator(
                    obstacles=obs_list,
                    start_pos=start_positions,
                    target_pos=target_positions,
                    params=params
                )

            with _measure("optimization"):
                # Przekazanie rozpakowanych argumentów bezpośrednio do matematycznego silnika algorytmu
                best_genotype_flat = sparrow_search_algorithm_vectorized(
                    X_init=X_init,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                    iter_max=n_gen,
                    n_drones=drone_swarm_size,
                    n_inner=n_inner,
                    n_out=number_of_waypoints,
                    starts=start_positions,
                    targets=target_positions,
                    evaluator=evaluator,
                    st=st,
                    pd_ratio=pd_ratio,
                    sd_ratio=sd_ratio,
                    penalty_weight=penalty_weight,
                    weights=weights,
                    _measure=_measure
                )
            
            with _measure("decision_and_reconstruction"):
                inner = best_genotype_flat.reshape(1, drone_swarm_size, n_inner, 3)
                s = start_positions[None, :, None, :]
                t = target_positions[None, :, None, :]
                sparse = np.concatenate([s, inner, t], axis=2)
                
                final_traj = resample_polyline_batch(sparse, num_samples=number_of_waypoints)
                return final_traj[0]
            
    except Exception as e:
        with _measure("fallback"):
            print(f"[SSA Strategy] Błąd optymalizacji: {e}")
            # Fallback na trajektorię liniową
            t_arr = np.linspace(0, 1, number_of_waypoints)
            out = np.empty((drone_swarm_size, number_of_waypoints, 3))
            for d_idx in range(drone_swarm_size):
                for i in range(2):
                    out[d_idx, :, i] = np.interp(t_arr, [0, 1], [start_positions[d_idx, i], target_positions[d_idx, i]])
                z_s = max(start_positions[d_idx, 2], MIN_Z_ALTITUDE)
                z_t = max(target_positions[d_idx, 2], MIN_Z_ALTITUDE)
                out[d_idx, :, 2] = np.interp(t_arr, [0, 1], [z_s, z_t])
            return out
            
    finally:
        if local_timing and timing is not None:
            try:
                out_dir = HydraConfig.get().runtime.output_dir
                timing.save_csv(os.path.join(out_dir, "optimization_timings.csv"))
            except Exception as save_err:
                print(f"[SSA Strategy] Nie udało się zapisać logów czasowych: {save_err}")