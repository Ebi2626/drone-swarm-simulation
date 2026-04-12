import numpy as np
import pytest
from src.algorithms.abstraction.trajectory.strategies.core_msffoa import MSFFOAOptimizer  # Zaktualizuj import stosownie do nazwy pliku z kodem

# ---------------------------------------------------------------------------
# Zestaw danych testowych (Fixtures)
# ---------------------------------------------------------------------------

@pytest.fixture
def base_optimizer_params():
    """Zwraca podstawowe parametry dla testów."""
    pop_size = 20
    n_drones = 3
    n_inner = 5
    n_out = 20
    
    world_min = np.array([0.0, 0.0, 0.0])
    world_max = np.array([100.0, 100.0, 50.0])
    
    start_pos = np.array([
        [10.0, 10.0, 1.0],
        [10.0, 15.0, 1.0],
        [10.0, 20.0, 1.0]
    ])
    target_pos = np.array([
        [90.0, 90.0, 5.0],
        [90.0, 95.0, 5.0],
        [90.0, 100.0, 5.0]
    ])
    
    return {
        "pop_size": pop_size,
        "n_drones": n_drones,
        "n_inner": n_inner,
        "n_output_samples": n_out,
        "world_min_bounds": world_min,
        "world_max_bounds": world_max,
        "start_positions": start_pos,
        "target_positions": target_pos,
        "max_generations": 15, # Niska wartość, aby testy działały szybko
        "seed": 42
    }


def dummy_fitness_function(dense_trajectories: np.ndarray) -> np.ndarray:
    """
    Prosta funkcja celu (zastępująca VectorizedEvaluator na etapie 1).
    Minimalizuje odległość całej trajektorii od idealnego środka świata [50, 50, 25].
    Kształt wejścia: (Pop_size, Drones, N_out, 3)
    Kształt wyjścia: (Pop_size,)
    """
    ideal_point = np.array([50.0, 50.0, 25.0])
    
    # Odległość euklidesowa każdego punktu od punktu idealnego
    distances = np.linalg.norm(dense_trajectories - ideal_point, axis=-1)
    
    # Sumujemy odległości po dronach i waypointach, by uzyskać 1 wartość na osobnika
    fitness = np.sum(distances, axis=(1, 2))
    return fitness


# ---------------------------------------------------------------------------
# Testy
# ---------------------------------------------------------------------------

def test_msffoa_initialization_shapes(base_optimizer_params):
    """Sprawdza, czy tensory populacji są generowane w prawidłowych wymiarach i na prawidłowych osiach Z."""
    params = base_optimizer_params.copy()
    params["fitness_function"] = dummy_fitness_function
    
    optimizer = MSFFOAOptimizer(**params)
    pop = optimizer._initialize_population()
    
    expected_shape = (params["pop_size"], params["n_drones"], params["n_inner"], 3)
    
    # Asercja 1: Poprawny kształt tensora 4D
    assert pop.shape == expected_shape, f"Oczekiwano kształtu {expected_shape}, otrzymano {pop.shape}"
    
    # Asercja 2: Zabezpieczenie przed zejściem pod ziemię (Z clipping)
    # Minimalne Z zdefiniowane w konstruktorze to 0.5
    min_z_in_pop = np.min(pop[:, :, :, 2])
    assert min_z_in_pop >= 0.5, f"Wykryto kolizję z ziemią (Z < 0.5) w populacji startowej: {min_z_in_pop}"


def test_msffoa_dense_trajectory_building(base_optimizer_params):
    """Weryfikuje czy mechanizm sklejania i zagęszczania trajektorii (resampling) dodaje start i cel."""
    params = base_optimizer_params.copy()
    params["fitness_function"] = dummy_fitness_function
    
    optimizer = MSFFOAOptimizer(**params)
    pop = optimizer._initialize_population()
    dense = optimizer._build_dense(pop)
    
    expected_dense_shape = (params["pop_size"], params["n_drones"], params["n_output_samples"], 3)
    assert dense.shape == expected_dense_shape, "Błąd w mechanizmie zagęszczania polilinii"
    
    # Upewnienie się, że początek i koniec gęstej trajektorii to dokładnie start i target
    # np.allclose omija problemy z precyzją floatów
    start_matches = np.allclose(dense[:, :, 0, :], params["start_positions"])
    target_matches = np.allclose(dense[:, :, -1, :], params["target_positions"])
    
    assert start_matches, "Pierwszy waypoint dense trajektorii nie pokrywa się z pozycją startową"
    assert target_matches, "Ostatni waypoint dense trajektorii nie pokrywa się z metą"


def test_msffoa_optimization_convergence(base_optimizer_params):
    """
    Najważniejszy test behawioralny: Uruchamia algorytm i sprawdza, czy po X generacjach 
    następuje poprawa (zbieżność) w stosunku do populacji początkowej.
    """
    params = base_optimizer_params.copy()
    params["fitness_function"] = dummy_fitness_function
    params["max_generations"] = 25 # Dajemy mu czas na zbiegnięcie
    
    optimizer = MSFFOAOptimizer(**params)
    
    # Wywołujemy optymalizację
    best_trajectory, final_fitness = optimizer.optimize()
    
    # Sprawdzamy ocenę generacji zerowej (aby mieć punkt odniesienia)
    initial_pop_fitness = dummy_fitness_function(optimizer._build_dense(optimizer.population))
    worst_initial_fitness = np.max(initial_pop_fitness)
    
    # Asercja 1: Sprawdzenie poprawności zwracanych danych
    expected_best_shape = (params["n_drones"], params["n_inner"], 3)
    assert best_trajectory.shape == expected_best_shape
    
    # Asercja 2: Sprawdzenie matematycznej zbieżności
    # Algorytm optymalizacyjny MUST poprawić (lub zachować najlepszy) wynik względem losowego szumu
    assert final_fitness < worst_initial_fitness, "Algorytm nie zbiega. Fitness na końcu jest gorszy niż najgorszy punkt startowy."
    
    # Upewnienie się, że pełna gęsta trajektoria też może zostać prawidłowo pobrana z API
    best_dense = optimizer.get_best_dense_trajectory()
    assert best_dense.shape == (params["n_drones"], params["n_output_samples"], 3)