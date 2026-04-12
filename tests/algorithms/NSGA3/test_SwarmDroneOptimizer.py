import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.algorithms.NSGA3.SwarmDroneOptimizer import SwarmDroneOptimizer

@pytest.fixture
def optimizer_config():
    """Podstawowa konfiguracja do inicjalizacji Optymalizatora."""
    return {
        "space_limits": [100.0, 100.0, 50.0],
        "n_drones": 2,
        "n_waypoints": 3,
        "start_positions": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        "end_positions": [[0.0, 100.0, 0.0], [10.0, 100.0, 0.0]],
        "obstacles": np.array([[5.0, 50.0, 2.0, 20.0]])
    }

# Tworzymy atrapę (Mock) dla obiektu Result zwracanego przez Pymoo
class DummyPymooResult:
    def __init__(self, X, F):
        self.X = X
        self.F = F

@patch("src.algorithms.NSGA3.SwarmDroneOptimizer.ElementWiseProblem")
def test_optimizer_initialization(mock_problem_class, optimizer_config):
    """Sprawdza, czy Optymalizator poprawnie przekazuje argumenty do Problem-u."""
    optimizer = SwarmDroneOptimizer(**optimizer_config)
    
    # Sprawdzamy czy przypisał zmienne
    assert optimizer.n_drones == 2
    assert optimizer.n_waypoints == 3
    
    # Sprawdzamy czy zainicjalizował Problem z prawidłowymi argumentami
    mock_problem_class.assert_called_once_with(
        space_limits=optimizer_config["space_limits"],
        n_drones=optimizer_config["n_drones"],
        n_waypoints=optimizer_config["n_waypoints"],
        start_positions=optimizer_config["start_positions"],
        end_positions=optimizer_config["end_positions"],
        obstacles=optimizer_config["obstacles"]
    )

@patch("src.algorithms.NSGA3.SwarmDroneOptimizer.minimize")
@patch("src.algorithms.NSGA3.SwarmDroneOptimizer.NSGA3")
@patch("src.algorithms.NSGA3.SwarmDroneOptimizer.ElementWiseProblem")
def test_run_optimization(mock_problem_class, mock_nsga3, mock_minimize, optimizer_config):
    """Sprawdza, czy funkcja minimalizująca Pymoo jest wywoływana z odpowiednimi flagami."""
    optimizer = SwarmDroneOptimizer(**optimizer_config)
    
    # Ustawiamy mock dla wyniku
    fake_result = MagicMock()
    mock_minimize.return_value = fake_result
    
    # Uruchamiamy
    res = optimizer.run_optimization(pop_size=50, n_gen=10)
    
    # 1. Weryfikacja czy NSGA3 dostało pop_size
    mock_nsga3.assert_called_once()
    assert mock_nsga3.call_args.kwargs['pop_size'] == 50
    
    # 2. Weryfikacja delegacji do funkcji minimize
    mock_minimize.assert_called_once()
    called_kwargs = mock_minimize.call_args.kwargs
    
    # Sprawdzamy kluczowe parametry konfiguracyjne
    assert called_kwargs['termination'] == ('n_gen', 10)
    assert called_kwargs['verbose'] is True
    assert called_kwargs['return_least_infeasible'] is True
    
    # 3. Zwracany jest ten sam obiekt wyniku
    assert res == fake_result

def test_get_best_trajectories_sorting_and_reshape(optimizer_config):
    """Sprawdza algorytm sortowania wyników i konwersji kształtów."""
    # Instancjujemy z użyciem "patch" jako context managera, żeby nie mockować globalnie dla testu
    with patch("src.algorithms.NSGA3.ElementWiseProblem.ElementWiseProblem"):
        optimizer = SwarmDroneOptimizer(**optimizer_config)
        
    n_drones = 2
    n_points = 3
    n_var = n_drones * n_points * 3 # 18
    
    # Tworzymy 3 sztuczne rozwiązania (X) z różnymi wartościami
    # Rozwiązanie A: same 1, Rozwiązanie B: same 2, Rozwiązanie C: same 3
    X_dummy = np.array([
        np.ones(n_var) * 1.0,  # Będzie miało F=50.0
        np.ones(n_var) * 2.0,  # Będzie miało F=10.0 (Najlepsze)
        np.ones(n_var) * 3.0   # Będzie miało F=30.0
    ])
    
    # Tworzymy sztuczne oceny (F). Zakładamy, że pierwsza kolumna to F1 (długość trasy).
    # Chcemy, aby algorytm posortował to tak: indeks 1, potem 2, potem 0.
    F_dummy = np.array([
        [50.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [30.0, 0.0, 0.0]
    ])
    
    mock_res = DummyPymooResult(X=X_dummy, F=F_dummy)
    
    # Żądamy 2 najlepszych z 3
    trajectories = optimizer.get_best_trajectories(mock_res, n=2)
    
    # WERYFIKACJA KSZTAŁTU
    # Oczekujemy: (Liczba rozwiązań=2, Liczba dronów=2, Liczba punktów=3, Współrzędne XYZ=3)
    assert trajectories.shape == (2, 2, 3, 3)
    
    # WERYFIKACJA SORTOWANIA
    # Pierwsze zwrócone rozwiązanie powinno pochodzić z indeksu 1 (wartości=2.0)
    assert np.all(trajectories[0] == 2.0)
    
    # Drugie zwrócone rozwiązanie powinno pochodzić z indeksu 2 (wartości=3.0)
    assert np.all(trajectories[1] == 3.0)

def test_get_best_trajectories_raises_value_error(optimizer_config):
    """Sprawdza zabezpieczenie przed pustym obiektem X."""
    with patch("src.algorithms.NSGA3.ElementWiseProblem.ElementWiseProblem"):
        optimizer = SwarmDroneOptimizer(**optimizer_config)
    
    mock_res_empty = DummyPymooResult(X=None, F=np.array([]))
    
    with pytest.raises(ValueError, match="Optymalizacja nie zwróciła wyników"):
        optimizer.get_best_trajectories(mock_res_empty)

def test_get_best_trajectories_less_results_than_requested(optimizer_config):
    """Sprawdza, czy funkcja prawidłowo obsłuży sytuację, gdy n > ilości znalezionych rozwiązań."""
    with patch("src.algorithms.NSGA3.ElementWiseProblem.ElementWiseProblem"):
        optimizer = SwarmDroneOptimizer(**optimizer_config)
        
    # Symulujemy tylko 1 znalezione rozwiązanie
    X_dummy = np.ones((1, 18))
    F_dummy = np.array([[10.0, 0.0, 0.0]])
    mock_res = DummyPymooResult(X=X_dummy, F=F_dummy)
    
    # Żądamy 5
    trajectories = optimizer.get_best_trajectories(mock_res, n=5)
    
    # System powinien "przyciąć" n do 1 i zwrócić po prostu wszystko, co ma
    assert len(trajectories) == 1
    assert trajectories.shape == (1, 2, 3, 3)