import pytest
import numpy as np

from src.algorithms.NSGA3.ElementWiseProblem import ElementWiseProblem

@pytest.fixture
def sample_problem():
    """Przygotowuje instancję problemu z małymi wymiarami do szybkich testów."""
    space_limits = [100.0, 100.0, 50.0]
    n_drones = 2
    n_waypoints = 3
    
    start_positions = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    end_positions = [[0.0, 100.0, 0.0], [10.0, 100.0, 0.0]]
    
    # Przeszkoda: x=5, y=50, r=2, h=20
    obstacles = np.array([[5.0, 50.0, 2.0, 20.0]])
    
    return ElementWiseProblem(
        space_limits=space_limits,
        n_drones=n_drones,
        n_waypoints=n_waypoints,
        start_positions=start_positions,
        end_positions=end_positions,
        obstacles=obstacles
    )

def test_initialization_bounds_and_vars(sample_problem):
    """Weryfikuje, czy granice przestrzeni i wymiary zostały poprawnie zainicjalizowane."""
    # Sprawdzamy liczbę zmiennych decyzyjnych: 2 drony * 3 waypointy * 3 osie = 18
    assert sample_problem.n_var == 18
    
    # Sprawdzamy l_max (limit baterii = 3 * dystans w linii prostej)
    # Dystans Start->End dla obu dronów to dokładnie 100.0 (wzdłuż osi Y)
    expected_l_max = np.array([300.0, 300.0])
    np.testing.assert_array_almost_equal(sample_problem.l_max, expected_l_max)
    
    # Sprawdzamy kształt upper bounds
    assert sample_problem.xu.shape == (18,)
    assert sample_problem.xl.shape == (18,)

def test_reshape_population(sample_problem):
    """Weryfikuje, czy płaski wektor populacji jest poprawnie zmieniany w tensor 4D z dodanymi punktami Start/End."""
    n_pop = 10
    n_var = sample_problem.n_var
    
    # Tworzymy sztuczną populację z jedynkami
    X_dummy = np.ones((n_pop, n_var))
    
    # Wywołanie testowanej metody
    trajectories = sample_problem._reshape_population(X_dummy)
    
    # Oczekiwany kształt: (N_pop, n_drones, n_waypoints + 2, 3)
    # n_pop=10, drones=2, points=5 (3 + start + end), coords=3
    assert trajectories.shape == (10, 2, 5, 3)
    
    # Sprawdzenie czy Start został poprawnie doklejony (na indeksie 0 dla punktów)
    # Dla drona 0 start to [0, 0, 0]
    np.testing.assert_array_equal(trajectories[0, 0, 0, :], [0.0, 0.0, 0.0])
    # Dla drona 1 start to [10, 0, 0]
    np.testing.assert_array_equal(trajectories[5, 1, 0, :], [10.0, 0.0, 0.0])
    
    # Sprawdzenie czy End został poprawnie doklejony (na indeksie -1)
    np.testing.assert_array_equal(trajectories[0, 0, -1, :], [0.0, 100.0, 0.0])

def test_calc_segment_lengths(sample_problem):
    """Sprawdza wektorowe liczenie długości tras z użyciem prostego układu geometrycznego."""
    # Konstrukcja ręczna trajektorii dla 1 osobnika i 1 drona: (N_pop=1, n_drones=1, n_points=3, coords=3)
    # Punkty: (0,0,0) -> (3,4,0) -> (3,4,12)
    # Odcinki: 5 (z Pitagorasa 3,4) + 12 (w pionie) = 17 całkowitej długości
    dummy_trajs = np.array([[[
        [0.0, 0.0, 0.0],
        [3.0, 4.0, 0.0],
        [3.0, 4.0, 12.0]
    ]]])
    
    lengths = sample_problem._calc_segment_lengths(dummy_trajs)
    
    assert lengths.shape == (1, 1) # (N_pop, n_drones)
    np.testing.assert_almost_equal(lengths[0, 0], 17.0)

def test_evaluate_returns_correct_dictionary_shapes(sample_problem):
    """
    Test integracyjny głównej metody _evaluate.
    Sprawdza, czy Pymoo dostanie z powrotem słownik z prawidłowymi wymiarami macierzy F i G.
    """
    n_pop = 5
    X_dummy = np.random.rand(n_pop, sample_problem.n_var) * 50.0
    
    # Pusty słownik, do którego Pymoo standardowo oczekuje wstrzyknięcia wyników
    out = {}
    
    # Wywołanie ewaluacji
    sample_problem._evaluate(x=X_dummy, out=out)
    
    # SPRAWDZENIE CELÓW (F)
    assert "F" in out, "Słownik musi zawierać klucz 'F'"
    # Mamy 3 funkcje celu, więc kształt F to (N_pop, 3)
    assert out["F"].shape == (n_pop, 3)
    
    # SPRAWDZENIE OGRANICZEŃ (G)
    assert "G" in out, "Słownik musi zawierać klucz 'G'"
    # Mamy n_drones * 3 ograniczeń, czyli 2 * 3 = 6
    assert out["G"].shape == (n_pop, 6)
    
    # Upewniamy się, że w wyjściu nie ma wartości NaN (złe wskaźniki matematyczne)
    assert not np.isnan(out["F"]).any()
    assert not np.isnan(out["G"]).any()