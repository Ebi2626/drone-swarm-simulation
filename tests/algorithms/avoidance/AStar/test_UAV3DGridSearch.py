import pytest
import numpy as np
from src.algorithms.avoidance.AStar.UAV3DGridSearch import UAV3DGridSearch

@pytest.fixture
def base_searcher():
    """Zwraca instancję A* bez biasu (równomierny koszt w każdym kierunku)."""
    return UAV3DGridSearch(
        obs_pos=np.array([5.0, 5.0, 5.0]),
        obs_radius=1.0,
        grid_res=1.0,
        bbox_min=np.array([0.0, 0.0, 0.0]),
        bbox_max=np.array([10.0, 10.0, 10.0])
    )

@pytest.fixture
def biased_searcher():
    """Zwraca instancję A* z preferencją ruchu w osi Z (w górę)."""
    return UAV3DGridSearch(
        obs_pos=np.array([5.0, 5.0, 5.0]),
        obs_radius=1.0,
        grid_res=1.0,
        bbox_min=np.array([0.0, 0.0, 0.0]),
        bbox_max=np.array([10.0, 10.0, 10.0]),
        preferred_dir=np.array([0.0, 0.0, 1.0]),
        bias_preferred=1.0,
        bias_perpendicular=1.5,
        bias_oppose=2.0
    )


def test_heuristic_cost_estimate(base_searcher):
    """Sprawdza klasyczną euklidesową heurystykę (bez uwzględniania przeszkód)."""
    current = (0.0, 0.0, 0.0)
    goal = (3.0, 4.0, 0.0)
    # sqrt(3^2 + 4^2) = 5.0
    assert base_searcher.heuristic_cost_estimate(current, goal) == pytest.approx(5.0)


def test_distance_between_biased(biased_searcher):
    """Sprawdza, czy koszty ruchu uwzględniają preferowany kierunek (directional bias)."""
    start = np.array([0.0, 0.0, 0.0])
    
    # 1. Ruch zgodny z preferowanym wektorem Z
    up = np.array([0.0, 0.0, 1.0])
    cost_up = biased_searcher.distance_between(start, up)
    assert cost_up == pytest.approx(1.0 * biased_searcher.bias_preferred)

    # 2. Ruch przeciwny do preferowanego wektora Z (w dół)
    down = np.array([0.0, 0.0, -1.0])
    cost_down = biased_searcher.distance_between(start, down)
    assert cost_down == pytest.approx(1.0 * biased_searcher.bias_oppose)

    # 3. Ruch prostopadły (w osi Y)
    lateral = np.array([0.0, 1.0, 0.0])
    cost_lateral = biased_searcher.distance_between(start, lateral)
    assert cost_lateral == pytest.approx(1.0 * biased_searcher.bias_perpendicular)


def test_neighbors_out_of_bounds(base_searcher):
    """Sprawdza twarde ograniczenia bounding boxa. Węzły poza bbox nie mogą zostać zwrócone."""
    node = (0.0, 0.0, 0.0)  # Na samej krawędzi bbox_min
    
    neighbors = list(base_searcher.neighbors(node))
    
    # Wokół (0,0,0) w pełnym zakresie 3D jest 26 sąsiadów.
    # Będąc na rogu bounding boxa (0-10) [X,Y,Z >= 0], możliwe są tylko kroki dodatnie (względem zera).
    # Więc w osi X możliwe wartości: 0, 1. Oś Y: 0, 1. Oś Z: 0, 1.
    # Kombinacji jest 2x2x2 = 8, minus wektor zerowy (my) = 7 ważnych sąsiadów.
    assert len(neighbors) == 7
    
    for n in neighbors:
        assert all(0.0 <= val <= 10.0 for val in n), f"Sąsiad poza BBoxem: {n}"


def test_neighbors_inside_obstacle(base_searcher):
    """Weryfikuje, czy punkt końcowy kroku jest odrzucany, jeśli leży w przeszkodzie."""
    # Startujemy obok przeszkody
    start_node = (4.0, 5.0, 5.0)
    
    neighbors = list(base_searcher.neighbors(start_node))
    
    # Punkt (5.0, 5.0, 5.0) leży w samym środku przeszkody (obs_radius=1.0)
    # i musi zostać absolutnie wykluczony.
    assert (5.0, 5.0, 5.0) not in neighbors
    # Punkt (4.5, 5.0, 5.0) przy grid_res 1.0 nie istnieje, ale punkt (6.0, 5.0, 5.0) jest już bezpieczny
    # i może być potencjalnym sąsiadem (gdybyśmy skakali z 5.0).
    

def test_neighbors_segment_intersection(base_searcher):
    """
    Kluczowy test: Continuous Collision Detection.
    Sprawdza zjawisko tunelowania: punkty końcowe A i B są bezpieczne (poza przeszkodą),
    ale sam odcinek AB przechodzi na wylot przez przeszkodę.
    """
    # Zwiększamy promień przeszkody by ułatwić przecięcie z szerokim marginesem
    base_searcher.obs_radius = 1.5
    base_searcher.obs_radius_sq = 1.5 ** 2
    base_searcher.grid_res = 3.0  # Duży krok siatki wymuszający przelot przez środek
    
    # Start po jednej stronie przeszkody
    node_a = (2.0, 5.0, 5.0)
    
    neighbors = list(base_searcher.neighbors(node_a))
    
    # Punkt docelowy (5.0, 5.0, 5.0) będzie wyłączony, bo jest w środku przeszkody.
    # Ale punkt (8.0, 5.0, 5.0) jest odległy o 3m od (5,5,5), więc technicznie sam w sobie
    # jest "poza" sferą R=1.5. Jednak odcinek A(2,5,5) -> B(8,5,5) przechodzi przez (5,5,5).
    # Algorytm ray-castingowy musi wykluczyć węzeł B z sąsiedztwa węzła A!
    assert (8.0, 5.0, 5.0) not in neighbors