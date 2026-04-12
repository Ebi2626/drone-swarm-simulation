import pytest
import numpy as np
import importlib
TARGET_MODULE = "src.algorithms.abstraction.trajectory.strategies.nsga3_utils.objective_constrains"
EvaluatorModule = importlib.import_module(TARGET_MODULE)

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def sample_trajectories():
    """
    Kształt wektora: (pop_size, n_drones, n_waypoints, 3)
    Pop 0: Prosta linia w osi X (0->1->2), Z rośnie i opada.
    Pop 1: Trójkąt (0,0,0) -> (0,3,0) -> (4,3,0) (Długość: 3 + 4 = 7).
    """
    pop0 = np.array([
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 2.0],
            [2.0, 0.0, 1.0]
        ]
    ])
    
    pop1 = np.array([
        [
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [4.0, 3.0, 0.0]
        ]
    ])
    
    return np.vstack([pop0[None, ...], pop1[None, ...]])

# ==========================================
# TESTY FUNKCJI CELU (F)
# ==========================================

def test_calc_path_length(sample_trajectories):
    """Intencja: Obliczenie sumy długości euklidesowych segmentów."""
    lengths = EvaluatorModule.calc_path_length(sample_trajectories)
    
    # Pop 0: 
    # Seg 1: (0,0,0)->(1,0,2) = sqrt(1^2 + 2^2) = sqrt(5) ~ 2.236
    # Seg 2: (1,0,2)->(2,0,1) = sqrt(1^2 + 1^2) = sqrt(2) ~ 1.414
    # Suma ~ 3.650
    expected_pop0 = np.sqrt(5) + np.sqrt(2)
    
    # Pop 1:
    # Seg 1: 3.0, Seg 2: 4.0. Suma = 7.0
    expected_pop1 = 7.0
    
    np.testing.assert_array_almost_equal(lengths, [expected_pop0, expected_pop1])

def test_calc_elevation_changes(sample_trajectories):
    """Intencja: Obliczenie sumy bezwzględnych zmian wysokości (osi Z)."""
    elevations = EvaluatorModule.calc_elevation_changes(sample_trajectories)
    
    # Pop 0: 0 -> 2 (zmiana 2), 2 -> 1 (zmiana 1). Suma = 3
    # Pop 1: 0 -> 0 -> 0. Suma = 0
    np.testing.assert_array_almost_equal(elevations, [3.0, 0.0])

# ==========================================
# TESTY NOWYCH OGRANICZEŃ GEOMETRYCZNYCH (G)
# ==========================================

def test_constr_segment_uniformity():
    """
    Intencja: Wykrycie "źle rozłożonych" waypointów za pomocą odchylenia standardowego.
    """
    # Pop 0: Punkty równomierne (długości segmentów: 1.0 i 1.0) -> std = 0.0
    pop_uniform = np.array([[[[0,0,0], [1,0,0], [2,0,0]]]])
    
    # Pop 1: Punkty "zlepione" (długości segmentów: 0.2 i 1.8) -> średnia = 1.0, std = 0.8
    pop_clustered = np.array([[[[0,0,0], [0.2,0,0], [2.0,0,0]]]])
    
    traj = np.vstack([pop_uniform, pop_clustered])
    
    # Ustawiamy tolerancję na 0.5. Pop_uniform powinno mieć CV=0, Pop_clustered CV=(0.8 - 0.5) = 0.3
    cv = EvaluatorModule.constr_segment_uniformity(traj, tolerance_std=0.5)
    
    np.testing.assert_array_almost_equal(cv, [0.0, 0.3])

def test_constr_path_smoothness():
    """
    Intencja: Wykrycie ostrych "zygzaków" za pomocą drugiej różnicy (krzywizny).
    """
    # Pop 0: Prosta linia. P(i+1) - 2P(i) + P(i-1) = [2,0,0] - [2,0,0] + [0,0,0] = [0,0,0]
    pop_straight = np.array([[[[0,0,0], [1,0,0], [2,0,0]]]])
    
    # Pop 1: Zygzak (kąt prosty). (0,0) -> (1,1) -> (2,0)
    # Druga różnica: [2,0,0] - 2*[1,1,0] + [0,0,0] = [0,-2,0]
    # Kwadrat (Jerk_sq): 4.0
    pop_zig_zag = np.array([[[[0,0,0], [1,1,0], [2,0,0]]]])
    
    traj = np.vstack([pop_straight, pop_zig_zag])
    
    # Tolerancja na 1.0. Prosta ma CV=0, Zygzak ma (4.0 - 1.0) = 3.0
    cv = EvaluatorModule.constr_path_smoothness(traj, max_turn_factor=1.0)
    
    np.testing.assert_array_almost_equal(cv, [0.0, 3.0])

# ==========================================
# TESTY DETEKCJI KOLIZJI (Odcinek vs Cylinder)
# ==========================================

def test_dist_segment_to_cylinder():
    """
    Intencja: Sprawdzenie matematyki wektorowej dla detekcji odcinka tnącego cylinder.
    """
    # Odcinek tnie środek układu współrzędnych w osi Y
    seg_start = np.array([[0.0, -2.0, 1.0]])
    seg_end = np.array([[0.0, 2.0, 1.0]])
    
    # Cylinder w środku (0,0), dół Z=0, promień=1, wysokość=2
    obs_center = np.array([[0.0, 0.0, 0.0]])
    obs_radius = np.array([1.0])
    obs_height = np.array([2.0])
    
    # Odcinek przechodzi centralnie przez środek (odległość XY od środka = 0).
    # R_sq = 1^2 = 1. Penetracja = max(0, 1 - 0) = 1.0
    violation = EvaluatorModule._dist_segment_to_cylinder(
        seg_start, seg_end, obs_center, obs_radius, obs_height
    )
    
    np.testing.assert_array_almost_equal(violation, [[1.0]])

def test_dist_segment_to_cylinder_above():
    """
    Edge case: Odcinek przelatuje DOKŁADNIE nad cylindrem (w osi XY tnie go, ale Z jest bezpieczne).
    """
    seg_start = np.array([[0.0, -2.0, 5.0]]) # Z = 5.0
    seg_end = np.array([[0.0, 2.0, 5.0]])
    
    obs_center = np.array([[0.0, 0.0, 0.0]])
    obs_radius = np.array([1.0])
    obs_height = np.array([2.0]) # Cylinder sięga tylko do Z = 2.0
    
    violation = EvaluatorModule._dist_segment_to_cylinder(
        seg_start, seg_end, obs_center, obs_radius, obs_height
    )
    
    # Całkowity brak penetracji
    np.testing.assert_array_almost_equal(violation, [[0.0]])

# ==========================================
# TESTY WRAPPERA (VectorizedEvaluator)
# ==========================================

def test_vectorized_evaluator(sample_trajectories):
    """
    Intencja: Upewnienie się, że klasa nadrzędna poprawnie składa wszystkie 
    funkcje F i G w macierze dla biblioteki pymoo.
    """
    start_pos = np.array([[0.0, 0.0, 0.0]])
    target_pos = np.array([[4.0, 3.0, 0.0]]) # Odległość 5.0
    
    # Zostawiamy listę przeszkód pustą dla uproszczenia
    evaluator = EvaluatorModule.VectorizedEvaluator(
        obstacles=[], 
        start_pos=start_pos, 
        target_pos=target_pos,
        params={"max_jerk": 100.0, "uniformity_std": 10.0} # Duża tolerancja by nie wchodzić w detale G
    )
    
    out = {}
    evaluator.evaluate(sample_trajectories, out)
    
    # Sprawdzamy, czy słownik został uzupełniony macierzami o poprawnych wymiarach
    assert "F" in out
    assert "G" in out
    
    # F ma 3 cele: (Length, Risk, Elevation)
    assert out["F"].shape == (2, 3) 
    
    # G ma 5 ograniczeń: (Battery, Separation, Obstacle_Hard, Uniformity, Smoothness)
    assert out["G"].shape == (2, 5)