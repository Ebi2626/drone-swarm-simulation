import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.environments.obstacles.ObstacleShape import ObstacleShape
from src.algorithms.abstraction.trajectory.objective_constrains import VectorizedEvaluator

TARGET_MODULE = "src.algorithms.abstraction.trajectory.objective_constrains"

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def basic_params():
    return {
        "k_factor": 2.0,
        "absolute_min_node_dist": 5.0,
        "obstacle_safety_margin": 0.5,
        "min_drone_distance": 2.0,
        "max_accel_limit": 5.0,
        # h_pref=0 + flat trajectory → f_height=0, f_angle=0 (testowalne).
        "preferred_height": 0.0,
        "coordination_penalty_factor": 1.0,
    }

@pytest.fixture
def mock_obstacles_cylinder():
    """Przeszkoda walcowa do testowania ekstrakcji promienia."""
    obs = MagicMock()
    obs.shape_type = ObstacleShape.CYLINDER
    # data: [x, y, z, promień, wysokość, ...]
    obs.data = np.array([[10.0, 10.0, 0.0, 2.0, 10.0, 0.0]])
    return obs

@pytest.fixture
def mock_obstacles_box():
    """Przeszkoda prostopadłościenna do testowania okręgu opisanego."""
    obs = MagicMock()
    obs.shape_type = ObstacleShape.BOX
    # data: [x, y, z, długość_x, szerokość_y, wysokość]
    obs.data = np.array([[20.0, 20.0, 0.0, 4.0, 3.0, 10.0]])
    return obs

# ==========================================
# TESTY INICJALIZACJI PRZESZKÓD
# ==========================================

@patch(f"{TARGET_MODULE}.calculate_dynamic_max_node_distance", return_value=10.0)
def test_evaluator_init_no_obstacles(mock_calc, basic_params):
    """Brak przeszkód nie powinien rzucać wyjątków, a tablice powinny być puste."""
    starts = np.array([[0.0, 0.0, 0.0]])
    targets = np.array([[100.0, 100.0, 0.0]])
    
    evaluator = VectorizedEvaluator(None, starts, targets, 3, basic_params)
    
    assert len(evaluator.obstacles_xy) == 0
    assert len(evaluator.obstacle_radii) == 0

@patch(f"{TARGET_MODULE}.calculate_dynamic_max_node_distance", return_value=10.0)
def test_evaluator_init_cylinder_obstacles(mock_calc, mock_obstacles_cylinder, basic_params):
    """Dla cylindra bierzemy jego promień i dodajemy margines bezpieczeństwa."""
    starts = np.array([[0.0, 0.0, 0.0]])
    targets = np.array([[100.0, 100.0, 0.0]])
    
    evaluator = VectorizedEvaluator(mock_obstacles_cylinder, starts, targets, 3, basic_params)
    
    assert evaluator.obstacles_xy.shape == (1, 2)
    np.testing.assert_array_almost_equal(evaluator.obstacles_xy[0], [10.0, 10.0])
    
    # Oczekiwany promień: 2.0 (z danych) + 0.5 (margines z parametrów) = 2.5
    np.testing.assert_array_almost_equal(evaluator.obstacle_radii, [2.5])

@patch(f"{TARGET_MODULE}.calculate_dynamic_max_node_distance", return_value=10.0)
def test_evaluator_init_box_obstacles(mock_calc, mock_obstacles_box, basic_params):
    """
    Dla pudełka liczymy promień okręgu opisanego na prostokącie XY, 
    a następnie dodajemy margines bezpieczeństwa.
    """
    starts = np.array([[0.0, 0.0, 0.0]])
    targets = np.array([[100.0, 100.0, 0.0]])
    
    evaluator = VectorizedEvaluator(mock_obstacles_box, starts, targets, 3, basic_params)
    
    # lx=4, wy=3 -> half_lx=2.0, half_wy=1.5
    # circumscribed_radius = sqrt(2^2 + 1.5^2) = sqrt(4 + 2.25) = sqrt(6.25) = 2.5
    # promień ostateczny: 2.5 + 0.5 (margines) = 3.0
    np.testing.assert_array_almost_equal(evaluator.obstacle_radii, [3.0])

# ==========================================
# TESTY FUNKCJI CELU I OGRANICZEŃ (EWALUACJA)
# ==========================================

@patch(f"{TARGET_MODULE}.calculate_dynamic_max_node_distance", return_value=20.0)
@patch(f"{TARGET_MODULE}.evaluate_bspline_trajectory_sync")
def test_evaluate_shapes_and_values(mock_sync, mock_calc, basic_params):
    """
    Nowy `VectorizedEvaluator` zwraca F.shape=(PopSize,5) i G.shape=(PopSize,3):
      F = [f1 trajectory, f2 height_angle, f3 threat, f4 turn, f5 coordination]
      G = [obs_collisions_sum, swarm_hard - 0.01, kinematic_penalty]
    """
    starts = np.array([[0.0, 0.0, 0.0]])
    targets = np.array([[100.0, 100.0, 0.0]])

    pop_size = 2
    n_drones = 1
    n_inner = 3

    # Mockujemy wyniki z zewnętrznej, zsynchronizowanej funkcji Numbą:
    # 1. obs_collisions: shape (PopSize, NDrones)
    # 2. lengths: shape (PopSize, NDrones)
    # 3. swarm_collisions: shape (PopSize,)
    mock_sync.return_value = (
        np.array([[1.0], [0.0]]),       # Agent 0 zderzył się raz. Agent 1 czysty.
        np.array([[50.0], [45.0]]),     # Długości tras (mock — niezależne od pkt)
        np.array([0.0, 2.0])            # Kolizje wewnątrz roju
    )

    evaluator = VectorizedEvaluator(None, starts, targets, n_inner, basic_params)

    # Wymuszamy "zdegenerowanie do zera" wszystkich wtórnych kar:
    # - punkty kolinearne na prostej start→target (xs=ys, z=0) → f_shape=0,
    #   f4_turn=0 (kolejne wektory w tym samym kierunku → arccos(1)=0).
    # - z=0 == h_pref=0 → f_height=0; brak skoków w z → f_angle=0.
    # - 1 dron → f5_coordination=0 (po wyzerowaniu diagonali).
    # - max_node_distance=20 (mock); spacing 5 < 20 → kinematic=0.
    n_control_points = n_inner + 2
    xs = np.linspace(0.0, 20.0, n_control_points)
    control_points = np.zeros((pop_size, n_drones, n_control_points, 3))
    control_points[..., 0] = xs[np.newaxis, np.newaxis, :]
    control_points[..., 1] = xs[np.newaxis, np.newaxis, :]

    out = {}
    evaluator.evaluate(control_points, out)

    # Wymiarowość: 5 obj + 3 ograniczenia (Pymoo NSGA-III).
    assert "F" in out
    assert "G" in out
    assert out["F"].shape == (pop_size, 5)
    assert out["G"].shape == (pop_size, 3)

    F = out["F"]
    G = out["G"]

    # --- Cele (F) ---
    # f1 = f_length (mock) + f_shape (=0 dla kolinearnych punktów na prostej).
    np.testing.assert_array_almost_equal(F[:, 0], [50.0, 45.0])
    # f2 = f_height (=0; z=h_pref=0) + f_angle (=0; brak skoków w z).
    np.testing.assert_array_almost_equal(F[:, 1], [0.0, 0.0])
    # f3 = threat — brak przeszkód.
    np.testing.assert_array_almost_equal(F[:, 2], [0.0, 0.0])
    # f4 = turn² — punkty kolinearne, dot=1, arccos(1)=0.
    np.testing.assert_array_almost_equal(F[:, 3], [0.0, 0.0])
    # f5 = coordination — pojedynczy dron, po wyzerowaniu diagonali =0.
    np.testing.assert_array_almost_equal(F[:, 4], [0.0, 0.0])

    # --- Ograniczenia (G) ---
    # G[0]: suma kar za uderzenia w przeszkody (mock).
    np.testing.assert_array_almost_equal(G[:, 0], [1.0, 0.0])
    # G[1]: swarm_hard - 0.01.
    np.testing.assert_array_almost_equal(G[:, 1], [-0.01, 1.99])
    # G[2]: kinematic — spacing 5, max_node_distance 20, accel limit 5 → 0.
    np.testing.assert_array_almost_equal(G[:, 2], [0.0, 0.0])


# ==========================================
# TWARDY KINEMATYCZNY CONSTRAINT (2026-05-07)
# ==========================================

@patch(f"{TARGET_MODULE}.calculate_dynamic_max_node_distance", return_value=50.0)
@patch(f"{TARGET_MODULE}.evaluate_bspline_trajectory_sync")
def test_kinematic_penalty_catches_physical_acceleration_violation(
    mock_sync, mock_calc, basic_params,
):
    """
    Twardy fizyczny constraint na lateral acceleration (2026-05-07).

    Obecny `accel_violations = max(0, ||diff2|| - max_accel_limit)` mierzy
    drugą różnicę skończoną control points B-spline'a w przestrzeni
    *geometrycznej* (m), NIE fizyczną acceleration (m/s²). Mapping:

        a_lateral_physical ≈ v_cruise² × ||diff2|| / ||diff1||²

    Test konstruuje B-spline z:
      - ||diff1|| = 1.0m  (clustered control points)
      - ||diff2|| ≈ 1.41m (poniżej max_accel_limit=5.0 — current geometric
        constraint NIE łapie)
      - cruise_speed = 6 m/s, max_accel = 2 m/s²

    Fizyczna lateral acceleration: a_lat ≈ 36 × 1.41 / 1 = 50.8 m/s² ≫ 2.0.
    Obecne G[2] = 0 (test FAIL). Po fix'ie sample-based: G[2] > 0
    (NSGA-III odrzuca rozwiązanie jako infeasible).

    Test reprodukuje scenariusz raportowany przez user'a z 2026-05-07:
    drony spadają bo trajectory planner generuje zakręty wymagające
    `|a_lat| > max_accel`, czego current G[2] nie wykrywa.
    """
    starts = np.array([[0.0, 0.0, 0.0]])
    targets = np.array([[100.0, 100.0, 0.0]])

    # Mock'ujemy zewnętrzną funkcję — testujemy CZYSTO kinematic constraint,
    # bez kolizji obstacle/swarm.
    mock_sync.return_value = (
        np.array([[0.0]]),    # obs_collisions
        np.array([[10.0]]),   # lengths
        np.array([0.0]),      # swarm_collisions
    )

    # Parametry krytyczne dla twardego constraint:
    params = dict(basic_params)
    params["max_accel_limit"] = 5.0       # GEOMETRIC limit (current, lenient)
    params["max_accel"] = 2.0             # PHYSICAL limit (m/s², hard)
    params["cruise_speed"] = 6.0          # do mapowania ||diff2||→a_lat

    evaluator = VectorizedEvaluator(None, starts, targets, 4, params)

    # B-spline z ostrym U-turn — clustered control points (||diff1||=1m)
    # robiące zakręt 90° (diff2 perpendicular do diff1):
    #   P0=(0,0,1), P1=(1,0,1), P2=(2,0,1), P3=(2,1,1), P4=(1,1,1), P5=(0,1,1)
    # ||diff1|| = [1, 1, 1, 1, 1] m  (uniform spacing)
    # ||diff2|| = [0, sqrt(2), sqrt(2), 0] m  (max ≈ 1.41 < 5.0 = limit)
    # Physical a_lat (z derivacji): 6² × 1.41 / 1² ≈ 50.8 m/s² ≫ 2.0.
    control_points = np.zeros((1, 1, 6, 3), dtype=np.float64)
    control_points[0, 0] = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)

    out: dict = {}
    evaluator.evaluate(control_points, out)

    # Pre-condition: weryfikujemy że scenariusz spełnia założenia (geometric
    # constraint NIE wykrywa, ale fizyczne łamanie jest realne).
    diff1 = np.diff(control_points, axis=2)
    diff2 = np.diff(diff1, axis=2)
    max_diff1 = float(np.max(np.linalg.norm(diff1, axis=-1)))
    max_diff2 = float(np.max(np.linalg.norm(diff2, axis=-1)))
    min_diff1 = float(np.min(np.linalg.norm(diff1, axis=-1)))
    a_lat_physical = (params["cruise_speed"] ** 2) * max_diff2 / (min_diff1 ** 2)

    assert max_diff2 < params["max_accel_limit"], (
        f"Setup error: max(||diff2||)={max_diff2:.2f} powinien być < "
        f"max_accel_limit={params['max_accel_limit']} (geometric NIE łapie)."
    )
    assert a_lat_physical > params["max_accel"], (
        f"Setup error: a_lat_physical={a_lat_physical:.1f} powinno przekraczać "
        f"max_accel={params['max_accel']} (fizyczny constraint POWINIEN łapać)."
    )

    # GŁÓWNA ASERCJA (oczekiwana fail pre-fix):
    # G[2] musi być > 0 — kinematic_penalty wykrywa fizyczne naruszenie
    # acceleration limit. Obecnie G[2]=0 bo current constraint mierzy
    # tylko diff2 w przestrzeni geometrycznej (lenient).
    g_kinematic = float(out["G"][0, 2])
    assert g_kinematic > 0.0, (
        f"❌ Twardy kinematic constraint nie łapie naruszenia. "
        f"max(||diff2||)={max_diff2:.2f}, min(||diff1||)={min_diff1:.2f}, "
        f"a_lat_physical={a_lat_physical:.1f} m/s² (limit={params['max_accel']}). "
        f"G[2]={g_kinematic} (oczekiwane > 0). "
        "Pipeline pozwala optimizer'owi proponować trajektorie wymagające "
        "fizycznie niemożliwych accelerations. Required fix: sample-based "
        "kinematic check w `objective_constrains.py:evaluate`."
    )


@patch(f"{TARGET_MODULE}.calculate_dynamic_max_node_distance", return_value=50.0)
@patch(f"{TARGET_MODULE}.evaluate_bspline_trajectory_sync")
def test_kinematic_penalty_zero_for_smooth_trajectory(
    mock_sync, mock_calc, basic_params,
):
    """Komplementarny test: gładka trajektoria (kolinearne control points,
    zerowe ||diff2||) NIE narusza kinematic constraint. Po fix'ie G[2]
    powinno wciąż być 0 dla rozsądnych trajektorii — w przeciwnym razie
    sample-based constraint jest *za bardzo restrykcyjny* i odrzuca też
    feasible rozwiązania.
    """
    starts = np.array([[0.0, 0.0, 0.0]])
    targets = np.array([[100.0, 100.0, 0.0]])

    mock_sync.return_value = (
        np.array([[0.0]]),
        np.array([[10.0]]),
        np.array([0.0]),
    )

    params = dict(basic_params)
    params["max_accel_limit"] = 5.0
    params["max_accel"] = 2.0
    params["cruise_speed"] = 6.0

    evaluator = VectorizedEvaluator(None, starts, targets, 4, params)

    # Kolinearne control points na prostej z=1 — zerowa krzywizna,
    # zerowe ||diff2||, zerowa fizyczna lateral acceleration.
    control_points = np.zeros((1, 1, 6, 3), dtype=np.float64)
    xs = np.linspace(0.0, 25.0, 6)
    control_points[0, 0, :, 0] = xs
    control_points[0, 0, :, 2] = 1.0

    out: dict = {}
    evaluator.evaluate(control_points, out)

    g_kinematic = float(out["G"][0, 2])
    assert g_kinematic == 0.0, (
        f"Smooth straight trajectory generates kinematic_penalty={g_kinematic} "
        "≠ 0. Fix is too restrictive."
    )