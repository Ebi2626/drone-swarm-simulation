import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.algorithms.avoidance.EvasionContextBuilder import EvasionContextBuilder
from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import KinematicState, ThreatAlert

@pytest.fixture
def base_spline_mock():
    """Mock bazowego splinu — flat API NumbaTrajectoryProfile (no nested .profile)."""
    mock_spline = MagicMock()
    mock_spline.arc_length = 100.0
    mock_spline.cruise_speed = 8.0
    mock_spline.max_accel = 2.0
    return mock_spline

@pytest.fixture
def drone_state():
    """Podstawowy stan kinematyczny drona (w locie z prędkością 10 m/s)."""
    return KinematicState(
        position=np.array([0.0, 0.0, 5.0]),
        velocity=np.array([10.0, 0.0, 0.0]),
        radius=0.4
    )

@pytest.fixture
def env_bounds():
    """Standardowe granice świata."""
    return (np.array([0.0, -10.0, 0.0]), np.array([50.0, 10.0, 10.0]))


@patch("src.algorithms.avoidance.EvasionContextBuilder.splev")
def test_build_dynamic_search_space_includes_velocity_obstacle(mock_splev, drone_state, base_spline_mock, env_bounds):
    """
    Weryfikuje najważniejszą modyfikację (Velocity Obstacles).
    Sprawdza, czy Search Space Bounding Box został sztucznie rozszerzony o 
    przewidywaną (przyszłą) pozycję przeszkody w horyzoncie t_max.
    """
    # Mock splev używanego do _sample_base_at_arc i _compute_forward_direction
    # Ustalamy że Rejoin Point będzie w [20.0, 0.0, 5.0] a kierunek to [1.0, 0.0, 0.0]
    mock_splev.side_effect = [
        np.array([20.0, 0.0, 5.0]), # Zwrot dla _sample_base_at_arc
        np.array([1.0, 0.0, 0.0])   # Zwrot dla _compute_forward_direction (der=1)
    ]
    
    # Przeszkoda przed dronem, poruszająca się z dużą prędkością prostopadle do drona!
    # Leci wzdłuż osi Y z prędkością 10 m/s
    obs_state = KinematicState(
        position=np.array([15.0, 0.0, 5.0]),
        velocity=np.array([0.0, 10.0, 0.0]),
        radius=0.5
    )
    threat = ThreatAlert(
        obstacle_state=obs_state,
        distance=15.0,
        time_to_collision=1.5,
        relative_velocity=np.array([10.0, -10.0, 0.0])
    )
    
    # t_max = 3.0 sekundy; wyłączamy adaptacyjny bufor (gain=0) by test sprawdzał
    # czystą geometrię VO niezależnie od skalowania do |rel_vel| (Faza 2.5 planu).
    # `lateral_max_offset_m=0` wyłącza nowy cap (Bug #2 Krok 3c) — ten test
    # weryfikuje przed-capowy bbox VO; cap jest pokryty osobnym testem.
    builder = EvasionContextBuilder(
        t_min=1.0, t_max=3.0,
        floor_margin=0.0, ceiling_margin=0.0,
        margin_velocity_gain=0.0,
        lateral_max_offset_m=0.0,
    )

    context = builder.build(
        drone_id=1,
        current_time=10.0,
        drone_state=drone_state,
        threat=threat,
        base_spline=base_spline_mock,
        base_arc_progress=5.0,
        env_bounds=env_bounds
    )

    # --- WERYFIKACJA LOGIKI VO (Velocity Obstacles) ---
    # Przewidywana pozycja przeszkody po czasie t_max (3.0s):
    # future_pos = [15.0, 0.0, 5.0] + [0.0, 10.0, 0.0] * 3.0 = [15.0, 30.0, 5.0]
    # To znacznie poza zadeklarowanym Bounding Boxem świata! (Y max to 10.0)

    bbox_max = context.search_space_max

    # Ponieważ przeszkoda "odlatuje" poza granicę świata (Y=30.0, ale granica env to Y=10.0),
    # Builder musi przyciąć BoundingBox do granicy świata i bufora.
    # Wartość Y_max nie może przekroczyć env_bounds[1][1] (10.0) pomniejszonego o bufor promienia (1.5)
    assert bbox_max[1] == pytest.approx(10.0 - (0.5 + 1.0)) # 8.5
    
    # Bounding Box powinien sięgnąć co najmniej punktu powrotu (X=20.0)
    assert bbox_max[0] >= 20.0


@patch("src.algorithms.avoidance.EvasionContextBuilder.splev")
def test_evaluate_collision_risk(mock_splev, drone_state, base_spline_mock, env_bounds):
    """
    Weryfikacja funkcji oceny ryzyka (fitness function), która jest podstawą 
    działania algorytmów roju (PSO, SPSA) w nowej architekturze CCOV.
    """
    mock_splev.side_effect = [
        np.array([20.0, 0.0, 5.0]),
        np.array([1.0, 0.0, 0.0])
    ]
    
    # Przeszkoda leci z prędkością 2 m/s w osi X
    obs_state = KinematicState(
        position=np.array([10.0, 0.0, 5.0]),
        velocity=np.array([2.0, 0.0, 0.0]),
        radius=0.5
    )
    threat = ThreatAlert(
        obstacle_state=obs_state,
        distance=10.0,
        time_to_collision=1.0,
        relative_velocity=np.array([8.0, 0.0, 0.0])
    )
    
    builder = EvasionContextBuilder(t_max=4.0)
    context = builder.build(
        drone_id=1,
        current_time=10.0,
        drone_state=drone_state,
        threat=threat,
        base_spline=base_spline_mock,
        base_arc_progress=0.0,
        env_bounds=env_bounds
    )
    
    # Próbkujemy punkt [14.0, 0.0, 5.0] z przesunięciem czasu = 2.0s
    # Gdzie wtedy będzie przeszkoda?
    # future_obs_pos = [10.0, 0.0, 5.0] + [2.0, 0.0, 0.0] * 2.0 = [14.0, 0.0, 5.0]
    
    # Odległość między kandydującym punktem a przyszłą pozycją przeszkody powinna wynieść 0!
    risk_distance = context.evaluate_collision_risk(candidate_pos=np.array([14.0, 0.0, 5.0]), time_offset=2.0)
    
    assert risk_distance == pytest.approx(0.0)
    
    # Próbkujemy inny punkt [14.0, 5.0, 5.0] przy tym samym czasie
    risk_distance_safe = context.evaluate_collision_risk(candidate_pos=np.array([14.0, 5.0, 5.0]), time_offset=2.0)
    
    # Powinno wynieść równe 5.0m
    assert risk_distance_safe == pytest.approx(5.0)

@patch("src.algorithms.avoidance.EvasionContextBuilder.splev")
def test_lateral_max_offset_m_caps_search_bbox(mock_splev, drone_state, base_spline_mock, env_bounds):
    """Krok 3c (Bug #2): `lateral_max_offset_m` ogranicza lateralne rozszerzenie BBOX-u
    przez VO/obs_future. Bez capa BBOX rośnie z `obs_pos + obs_vel * t_max`,
    AStar wybiera waypointy odlegające o dziesiątki metrów → ostre zakrzywienia.
    """
    mock_splev.side_effect = [
        np.array([20.0, 0.0, 5.0]),
        np.array([1.0, 0.0, 0.0]),
    ]

    obs_state = KinematicState(
        position=np.array([15.0, 0.0, 5.0]),
        velocity=np.array([0.0, 10.0, 0.0]),  # przeszkoda leci wzdłuż Y
        radius=0.5,
    )
    threat = ThreatAlert(
        obstacle_state=obs_state,
        distance=15.0,
        time_to_collision=1.5,
        relative_velocity=np.array([10.0, -10.0, 0.0]),
    )

    # Drone leci wzdłuż X (forward_xy ≈ [1,0,0] → lateral_xy ≈ [0,1,0]),
    # więc cap powinien clamp'ować Y BBOX-u (axis dominujący lateralu = 1).
    builder = EvasionContextBuilder(
        t_min=1.0, t_max=3.0,
        floor_margin=0.0, ceiling_margin=0.0,
        margin_velocity_gain=0.0,
        lateral_max_offset_m=5.0,
    )

    context = builder.build(
        drone_id=1,
        current_time=10.0,
        drone_state=drone_state,
        threat=threat,
        base_spline=base_spline_mock,
        base_arc_progress=5.0,
        env_bounds=env_bounds,
    )

    # Drone Y = 0.0, cap 5.0 → BBOX Y nie może przekroczyć ±5.0 od drona.
    # (Z drobnym marginesem +1m na re-inkluzję rejoin/drone — patrz `keep_pt`.)
    assert context.search_space_max[1] <= 5.0 + 1e-6, (
        f"cap nie zadziałał: bbox_max[1]={context.search_space_max[1]:.2f}"
    )
    assert context.search_space_min[1] >= -5.0 - 1e-6, (
        f"cap nie zadziałał: bbox_min[1]={context.search_space_min[1]:.2f}"
    )
    # Forward (X) NIE jest cap'owany — rejoin point musi być osiągalny.
    assert context.search_space_max[0] >= 20.0


@patch("src.algorithms.avoidance.EvasionContextBuilder.splev")
def test_build_with_real_numba_trajectory_profile(mock_splev, drone_state, env_bounds):
    """Regresja (2026-04-29): builder.build() musi czytać `cruise_speed` z NumbaTrajectoryProfile
    bezpośrednio (atrybut top-level), a NIE z nieistniejącego `.profile.cruise_speed`.

    Pre-fix crashował z `AttributeError: 'NumbaTrajectoryProfile' object has no attribute 'profile'`
    gdy LIDAR wykrył pierwszą dynamiczną przeszkodę (stack: main.py:318 → compute_actions
    → _run_lidar_and_detect → _maybe_trigger_evasion → builder.build → linia 165).
    """
    from src.algorithms.abstraction.trajectory.strategies.shared.NumbaTrajectoryProfile import (
        NumbaTrajectoryProfile,
    )

    mock_splev.side_effect = [
        np.array([20.0, 0.0, 5.0]),
        np.array([1.0, 0.0, 0.0]),
    ]

    real_profile = NumbaTrajectoryProfile(
        waypoints=np.array([[0.0, 0.0, 5.0], [25.0, 0.0, 5.0]], dtype=np.float64),
        cruise_speed=8.0,
        max_accel=2.0,
    )
    # Builder pyta o `arc_length` w `_sample_base_at_arc` → mock by uniknąć splev fixturowania.
    real_profile.arc_length = float(real_profile.total_distance)

    obs_state = KinematicState(
        position=np.array([15.0, 0.0, 5.0]),
        velocity=np.array([0.0, 5.0, 0.0]),
        radius=0.5,
    )
    threat = ThreatAlert(
        obstacle_state=obs_state,
        distance=15.0,
        time_to_collision=2.0,
        relative_velocity=np.array([10.0, -5.0, 0.0]),
    )

    builder = EvasionContextBuilder(t_min=1.0, t_max=3.0, margin_velocity_gain=0.0)

    # Nie powinno crashować na AttributeError.
    context = builder.build(
        drone_id=0,
        current_time=10.0,
        drone_state=drone_state,
        threat=threat,
        base_spline=real_profile,
        base_arc_progress=0.0,
        env_bounds=env_bounds,
    )

    assert context is not None
    assert context.search_space_max[0] >= 20.0


@patch("src.algorithms.avoidance.EvasionContextBuilder.splev")
def test_static_methods_without_spline_mock(mock_splev):
    """Weryfikuje _compute_forward_direction z wbudowanym zabezpieczeniem na niskie prędkości."""
    
    # Zabezpieczamy zewnętrzne wywołanie biblioteki Scipy
    mock_splev.return_value = np.array([1.0, 0.0, 0.0])
    
    builder = EvasionContextBuilder()
    
    # Kiedy dron stoi, funkcja musi zfallbackować do wektora stycznej bazy,
    # ale tu testujemy wyjście gdy B-Spline też jest zepsuty/pusty.
    mock_bad_spline = MagicMock()
    mock_bad_spline.arc_length = 0.0
    
    # Dron w hover (prędkość: norm([0.1, 0.1, 0.1]) ≈ 0.17 < 0.5)
    fwd = builder._compute_forward_direction(np.array([0.1, 0.1, 0.1]), mock_bad_spline, 0.0)
    
    # Powinno zwrócić to, co wymusiliśmy mockiem splev (znormalizowane [1,0,0])
    np.testing.assert_array_equal(fwd, np.array([1.0, 0.0, 0.0]))
    
    # Gdy dron leci szybko (> 0.5 m/s) w osi Y
    # Algorytm w ogóle nie powinien pytać splinu, tylko użyć wektora prędkości!
    fwd_fast = builder._compute_forward_direction(np.array([0.0, 10.0, 0.0]), mock_bad_spline, 0.0)
    
    # Znormalizowana prędkość [0, 10, 0] to [0, 1, 0]
    np.testing.assert_array_equal(fwd_fast, np.array([0.0, 1.0, 0.0]))
    
    # Dodatkowo możemy upewnić się, że dla szybkiego drona splev nie był wołany po raz drugi
    assert mock_splev.call_count == 1