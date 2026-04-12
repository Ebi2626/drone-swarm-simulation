import pytest
import numpy as np
from unittest.mock import patch

# UWAGA: Zmień 'src.trajectory.BSplineTrajectory' na faktyczną ścieżkę do pliku z klasą BSplineTrajectory.
from src.trajectory.BSplineTrajectory import BSplineTrajectory

# ==========================================
# FIXTURES (Przygotowanie środowiska testowego)
# ==========================================

@pytest.fixture
def sample_waypoints():
    """
    Zwraca proste, liniowe punkty w 3D na osi X.
    Krzywa k=3 (cubic) wymaga co najmniej 4 punktów wejściowych.
    Długość takiej trasy wynosi dokładnie 3.0 metry.
    """
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0]
    ])

@pytest.fixture
def mock_profile():
    """
    Mockujemy klasę TrapezoidalProfile, aby testy BSplineTrajectory 
    nie zależały od implementacji profilu prędkości.
    Patchujemy ją w module, w którym jest UŻYWANA (czyli w pliku BSplineTrajectory).
    """
    with patch('src.trajectory.BSplineTrajectory.TrapezoidalProfile') as MockClass:
        # Tworzymy atrapę instancji
        instance = MockClass.return_value
        instance.total_duration = 5.0
        yield instance

# ==========================================
# TESTY
# ==========================================

def test_initialization_and_arc_length(sample_waypoints, mock_profile):
    """
    Intencja: Sprawdzenie, czy dla prostych punktów długość łuku wylicza się
    poprawnie za pomocą całkowania numerycznego.
    """
    trajectory = BSplineTrajectory(sample_waypoints, cruise_speed=1.0, max_accel=0.5)
    
    # Oczekiwana długość prostej od (0,0,0) do (3,0,0) to 3.0.
    # Używamy pytest.approx z uwagi na drobne niedokładności całkowania numerycznego.
    assert trajectory.arc_length == pytest.approx(3.0, rel=1e-3)
    assert trajectory.total_duration == 5.0
    

def test_get_state_at_time_normal_movement(sample_waypoints, mock_profile):
    """
    Intencja: Poprawne zmapowanie dystansu z profilu na parametr 'u' krzywej 
    oraz poprawne wyliczenie wektora 3D prędkości ze stycznej.
    """
    trajectory = BSplineTrajectory(sample_waypoints, cruise_speed=1.0, max_accel=0.5)
    
    # Symulujemy, że dron jest w połowie trasy (dystans 1.5m), leci z prędkością 2.0 m/s
    mock_profile.get_state.return_value = (1.5, 2.0)
    
    pos, vel = trajectory.get_state_at_time(t_flight=2.5)
    
    # Pozycja powinna wynosić [1.5, 0.0, 0.0]
    np.testing.assert_allclose(pos, [1.5, 0.0, 0.0], atol=1e-3)
    # Prędkość wzdłuż osi X powinna mieć wartość skalaru speed (2.0)
    np.testing.assert_allclose(vel, [2.0, 0.0, 0.0], atol=1e-3)

def test_get_state_at_time_stopped_at_end(sample_waypoints, mock_profile):
    """
    Edge case: Dron doleciał do końca trasy (u >= 1.0).
    Zabezpiecza przed wyjazdem poza krzywą.
    """
    trajectory = BSplineTrajectory(sample_waypoints, cruise_speed=1.0, max_accel=0.5)
    
    # Dron przebył całą trasę (3.0m), z prędkością 0.0
    mock_profile.get_state.return_value = (3.0, 0.0)
    
    pos, vel = trajectory.get_state_at_time(t_flight=10.0)
    
    np.testing.assert_allclose(pos, [3.0, 0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(vel, [0.0, 0.0, 0.0], atol=1e-6)

def test_get_state_at_time_mid_flight_stop(sample_waypoints, mock_profile):
    """
    Edge case: Dron jest w trakcie lotu (u < 1.0), ale profil prędkości zwrócił 0 
    (np. gwałtowne zatrzymanie lub najechanie na przeszkodę).
    """
    trajectory = BSplineTrajectory(sample_waypoints, cruise_speed=1.0, max_accel=0.5)
    
    # Zatrzymanie w połowie (dystans 1.5, speed 0.0)
    mock_profile.get_state.return_value = (1.5, 0.0)
    
    pos, vel = trajectory.get_state_at_time(t_flight=3.0)
    
    # Prędkość powinna być rygorystycznie wyzerowana przez warunek `current_speed <= 1e-6`
    np.testing.assert_allclose(vel, [0.0, 0.0, 0.0], atol=1e-6)

def test_extremely_short_trajectory(mock_profile):
    """
    Edge case: Trasa składa się z punktów prawie nakładających się na siebie.
    Sprawdza, czy nie wystąpi błąd dzielenia przez zero przy mapowaniu 'u'.
    """
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [1e-8, 0.0, 0.0],
        [2e-8, 0.0, 0.0],
        [3e-8, 0.0, 0.0]
    ])
    
    trajectory = BSplineTrajectory(waypoints, cruise_speed=1.0, max_accel=0.5)
    
    # Profil mówi, że przebyliśmy ułamek milimetra, v=0
    mock_profile.get_state.return_value = (1e-8, 0.0)
    
    # Powinien wejść w instrukcję warunkową zabezpieczającą dzielenie:
    # if self.arc_length <= 1e-6: u = 1.0
    pos, vel = trajectory.get_state_at_time(t_flight=1.0)
    
    # Sprawdzamy czy poprawnie zabezpieczył się przed dzieleniem i rzucił pozycję końcową
    assert trajectory.arc_length <= 1e-6
    np.testing.assert_allclose(vel, [0.0, 0.0, 0.0], atol=1e-6)