import pytest
import numpy as np

# UWAGA: Upewnij się, że ścieżka importu jest zgodna z układem Twojego projektu
from src.trajectory.TrapezoidalProfile import TrapezoidalProfile

# ==========================================
# TESTY INICJALIZACJI I OBLICZANIA PROFILU
# ==========================================

def test_trapezoidal_profile_computation():
    """
    Intencja: Sprawdzenie poprawności wyliczania czasów i dystansów dla klasycznego 
    profilu trapezowego (jest wystarczająco dużo miejsca na osiągnięcie v_cruise).
    """
    # Dystans: 10m, v_cruise: 2 m/s, max_accel: 1 m/s^2
    # Oczekiwane: 
    # Przyspieszanie (t_a): do v=2 zajmie 2s. Pokonany dystans (s = 0.5*a*t^2) = 2m.
    # Zwalnianie (t_d): zajmie 2s, dystans = 2m.
    # Cruise (s_c): 10 - 2 - 2 = 6m. Czas cruise (t_c) = 6 / 2 = 3s.
    # Czas całkowity: 2 + 3 + 2 = 7s.
    
    profile = TrapezoidalProfile(total_distance=10.0, cruise_speed=2.0, max_accel=1.0)
    
    assert profile.t_a == 2.0
    assert profile.s_a == 2.0
    assert profile.t_c == 3.0
    assert profile.s_c == 6.0
    assert profile.t_d == 2.0
    assert profile.v_peak == 2.0
    assert profile.total_duration == 7.0

def test_triangular_profile_computation():
    """
    Intencja: Sprawdzenie profilu trójkątnego. Dystans jest zbyt krótki, 
    aby osiągnąć zadaną prędkość przelotową (brak fazy cruise).
    """
    # Dystans: 2m, v_cruise: 5 m/s, max_accel: 1 m/s^2
    # Przyspieszanie na dystansie s_a = 1m z a=1 da v_peak = sqrt(2 * a * s) = sqrt(2)
    
    profile = TrapezoidalProfile(total_distance=2.0, cruise_speed=5.0, max_accel=1.0)
    
    assert profile.s_a == 1.0
    assert profile.s_c == 0.0
    assert profile.t_c == 0.0
    assert profile.v_peak == pytest.approx(np.sqrt(2.0))
    # Czas całkowity to 2 * czas przyspieszania
    expected_t_a = np.sqrt(2.0) / 1.0
    assert profile.total_duration == pytest.approx(2 * expected_t_a)

def test_zero_distance_profile():
    """
    Edge case: Całkowity dystans wynosi 0 (lub jest ekstremalnie mały).
    """
    profile = TrapezoidalProfile(total_distance=1e-7, cruise_speed=2.0, max_accel=1.0)
    
    assert profile.total_duration == 0.0
    assert profile.v_peak == 0.0

# ==========================================
# TESTY METODY GET_STATE (Stany w czasie)
# ==========================================

def test_get_state_acceleration_phase():
    """
    Intencja: Sprawdzenie pozycji i prędkości w fazie przyspieszania.
    Dla profilu: dist=10, v=2, a=1 (t_a = 2s)
    """
    profile = TrapezoidalProfile(total_distance=10.0, cruise_speed=2.0, max_accel=1.0)
    
    # W czasie t=1.0s: v = a*t = 1.0, s = 0.5*a*t^2 = 0.5
    dist, speed = profile.get_state(t=1.0)
    
    assert speed == pytest.approx(1.0)
    assert dist == pytest.approx(0.5)

def test_get_state_cruise_phase():
    """
    Intencja: Sprawdzenie pozycji i prędkości w fazie stałej prędkości.
    Dla profilu: dist=10, v=2, a=1 (t_a = 2s, t_c = 3s)
    """
    profile = TrapezoidalProfile(total_distance=10.0, cruise_speed=2.0, max_accel=1.0)
    
    # W czasie t=3.5s (1.5s po rozpoczęciu fazy cruise):
    # s = s_a + v_cruise * dt = 2.0 + 2.0 * 1.5 = 5.0
    dist, speed = profile.get_state(t=3.5)
    
    assert speed == pytest.approx(2.0)
    assert dist == pytest.approx(5.0)

def test_get_state_deceleration_phase():
    """
    Intencja: Sprawdzenie pozycji i prędkości w fazie hamowania.
    Dla profilu: dist=10, v=2, a=1 (zaczyna hamować w t=5.0s, kończy w 7.0s)
    """
    profile = TrapezoidalProfile(total_distance=10.0, cruise_speed=2.0, max_accel=1.0)
    
    # W czasie t=6.0s (1.0s po rozpoczęciu zwalniania):
    # speed = 2.0 - 1.0 * 1.0 = 1.0
    # dist = (s_a + s_c) + v_0*dt - 0.5*a*dt^2 = 8.0 + 2.0(1) - 0.5(1)(1)^2 = 9.5
    dist, speed = profile.get_state(t=6.0)
    
    assert speed == pytest.approx(1.0)
    assert dist == pytest.approx(9.5)

def test_get_state_out_of_bounds_time():
    """
    Edge case: Żądanie stanu dla czasu poniżej zera lub przekraczającego czas misji.
    """
    profile = TrapezoidalProfile(total_distance=10.0, cruise_speed=2.0, max_accel=1.0)
    
    # Czas ujemny powinien być przycięty do 0.0
    dist_neg, speed_neg = profile.get_state(t=-2.0)
    assert dist_neg == 0.0
    assert speed_neg == 0.0
    
    # Czas powyżej total_duration (7.0s) powinien zwrócić koniec trasy i zatrzymanie
    dist_over, speed_over = profile.get_state(t=10.0)
    assert dist_over == profile.total_distance
    assert speed_over == 0.0

def test_get_state_precision_errors():
    """
    Edge case: Błędy zmiennoprzecinkowe przy samym końcu zwalniania.
    Zabezpieczenia w kodzie gwarantują nierzekraczanie total_distance i speed >= 0.
    """
    profile = TrapezoidalProfile(total_distance=10.0, cruise_speed=2.0, max_accel=1.0)
    
    # Przekazujemy czas o epsilon ułamek sekundy mniejszy/większy od końca trasy
    dist, speed = profile.get_state(t=6.99999999999)
    assert speed >= 0.0
    assert dist <= 10.0