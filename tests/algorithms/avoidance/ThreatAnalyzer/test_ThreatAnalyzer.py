import pytest
import numpy as np
from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import ThreatAnalyzer, KinematicState, ThreatAlert

class MockLidarHit:
    """Mock struktury LidarHit do testowania niezależnie od PyBullet."""
    def __init__(self, distance: float, hit_position: list, velocity: list):
        self.distance = distance
        self.hit_position = np.array(hit_position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)

@pytest.fixture
def analyzer():
    """Domyślna instancja analizatora dla scenariuszy akademickich."""
    return ThreatAnalyzer(trigger_ttc=1.5, trigger_dist=6.0, critical_dist=25.0)

def test_no_hits(analyzer):
    """Brak wyników z Lidaru powinien zwracać None."""
    drone_state = KinematicState(
        position=np.array([0, 0, 0]), 
        velocity=np.array([5, 0, 0]), 
        radius=0.4
    )
    assert analyzer.analyze([], drone_state) is None

def test_ignore_hits_beyond_critical_distance(analyzer):
    """Zagrożenia poza zasięgiem krytycznym (critical_dist) muszą być ignorowane."""
    drone_state = KinematicState(
        position=np.array([0, 0, 0]), 
        velocity=np.array([5, 0, 0]), 
        radius=0.4
    )
    hit = MockLidarHit(distance=30.0, hit_position=[30, 0, 0], velocity=[0, 0, 0])
    
    assert analyzer.analyze([hit], drone_state) is None

def test_ignore_fleeing_obstacle(analyzer):
    """
    Test scenariusza 'fleeing obstacle' (przeszkoda uciekająca).
    Zgodnie z teorią Velocity Obstacles, jeśli prędkość zamykania (closing speed)
    jest ujemna lub zerowa, nie ma ryzyka kolizji czołowej.
    """
    drone_state = KinematicState(
        position=np.array([0, 0, 0]), 
        velocity=np.array([5, 0, 0]), 
        radius=0.4
    )
    # Przeszkoda leci w tym samym kierunku, ale szybciej (10 m/s > 5 m/s drona)
    hit = MockLidarHit(distance=10.0, hit_position=[10, 0, 0], velocity=[10, 0, 0])
    
    assert analyzer.analyze([hit], drone_state) is None

def test_head_on_collision_triggers_ttc(analyzer):
    """
    Klasyczny scenariusz kolizji czołowej (head-on collision).
    Prędkości się sumują, co powinno drastycznie zmniejszyć TTC i wyzwolić unik.
    """
    drone_state = KinematicState(
        position=np.array([0, 0, 0]), 
        velocity=np.array([5, 0, 0]), 
        radius=0.4
    )
    # Przeszkoda leci prosto na drona (prędkość względna = 10 m/s)
    hit = MockLidarHit(distance=10.0, hit_position=[10, 0, 0], velocity=[-5, 0, 0])
    
    alert = analyzer.analyze([hit], drone_state)
    
    assert alert is not None
    assert isinstance(alert, ThreatAlert)
    # Dystans 10m / prędkość względna 10m/s = TTC 1.0s
    assert alert.time_to_collision == pytest.approx(1.0)
    np.testing.assert_array_almost_equal(alert.relative_velocity, np.array([10, 0, 0]))

def test_static_obstacle_trigger_distance(analyzer):
    """
    Test wyzwolenia przestrzennego. Jeśli dron leci bardzo wolno, TTC może być duże,
    ale jeśli dystans spadnie poniżej trigger_dist (6.0m), system musi zareagować.
    """
    drone_state = KinematicState(
        position=np.array([0, 0, 0]), 
        velocity=np.array([1, 0, 0]), 
        radius=0.4
    )
    # Przeszkoda statyczna z odległości 4m. TTC = 4.0s (większe niż trigger_ttc=1.5), 
    # ale dystans 4.0m jest mniejszy niż trigger_dist=6.0
    hit = MockLidarHit(distance=4.0, hit_position=[4, 0, 0], velocity=[0, 0, 0])
    
    alert = analyzer.analyze([hit], drone_state)
    
    assert alert is not None
    assert alert.distance == 4.0
    assert alert.time_to_collision == pytest.approx(4.0)

def test_select_most_critical_threat(analyzer):
    """
    Rozdzielczość wielocelowa (Multi-target resolution).
    Algorytm musi wybrać przeszkodę o najmniejszym TTC, a nie tę położoną fizycznie najbliżej.
    To fundamentalne wymaganie dla algorytmów roju w środowiskach o wysokiej dynamice.
    """
    drone_state = KinematicState(
        position=np.array([0, 0, 0]), 
        velocity=np.array([5, 0, 0]), 
        radius=0.4
    )
    
    # Zagrożenie 1: Bardzo blisko (8m), ale wolno się zbliża (rel_vel = 4 m/s) -> TTC = 2.0s
    hit_1 = MockLidarHit(distance=8.0, hit_position=[8, 0, 0], velocity=[1, 0, 0])
    
    # Zagrożenie 2: Dalej (12m), ale leci z ogromną prędkością na drona (rel_vel = 15 m/s) -> TTC = 0.8s
    hit_2 = MockLidarHit(distance=12.0, hit_position=[12, 0, 0], velocity=[-10, 0, 0])
    
    alert = analyzer.analyze([hit_1, hit_2], drone_state)
    
    assert alert is not None
    assert alert.distance == 12.0
    assert alert.time_to_collision == pytest.approx(0.8)
    np.testing.assert_array_almost_equal(alert.relative_velocity, np.array([15, 0, 0]))

def test_oblique_collision_path(analyzer):
    """
    Test trajektorii ukośnej. Sprawdza poprawność rzutowania wektorów 
    (iloczyn skalarny) do obliczania closing_speed.
    Prędkość została dobrana tak, by TTC spadło poniżej progu 1.5s.
    """
    drone_state = KinematicState(
        position=np.array([0, 0, 0]), 
        velocity=np.array([10, 0, 0]), # Zwiększamy prędkość drona do 10 m/s
        radius=0.4
    )
    # Przeszkoda na osi X, poruszająca się w osi Y.
    # Wektor do przeszkody: [1, 0, 0].
    # Relatywna prędkość: [10 - 0, 0 - 5, 0] = [10, -5, 0].
    # Closing speed (iloczyn skalarny): 10.0 m/s.
    # TTC = 10.0 m / 10.0 m/s = 1.0 s.
    hit = MockLidarHit(distance=10.0, hit_position=[10, 0, 0], velocity=[0, 5, 0])
    
    alert = analyzer.analyze([hit], drone_state)
    
    # Przy TTC = 1.0s (< 1.5s) alarm MUSI zostać wyzwolony
    assert alert is not None
    assert isinstance(alert, ThreatAlert)
    assert alert.time_to_collision == pytest.approx(1.0)
    np.testing.assert_array_almost_equal(alert.relative_velocity, np.array([10, -5, 0]))