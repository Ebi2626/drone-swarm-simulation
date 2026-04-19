import pytest
import numpy as np
from unittest.mock import MagicMock
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.abstraction.generate_obstacles import ObstaclesData

import importlib

TARGET_MODULE = "src.environments.SwarmBaseWorld"
SwarmBaseWorld = importlib.import_module(TARGET_MODULE).SwarmBaseWorld


# ==========================================
# FIXTURES (Przygotowanie)
# ==========================================

class DummySwarmWorld(SwarmBaseWorld):
    """
    Ponieważ SwarmBaseWorld ma metodę abstrakcyjną (draw_obstacles),
    musimy stworzyć tę sztuczną klasę potomną, aby w ogóle móc ją zainicjować w testach.
    """
    def draw_obstacles(self) -> None:
        pass


@pytest.fixture
def mock_world_data():
    return WorldData(
        dimensions=np.array([10.0, 20.0, 5.0]),  # width=10, length=20, height=5
        min_bounds=np.array([0.0, 0.0, 0.0]),    # ground_position = 0.0
        max_bounds=np.array([10.0, 20.0, 5.0]),
        bounds=np.zeros((3, 2)),
        center=np.array([5.0, 10.0, 2.5])
    )

@pytest.fixture
def mock_obstacles_data():
    return MagicMock(spec=ObstaclesData)

@pytest.fixture
def dummy_world(mock_world_data, mock_obstacles_data, mocker):
    """
    Inicjalizuje środowisko unikając uruchamiania faktycznego PyBulleta.
    """
    # 1. Zatrzymujemy BaseAviary przed próbą łączenia się z silnikiem fizyki
    mocker.patch("gym_pybullet_drones.envs.BaseAviary.BaseAviary.__init__", return_value=None)
    
    # 2. Tworzymy naszą atrapę świata
    world = DummySwarmWorld(
        world_data=mock_world_data,
        obstacles_data=mock_obstacles_data,
        num_drones=2,
        fake_argument=123 # Testujemy od razu odfiltrowanie nieznanych kwargs
    )
    
    # 3. Dodajemy parametry, które normalnie utworzyłby BaseAviary.__init__
    world.GUI = False
    world.NUM_DRONES = 2
    world.MAX_RPM = 15000
    # Zakładamy, że PyBullet nadał dronom ID fizyczne: 101 i 102
    world.DRONE_IDS = [101, 102] 
    
    return world

# ==========================================
# TESTY GEOMETRII (Podłoga i Sufit)
# ==========================================

def test_create_ground(dummy_world, mocker):
    """
    Intencja: Sprawdzenie, czy wymiary podłogi oraz punkt jej zespawnowania (createMultiBody)
    są poprawnie wyliczane z danych WorldData z uwzględnieniem offsetu grubości i marginesów.
    """
    mock_p = mocker.patch(f"{TARGET_MODULE}.p")
    mock_p.createCollisionShape.return_value = 1
    mock_p.createVisualShape.return_value = 2

    # Wywołujemy tworzenie ziemi o grubości 0.2
    dummy_world._create_ground(thickness=0.2)

    # Z WorldData: width=10.0, length=20.0, ground_position=0.0
    # Oczekiwane środki na osiach: X = 10/2 = 5.0, Y = 20/2 = 10.0
    # Oczekiwana pozycja Z: ground_position - (thickness / 2.0) = 0.0 - 0.1 = -0.1
    expected_pos = [5.0, 10.0, -0.1]
    
    # Sprawdzamy czy MultiBody zostało postawione w odpowiednim miejscu
    mock_p.createMultiBody.assert_called_once_with(0, 1, 2, expected_pos)
    
    # Dodatkowo możemy sprawdzić halfExtents z marginesem 100
    # half_wid = (10 + 100)/2 = 55.0, half_len = (20 + 100)/2 = 60.0
    kwargs = mock_p.createCollisionShape.call_args[1]
    assert kwargs["halfExtents"] == [55.0, 60.0, 0.1] # 0.1 bo thickness/2

def test_create_ceiling(dummy_world, mocker):
    """
    Intencja: Sprawdzenie, czy sufit ląduje na poprawnej wysokości 
    bazując na max_bounds/dimensions.
    """
    mock_p = mocker.patch(f"{TARGET_MODULE}.p")
    mock_p.createCollisionShape.return_value = 3
    mock_p.createVisualShape.return_value = 4

    dummy_world._create_ceiling()

    # Z WorldData: width=10, length=20, height=5.0
    expected_pos = [5.0, 10.0, 5.0]
    
    mock_p.createMultiBody.assert_called_once_with(0, 3, 4, expected_pos)

# ==========================================
# TESTY LOGIKI KOLIZJI
# ==========================================

def test_get_detailed_collisions(dummy_world, mocker):
    """
    Intencja: Weryfikacja, czy algorytm poprawnie zgłasza kolizje, ignoruje
    zderzenia drona z samym sobą oraz odpowiednio deduplikuje punkty styku.
    """
    mock_p = mocker.patch(f"{TARGET_MODULE}.p")

    # Tworzymy fałszywą funkcję zwracającą punkty kolizji zależnie od zadanego ID ciała
    def mock_getContactPoints(bodyA):
        if bodyA == 101: # Dron 0
            return [
                # Format krotki w pybullet (bodyA, bodyB(ignorowane), bodyC(target), ...)
                (0, 0, 999, "punkt styku 1"), # Kolizja z sufitem (ID 999)
                (0, 0, 999, "punkt styku 2"), # Duplikat (np. uderzenie dwoma śmigłami naraz)
                (0, 0, 101, "punkt styku 3")  # Dron uderzył w siebie samego (powinno być zignorowane)
            ]
        elif bodyA == 102: # Dron 1
            return [] # Brak kolizji
        return []

    mock_p.getContactPoints.side_effect = mock_getContactPoints

    # Wykonujemy testowaną logikę
    collisions = dummy_world.get_detailed_collisions()

    # Oczekujemy dokładnie 1 kolizji: Dron nr 0 (index 0) uderzył w obiekt 999
    assert len(collisions) == 1
    assert collisions[0] == (0, 999)

# ==========================================
# TESTY PRZESTRZENI GYMNASIUM
# ==========================================

def test_action_space_shape(dummy_world):
    """
    Intencja: Poprawność definicji przestrzeni akcji dla NUM_DRONES = 2.
    """
    space = dummy_world._actionSpace()
    # Oczekiwany kształt boxa: 2 drony, każdy po 4 silniki = kształt (2, 4)
    assert space.shape == (2, 4)
    assert space.high[0][0] == 15000 # MAX_RPM

def test_observation_space_shape(dummy_world):
    """
    Intencja: Poprawność definicji przestrzeni obserwacji.
    Zgodnie z kodem powinno być (NUM_DRONES, 20).
    """
    space = dummy_world._observationSpace()
    assert space.shape == (2, 20)