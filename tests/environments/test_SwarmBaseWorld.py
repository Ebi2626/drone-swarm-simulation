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


# ==========================================
# TESTY LOGIKI DYNAMICZNYCH PRZESZKÓD
# ==========================================

@pytest.fixture
def dummy_world_with_obstacles(mock_world_data, mock_obstacles_data, mocker):
    """Świat z włączonymi dynamicznymi przeszkodami: 2 drony główne + 2 przeszkody."""
    mocker.patch("gym_pybullet_drones.envs.BaseAviary.BaseAviary.__init__", return_value=None)
    world = DummySwarmWorld(
        world_data=mock_world_data,
        obstacles_data=mock_obstacles_data,
        primary_num_drones=2,
        dynamic_obstacles_enabled=True,
        num_dynamic_obstacles=2,
    )
    world.GUI = False
    world.NUM_DRONES = 4
    world.MAX_RPM = 15000
    # Drony główne: 100, 101; przeszkody dynamiczne: 200, 201
    world.DRONE_IDS = [100, 101, 200, 201]
    return world


def test_dynamic_obstacles_enabled_fields(mock_world_data, mock_obstacles_data, mocker):
    """Po włączeniu flagi wszystkie pola są ustawione zgodnie z rolą agentów."""
    mocker.patch("gym_pybullet_drones.envs.BaseAviary.BaseAviary.__init__", return_value=None)
    world = DummySwarmWorld(
        world_data=mock_world_data,
        obstacles_data=mock_obstacles_data,
        primary_num_drones=2,
        dynamic_obstacles_enabled=True,
        num_dynamic_obstacles=2,
    )
    assert world.primary_num_drones == 2
    assert world.dynamic_obstacles_enabled is True
    assert world.num_dynamic_obstacles == 2
    # total = 2 (primary) + 2 (obstacles) = 4
    assert world.total_agents == 4


def test_dynamic_obstacles_disabled_zeroes_count(mock_world_data, mock_obstacles_data, mocker):
    """Gdy flaga False, liczba przeszkód to 0 nawet jeśli ktoś przekazał inną wartość."""
    mocker.patch("gym_pybullet_drones.envs.BaseAviary.BaseAviary.__init__", return_value=None)
    world = DummySwarmWorld(
        world_data=mock_world_data,
        obstacles_data=mock_obstacles_data,
        num_drones=3,
        dynamic_obstacles_enabled=False,
        num_dynamic_obstacles=5,  # ignorowane, bo flaga False
    )
    assert world.num_dynamic_obstacles == 0
    assert world.total_agents == 3


def test_raises_without_any_drone_count(mock_world_data, mock_obstacles_data, mocker):
    """Brak informacji o liczbie agentów to twardy błąd — chroni przed cichym pominięciem."""
    mocker.patch("gym_pybullet_drones.envs.BaseAviary.BaseAviary.__init__", return_value=None)
    with pytest.raises(ValueError, match="Brak informacji o liczbie agentów"):
        DummySwarmWorld(
            world_data=mock_world_data,
            obstacles_data=mock_obstacles_data,
        )


def test_agent_indices_without_dynamic_obstacles(dummy_world):
    """Bez przeszkód wszyscy agenci to drony główne, lista przeszkód jest pusta."""
    assert dummy_world.get_primary_agent_indices() == [0, 1]
    assert dummy_world.get_dynamic_obstacle_indices() == []
    assert dummy_world.is_dynamic_obstacle(0) is False
    assert dummy_world.is_dynamic_obstacle(1) is False


def test_agent_indices_with_dynamic_obstacles(dummy_world_with_obstacles):
    """Indeksy są pogrupowane: najpierw drony główne (0..N-1), potem przeszkody."""
    world = dummy_world_with_obstacles
    assert world.get_primary_agent_indices() == [0, 1]
    assert world.get_dynamic_obstacle_indices() == [2, 3]
    assert world.is_dynamic_obstacle(0) is False
    assert world.is_dynamic_obstacle(1) is False
    assert world.is_dynamic_obstacle(2) is True
    assert world.is_dynamic_obstacle(3) is True


def test_get_body_role_classifies_all_body_types(dummy_world_with_obstacles):
    """body_id jest poprawnie mapowany na rolę: drone / dynamic_obstacle / ground / ceiling / static_obstacle."""
    world = dummy_world_with_obstacles
    world.ground_body_id = 5
    world.ceiling_body_id = 6

    assert world.get_body_role(100) == "drone"
    assert world.get_body_role(101) == "drone"
    assert world.get_body_role(200) == "dynamic_obstacle"
    assert world.get_body_role(201) == "dynamic_obstacle"
    assert world.get_body_role(5) == "ground"
    assert world.get_body_role(6) == "ceiling"
    # Cokolwiek nieznanego — uznawane za statyczną przeszkodę świata
    assert world.get_body_role(999) == "static_obstacle"


def test_detailed_collisions_filters_obstacles_by_default(dummy_world_with_obstacles, mocker):
    """Domyślnie get_detailed_collisions zgłasza kolizje WYŁĄCZNIE dla dronów głównych."""
    world = dummy_world_with_obstacles
    mock_p = mocker.patch(f"{TARGET_MODULE}.p")

    def mock_cp(bodyA):
        # Dron 0 (100) oraz przeszkoda 0 (200) uderzają w obiekt 999
        if bodyA in (100, 200):
            return [(0, 0, 999, "kontakt")]
        return []

    mock_p.getContactPoints.side_effect = mock_cp

    collisions = world.get_detailed_collisions()
    # Tylko dron główny o indeksie 0; przeszkoda (indeks 2) jest pominięta
    assert collisions == [(0, 999)]


def test_detailed_collisions_with_obstacles_when_requested(dummy_world_with_obstacles, mocker):
    """Z flagą include_dynamic_obstacles=True sprawdzamy wszystkich agentów."""
    world = dummy_world_with_obstacles
    mock_p = mocker.patch(f"{TARGET_MODULE}.p")

    def mock_cp(bodyA):
        if bodyA in (100, 200):
            return [(0, 0, 999, "kontakt")]
        return []

    mock_p.getContactPoints.side_effect = mock_cp

    collisions = sorted(world.get_detailed_collisions(include_dynamic_obstacles=True))
    # Oba: dron 0 (idx 0) oraz przeszkoda 0 (idx 2)
    assert collisions == [(0, 999), (2, 999)]


def test_action_and_observation_space_scale_with_total_agents(dummy_world_with_obstacles):
    """Kształt przestrzeni akcji i obserwacji odzwierciedla total_agents, nie primary_num_drones."""
    world = dummy_world_with_obstacles
    assert world._actionSpace().shape == (4, 4)
    assert world._observationSpace().shape == (4, 20)