import pytest
from unittest.mock import patch

# UWAGA: Podmień "src.input.handler" na ścieżkę do pliku z InputHandler.
from src.utils.input_utils import InputHandler, CommandType

# ==========================================
# FIXTURES (Zastępowanie PyBulleta)
# ==========================================

@pytest.fixture
def mock_p():
    """
    Mockuje moduł 'p' (pybullet) w pliku, z którego importujemy InputHandler.
    Ustawiamy w nim sztuczne stałe klawiszy, aby operacje bitowe (&) działały poprawnie.
    """
    # Zmień 'src.input.handler.p' na dokładną ścieżkę do modułu, gdzie żyje kod
    with patch('src.utils.input_utils.p') as mock_pybullet:
        mock_pybullet.KEY_WAS_TRIGGERED = 1
        mock_pybullet.KEY_IS_DOWN = 2
        mock_pybullet.KEY_WAS_RELEASED = 4
        yield mock_pybullet

# ==========================================
# TESTY
# ==========================================

def test_no_keys_pressed(mock_p):
    """ Intencja: Gdy nikt nie naciska klawiatury, nie zwracamy żadnej komendy. """
    mock_p.getKeyboardEvents.return_value = {}
    
    handler = InputHandler(num_drones=5)
    command = handler.get_command()
    
    assert command is None

def test_spacebar_toggles_simulation(mock_p):
    """ Intencja: Wciśnięcie spacji uruchamia/pauzuje symulację. """
    # Słownik symulujący, że kody klawisza spacji mają ustawiony bit 'triggered' (1)
    mock_p.getKeyboardEvents.return_value = {ord(' '): 1}
    
    handler = InputHandler(num_drones=5)
    command = handler.get_command()
    
    assert command is not None
    assert command.type == CommandType.TOGGLE_SIMULATION
    assert command.payload is None

def test_key_held_down_is_ignored(mock_p):
    """
    Edge case: Klawisz spacji jest przytrzymany (KEY_IS_DOWN = 2), 
    ale NIE został właśnie wyzwolony (brak flagi KEY_WAS_TRIGGERED = 1).
    Operacja bitowa (2 & 1) zwraca 0.
    """
    mock_p.getKeyboardEvents.return_value = {ord(' '): 2}
    
    handler = InputHandler(num_drones=5)
    command = handler.get_command()
    
    assert command is None

def test_switch_drone_camera_valid_id(mock_p):
    """ Intencja: Klawisze od 1 do 8 mapują się poprawnie na indexy (0-7). """
    # Symulujemy wciśnięcie "3"
    mock_p.getKeyboardEvents.return_value = {ord('3'): 1}
    
    handler = InputHandler(num_drones=5)
    command = handler.get_command()
    
    assert command is not None
    assert command.type == CommandType.SWITCH_DRONE_CAMERA
    # '3' - '1' = 2 (czyli trzeci dron, bo indeksujemy od 0)
    assert command.payload == 2

def test_switch_drone_camera_out_of_bounds_ignored(mock_p):
    """ Edge case: Próba przełączenia na drona, którego nie ma (np. wciśnięto "5", a dronów jest 3). """
    mock_p.getKeyboardEvents.return_value = {ord('5'): 1}
    
    handler = InputHandler(num_drones=3)
    command = handler.get_command()
    
    assert command is None

def test_switch_camera_to_drone_zero_key(mock_p):
    """ Intencja: Klawisz "0" obsługuje drona z indeksem 9, pod warunkiem, że flota jest wystarczająco duża. """
    mock_p.getKeyboardEvents.return_value = {ord('0'): 1}
    
    # Próba przy zbyt małej flocie
    handler_small = InputHandler(num_drones=5)
    assert handler_small.get_command() is None

    # Próba przy wystarczająco dużej flocie
    handler_large = InputHandler(num_drones=10)
    command = handler_large.get_command()
    
    assert command is not None
    assert command.type == CommandType.SWITCH_DRONE_CAMERA
    assert command.payload == 9

def test_unsupported_key_ignored(mock_p):
    """ Edge case: Klawisz, który nie ma przypisanej funkcji (np. 'A') zostaje zignorowany. """
    mock_p.getKeyboardEvents.return_value = {ord('A'): 1}
    
    handler = InputHandler(num_drones=5)
    assert handler.get_command() is None

def test_switch_drone_camera_key_nine(mock_p):
    """ 
    Ten test aktualnie 'nie przejdzie' (xfail - expected to fail), 
    dopóki nie zamienisz '< ord('9')' na '<= ord('9')' w pliku InputHandler.
    """
    mock_p.getKeyboardEvents.return_value = {ord('9'): 1}
    
    handler = InputHandler(num_drones=10)
    command = handler.get_command()
    
    assert command is not None
    assert command.type == CommandType.SWITCH_DRONE_CAMERA
    assert command.payload == 8