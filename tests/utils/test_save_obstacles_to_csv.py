import pytest
import numpy as np
from unittest.mock import MagicMock

# UWAGA: Zmień 'src.utils.zapis' na faktyczny moduł, w którym leży Twoja funkcja
from src.utils.save_obstacles_to_csv import save_obstacles_to_csv

# UWAGA: Upewnij się, że ścieżki importu pokrywają się z Twoim kodem
from src.environments.obstacles.ObstacleShape import ObstacleShape
# Jeśli ObstaclesData to prosta klasa/dataclass, zrobimy dla niej atrapę w testach, 
# ale możesz też zaimportować prawdziwą z src.environments.abstraction.generate_obstacles

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_obstacles_data():
    """
    Tworzy uniwersalną atrapę (mocka) dla obiektu ObstaclesData, 
    abyśmy nie musieli inicjalizować pełnego środowiska generowania przeszkód.
    """
    def _create_data(shape_type, data_matrix):
        mock_obj = MagicMock()
        mock_obj.shape_type = shape_type
        mock_obj.data = data_matrix
        return mock_obj
    return _create_data

@pytest.fixture
def mock_hydra(mocker):
    """
    Mockuje Singleton HydraConfig. Dzięki temu omijamy błędy typu 
    'HydraConfig is not initialized' i możemy narzucić sztywną ścieżkę.
    """
    mock_config = mocker.patch('src.utils.save_obstacles_to_csv.HydraConfig')
    fake_dir = "/fake/hydra/outputs/2026-04-10"
    mock_config.get.return_value.runtime.output_dir = fake_dir
    return fake_dir

# ==========================================
# TESTY
# ==========================================

def test_save_cylinder_obstacles(mock_obstacles_data, mock_hydra, mocker, capsys):
    """
    Intencja: Sprawdzenie dedykowanej logiki dla CYLINDER.
    Czy usuwa kolumnę 'dim3_unused' i nadaje poprawne nazwy 'radius' i 'height'?
    """
    # 1 wymyślony cylinder: x=1, y=2, z=3, r=0.5, h=2.0, dim3=0.0
    data_matrix = np.array([[1.0, 2.0, 3.0, 0.5, 2.0, 0.0]])
    obs_data = mock_obstacles_data(ObstacleShape.CYLINDER, data_matrix)
    
    # Tworzymy funkcję, która zastąpi domyślne df.to_csv()
    # 'self' w tym kontekście to DataFrame stworzony wewnątrz testowanej funkcji!
    def mock_to_csv(self_df, filepath, index):
        assert index is False
        assert filepath == f"{mock_hydra}/test_cylinders.csv"
        
        # Weryfikacja struktury kolumn (czy wyrzuciło dim3_unused)
        expected_columns = ['shape_type', 'pos_x', 'pos_y', 'pos_z', 'radius', 'height']
        assert list(self_df.columns) == expected_columns
        
        # Weryfikacja wpisanych danych
        assert self_df.iloc[0]['shape_type'] == ObstacleShape.CYLINDER
        assert self_df.iloc[0]['radius'] == 0.5
        assert 'dim3_unused' not in self_df.columns

    # Podmieniamy metodę to_csv z pandas na naszą asercję
    mocker.patch('pandas.DataFrame.to_csv', new=mock_to_csv)
    
    # Uruchamiamy funkcję
    save_obstacles_to_csv(obs_data, filename="test_cylinders.csv")
    
    # Sprawdzamy czy print() zadziałał prawidłowo
    captured = capsys.readouterr()
    assert "Pomyślnie zapisano konfigurację środowiska do:" in captured.out
    assert "test_cylinders.csv" in captured.out

def test_save_box_obstacles(mock_obstacles_data, mock_hydra, mocker):
    """
    Intencja: Sprawdzenie dedykowanej logiki dla BOX.
    Powinny zostać użyte kolumny 'length', 'width', 'height'. Brak usuwania kolumn.
    """
    data_matrix = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    obs_data = mock_obstacles_data(ObstacleShape.BOX, data_matrix)
    
    def mock_to_csv(self_df, filepath, index):
        expected_columns = ['shape_type', 'pos_x', 'pos_y', 'pos_z', 'length', 'width', 'height']
        assert list(self_df.columns) == expected_columns
        assert self_df.iloc[0]['shape_type'] == ObstacleShape.BOX
        assert self_df.iloc[0]['width'] == 1.0

    mocker.patch('pandas.DataFrame.to_csv', new=mock_to_csv)
    save_obstacles_to_csv(obs_data, filename="boxes.csv")

def test_save_fallback_shape_obstacles(mock_obstacles_data, mock_hydra, mocker):
    """
    Edge case: Przekazanie nieznanego/nowego typu kształtu (fallback na else).
    Funkcja powinna uciec do domyślnych nazw 'dim1', 'dim2', 'dim3'.
    """
    data_matrix = np.array([[5.0, 5.0, 5.0, 9.0, 9.0, 9.0]])
    # Symulujemy kształt, który nie wpada w CYLINDER ani BOX
    obs_data = mock_obstacles_data("NEW_UNKNOWN_SHAPE", data_matrix)
    
    def mock_to_csv(self_df, filepath, index):
        expected_columns = ['shape_type', 'pos_x', 'pos_y', 'pos_z', 'dim1', 'dim2', 'dim3']
        assert list(self_df.columns) == expected_columns
        assert self_df.iloc[0]['shape_type'] == "NEW_UNKNOWN_SHAPE"

    mocker.patch('pandas.DataFrame.to_csv', new=mock_to_csv)
    save_obstacles_to_csv(obs_data, filename="unknown.csv")

def test_empty_data_matrix(mock_obstacles_data, mock_hydra, mocker):
    """
    Edge case: Funkcja otrzymuje tablicę numpy bez żadnych rzędów (np. środowisko bez przeszkód).
    Powinno zapisać plik CSV wyłącznie z samymi nagłówkami.
    """
    # Tablica 0x6 (0 przeszkód, 6 wymiarów)
    data_matrix = np.empty((0, 6))
    obs_data = mock_obstacles_data(ObstacleShape.BOX, data_matrix)
    
    def mock_to_csv(self_df, filepath, index):
        # Sprawdzamy czy DataFrame ma kolumny, ale 0 rzędów
        assert len(self_df) == 0
        assert list(self_df.columns) == ['shape_type', 'pos_x', 'pos_y', 'pos_z', 'length', 'width', 'height']

    mocker.patch('pandas.DataFrame.to_csv', new=mock_to_csv)
    save_obstacles_to_csv(obs_data, filename="empty.csv")