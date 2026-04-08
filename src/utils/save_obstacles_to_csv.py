import pandas as pd
import os
from hydra.core.hydra_config import HydraConfig

from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.obstacles.ObstacleShape import ObstacleShape

def save_obstacles_to_csv(obstacles_data: ObstaclesData, filename: str = "obstacles_scenario.csv"):
    """
    Function to save obstacles data to a CSV file with Hydra.

    Args:
        obstacles_data (ObstaclesData): object with information about obstacles 
        filename: name of csv file with obstacles postions and dimensions

    Returns:
        None
    """
    # Wydobywamy dane z macierzy
    data_matrix = obstacles_data.data
    shape_type = obstacles_data.shape_type
    
    # Zamieniamy macierz numpy na DataFrame
    if shape_type == ObstacleShape.CYLINDER:
        columns = ['pos_x', 'pos_y', 'pos_z', 'radius', 'height', 'dim3_unused']
    elif shape_type == ObstacleShape.BOX:
        columns = ['pos_x', 'pos_y', 'pos_z', 'length', 'width', 'height']
    else:
        columns = ['pos_x', 'pos_y', 'pos_z', 'dim1', 'dim2', 'dim3']
        
    df = pd.DataFrame(data_matrix, columns=columns)
    
    # Dodajemy czytelną kolumnę z typem kształtu
    df.insert(0, 'shape_type', shape_type)
    
    # Usuwamy nieużywane kolumny (np. 3. wymiar dla cylindra, który celowo był zerem)
    if shape_type == ObstacleShape.CYLINDER:
        df = df.drop(columns=['dim3_unused'])
        
    # Pobieramy ścieżkę wyjściową bieżącego RUN'a z Hydry
    # Hydra domyślnie tworzy unikalny folder dla każdego uruchomienia (np. outputs/2026-02-25/10-00-00/)
    output_dir = HydraConfig.get().runtime.output_dir
    filepath = os.path.join(output_dir, filename)
    
    # Zapisujemy plik (bez numerycznego indeksu po lewej stronie, bo jest zbędny)
    df.to_csv(filepath, index=False)
    print(f"Pomyślnie zapisano konfigurację środowiska do: {filepath}")
