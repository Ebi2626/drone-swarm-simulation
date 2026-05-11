import pandas as pd
import os
from hydra.core.hydra_config import HydraConfig

from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.obstacles.ObstacleShape import ObstacleShape

def save_obstacles_to_csv(obstacles_data: ObstaclesData, filename: str = "obstacles_scenario.csv"):
    """Zapisz `obstacles_data` do CSV w katalogu wyjściowym Hydry.

    Args:
        obstacles_data: Macierz przeszkód `(N, 6)` z typem kształtu.
        filename: Nazwa pliku CSV w `HydraConfig.runtime.output_dir`.

    Efekty uboczne:
        Tworzy plik CSV z kolumnami zależnymi od `shape_type`
        (CYLINDER usuwa nieużywaną 6. kolumnę).
    """
    data_matrix = obstacles_data.data
    shape_type = obstacles_data.shape_type

    if shape_type == ObstacleShape.CYLINDER:
        columns = ['pos_x', 'pos_y', 'pos_z', 'radius', 'height', 'dim3_unused']
    elif shape_type == ObstacleShape.BOX:
        columns = ['pos_x', 'pos_y', 'pos_z', 'length', 'width', 'height']
    else:
        columns = ['pos_x', 'pos_y', 'pos_z', 'dim1', 'dim2', 'dim3']

    df = pd.DataFrame(data_matrix, columns=columns)
    df.insert(0, 'shape_type', shape_type)

    if shape_type == ObstacleShape.CYLINDER:
        df = df.drop(columns=['dim3_unused'])

    output_dir = HydraConfig.get().runtime.output_dir
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Pomyślnie zapisano konfigurację środowiska do: {filepath}")
