import pandas as pd
import numpy as np
from pathlib import Path
from src.environments.obstacles.ObstacleShape import ObstacleShape
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.runner.ExperimentDataStrategy import ExperimentDataStrategy
from typing import TYPE_CHECKING

from src.utils.SeedRegistry import SeedRegistry

if TYPE_CHECKING:
    from main import ExperimentRunner

class ReplayDataStrategy(ExperimentDataStrategy):
    """Strategia odtwarzająca eksperyment z archiwum CSV (świat + przeszkody + trajektorie)."""

    def __init__(self, results_path: str):
        """Powiąż strategię z katalogiem zawierającym pliki CSV runa do odtworzenia."""
        self.results_path = Path(results_path)

    def _map_to_world_data(self, df: pd.DataFrame) -> WorldData:
        """Zrekonstruuj `WorldData` z CSV; wymusza kolejność osi X/Y/Z."""
        df_sorted = df.set_index('Axis').loc[['X', 'Y', 'Z']]

        dimensions = df_sorted['Dimension'].to_numpy(dtype=np.float64)
        min_bounds = df_sorted['Min_Bound'].to_numpy(dtype=np.float64)
        max_bounds = df_sorted['Max_Bound'].to_numpy(dtype=np.float64)
        center = df_sorted['Center'].to_numpy(dtype=np.float64)

        bounds = np.column_stack((min_bounds, max_bounds))

        return WorldData(
            dimensions=dimensions,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            bounds=bounds,
            center=center
        )
    
    def _map_to_obstacles_data(self, df: pd.DataFrame, shape_type_str: str) -> ObstaclesData:
        """Zrekonstruuj `ObstaclesData (N, 6)` z CSV, dobierając kolumny wg `shape_type`.

        Args:
            df: DataFrame z `generated_obstacles.csv`.
            shape_type_str: `"CYLINDER"` lub `"BOX"` (case-insensitive).

        Returns:
            `ObstaclesData` z `(N, 6)` macierzą; CYLINDER pad-uje 6. kolumnę zerami.

        Raises:
            KeyError: Gdy w CSV brakuje wymaganych kolumn dla danego shape_type.
        """
        shape_type = ObstacleShape[shape_type_str.upper()] 
        
        if shape_type == ObstacleShape.CYLINDER:
            # Cylinder: CSV ma 5 kolumn (SimulationLogger usuwa `unused_dim`),
            # `ObstaclesData.data` kanonicznie shape=(N, 6) — pad zero poniżej.
            expected_columns = ['x', 'y', 'z', 'radius', 'height']
        elif shape_type == ObstacleShape.BOX:
            expected_columns = ['x', 'y', 'z', 'length', 'width', 'height']
        else:
            expected_columns = df.columns[:6].tolist()
            print(f"[WARNING] Nierozpoznany kształt przeszkody: {shape_type_str}. Użyto domyślnych kolumn.")

        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Brakuje następujących kolumn w archiwalnym pliku przeszkód: {missing_cols}")

        data_matrix = df[expected_columns].to_numpy(dtype=np.float64)
        if shape_type == ObstacleShape.CYLINDER and data_matrix.shape[1] == 5:
            pad = np.zeros((data_matrix.shape[0], 1), dtype=np.float64)
            data_matrix = np.hstack([data_matrix, pad])

        return ObstaclesData(data=data_matrix, shape_type=shape_type)

    def _map_to_trajectories(self, df: pd.DataFrame) -> np.ndarray:
        """Zamień długą tabelę `(drone_id, waypoint_id, x, y, z)` na tensor `(num_drones, num_waypoints, 3)`."""
        num_drones = df['drone_id'].nunique()
        num_waypoints = df['waypoint_id'].nunique()

        trajectories = np.zeros((num_drones, num_waypoints, 3), dtype=np.float64)

        for d_id in range(num_drones):
            drone_data = df[df['drone_id'] == d_id].sort_values('waypoint_id')
            trajectories[d_id] = drone_data[['x', 'y', 'z']].to_numpy(dtype=np.float64)

        return trajectories

    def prepare_data(self, runner: "ExperimentRunner", seeds: SeedRegistry):
        """Wczytaj świat / przeszkody / trajektorie z `results_path` i zapisz do `runner`.

        Args:
            runner: `ExperimentRunner` — modyfikowany in-place.
            seeds: Rejestr ziaren (nieużywany przy replayu — zachowany dla
                spójności kontraktu).

        Raises:
            ValueError: Gdy któryś z wczytanych obiektów ma `None`.
        """
        print(f"[INFO] Odtwarzanie eksperymentu z archiwum: {self.results_path}")

        world_df = pd.read_csv(self.results_path / "world_boundaries.csv")
        runner.world_data = self._map_to_world_data(world_df)

        if runner.world_data is None:
            raise ValueError("[BŁĄD KRYTYCZNY] Obiekt 'world_data' nie został poprawnie zainicjalizowany!")

        obstacles_df = pd.read_csv(self.results_path / "generated_obstacles.csv")
        shape_type_str = runner.cfg.environment.params.get("shape_type", "CYLINDER")
        runner.obstacles_data = self._map_to_obstacles_data(obstacles_df, shape_type_str)

        if runner.obstacles_data is None:
            raise ValueError("[BŁĄD KRYTYCZNY] Obiekt 'obstacles_data' nie został poprawnie zainicjalizowany!")

        trajectories_df = pd.read_csv(self.results_path / "counted_trajectories.csv")
        runner.drones_trajectories = self._map_to_trajectories(trajectories_df)

        # Pozycje start/end pochodzą z trajektorii archiwalnej, nie z YAML —
        # gwarantuje to 100% determinizmu replayu nawet gdy YAML się zmienił.
        runner.start_positions = runner.drones_trajectories[:, 0, :]
        runner.end_positions = runner.drones_trajectories[:, -1, :]

        print(f"[DEBUG] Pomyślnie zrekonstruowano WorldData (wymiary: {runner.world_data.dimensions})")
        print(f"[DEBUG] Pomyślnie zrekonstruowano ObstaclesData (liczba: {runner.obstacles_data.count})")
        print(f"[DEBUG] Tensor trajektorii gotowy do śledzenia. Kształt: {runner.drones_trajectories.shape}")