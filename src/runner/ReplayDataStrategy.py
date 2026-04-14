import pandas as pd
import numpy as np
from pathlib import Path
from src.environments.obstacles.ObstacleShape import ObstacleShape
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.runner.ExperimentDataStrategy import ExperimentDataStrategy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import ExperimentRunner

class ReplayDataStrategy(ExperimentDataStrategy):
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)

    def _map_to_world_data(self, df: pd.DataFrame) -> WorldData:
        """
        Rekonstrukcja obiektu WorldData z archiwum CSV.
        Plik zorganizowany jest osiami (X, Y, Z) w wierszach.
        """
        # Upewniamy się, że dane są posortowane/wyciągane w prawidłowej kolejności osi: X, Y, Z
        # Zabezpiecza to przed błędami w przypadku, gdyby wiersze w pliku CSV zmieniły kolejność
        df_sorted = df.set_index('Axis').loc[['X', 'Y', 'Z']]
        
        # Dzięki temu, że kolumny to od razu wektory 3D (dla X, Y, Z),
        # możemy bezstratnie zrzutować całe serie pandas na macierze numpy.
        dimensions = df_sorted['Dimension'].to_numpy(dtype=np.float64)
        min_bounds = df_sorted['Min_Bound'].to_numpy(dtype=np.float64)
        max_bounds = df_sorted['Max_Bound'].to_numpy(dtype=np.float64)
        center = df_sorted['Center'].to_numpy(dtype=np.float64)
        
        # Odtworzenie macierzy bounds (3x2) - łączenie wektorów min i max
        bounds = np.column_stack((min_bounds, max_bounds))

        return WorldData(
            dimensions=dimensions,
            min_bounds=min_bounds,
            max_bounds=max_bounds,
            bounds=bounds,
            center=center
        )
    
    def _map_to_obstacles_data(self, df: pd.DataFrame, shape_type_str: str) -> ObstaclesData:
        """
        Rekonstrukcja macierzy przeszkód z uwzględnieniem kształtu.
        Obsługuje dynamicznie format dla lasu (CYLINDER) oraz miasta (BOX).
        """
        shape_type = ObstacleShape[shape_type_str.upper()] 
        
        if shape_type == ObstacleShape.CYLINDER:
            # Model leśny: [x, y, z, radius, height, unused_dim]
            expected_columns = ['x', 'y', 'z', 'radius', 'height', 'unused_dim']
        elif shape_type == ObstacleShape.BOX:
            # Model zurbanizowany: [x, y, z, length, width, height]
            expected_columns = ['x', 'y', 'z', 'length', 'width', 'height']
        else:
            # Fallback dla nieprzewidzianych kształtów - pobiera pierwsze 6 kolumn
            expected_columns = df.columns[:6].tolist()
            print(f"[WARNING] Nierozpoznany kształt przeszkody: {shape_type_str}. Użyto domyślnych kolumn.")
            
        # Zabezpieczenie przed brakiem kolumn w pliku
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Brakuje następujących kolumn w archiwalnym pliku przeszkód: {missing_cols}")
            
        # Wyciągnięcie i rzutowanie wartości prosto do postaci tensora (N, 6)
        data_matrix = df[expected_columns].to_numpy(dtype=np.float64)
        
        return ObstaclesData(data=data_matrix, shape_type=shape_type)

    def _map_to_trajectories(self, df: pd.DataFrame) -> np.ndarray:
        """
        Przekształca płaską tabelę CSV (drone_id, waypoint_id, x, y, z) 
        w trójwymiarowy tensor o kształcie: (liczba_dronów, liczba_punktów, 3).
        """
        # Obliczenie właściwych wymiarów tensora
        num_drones = df['drone_id'].nunique()
        num_waypoints = df['waypoint_id'].nunique()
        
        # Alokacja pamięci na tensor (Drones x Waypoints x 3)
        trajectories = np.zeros((num_drones, num_waypoints, 3), dtype=np.float64)
        
        for d_id in range(num_drones):
            # 1. Filtrowanie danych dla konkretnego drona
            # 2. Gwarancja sortowania topologicznego (chronologicznego) po waypoint_id
            drone_data = df[df['drone_id'] == d_id].sort_values('waypoint_id')
            
            # 3. Zrzutowanie wyłącznie kolumn przestrzennych do macierzy 2D i przypisanie do drona
            trajectories[d_id] = drone_data[['x', 'y', 'z']].to_numpy(dtype=np.float64)
            
        return trajectories

    def prepare_data(self, runner: "ExperimentRunner"):
        print(f"[INFO] Odtwarzanie eksperymentu z archiwum: {self.results_path}")
        
        # 1. Rekonstrukcja WorldData (Granice Świata)
        world_df = pd.read_csv(self.results_path / "world_boundaries.csv")
        runner.world_data = self._map_to_world_data(world_df)
        
        # Zabezpieczenie na wypadek niepowodzenia deserializacji
        if runner.world_data is None:
            raise ValueError("[BŁĄD KRYTYCZNY] Obiekt 'world_data' nie został poprawnie zainicjalizowany!")
            
        # 2. Rekonstrukcja ObstaclesData (Przeszkody)
        obstacles_df = pd.read_csv(self.results_path / "generated_obstacles.csv")
        shape_type_str = runner.cfg.environment.params.get("shape_type", "CYLINDER") 
        runner.obstacles_data = self._map_to_obstacles_data(obstacles_df, shape_type_str)
        
        if runner.obstacles_data is None:
            raise ValueError("[BŁĄD KRYTYCZNY] Obiekt 'obstacles_data' nie został poprawnie zainicjalizowany!")
        
        # 3. Rekonstrukcja Trajektorii (Z płaskiej tabeli do tensora N x W x 3)
        trajectories_df = pd.read_csv(self.results_path / "counted_trajectories.csv")
        runner.trajectories = self._map_to_trajectories(trajectories_df)
        
        # 4. Iniekcja archiwalnych pozycji początkowych i końcowych do głównego procesu symulacji
        # Odcina to symulację od jakichkolwiek losowych wartości zaszytych w pliku YAML, 
        # zachowując 100% determinizmu eksperymentu.
        runner.start_positions = runner.trajectories[:, 0, :]
        runner.end_positions = runner.trajectories[:, -1, :]
        
        # Raportowanie stanu do logów (dobra praktyka ewaluacyjna)
        print(f"[DEBUG] Pomyślnie zrekonstruowano WorldData (wymiary: {runner.world_data.dimensions})")
        print(f"[DEBUG] Pomyślnie zrekonstruowano ObstaclesData (liczba: {runner.obstacles_data.count})")
        print(f"[DEBUG] Tensor trajektorii gotowy do śledzenia. Kształt: {runner.trajectories.shape}")