import os
import csv
from numpy.typing import NDArray
import pandas as pd
import numpy as np

from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.obstacles.ObstacleShape import ObstacleShape

class SimulationLogger:
    def __init__(self, output_dir, log_freq, ctrl_freq, num_drones):
        self.output_dir = output_dir
        self.log_step_interval = max(1, int(ctrl_freq / log_freq))
        self.num_drones = num_drones        
        self.trajectory_buffer = []
        self.collision_buffer = []        
        self.crashed_drones = set()
        print("[LOGGER] Buffering in RAM. Writing to disk after completion.")

    def log_step(self, step_idx, current_time, all_states):
        if step_idx % self.log_step_interval == 0:
            for drone_id, state in enumerate(all_states):
                if drone_id in self.crashed_drones:
                    continue
                
                record = (
                    round(current_time, 3),
                    drone_id,
                    round(state[0], 3),
                    round(state[1], 3),
                    round(state[2], 3),
                    round(state[7], 3),
                    round(state[8], 3),
                    round(state[9], 3),
                    round(state[10], 3),
                    round(state[11], 3),
                    round(state[12], 3)
                )
                self.trajectory_buffer.append(record)

    def log_collision(self, current_time, drone_id, other_body_id):
        if current_time < 1:
            return
        
        if drone_id not in self.crashed_drones:
            self.crashed_drones.add(drone_id)
            
            self.collision_buffer.append((
                round(current_time, 3),
                drone_id,
                other_body_id
            ))
            
            print(f"[LOGGER] Collision! Drone {drone_id} hit object {other_body_id} (t={current_time:.2f}s)")
    
    def _trajectory_to_dataframe(self, trajectory: NDArray) -> pd.DataFrame:
        """
        Konwertuje tablicę trajektorii do DataFrame gotowego do zapisu CSV.
        
        Args:
            trajectory: NDArray kształtu (n_drones, n_waypoints, 3)
            
        Returns:
            pd.DataFrame z kolumnami: drone_id, waypoint_id, x, y, z
        """
        n_drones, n_waypoints, _ = trajectory.shape
        
        # Tworzymy indeksy za pomocą meshgrid (w pełni wektoryzowane)
        drone_ids, waypoint_ids = np.meshgrid(
            np.arange(n_drones),
            np.arange(n_waypoints),
            indexing='ij'
        )
        
        return pd.DataFrame({
            "drone_id":    drone_ids.ravel(),
            "waypoint_id": waypoint_ids.ravel(),
            "x":           trajectory[:, :, 0].ravel(),
            "y":           trajectory[:, :, 1].ravel(),
            "z":           trajectory[:, :, 2].ravel(),
        })

    def _obstacles_to_dataframe(self, obstacles: ObstaclesData) -> pd.DataFrame:
        columns = ['x', 'y', 'z']
        shape_type = obstacles.shape_type
        if shape_type == ObstacleShape.BOX:
            columns.extend(["length", "width", "height"])
        elif shape_type == ObstacleShape.CYLINDER:
            columns.extend(["radius", "height", "unused_dim"])
        else:
            raise ValueError("Wrong obstacles shape type: ", shape_type)
        
        df = pd.DataFrame(obstacles.data, columns=columns)

        if shape_type is ObstacleShape.CYLINDER:
            df = df.drop(columns=['dim3_unused'])

        return df
    
    def _world_to_dataframe(self, world: WorldData) -> pd.DataFrame:
        data = {
            'Dimension': world.dimensions,
            'Min_Bound': world.min_bounds,
            'Max_Bound': world.max_bounds,
            'Center': world.center
        }
        
        # Przekazanie etykiet osi X, Y, Z jako indeksów wierszy
        df = pd.DataFrame(data, index=['X', 'Y', 'Z'])
        
        return df
        

    def log_chosen_trajectories(self, trajectories: NDArray):
        trajectories_data_frame = self._trajectory_to_dataframe(trajectories)
        path = os.path.join(self.output_dir, "counted_trajectories.csv")
        trajectories_data_frame.to_csv(path, index=False, float_format="%.4f")
        print(f"Zapisano {len(trajectories_data_frame)} punktów do: {path}")

    def log_world_dimensions(self, world: WorldData):
        world_data_frame = self._world_to_dataframe(world)
        path = os.path.join(self.output_dir, "world_boundaries.csv")
        world_data_frame.to_csv(path, index=True, index_label="Axis", float_format="%.4f")
        print(f"Zapisano {len(world_data_frame)} punktów do: {path}")

    def log_obstacles(self, obstacles: ObstaclesData):
        if obstacles is None:
            print("Brak przeszkód - plik z logami dotyczącymi pozycji przeszkód nie zostanie utworzony")
            return
        obstacles_data_frame = self._obstacles_to_dataframe(obstacles)
        path = os.path.join(self.output_dir, "generated_obstacles.csv")
        obstacles_data_frame.to_csv(path, index=False, float_format="%.4f")
        print(f"Zapisano {len(obstacles_data_frame)} pozycji przeszkód do: {path}")

    def save(self):
        print("[LOGGER] Saving data to disk...")
        
        if self.trajectory_buffer:
            path = os.path.join(self.output_dir, "trajectories.csv")
            headers = ["time", "drone_id", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz"]
            
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(self.trajectory_buffer)
            print("[LOGGER] Trajectories saved: trajectories.csv")
            self.trajectory_buffer.clear()
        
        if self.collision_buffer:
            path = os.path.join(self.output_dir, "collisions.csv")
            headers = ["time", "drone_id", "other_body_id"]
            
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(self.collision_buffer)
                
            print(f"[LOGGER] Collisions saved: collisions.csv ({len(self.collision_buffer)} events)")
            self.collision_buffer.clear()
        else:
            print("[LOGGER] No collisions - file collisions.csv not created.")
