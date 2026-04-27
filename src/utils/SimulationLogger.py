import json
import os
import csv
from typing import Any, Dict, List, Optional

from numpy.typing import NDArray
import pandas as pd
import numpy as np

from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.obstacles.ObstacleShape import ObstacleShape
from src.utils.lidar_log_writer import LidarHDF5Writer

_TIMING_HEADERS = [
    "run_id",
    "algorithm_name",
    "stage_name",
    "wall_time_s",
    "cpu_time_s",
    "success",
    "n_drones",
    "number_of_waypoints",
    "population_size",
    "max_generations",
    "extra_params_json",
    "created_at_utc",
]

_EVASION_HEADERS = [
    "time",
    "drone_id",
    "event_type",
    "mode",
    "ttc",
    "dist_to_threat",
    "threat_x", "threat_y", "threat_z",
    "threat_vx", "threat_vy", "threat_vz",
    "rejoin_x", "rejoin_y", "rejoin_z",
    "rejoin_arc",
    "astar_success",
    "fallback_used",
    "pos_error_at_rejoin",
    "vel_error_at_rejoin",
    "planning_wall_time_s",
    "notes",
]

class SimulationLogger:
    def __init__(self, output_dir, log_freq, ctrl_freq, num_drones):
        self.output_dir = output_dir
        self.log_step_interval = max(1, int(ctrl_freq / log_freq))
        self.num_drones = num_drones
        self.trajectory_buffer = []
        self.collision_buffer = []
        self.optimization_timing_buffer: List[Dict[str, Any]] = []
        self.crashed_drones = set()
        
        # Nowy bufor na logi z sensorów LiDAR
        self._lidar_writer = LidarHDF5Writer(output_dir)

        # Bufor dla diagnostyki uniku (Faza 0 planu). Rekordy są słownikami,
        # zapisywane jako CSV z _EVASION_HEADERS w save().
        self.evasion_buffer: List[Dict[str, Any]] = []

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
            
    # --- NOWA METODA LOGOWANIA LIDARU ---
    def log_lidar_hit(self, current_time: float, drone_id: int, hit):
        """Zapisuje w buforze konkretne uderzenie promienia LiDARu."""
        # Ze względu na ogromną liczbę promieni, zaokrąglamy dla oszczędności pamięci
        self._lidar_writer.put((
            round(current_time, 3),
            drone_id,
            hit.object_id,
            round(hit.distance, 3),
            round(hit.hit_position[0], 3),
            round(hit.hit_position[1], 3),
            round(hit.hit_position[2], 3),
        ))

    def log_evasion_event(
        self,
        *,
        current_time: float,
        drone_id: int,
        event_type: str,
        mode: int = -1,
        ttc: float = float("nan"),
        dist_to_threat: float = float("nan"),
        threat_pos: Optional[NDArray] = None,
        threat_vel: Optional[NDArray] = None,
        rejoin_point: Optional[NDArray] = None,
        rejoin_arc: float = float("nan"),
        astar_success: Optional[bool] = None,
        fallback_used: Optional[bool] = None,
        pos_error_at_rejoin: float = float("nan"),
        vel_error_at_rejoin: float = float("nan"),
        planning_wall_time_s: float = float("nan"),
        notes: str = "",
    ) -> None:
        """
        Rekord diagnostyczny fazy uniku — pozwala mierzyć opóźnienie triggera,
        skuteczność A*, błędy pozycji/prędkości przy rejoin. Fields mogą być
        NaN jeśli nieznane w danym zdarzeniu.
        """
        def _xyz(v: Optional[NDArray]) -> tuple:
            if v is None:
                return (float("nan"), float("nan"), float("nan"))
            return (float(v[0]), float(v[1]), float(v[2]))

        tx, ty, tz = _xyz(threat_pos)
        tvx, tvy, tvz = _xyz(threat_vel)
        rx, ry, rz = _xyz(rejoin_point)

        self.evasion_buffer.append({
            "time": round(current_time, 3),
            "drone_id": drone_id,
            "event_type": event_type,
            "mode": mode,
            "ttc": ttc,
            "dist_to_threat": dist_to_threat,
            "threat_x": tx, "threat_y": ty, "threat_z": tz,
            "threat_vx": tvx, "threat_vy": tvy, "threat_vz": tvz,
            "rejoin_x": rx, "rejoin_y": ry, "rejoin_z": rz,
            "rejoin_arc": rejoin_arc,
            "astar_success": astar_success if astar_success is not None else "",
            "fallback_used": fallback_used if fallback_used is not None else "",
            "pos_error_at_rejoin": pos_error_at_rejoin,
            "vel_error_at_rejoin": vel_error_at_rejoin,
            "planning_wall_time_s": planning_wall_time_s,
            "notes": notes,
        })

    def _trajectory_to_dataframe(self, trajectory: NDArray) -> pd.DataFrame:
        n_drones, n_waypoints, _ = trajectory.shape
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
            df = df.drop(columns=['unused_dim'])
        return df
    
    def _world_to_dataframe(self, world: WorldData) -> pd.DataFrame:
        data = {
            'Dimension': world.dimensions,
            'Min_Bound': world.min_bounds,
            'Max_Bound': world.max_bounds,
            'Center': world.center
        }
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

    def log_optimization_timing(
        self, *, run_id: str = "", algorithm_name: str = "", stage_name: str = "",
        wall_time_s: Optional[float] = None, cpu_time_s: Optional[float] = None,
        success: Optional[bool] = None, n_drones: Optional[int] = None,
        number_of_waypoints: Optional[int] = None, population_size: Optional[int] = None,
        max_generations: Optional[int] = None, extra_params: Optional[Dict[str, Any]] = None,
        created_at_utc: str = "",
    ) -> None:
        self.optimization_timing_buffer.append({
            "run_id": run_id, "algorithm_name": algorithm_name, "stage_name": stage_name,
            "wall_time_s": wall_time_s, "cpu_time_s": cpu_time_s, "success": success,
            "n_drones": n_drones, "number_of_waypoints": number_of_waypoints,
            "population_size": population_size, "max_generations": max_generations,
            "extra_params_json": json.dumps(extra_params) if extra_params else "",
            "created_at_utc": created_at_utc,
        })

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

        if self.evasion_buffer:
            path = os.path.join(self.output_dir, "evasion_events.csv")
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=_EVASION_HEADERS)
                writer.writeheader()
                writer.writerows(self.evasion_buffer)
            print(f"[LOGGER] Evasion events saved: evasion_events.csv ({len(self.evasion_buffer)} events)")
            self.evasion_buffer.clear()

        if self.optimization_timing_buffer:
            path = os.path.join(self.output_dir, "optimization_timings.csv")
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=_TIMING_HEADERS)
                writer.writeheader()
                writer.writerows(self.optimization_timing_buffer)
            print(f"[LOGGER] Optimization timings saved: optimization_timings.csv")
            self.optimization_timing_buffer.clear()