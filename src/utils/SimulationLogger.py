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
from src.utils.optimization_metrics import (
    OUTCOME_PENDING,
    ConvergenceSample,
    OnlineOptimizationRecord,
    convergence_sample_headers,
    online_record_headers,
    record_to_dict,
)

# PK match tolerance dla update_online_optimization_outcome — float trigger_time
# zachowuje binarną reprezentację tylko wtedy gdy ten sam string→float konwert.
# Praktycznie tu może wpaść zaokrąglenie do 3 cyfr w timer'ze SwarmFlightController
# vs surowy float w `compute_evasion_plan`. 1e-6 to bezpieczna granica (1 µs).
_PK_FLOAT_TOL_S = 1e-6

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
    "ttc_source",          # 'oracle_discrete' | 'continuous' | empty
    "dist_to_threat",
    "threat_x", "threat_y", "threat_z",
    "threat_vx", "threat_vy", "threat_vz",
    "rejoin_x", "rejoin_y", "rejoin_z",
    "rejoin_arc",
    "preferred_axis",       # 'right' | 'left' | 'up' | 'down' | empty (NULL gdy nieokreślone)
    "fallback_used",
    "pos_error_at_rejoin",
    "vel_error_at_rejoin",
    "planning_wall_time_s",
    "notes",
]

class SimulationLogger:
    """Centralny logger symulacji — bufory RAM dla wszystkich CSV/h5 (flush w `save()`).

    Bufory zapisywane przy `save()` do plików w `output_dir`:
      - `trajectories.csv`, `collisions.csv`, `evasion_events.csv`,
      - `optimization_timings.csv`, `online_optimization.csv`,
      - `convergence_traces.csv`, `world_boundaries.csv`,
      - `generated_obstacles.csv`, `counted_trajectories.csv`,
      - `lidar_hits.h5` (asynchroniczny `LidarHDF5Writer`).
    """

    def __init__(self, output_dir, log_freq, ctrl_freq, num_drones, log_lidar_hits: bool = False):
        """Skonfiguruj bufory, częstotliwości próbkowania i writer LiDAR.

        Args:
            output_dir: Katalog docelowy artefaktów runa.
            log_freq: Pożądana częstotliwość logu trajektorii [Hz].
            ctrl_freq: Częstotliwość symulacji [Hz]; wyznacza `log_step_interval`.
            num_drones: Liczba dronów (do walidacji).
            log_lidar_hits: `True` ⇒ logowanie pełnej historii LiDAR do h5.
        """
        self.output_dir = output_dir
        # `run_id` używane w online_optimization.csv / convergence_traces.csv —
        # default = basename katalogu (zwykle Hydra timestamp). Może być nadpisane
        # przez integrator (`logger.run_id = "..."`).
        self.run_id = os.path.basename(os.path.normpath(str(output_dir)))
        self.log_step_interval = max(1, int(ctrl_freq / log_freq))
        self.num_drones = num_drones
        self.trajectory_buffer = []
        self.collision_buffer = []
        self.optimization_timing_buffer: List[Dict[str, Any]] = []
        self.crashed_drones = set()
        self.log_lidar_hits = log_lidar_hits

        # Nowy bufor na logi z sensorów LiDAR
        self._lidar_writer = LidarHDF5Writer(output_dir)

        # Bufor dla diagnostyki uniku — rekordy są słownikami zapisywanymi jako
        # CSV z `_EVASION_HEADERS` w `save()`.
        self.evasion_buffer: List[Dict[str, Any]] = []

        # Bufory metryk online optymalizacji — każdy wiersz to dict zgodny
        # ze schema dataclass'ów z `optimization_metrics`.
        self.online_optimization_buffer: List[Dict[str, Any]] = []
        self.convergence_traces_buffer: List[Dict[str, Any]] = []

        print("[LOGGER] Buffering in RAM. Writing to disk after completion.")

    def log_step(self, step_idx, current_time, all_states):
        """Zaloguj pozycje, RPY i velocity wszystkich dronów co `log_step_interval` kroków."""
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
        """Zaloguj kolizję `drone_id` z `other_body_id` (raz per dron, ignoruje `t < 1s`)."""
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
        """Asynchronicznie zapisz pojedyncze trafienie LiDAR do `lidar_hits.h5` (gdy włączone)."""
        # Ze względu na ogromną liczbę promieni, zaokrąglamy dla oszczędności pamięci
        if(self.log_lidar_hits is True):
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
        ttc_source: Optional[str] = None,
        dist_to_threat: float = float("nan"),
        threat_pos: Optional[NDArray] = None,
        threat_vel: Optional[NDArray] = None,
        rejoin_point: Optional[NDArray] = None,
        rejoin_arc: float = float("nan"),
        preferred_axis: Optional[str] = None,
        fallback_used: Optional[bool] = None,
        pos_error_at_rejoin: float = float("nan"),
        vel_error_at_rejoin: float = float("nan"),
        planning_wall_time_s: float = float("nan"),
        notes: str = "",
        # Backward-compat alias: starsze callsity przekazują `astar_success`,
        # które semantycznie = `NOT fallback_used`. Konwertujemy gdy
        # `fallback_used` nie podane.
        astar_success: Optional[bool] = None,
    ) -> None:
        """Dorzuć do bufora rekord diagnostyczny zdarzenia uniku.

        Args:
            current_time: Czas symulacji [s].
            drone_id: Indeks drona.
            event_type: `trigger / blend_start / rejoin / fallback / …`.
            mode: `MODE_*` z `SwarmFlightController`.
            ttc, dist_to_threat: Time-to-collision i dystans do zagrożenia.
            ttc_source: `'oracle_discrete'` lub `'continuous'`.
            threat_pos, threat_vel: Stan zagrożenia (`(3,)` lub `None`).
            rejoin_point, rejoin_arc: Punkt powrotu i odpowiadający łuk
                bazowego splajnu.
            preferred_axis: `right / left / up / down / None` — wybrana oś.
            fallback_used: `True`, gdy plan z fallbacku zamiast optymalizatora.
            pos_error_at_rejoin, vel_error_at_rejoin, planning_wall_time_s,
            notes: Dodatkowe pola diagnostyczne.
            astar_success: DEPRECATED — używać `fallback_used` (odwrotne).

        Efekty uboczne:
            Append do `evasion_buffer` (zapisywany w `save`).
        """
        def _xyz(v: Optional[NDArray]) -> tuple:
            if v is None:
                return (float("nan"), float("nan"), float("nan"))
            return (float(v[0]), float(v[1]), float(v[2]))

        tx, ty, tz = _xyz(threat_pos)
        tvx, tvy, tvz = _xyz(threat_vel)
        rx, ry, rz = _xyz(rejoin_point)

        # Backward-compat: jeśli ktoś wciąż przekazuje astar_success a nie
        # fallback_used, wywodzimy fallback_used jako negację.
        if fallback_used is None and astar_success is not None:
            fallback_used = not astar_success

        self.evasion_buffer.append({
            "time": round(current_time, 3),
            "drone_id": drone_id,
            "event_type": event_type,
            "mode": mode,
            "ttc": ttc,
            "ttc_source": ttc_source if ttc_source is not None else "",
            "dist_to_threat": dist_to_threat,
            "threat_x": tx, "threat_y": ty, "threat_z": tz,
            "threat_vx": tvx, "threat_vy": tvy, "threat_vz": tvz,
            "rejoin_x": rx, "rejoin_y": ry, "rejoin_z": rz,
            "rejoin_arc": rejoin_arc,
            "preferred_axis": preferred_axis if preferred_axis in ("X", "Y", "Z") else "",
            "fallback_used": fallback_used if fallback_used is not None else "",
            "pos_error_at_rejoin": pos_error_at_rejoin,
            "vel_error_at_rejoin": vel_error_at_rejoin,
            "planning_wall_time_s": planning_wall_time_s,
            "notes": notes,
        })

    def log_online_optimization_trigger(
        self, record: OnlineOptimizationRecord
    ) -> None:
        """Dorzuć `record` (per-trigger summary) do bufora online optymalizacji.

        Outcome (grupa D) pozostaje `OUTCOME_PENDING` aż do BLEND_END /
        kolizji — wtedy `update_online_optimization_outcome` wypełnia grupę D
        in-place po PK `(drone_id, trigger_time)`.
        """
        self.online_optimization_buffer.append(record_to_dict(record))

    def update_online_optimization_outcome(
        self,
        *,
        drone_id: int,
        trigger_time: float,
        outcome: str,
        pos_err_at_rejoin_m: float = float("nan"),
        vel_err_at_rejoin_mps: float = float("nan"),
        time_to_rejoin_s: float = float("nan"),
    ) -> None:
        """Uzupełnij grupę D rekordu pasującego do `(drone_id, trigger_time)` z tolerancją 1 µs.

        Args:
            drone_id, trigger_time: Klucz wyszukiwania (`_PK_FLOAT_TOL_S`).
            outcome: `OUTCOME_*` z `optimization_metrics`.
            pos_err_at_rejoin_m, vel_err_at_rejoin_mps, time_to_rejoin_s:
                Wartości grupy D do wpisania.

        Efekty uboczne:
            Aktualizuje rekord in-place. Brak match-a ⇒ `print` z ostrzeżeniem
            (race lub outcome dla `plan=None`).
        """
        for rec in reversed(self.online_optimization_buffer):
            if rec["drone_id"] != drone_id:
                continue
            if abs(float(rec["trigger_time"]) - float(trigger_time)) > _PK_FLOAT_TOL_S:
                continue
            if rec["outcome"] != OUTCOME_PENDING:
                # Już wypełnione (np. drugi callback) — nie nadpisuj.
                return
            rec["outcome"] = outcome
            rec["pos_err_at_rejoin_m"] = pos_err_at_rejoin_m
            rec["vel_err_at_rejoin_mps"] = vel_err_at_rejoin_mps
            rec["time_to_rejoin_s"] = time_to_rejoin_s
            return
        print(
            f"[LOGGER] update_online_optimization_outcome: brak match dla "
            f"drone_id={drone_id}, trigger_time={trigger_time:.6f}s "
            f"(outcome={outcome}). Możliwy race lub outcome dla planu=None."
        )

    def log_convergence_trace(
        self,
        *,
        run_id: str,
        drone_id: int,
        trigger_time: float,
        algorithm: str,
        trace: List[float],
    ) -> None:
        """Append `len(trace)` rekordów long-form (1 wiersz / generacja) dla 1 triggera.

        Args:
            run_id, drone_id, trigger_time, algorithm: Pola identyfikacyjne
                (FK do `OnlineOptimizationRecord`).
            trace: Lista `best_fitness` per generacja; pusta ⇒ zero rekordów.
        """
        for gen, fit in enumerate(trace):
            sample = ConvergenceSample(
                run_id=run_id,
                drone_id=drone_id,
                trigger_time=float(trigger_time),
                algorithm=algorithm,
                generation=int(gen),
                best_fitness=float(fit),
            )
            self.convergence_traces_buffer.append(record_to_dict(sample))

    def _trajectory_to_dataframe(self, trajectory: NDArray) -> pd.DataFrame:
        """Spłaszcz `(n_drones, n_waypoints, 3)` do tidy DataFrame `(drone_id, waypoint_id, x, y, z)`."""
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
        """Zamień `ObstaclesData` na DataFrame z kolumnami zależnymi od `shape_type`."""
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
        """Zamień `WorldData` na DataFrame z indeksem `[X, Y, Z]` i 4 kolumnami."""
        data = {
            'Dimension': world.dimensions,
            'Min_Bound': world.min_bounds,
            'Max_Bound': world.max_bounds,
            'Center': world.center
        }
        df = pd.DataFrame(data, index=['X', 'Y', 'Z'])
        return df

    def log_chosen_trajectories(self, trajectories: NDArray):
        """Zapisz wybrane trajektorie offline do `counted_trajectories.csv`."""
        trajectories_data_frame = self._trajectory_to_dataframe(trajectories)
        path = os.path.join(self.output_dir, "counted_trajectories.csv")
        trajectories_data_frame.to_csv(path, index=False, float_format="%.4f")
        print(f"Zapisano {len(trajectories_data_frame)} punktów do: {path}")

    def log_world_dimensions(self, world: WorldData):
        """Zapisz `world` jako `world_boundaries.csv` (do replayu)."""
        world_data_frame = self._world_to_dataframe(world)
        path = os.path.join(self.output_dir, "world_boundaries.csv")
        world_data_frame.to_csv(path, index=True, index_label="Axis", float_format="%.4f")
        print(f"Zapisano {len(world_data_frame)} punktów do: {path}")

    def log_obstacles(self, obstacles: ObstaclesData):
        """Zapisz `obstacles` jako `generated_obstacles.csv`; `None` ⇒ skip z komunikatem."""
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
        """Dorzuć rekord wall/cpu time fazy `stage_name` (offline) do bufora timings."""
        self.optimization_timing_buffer.append({
            "run_id": run_id, "algorithm_name": algorithm_name, "stage_name": stage_name,
            "wall_time_s": wall_time_s, "cpu_time_s": cpu_time_s, "success": success,
            "n_drones": n_drones, "number_of_waypoints": number_of_waypoints,
            "population_size": population_size, "max_generations": max_generations,
            "extra_params_json": json.dumps(extra_params) if extra_params else "",
            "created_at_utc": created_at_utc,
        })

    def save(self):
        """Sflushuj wszystkie bufory na dysk i zamknij asynchroniczny LiDAR writer.

        Efekty uboczne:
            Zapisuje `trajectories.csv`, `collisions.csv`, `evasion_events.csv`,
            `optimization_timings.csv`, `online_optimization.csv`,
            `convergence_traces.csv` (gdy odpowiednie bufory niepuste)
            oraz finalizuje `lidar_hits.h5`.
        """
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

        if self.online_optimization_buffer:
            path = os.path.join(self.output_dir, "online_optimization.csv")
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=online_record_headers())
                writer.writeheader()
                writer.writerows(self.online_optimization_buffer)
            print(
                f"[LOGGER] Online optimization saved: online_optimization.csv "
                f"({len(self.online_optimization_buffer)} triggers)"
            )
            self.online_optimization_buffer.clear()

        if self.convergence_traces_buffer:
            path = os.path.join(self.output_dir, "convergence_traces.csv")
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=convergence_sample_headers())
                writer.writeheader()
                writer.writerows(self.convergence_traces_buffer)
            print(
                f"[LOGGER] Convergence traces saved: convergence_traces.csv "
                f"({len(self.convergence_traces_buffer)} samples)"
            )
            self.convergence_traces_buffer.clear()