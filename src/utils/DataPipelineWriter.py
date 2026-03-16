import csv
from pathlib import Path
from typing import Union

import numpy as np

from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.abstraction.generate_world_boundaries import WorldData


PathLike = Union[str, Path]


def _ensure_dir(output_dir: PathLike) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_world_data_csv(world_data: WorldData, output_dir: PathLike) -> Path:
    output_path = _ensure_dir(output_dir) / "world_data.csv"

    rows = [
        ("dimension_x", float(world_data.dimensions[0])),
        ("dimension_y", float(world_data.dimensions[1])),
        ("dimension_z", float(world_data.dimensions[2])),
        ("min_x", float(world_data.min_bounds[0])),
        ("min_y", float(world_data.min_bounds[1])),
        ("min_z", float(world_data.min_bounds[2])),
        ("max_x", float(world_data.max_bounds[0])),
        ("max_y", float(world_data.max_bounds[1])),
        ("max_z", float(world_data.max_bounds[2])),
        ("center_x", float(world_data.center[0])),
        ("center_y", float(world_data.center[1])),
        ("center_z", float(world_data.center[2])),
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["param", "value"])
        writer.writerows(rows)

    return output_path


def save_obstacles_csv(obstacles: ObstaclesData, output_dir: PathLike) -> Path:
    output_path = _ensure_dir(output_dir) / "obstacles.csv"

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "z", "dim1", "dim2", "dim3", "shape_type"])
        for obstacle in obstacles.data:
            writer.writerow([
                float(obstacle[0]),
                float(obstacle[1]),
                float(obstacle[2]),
                float(obstacle[3]),
                float(obstacle[4]),
                float(obstacle[5]),
                obstacles.shape_type.value,
            ])

    return output_path


def save_trajectories_csv(trajectories: np.ndarray, output_dir: PathLike) -> Path:
    output_path = _ensure_dir(output_dir) / "planned_trajectories.csv"

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["drone_id", "waypoint_idx", "x", "y", "z"])
        for drone_id in range(trajectories.shape[0]):
            for waypoint_idx in range(trajectories.shape[1]):
                waypoint = trajectories[drone_id, waypoint_idx]
                writer.writerow([
                    drone_id,
                    waypoint_idx,
                    float(waypoint[0]),
                    float(waypoint[1]),
                    float(waypoint[2]),
                ])

    return output_path


def save_start_end_positions_csv(start_positions: np.ndarray, end_positions: np.ndarray, output_dir: PathLike) -> Path:
    output_path = _ensure_dir(output_dir) / "start_end_positions.csv"

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["drone_id", "start_x", "start_y", "start_z", "end_x", "end_y", "end_z"])
        for drone_id in range(start_positions.shape[0]):
            start = start_positions[drone_id]
            end = end_positions[drone_id]
            writer.writerow([
                drone_id,
                float(start[0]),
                float(start[1]),
                float(start[2]),
                float(end[0]),
                float(end[1]),
                float(end[2]),
            ])

    return output_path
