import csv
from pathlib import Path
from typing import Union

import numpy as np

from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.obstacles.ObstacleShape import ObstacleShape


PathLike = Union[str, Path]


def load_obstacles_csv(input_dir: PathLike) -> ObstaclesData:
    input_path = Path(input_dir) / "obstacles.csv"
    rows = []
    shape_type = None

    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append([
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
                float(row["dim1"]),
                float(row["dim2"]),
                float(row["dim3"]),
            ])
            shape_type = row["shape_type"]

    if shape_type is None:
        shape_type = ObstacleShape.CYLINDER.value

    data = np.array(rows, dtype=np.float64) if rows else np.empty((0, 6), dtype=np.float64)
    return ObstaclesData(data=data, shape_type=ObstacleShape(shape_type))


def load_trajectories_csv(input_dir: PathLike, num_drones: int) -> np.ndarray:
    input_path = Path(input_dir) / "planned_trajectories.csv"
    rows = []
    max_waypoint_idx = -1

    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            drone_id = int(row["drone_id"])
            waypoint_idx = int(row["waypoint_idx"])
            rows.append((drone_id, waypoint_idx, float(row["x"]), float(row["y"]), float(row["z"])))
            max_waypoint_idx = max(max_waypoint_idx, waypoint_idx)

    if not rows:
        return np.empty((num_drones, 0, 3), dtype=np.float64)

    trajectories = np.zeros((num_drones, max_waypoint_idx + 1, 3), dtype=np.float64)
    for drone_id, waypoint_idx, x, y, z in rows:
        trajectories[drone_id, waypoint_idx] = [x, y, z]

    return trajectories
