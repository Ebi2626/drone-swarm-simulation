# src/analysis/db/utils.py
from pathlib import Path
import re

RUN_DIR_PATTERN = re.compile(
    r"^(?P<optimizer>.+?)_(?P<environment>forest|urban)_(?P<avoidance>.+?)_seed(?P<seed>\d+)$"
)

def parse_run_dir_name(run_dir_name: str) -> dict:
    match = RUN_DIR_PATTERN.match(run_dir_name)
    if not match:
        raise ValueError(f"Niepoprawna nazwa runu: {run_dir_name}")

    data = match.groupdict()
    data["seed"] = int(data["seed"])
    data["algorithm_pair"] = f"{data['optimizer']} + {data['avoidance']}"
    return data


def list_run_directories(experiment_dir: str | Path) -> list[Path]:
    experiment_dir = Path(experiment_dir).expanduser().resolve()
    return sorted(
        path for path in experiment_dir.iterdir()
        if path.is_dir() and RUN_DIR_PATTERN.match(path.name)
    )