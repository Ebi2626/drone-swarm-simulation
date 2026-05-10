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

def _to_int(value):
    return None if value in (None, "") else int(value)


def _to_float(value):
    return None if value in (None, "") else float(value)

def _to_bool(value):
    if value is None:
        return None

    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    return None

def _to_int_bool(value):
    if value is None:
        return None

    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return 1
    if value in {"false", "0", "no", "n"}:
        return 0
    return None

def _to_str_nullable(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() == "nan":
        return None
    return value

def _to_int_nullable(value):
    value = _to_str_nullable(value)
    return None if value is None else int(value)

def _to_float_nullable(value):
    value = _to_str_nullable(value)
    return None if value is None else float(value)

def _to_int_bool_nullable(value):
    value = _to_str_nullable(value)
    if value is None:
        return None
    if value.lower() in {"true", "1", "yes", "y"}:
        return 1
    if value.lower() in {"false", "0", "no", "n"}:
        return 0
    return None