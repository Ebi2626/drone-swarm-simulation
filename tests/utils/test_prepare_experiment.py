"""
Test integracyjny: prepare_experiment.py + RunRegistry.

Regression dla bugu z plan.md, Krok 6 — registry.db po zakończonym
12-runowym eksperymencie miał 0 wpisów (`exp_20260426_b9b56922_complex_test`).
Przyczyna: hardcoded `sweep_params` (4×2×2×100=1600) niezgodne z `definition`,
plus brak walidacji `pending == planned`.

Test puszcza prawdziwy `prepare_experiment.py` jako subprocess, sprawdza,
że registry.db zawiera DOKŁADNIE liczbę wierszy wynikającą z mini-fixture
YAML (2 optimizers × 1 env × 1 avoid × 2 seedy = 4). Cleanup po każdym teście
usuwa generated configs i results dir, żeby nie zostawiać śmieci w repo.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREPARE_SCRIPT = PROJECT_ROOT / "experiments" / "prepare_experiment.py"


@pytest.fixture
def mini_definition(tmp_path: Path) -> Path:
    """Minimalna definicja eksperymentu = 4 unikalne joby."""
    definition = {
        "name": "test_pe_integration",
        "runs_per_configuration": 2,
        "parameters_grid": {
            "optimizers": ["msffoa", "ssa"],
            "environments": ["forest"],
            "avoidances": ["none"],
        },
    }
    yaml_path = tmp_path / "mini_definition.yaml"
    yaml_path.write_text(yaml.safe_dump(definition))
    return yaml_path


@pytest.fixture
def cleanup_artifacts():
    """
    Buforuje listy ścieżek do usunięcia po teście. Skrypt
    `prepare_experiment.py` zapisuje do globalnych folderów projektu
    (configs/, results/) — używamy CWD-niezależnych Path-ów z PROJECT_ROOT.
    """
    created: dict[str, list[Path]] = {
        "results_dirs": [],
        "config_files": [],
    }
    yield created

    for path in created["results_dirs"]:
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
    for path in created["config_files"]:
        if path.exists() and path.is_file():
            path.unlink()


def _register_artifacts(cleanup: dict, exp_id: str) -> None:
    """Dopisuje do listy cleanup wszystkie ścieżki utworzone przez prepare."""
    cleanup["results_dirs"].append(PROJECT_ROOT / "results" / exp_id)
    cleanup["config_files"].append(
        PROJECT_ROOT / "configs" / "experiment_generated" / f"{exp_id}.yaml"
    )
    proxy_dir = PROJECT_ROOT / "configs" / "optimizer" / "tmp"
    if proxy_dir.exists():
        for p in proxy_dir.glob(f"{exp_id}_*.yaml"):
            cleanup["config_files"].append(p)


def test_prepare_experiment_populates_registry(
    mini_definition: Path, cleanup_artifacts: dict
) -> None:
    """
    Po wywołaniu prepare_experiment.py registry.db zawiera DOKŁADNIE
    liczbę wpisów wynikającą z definition. Regression dla bugu „0 wpisów".
    """
    expected_runs = 2 * 1 * 1 * 2  # = 4

    proc = subprocess.run(
        [sys.executable, str(PREPARE_SCRIPT), str(mini_definition)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"prepare_experiment.py failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert "Registry initialized" in proc.stdout
    assert f"({expected_runs} entries planned)" in proc.stdout

    # Identyfikacja wygenerowanego folderu (exp_id zawiera datę + uuid + name).
    candidates = sorted((PROJECT_ROOT / "results").glob("exp_*_test_pe_integration"))
    assert candidates, "Brak wygenerowanego folderu results/exp_*_test_pe_integration"
    exp_dir = candidates[-1]
    _register_artifacts(cleanup_artifacts, exp_dir.name)

    registry_path = exp_dir / "registry.db"
    assert registry_path.exists(), "registry.db nie powstał"

    with sqlite3.connect(registry_path) as conn:
        rows = conn.execute(
            "SELECT optimizer, environment, avoidance, seed, status "
            "FROM runs ORDER BY optimizer, seed"
        ).fetchall()

    assert len(rows) == expected_runs, (
        f"oczekiwane {expected_runs} wpisów, dostałem {len(rows)}: {rows}"
    )

    statuses = {r[4] for r in rows}
    assert statuses == {"PENDING"}, f"oczekiwano tylko PENDING, dostałem: {statuses}"

    expected_combos = {
        ("msffoa", "forest", "none", 1),
        ("msffoa", "forest", "none", 2),
        ("ssa", "forest", "none", 1),
        ("ssa", "forest", "none", 2),
    }
    actual_combos = {(r[0], r[1], r[2], r[3]) for r in rows}
    assert actual_combos == expected_combos, (
        f"diff: brakuje {expected_combos - actual_combos}, "
        f"nadmiarowe {actual_combos - expected_combos}"
    )


def test_prepare_experiment_empty_grid_fails(
    tmp_path: Path, cleanup_artifacts: dict
) -> None:
    """
    Pusta siatka eksperymentu MUSI zwrócić niezerowy exit code.
    Bez tej walidacji generowalibyśmy registry.db z 0 wpisami i
    eksperyment by wystartował, ale wszystkie joby UPSERT-owałyby
    wpisy w registry bez `PENDING` baseline'u — niespójność.
    """
    bad_definition = {
        "name": "test_empty_grid",
        "runs_per_configuration": 1,
        "parameters_grid": {
            "optimizers": ["msffoa"],
            "environments": [],  # PUSTE!
            "avoidances": ["none"],
        },
    }
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(yaml.safe_dump(bad_definition))

    proc = subprocess.run(
        [sys.executable, str(PREPARE_SCRIPT), str(yaml_path)],
        capture_output=True,
        text=True,
    )
    # Sprzątamy nawet po failach — exp_id mógł być przypisany przed failem
    candidates = sorted((PROJECT_ROOT / "results").glob("exp_*_test_empty_grid"))
    for c in candidates:
        _register_artifacts(cleanup_artifacts, c.name)

    assert proc.returncode != 0, (
        f"oczekiwano błędu dla pustej siatki, stdout={proc.stdout}"
    )
