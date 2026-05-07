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
    # Proxy pliki: `configs/optimizer/_proxy_<exp_id>_*.yaml` (flat, bez slasha
    # — zob. `_write_proxy_yaml` w `prepare_experiment.py`).
    proxy_dir = PROJECT_ROOT / "configs" / "optimizer"
    if proxy_dir.exists():
        for p in proxy_dir.glob(f"_proxy_{exp_id}_*.yaml"):
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


def test_static_overrides_from_parameters_grid(tmp_path: Path) -> None:
    """`parameters_grid.<non-sweep>` (np. `simulation`) powinno wylądować
    na top-level manifestu jako static override (deep-merge przez Hydra).

    Regression: do 2026-05-06 `prepare_experiment.generate_yaml_content`
    czytało static overrides tylko z top-level klucza `static_overrides`,
    a definicje w repo (`per_env_test.yaml`, `complex_test.yaml`) używają
    konwencji `parameters_grid.simulation`. Efekt: blok był silnie
    upuszczany.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.prepare_experiment import (  # noqa: E402
        _collect_static_overrides,
        generate_yaml_content,
    )

    definition = {
        "name": "test_static",
        "runs_per_configuration": 1,
        "parameters_grid": {
            "optimizers": ["msffoa"],
            "environments": ["forest"],
            "avoidances": ["none"],
            "simulation": {"dynamic_obstacles": True, "gui": False},
            "logging": {"enabled": False},
        },
        "static_overrides": {"seed": 7},  # ma wygrać nad grid w razie kolizji
    }

    overrides = _collect_static_overrides(definition)
    assert overrides == {
        "simulation": {"dynamic_obstacles": True, "gui": False},
        "logging": {"enabled": False},
        "seed": 7,
    }

    fake_jm = [{"optimizer": "msffoa", "environment": "forest", "base_name": "msffoa"}]
    content = generate_yaml_content("exp_test", definition, fake_jm)
    parsed = yaml.safe_load(content)
    # Static overrides siedzą na top-level, nie wewnątrz `hydra:`.
    assert parsed["simulation"] == {"dynamic_obstacles": True, "gui": False}
    assert parsed["logging"] == {"enabled": False}
    assert parsed["seed"] == 7
    # Sweep wymiary nie wchodzą w static.
    assert "optimizers" not in parsed
    assert "environments" not in parsed
    assert "avoidances" not in parsed


def test_proxy_optimizer_refs_have_no_slash(tmp_path: Path) -> None:
    """Proxy YAML-e (z `overrides` lub `env_overrides`) muszą produkować
    optimizer-ref BEZ slasha — inaczej Hydra-multirun w `sweep.subdir`
    template `${hydra:runtime.choices.optimizer}_..._seed${seed}` rozbija
    wyniki na `results/<exp>/<prefix>/<rest>/`, a ETL
    (`list_run_directories` iterdir top-level) nie wykrywa zagnieżdżonych
    runów.

    Regression: do 2026-05-06 proxy lądowały w `configs/optimizer/tmp/`,
    a `optimizer` w `job_matrix` był `tmp/<proxy_name>` → `/` w referencji.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.prepare_experiment import (  # noqa: E402
        PROXY_PREFIX,
        expand_optimizers_for_environments,
    )

    optimizers = [
        # Bez overrides — flat ref (bare base_name).
        "msffoa",
        # base_overrides only — base proxy.
        {"name": "ssa", "overrides": {"algorithm_params": {"epochs": 80}}},
        # env_overrides — per-env proxy.
        {"name": "msffoa", "env_overrides": {
            "forest": {"algorithm_params": {"pop_size": 200}},
        }},
    ]
    environments = ["forest", "urban"]

    # Tymczasowy `configs/optimizer/` z minimalnymi base files żeby _write_proxy_yaml
    # miał co czytać.
    tmp_configs = tmp_path / "configs"
    opt_dir = tmp_configs / "optimizer"
    opt_dir.mkdir(parents=True)
    for name in ("msffoa", "ssa"):
        (opt_dir / f"{name}.yaml").write_text(
            "_target_: src.algorithms.dummy.Dummy\nalgorithm_params:\n  pop_size: 50\n"
        )

    job_matrix, base_names = expand_optimizers_for_environments(
        optimizers, environments, "exp_test_xxx", tmp_configs,
    )

    for entry in job_matrix:
        assert "/" not in entry["optimizer"], (
            f"optimizer ref `{entry['optimizer']}` zawiera slash — "
            f"powodowałoby zagnieżdżenie w results/<exp>/.../"
        )

    # Dodatkowo: proxy które trzeba było wygenerować mają prefix `_proxy_`,
    # a bare base_name pozostaje bare.
    proxy_refs = [e["optimizer"] for e in job_matrix if e["optimizer"] != e["base_name"]]
    assert proxy_refs, "Test fixtures wymuszają min. 1 proxy"
    for ref in proxy_refs:
        assert ref.startswith(PROXY_PREFIX), f"{ref} nie ma prefiksu {PROXY_PREFIX}"

    # Sanity: bezpośrednia referencja `msffoa` (bez overrides) jest bare.
    bare_refs = [e for e in job_matrix if e["optimizer"] == "msffoa" and e["base_name"] == "msffoa"]
    assert bare_refs, "Bare msffoa-bez-overrides powinien się znaleźć w job_matrix"


def test_pairing_paired_only_filters_to_diagonal(
    tmp_path: Path, cleanup_artifacts: dict
) -> None:
    """
    `pairing: paired_only` → registry zawiera tylko wpisy z optimizer == avoidance.

    Dla 2 optimizers × 1 env × 2 avoidances × 2 seedy:
      - default 'crossed': 2×1×2×2 = 8 wpisów (wszystkie pary opt-avo)
      - 'paired_only': 2×1×2 = 4 wpisy (tylko msffoa-msffoa, ssa-ssa, ×2 seedy)

    Plus weryfikacja, że manifest.yaml ma `experiment_meta.pairing: paired_only`
    — `run_subprocess.py` polega na tym przy filtrowaniu jobs.
    """
    definition = {
        "name": "test_paired",
        "runs_per_configuration": 2,
        "pairing": "paired_only",
        "parameters_grid": {
            "optimizers": ["msffoa", "ssa"],
            "environments": ["forest"],
            "avoidances": ["msffoa", "ssa"],
        },
    }
    yaml_path = tmp_path / "paired.yaml"
    yaml_path.write_text(yaml.safe_dump(definition))

    proc = subprocess.run(
        [sys.executable, str(PREPARE_SCRIPT), str(yaml_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"prepare_experiment.py failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )

    candidates = sorted((PROJECT_ROOT / "results").glob("exp_*_test_paired"))
    assert candidates
    exp_dir = candidates[-1]
    _register_artifacts(cleanup_artifacts, exp_dir.name)

    # Manifest yaml: pairing przekazane do run_subprocess.py.
    manifest = yaml.safe_load((exp_dir / "manifest.yaml").read_text())
    assert manifest["experiment_meta"]["pairing"] == "paired_only"
    assert manifest["experiment_meta"]["total_runs"] == 4

    # Registry: dokładnie 4 wpisy, każdy z opt == avo.
    registry_path = exp_dir / "registry.db"
    with sqlite3.connect(registry_path) as conn:
        rows = conn.execute(
            "SELECT optimizer, environment, avoidance, seed FROM runs "
            "ORDER BY optimizer, seed"
        ).fetchall()

    assert len(rows) == 4, f"expected 4 paired runs, got {len(rows)}: {rows}"
    expected = {
        ("msffoa", "forest", "msffoa", 1),
        ("msffoa", "forest", "msffoa", 2),
        ("ssa", "forest", "ssa", 1),
        ("ssa", "forest", "ssa", 2),
    }
    assert set(rows) == expected, f"got {set(rows)}, expected {expected}"


def test_pairing_paired_only_rejects_unmatched_optimizer(
    tmp_path: Path, cleanup_artifacts: dict
) -> None:
    """
    `pairing: paired_only` z optimizer-em bez odpowiednika w avoidances → fail
    z czytelnym komunikatem (zamiast cicho generować 0 runów).
    """
    definition = {
        "name": "test_paired_unmatched",
        "runs_per_configuration": 1,
        "pairing": "paired_only",
        "parameters_grid": {
            "optimizers": ["msffoa", "ooa"],   # ooa nie jest w avoidances
            "environments": ["forest"],
            "avoidances": ["msffoa"],
        },
    }
    yaml_path = tmp_path / "bad_paired.yaml"
    yaml_path.write_text(yaml.safe_dump(definition))

    proc = subprocess.run(
        [sys.executable, str(PREPARE_SCRIPT), str(yaml_path)],
        capture_output=True,
        text=True,
    )
    candidates = sorted((PROJECT_ROOT / "results").glob("exp_*_test_paired_unmatched"))
    for c in candidates:
        _register_artifacts(cleanup_artifacts, c.name)

    assert proc.returncode != 0
    assert "paired_only" in proc.stdout or "paired_only" in proc.stderr
    assert "ooa" in proc.stdout or "ooa" in proc.stderr


def test_pairing_invalid_value_fails(
    tmp_path: Path, cleanup_artifacts: dict
) -> None:
    """Wartość spoza {'crossed', 'paired_only'} → błąd."""
    definition = {
        "name": "test_paired_invalid",
        "runs_per_configuration": 1,
        "pairing": "diagonal",   # nieznane
        "parameters_grid": {
            "optimizers": ["msffoa"],
            "environments": ["forest"],
            "avoidances": ["msffoa"],
        },
    }
    yaml_path = tmp_path / "invalid_pairing.yaml"
    yaml_path.write_text(yaml.safe_dump(definition))

    proc = subprocess.run(
        [sys.executable, str(PREPARE_SCRIPT), str(yaml_path)],
        capture_output=True,
        text=True,
    )
    candidates = sorted((PROJECT_ROOT / "results").glob("exp_*_test_paired_invalid"))
    for c in candidates:
        _register_artifacts(cleanup_artifacts, c.name)

    assert proc.returncode != 0
    assert "pairing" in proc.stdout or "pairing" in proc.stderr


def test_pairing_default_remains_crossed(
    tmp_path: Path, cleanup_artifacts: dict
) -> None:
    """
    Brak klucza `pairing` w definicji → fallback do 'crossed' (back-compat).

    Krytyczne dla istniejących plików w experiments/definitions/ — żaden
    z nich nie ma klucza `pairing` i nie powinno to zmieniać semantyki.
    """
    definition = {
        "name": "test_default_crossed",
        "runs_per_configuration": 1,
        "parameters_grid": {
            "optimizers": ["msffoa", "ssa"],
            "environments": ["forest"],
            "avoidances": ["msffoa", "ssa"],
        },
    }
    yaml_path = tmp_path / "default.yaml"
    yaml_path.write_text(yaml.safe_dump(definition))

    proc = subprocess.run(
        [sys.executable, str(PREPARE_SCRIPT), str(yaml_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    candidates = sorted((PROJECT_ROOT / "results").glob("exp_*_test_default_crossed"))
    exp_dir = candidates[-1]
    _register_artifacts(cleanup_artifacts, exp_dir.name)

    # Default: pairing == crossed, total_runs = 2×1×2×1 = 4
    manifest = yaml.safe_load((exp_dir / "manifest.yaml").read_text())
    assert manifest["experiment_meta"]["pairing"] == "crossed"
    assert manifest["experiment_meta"]["total_runs"] == 4

    # Registry zawiera wszystkie 4 pary (crossed): msffoa-msffoa, msffoa-ssa,
    # ssa-msffoa, ssa-ssa. Czyli 2 opt × 2 avo × 1 env × 1 seed.
    with sqlite3.connect(exp_dir / "registry.db") as conn:
        rows = conn.execute(
            "SELECT optimizer, avoidance FROM runs ORDER BY optimizer, avoidance"
        ).fetchall()
    assert sorted(rows) == sorted([
        ("msffoa", "msffoa"), ("msffoa", "ssa"),
        ("ssa", "msffoa"), ("ssa", "ssa"),
    ])


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
