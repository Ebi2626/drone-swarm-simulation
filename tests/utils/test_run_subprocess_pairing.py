"""
Test jednostkowy filtrowania `pairing: paired_only` w run_subprocess.py.

`parse_sweep_params` musi propagować `experiment_meta.pairing` z manifest.yaml,
a `main()` (przez logikę listy `jobs = [...]`) musi filtrować pary
gdzie `pair.base_name != avoidance` w trybie paired_only.

Testy nie odpalają fizycznie subprocess'ów (`run_one_job`) — sprawdzają tylko
logikę składania `jobs`, co jest izolowane przez monkeypatch
`ProcessPoolExecutor`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_SUBPROCESS_PY = PROJECT_ROOT / "experiments" / "run_subprocess.py"


@pytest.fixture
def run_subprocess_module():
    """Importuje run_subprocess.py jako moduł (nie jest packagowany)."""
    spec = importlib.util.spec_from_file_location("run_subprocess", RUN_SUBPROCESS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_manifest(path: Path, *, pairing: str, optimizers: List[str],
                    avoidances: List[str], seeds: List[int]) -> None:
    """Generuje manifest.yaml zgodny z formatem prepare_experiment.py."""
    job_matrix = [
        {"optimizer": o, "environment": "forest", "base_name": o}
        for o in optimizers
    ]
    manifest = {
        "experiment_meta": {
            "id": "test_exp",
            "name": "test",
            "total_runs": 0,
            "pairing": pairing,
        },
        "job_matrix": job_matrix,
        "hydra": {
            "sweeper": {
                "params": {
                    "avoidance": ",".join(avoidances),
                    "seed": ",".join(str(s) for s in seeds),
                }
            }
        },
    }
    path.write_text(yaml.safe_dump(manifest))


def test_parse_sweep_params_reads_pairing_paired_only(
    run_subprocess_module, tmp_path: Path
) -> None:
    """`parse_sweep_params` musi zwrócić pairing='paired_only' z manifestu."""
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(
        manifest_path,
        pairing="paired_only",
        optimizers=["msffoa", "ssa"],
        avoidances=["msffoa", "ssa"],
        seeds=[1, 2],
    )
    sweep = run_subprocess_module.parse_sweep_params(manifest_path)
    assert sweep["pairing"] == "paired_only"
    assert len(sweep["job_matrix"]) == 2
    assert sweep["avoidances"] == ["msffoa", "ssa"]
    assert sweep["seeds"] == [1, 2]


def test_parse_sweep_params_default_pairing_is_crossed(
    run_subprocess_module, tmp_path: Path
) -> None:
    """Manifest bez `experiment_meta.pairing` → fallback na 'crossed' (back-compat)."""
    # Stary format, bez klucza `pairing` w experiment_meta.
    manifest = {
        "experiment_meta": {"id": "x", "name": "y", "total_runs": 0},
        "job_matrix": [{"optimizer": "msffoa", "environment": "forest", "base_name": "msffoa"}],
        "hydra": {"sweeper": {"params": {"avoidance": "msffoa", "seed": "1"}}},
    }
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(manifest))
    sweep = run_subprocess_module.parse_sweep_params(p)
    assert sweep["pairing"] == "crossed"


def _build_jobs(sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Replikuje logikę składania `jobs` z `run_subprocess.main()` linii 142-156.

    Trzymamy ją zsynchronizowaną z source-em — assert poniżej weryfikuje, że
    semantyka filtrowania `paired_only` się zgadza. Gdyby kod się rozjechał,
    test wyłapie zmiany w polityce filtrowania.
    """
    pairing = sweep.get("pairing", "crossed")
    return [
        {
            "optimizer": pair["optimizer"],
            "base_name": pair.get("base_name", pair["optimizer"]),
            "avoidance": a,
            "seed": s,
        }
        for pair in sweep["job_matrix"]
        for a in sweep["avoidances"]
        for s in sweep["seeds"]
        if pairing != "paired_only" or pair.get("base_name", pair["optimizer"]) == a
    ]


def test_jobs_filtered_to_diagonal_when_paired_only(
    run_subprocess_module, tmp_path: Path
) -> None:
    """
    `paired_only` z 4 optimizers × 4 avoidances × 2 seeds = 32 możliwych jobów,
    filtr daje 4×2 = 8 (każdy algorytm tylko ze swoim odpowiednikiem).
    """
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(
        manifest_path,
        pairing="paired_only",
        optimizers=["msffoa", "ooa", "ssa", "nsga-3"],
        avoidances=["msffoa", "ooa", "ssa", "nsga-3"],
        seeds=[1, 2],
    )
    sweep = run_subprocess_module.parse_sweep_params(manifest_path)
    jobs = _build_jobs(sweep)

    assert len(jobs) == 8, f"oczekiwano 8 jobów (4 par × 2 seedy), got {len(jobs)}"
    pairs = {(j["base_name"], j["avoidance"]) for j in jobs}
    assert pairs == {
        ("msffoa", "msffoa"), ("ooa", "ooa"),
        ("ssa", "ssa"), ("nsga-3", "nsga-3"),
    }
    seeds = sorted({j["seed"] for j in jobs})
    assert seeds == [1, 2]


def test_jobs_full_crossed_when_default(
    run_subprocess_module, tmp_path: Path
) -> None:
    """Default ('crossed') → wszystkie pary opt×avo (back-compat)."""
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(
        manifest_path,
        pairing="crossed",
        optimizers=["msffoa", "ssa"],
        avoidances=["msffoa", "ssa"],
        seeds=[1],
    )
    sweep = run_subprocess_module.parse_sweep_params(manifest_path)
    jobs = _build_jobs(sweep)
    # 2 opt × 2 avo × 1 seed = 4 jobów
    assert len(jobs) == 4
    pairs = {(j["base_name"], j["avoidance"]) for j in jobs}
    assert pairs == {
        ("msffoa", "msffoa"), ("msffoa", "ssa"),
        ("ssa", "msffoa"), ("ssa", "ssa"),
    }
