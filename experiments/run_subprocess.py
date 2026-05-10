#!/usr/bin/env python3
"""Subprocess-based experiment runner — workaround dla globalnego stanu PyBullet.

Multi-run w jednym procesie Python (Hydra multirun + joblib, niezależnie od
backend / n_jobs / `reuse=False`) prowadzi do patologii „drony stoją
w PyBullet" w drugim i kolejnych jobach. Force-cleanup `p.resetSimulation` /
`p.disconnect` / `gc.collect` nie pomaga — globalny stan akumuluje się
w libbullet C-level lub innym module zewnętrznym (gym-pybullet-drones,
Numba JIT cache itp.).

Workaround: każde zadanie w fresh subprocess `python main.py`, koordynowane
przez `ProcessPoolExecutor`. Narzut ~1-2s startup per job — pomijalny przy
~10-20 min jobach.

Usage:
    python experiments/run_subprocess.py exp_20260426_xxx_complex_test \
        [--n-jobs 6]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_one_job(args: Dict[str, str]) -> tuple[Dict[str, str], int]:
    """Wywołuje single `python main.py` w fresh subprocess.

    Args są przekazywane jako Hydra overrides. `hydra.run.dir` jest
    wymuszony, by trafiał do `results/{exp_id}/{base_name}_{env}_{avoid}_seed{seed}/`,
    czyli tej samej struktury co Hydra-multirun produkował dotychczas.
    Pole `base_name` (krótka nazwa, np. "msffoa") jest używane do nazewnictwa
    katalogu zamiast pełnej ścieżki proxy (`tmp/exp_..._msffoa_forest_0`),
    żeby `parse_run_dir_name` regex pasował i nie wprowadzać `/` do path
    component.
    """
    exp_id = args["exp_id"]
    base_name = args.get("base_name", args["optimizer"])
    subdir = (
        f"{base_name}_{args['environment']}_"
        f"{args['avoidance']}_seed{args['seed']}"
    )
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        f"+experiment_generated={exp_id}",
        f"optimizer={args['optimizer']}",
        f"environment={args['environment']}",
        f"avoidance={args['avoidance']}",
        f"seed={args['seed']}",
        # `+experiment_generated=...` wstrzykuje `hydra.mode: MULTIRUN` z manifestu.
        # Bez wymuszenia `RUN` Hydra wchodzi w tryb sweep nawet dla jednego
        # joba i używa `sweep.subdir` template `${hydra:runtime.choices.optimizer}_...`,
        # ignorując jawne `hydra.run.dir`. Skutek przy proxy optimizera typu
        # `tmp/exp_xxx_msffoa_forest_0`: `/` tworzy zagnieżdżony katalog
        # `results/exp_xxx/tmp/exp_xxx_msffoa_forest_0_<env>_<avoid>_seed<N>/`,
        # czego ETL (`list_run_directories` używa iterdir top-level + regex
        # `^opt_(forest|urban)_avoid_seedN$`) nie podchwytuje — runy giną.
        # `mode=RUN` przywraca pojedynczy katalog z `hydra.run.dir`.
        "hydra.mode=RUN",
        f"hydra.run.dir=results/{exp_id}/{subdir}",
    ]
    cmd.extend(args.get("extra_overrides", []))

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return args, proc.returncode


def parse_sweep_params(manifest_path: Path) -> Dict[str, Any]:
    """Wczytuje manifest.yaml i wyciąga listę par (optimizer, environment) +
    pozostałe wymiary kartezjańskie (avoidance, seed).

    Format Option A (preferowany): top-level `job_matrix` z jawnymi parami,
    `hydra.sweeper.params` zawiera tylko `avoidance` + `seed`.

    Format legacy (back-compat): `hydra.sweeper.params` zawiera
    `optimizer` + `environment` + `avoidance` + `seed`. Pary tworzone jako
    pełny iloczyn kartezjański.
    """
    manifest = yaml.safe_load(manifest_path.read_text())
    sweep = manifest.get("hydra", {}).get("sweeper", {}).get("params", {})

    avoidances = str(sweep["avoidance"]).split(",")
    seeds = [int(s) for s in str(sweep["seed"]).split(",")]

    if "job_matrix" in manifest:
        # Option A: pary (optimizer, environment) zdefiniowane jawnie.
        job_matrix = list(manifest["job_matrix"])
        for pair in job_matrix:
            if not isinstance(pair, dict) or "optimizer" not in pair or "environment" not in pair:
                raise ValueError(
                    f"Nieprawidłowy wpis w job_matrix: {pair!r}. "
                    f"Wymagane klucze: optimizer, environment."
                )
    else:
        # Legacy: kartezjański optimizer × environment.
        optimizers = str(sweep["optimizer"]).split(",")
        environments = str(sweep["environment"]).split(",")
        job_matrix = [
            {"optimizer": o, "environment": e}
            for o in optimizers for e in environments
        ]

    # `experiment_meta.pairing` zapisany przez prepare_experiment.py:
    #   - 'crossed' (default, back-compat): pełen kartezjan job_matrix × avoidance.
    #   - 'paired_only': filtr do par pair["base_name"] == avoidance.
    pairing = manifest.get("experiment_meta", {}).get("pairing", "crossed")

    return {
        "job_matrix": job_matrix,
        "avoidances": avoidances,
        "seeds": seeds,
        "pairing": pairing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_id", help="ID eksperymentu wygenerowane przez prepare_experiment.py")
    parser.add_argument("--n-jobs", type=int, default=6,
                        help="Liczba równoległych subprocess (default 6).")
    parser.add_argument("--override", action="append", default=[],
                        help="Dodatkowy Hydra override przekazany do każdego "
                             "subprocess (np. simulation.gui=false). Wielokrotnie.")
    parsed = parser.parse_args()

    exp_dir = PROJECT_ROOT / "results" / parsed.exp_id
    manifest_path = exp_dir / "manifest.yaml"
    if not manifest_path.exists():
        print(f"❌ Brak manifest.yaml w {exp_dir}. Uruchom najpierw "
              f"prepare_experiment.py.", file=sys.stderr)
        return 1

    sweep = parse_sweep_params(manifest_path)
    pairing = sweep.get("pairing", "crossed")
    # `paired_only`: każdy run używa SWOJEGO algorytmu zarówno offline (optimizer)
    # jak i online (avoidance). Filtr: pair["base_name"] == a. Bez tej flagi
    # (default 'crossed') odpalamy pełen kartezjan jak dotąd.
    jobs = [
        {
            "exp_id": parsed.exp_id,
            "optimizer": pair["optimizer"],
            "environment": pair["environment"],
            "base_name": pair.get("base_name", pair["optimizer"]),
            "avoidance": a,
            "seed": s,
            "extra_overrides": parsed.override,
        }
        for pair in sweep["job_matrix"]
        for a in sweep["avoidances"]
        for s in sweep["seeds"]
        if pairing != "paired_only" or pair.get("base_name", pair["optimizer"]) == a
    ]

    print(f"[run_subprocess] {len(jobs)} jobs, n_jobs={parsed.n_jobs}, "
          f"exp_id={parsed.exp_id}, pairing={pairing}")

    failures = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=parsed.n_jobs) as ex:
        for args, rc in ex.map(run_one_job, jobs):
            completed += 1
            tag = f"{args['base_name']}/{args['environment']}/seed{args['seed']}"
            if rc != 0:
                failures += 1
                print(f"[run_subprocess] {completed}/{len(jobs)} ❌ {tag} (rc={rc})")
            else:
                print(f"[run_subprocess] {completed}/{len(jobs)} ✓ {tag}")

    print(f"\n[run_subprocess] Done: {completed - failures}/{len(jobs)} OK, "
          f"{failures} failed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
