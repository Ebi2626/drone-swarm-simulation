#!/usr/bin/env python3
"""
Subprocess-based experiment runner — workaround dla H1'' (plan.md, Krok 5c).

Multi-run w jednym procesie Python (Hydra multirun + joblib, niezależnie od
backend / n_jobs / `reuse=False`) prowadzi do patologii "drony stoją
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
from typing import Dict, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_one_job(args: Dict[str, str]) -> tuple[Dict[str, str], int]:
    """Wywołuje single `python main.py` w fresh subprocess.

    Args są przekazywane jako Hydra overrides. `hydra.run.dir` jest
    wymuszony, by trafiał do `results/{exp_id}/{opt}_{env}_{avoid}_seed{seed}/`,
    czyli tej samej struktury co Hydra-multirun produkował dotychczas.
    """
    exp_id = args["exp_id"]
    subdir = (
        f"{args['optimizer']}_{args['environment']}_"
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
        f"hydra.run.dir=results/{exp_id}/{subdir}",
    ]
    # extra_overrides przekazane przez --override z CLI
    cmd.extend(args.get("extra_overrides", []))

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return args, proc.returncode


def parse_sweep_params(manifest_path: Path) -> Dict[str, List[str]]:
    """Wczytuje multirun.yaml/manifest.yaml i wyciąga sweep params."""
    manifest = yaml.safe_load(manifest_path.read_text())
    sweep = manifest["hydra"]["sweeper"]["params"]
    return {
        "optimizers": str(sweep["optimizer"]).split(","),
        "environments": str(sweep["environment"]).split(","),
        "avoidances": str(sweep["avoidance"]).split(","),
        "seeds": [int(s) for s in str(sweep["seed"]).split(",")],
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
    jobs = [
        {
            "exp_id": parsed.exp_id,
            "optimizer": o,
            "environment": e,
            "avoidance": a,
            "seed": s,
            "extra_overrides": parsed.override,
        }
        for o, e, a, s in product(
            sweep["optimizers"], sweep["environments"],
            sweep["avoidances"], sweep["seeds"],
        )
    ]

    print(f"[run_subprocess] {len(jobs)} jobs, n_jobs={parsed.n_jobs}, "
          f"exp_id={parsed.exp_id}")

    failures = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=parsed.n_jobs) as ex:
        for args, rc in ex.map(run_one_job, jobs):
            completed += 1
            tag = f"{args['optimizer']}/{args['environment']}/seed{args['seed']}"
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
