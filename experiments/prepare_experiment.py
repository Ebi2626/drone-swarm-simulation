#!/usr/bin/env python3
import argparse
import uuid
import shutil
import sys
from datetime import datetime
from pathlib import Path
from itertools import product

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.RunRegistry import RunRegistry

def process_optimizers(optimizers_list: list, exp_id: str, configs_dir: Path) -> list:
    """Generuje pliki proxy, które poprawnie dziedziczą i instancjują obiekty Hydry."""
    processed_names = []
    
    # Źródło bazowych configów
    source_optimizer_dir = configs_dir / "optimizer"
    
    # Cel dla wygenerowanych proxy (katalog tmp wewnątrz optimizer)
    target_optimizer_dir = source_optimizer_dir / "tmp"
    target_optimizer_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, opt in enumerate(optimizers_list):
        if isinstance(opt, str):
            processed_names.append(opt)
        elif isinstance(opt, dict) and "name" in opt:
            base_name = opt["name"]
            overrides = opt.get("overrides", {})
            
            if not overrides:
                processed_names.append(base_name)
                continue
            
            # KROK 1: Wczytanie oryginalnego pliku
            base_file_path = source_optimizer_dir / f"{base_name}.yaml"
            if not base_file_path.exists():
                raise FileNotFoundError(f"Nie znaleziono pliku bazowego: {base_file_path}")
                
            with open(base_file_path, "r", encoding="utf-8") as bf:
                base_content = yaml.safe_load(bf)
                
            # KROK 2: Budowanie struktury pliku proxy
            proxy_content = {}
            for magic_key in ["_target_", "_partial_", "_convert_"]:
                if magic_key in base_content:
                    proxy_content[magic_key] = base_content[magic_key]
                    
            # KROK 3: Rekurencyjna aktualizacja parametrów z naszymi nadpisaniami
            def update_dict(d, u):
                import collections.abc
                for k, v in u.items():
                    if isinstance(v, collections.abc.Mapping):
                        d[k] = update_dict(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
            
            for magic_key in ["_target_", "_partial_", "_convert_"]:
                base_content.pop(magic_key, None)
                
            merged_content = update_dict(base_content, overrides)
            proxy_content.update(merged_content)
            
            # KROK 4: Zapis pliku proxy do podkatalogu tmp
            proxy_name = f"{exp_id}_{base_name}_{idx}"
            proxy_file = target_optimizer_dir / f"{proxy_name}.yaml"
            with open(proxy_file, "w", encoding="utf-8") as f:
                f.write(f"# WYGENEROWANO AUTOMATYCZNIE - PROXY DLA {exp_id}\n")
                yaml.dump(proxy_content, f, default_flow_style=False, sort_keys=False)
                
            # Dopisujemy 'tmp/' przed nazwą, żeby Hydra wiedziała, w którym podkatalogu szukać
            processed_names.append(f"tmp/{proxy_name}")
        else:
            raise ValueError(f"Nieprawidłowy format definicji algorytmu: {opt}")
            
    return processed_names

def generate_yaml_content(exp_id: str, definition: dict) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = definition.get("name", "unnamed_experiment")
    runs = definition.get("runs_per_configuration", 1)
    grid = definition.get("parameters_grid", {})
    
    optimizers = grid.get("optimizers", [])
    environments = grid.get("environments", [])
    avoidances = grid.get("avoidances", [])
    
    if not all([optimizers, environments, avoidances]):
        raise ValueError("Siatka parametrów (optimizers, environments, avoidances) nie może być pusta!")

    seeds = ",".join(str(i) for i in range(1, runs + 1))
    
    opt_str = ",".join(optimizers)
    env_str = ",".join(environments)
    avoid_str = ",".join(avoidances)
    
    total_runs = runs * len(optimizers) * len(environments) * len(avoidances)
    
    static_overrides = definition.get("static_overrides", {})
    static_overrides_str = ""
    for key, value in static_overrides.items():
        static_overrides_str += f"\n  {key}: {value}"

    return f"""# @package _global_
# AUTOMATICALLY GENERATED EXPERIMENT CONFIGURATION
# Date: {date_str}
# Experiment ID: {exp_id}
# Based on definition: {name}

defaults:
  - override /hydra/launcher: joblib

experiment_meta:
  id: "{exp_id}"
  name: "{name}"
  total_runs: {total_runs}

hydra:
  mode: MULTIRUN
  sweep:
    dir: results/${{experiment_meta.id}}
    subdir: ${{hydra:runtime.choices.optimizer}}_${{hydra:runtime.choices.environment}}_${{hydra:runtime.choices.avoidance}}_seed${{seed}}  
  launcher:
    # UWAGA: launcher Hydra-Joblib jest tutaj zachowany dla wstecznej
    # kompatybilności (np. ad-hoc multirun przez `python main.py -m`).
    # ALE pełny eksperyment idzie przez `experiments/run_subprocess.py`
    # (uruchamiane przez `./run.sh`), bo Hydra-multirun + joblib akumuluje
    # globalny stan PyBullet między uruchomieniami w tym samym procesie
    # Python — patologia "drony stoją" (plan.md, Krok 5c, H1'').
    n_jobs: 6
    backend: loky
    prefer: processes
  sweeper:
    params:
      optimizer: {opt_str}
      environment: {env_str}
      avoidance: {avoid_str}
      seed: {seeds}
""" + static_overrides_str

def main():
    parser = argparse.ArgumentParser(description="Generator statycznych konfiguracji z pliku definicji YAML.")
    parser.add_argument("definition_file", type=str, help="Ścieżka do pliku YAML z definicją eksperymentu")
    args = parser.parse_args()
    
    def_path = Path(args.definition_file)
    if not def_path.exists():
        print(f"❌ Błąd: Plik {def_path} nie istnieje.")
        sys.exit(1)
        
    with open(def_path, "r", encoding="utf-8") as f:
        try:
            definition = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(f"❌ Błąd parsowania YAML: {exc}")
            sys.exit(1)
            
    name = definition.get("name", def_path.stem)
    short_uuid = uuid.uuid4().hex[:8]
    date_prefix = datetime.now().strftime("%Y%m%d")
    exp_id = f"exp_{date_prefix}_{short_uuid}_{name}"
    
    project_root = Path(__file__).resolve().parent.parent
    configs_dir = project_root / "configs"
    generated_configs_dir = configs_dir / "experiment_generated"
    results_dir = project_root / "results" / exp_id
    
    raw_optimizers = definition.get("parameters_grid", {}).get("optimizers", [])
    try:
        processed_optimizers = process_optimizers(raw_optimizers, exp_id, configs_dir)
        definition["parameters_grid"]["optimizers"] = processed_optimizers
    except Exception as e:
        print(f"❌ Błąd przetwarzania algorytmów: {e}")
        sys.exit(1)

    generated_configs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        yaml_content = generate_yaml_content(exp_id, definition)
    except ValueError as e:
        print(f"❌ Błąd generowania zawartości: {e}")
        sys.exit(1)
    
    yaml_filename = f"{exp_id}.yaml"
    config_path = generated_configs_dir / yaml_filename
    manifest_path = results_dir / "manifest.yaml"
    
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    shutil.copy(def_path, results_dir / "original_definition.yaml")
    shutil.copy(config_path, manifest_path)
    
    total_runs = definition.get("runs_per_configuration", 1) * \
                 len(processed_optimizers) * \
                 len(definition.get("parameters_grid", {}).get("environments", [])) * \
                 len(definition.get("parameters_grid", {}).get("avoidances", []))

    # --- Sweep params dla RunRegistry: generowane DOKŁADNIE z definicji
    # eksperymentu YAML, nie z hardcoded list. Liczba wpisów PENDING musi się
    # zgadzać z liczbą Hydra-multirun jobów, inaczej `mark_started/completed`
    # (UPDATE w RunRegistry) trafią w nieistniejące wiersze i registry pozostanie
    # niespójny z parquet-em. Patrz plan.md, Krok 6 — bug z 0 wpisami w
    # results/exp_20260426_b9b56922_complex_test/registry.db.
    grid = definition.get("parameters_grid", {})
    runs = int(definition.get("runs_per_configuration", 1))

    # Bazowe nazwy optymalizatorów (string lub dict-with-name) — spójne
    # ze sposobem, w jaki main.py:_get_registry_job_key ekstraktuje krótką
    # nazwę z `cfg.optimizer._target_`. Proxy override'y (`tmp/{exp_id}_X_N`)
    # dzielą bazową nazwę i nie są rozróżniane przez registry — dla wariantów
    # tego samego algorytmu z różnymi overrides obecnie współdzielimy wpis
    # (znane ograniczenie do rozwiązania w przyszłym kroku).
    base_optimizer_names: list[str] = []
    for o in raw_optimizers:
        if isinstance(o, str):
            base_optimizer_names.append(o)
        elif isinstance(o, dict) and "name" in o:
            base_optimizer_names.append(o["name"])
        else:
            raise ValueError(f"Nieprawidłowa definicja optymalizatora: {o}")

    unique_optimizers = list(dict.fromkeys(base_optimizer_names))
    if len(unique_optimizers) != len(base_optimizer_names):
        print(
            f"⚠ Wykryto powtarzające się bazowe nazwy optymalizatorów: "
            f"{base_optimizer_names} — registry użyje unikatów {unique_optimizers}. "
            f"Warianty z overrides współdzielą wpisy."
        )

    env_names = list(grid.get("environments", []))
    avoid_names = list(grid.get("avoidances", []))
    seed_values = list(range(1, runs + 1))

    sweep_params = [
        {"optimizer": o, "environment": e, "avoidance": a, "seed": s}
        for o, e, a, s in product(unique_optimizers, env_names, avoid_names, seed_values)
    ]

    if not sweep_params:
        raise ValueError(
            "Pusty sweep — `parameters_grid` w YAML nie definiuje żadnej "
            "kombinacji (optimizer × environment × avoidance × seed)."
        )

    registry = RunRegistry(f"results/{exp_id}/registry.db")
    registry.populate(sweep_params)
    summary = registry.get_summary()
    print(
        f"Registry initialized: {summary} "
        f"({len(sweep_params)} entries planned)"
    )

    # Sanity check: po populate liczba PENDING musi się zgadzać z planem.
    pending_in_db = summary.get("PENDING", 0)
    if pending_in_db != len(sweep_params):
        print(
            f"⚠ Registry niespójny: PENDING={pending_in_db} ≠ "
            f"sweep_params={len(sweep_params)}. Sprawdź czy baza nie zawiera "
            f"starych wpisów z innego eksperymentu."
        )

    print("✅ Pomyślnie wygenerowano plik konfiguracji!")
    print(f"ID eksperymentu: {exp_id}")
    print(f"Wygenerowane pliki proxy w: {configs_dir}/optimizer/tmp/")
    print(f"Zaplanowane symulacje: {total_runs}")
    print("\nAby usunąć pliki po wykonaniu eksperymentu:")
    print(f"rm -rf {configs_dir}/optimizer/tmp/* && rm -rf {configs_dir}/experiment_generated/*")
    print("\nAby uruchomić eksperyment z optymalizacją wieloprocesorową, wywołaj:\n")
    print(f"./run.sh {exp_id}")

if __name__ == "__main__":
    main()