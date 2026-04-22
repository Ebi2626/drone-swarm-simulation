#!/usr/bin/env python3
import argparse
import uuid
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

def process_optimizers(optimizers_list: list, exp_id: str, configs_dir: Path) -> list:
    """Generuje pliki proxy, które poprawnie dziedziczą i instancjują obiekty Hydry."""
    processed_names = []
    
    # Źródło bazowych configów
    source_optimizer_dir = configs_dir / "optimizer"
    
    # Cel dla wygenerowanych proxy (katalog tmp wewnątrz optimizer)
    target_optimizer_dir = source_optimizer_dir / "tmp"
    target_optimizer_dir.mkdir(parents=True, exist_ok=True)
    
    for opt in optimizers_list:
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
            proxy_name = f"{exp_id}_{base_name}"
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
    n_jobs: 6
    backend: multiprocessing
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

    print("✅ Pomyślnie wygenerowano plik konfiguracji!")
    print(f"ID eksperymentu: {exp_id}")
    print(f"Wygenerowane pliki proxy w: {configs_dir}/optimizer/tmp/")
    print(f"Zaplanowane symulacje: {total_runs}")
    print("\nAby uruchomić eksperyment z optymalizacją wieloprocesorową, wywołaj:\n")
    print(f"./run.sh {exp_id}")
    print("\nAby usunąć pliki po wykonaniu eksperymentu:")
    print(f"rm -rf {configs_dir}/optimizer/tmp/*")

if __name__ == "__main__":
    main()