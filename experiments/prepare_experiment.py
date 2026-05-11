#!/usr/bin/env python3
"""Generator konfiguracji eksperymentów z plików definicji YAML.

Wczytuje `definition.yaml` opisujący `parameters_grid`, rozwija pary
(optimizer × environment) z opcjonalnymi `env_overrides`, generuje
pliki proxy w `configs/optimizer/_proxy_*.yaml`, zapisuje `manifest.yaml`
w `results/<exp_id>/` i inicjalizuje `RunRegistry` rekordami PENDING.
"""
import argparse
import collections.abc
import copy
import uuid
import shutil
import sys
from datetime import datetime
from pathlib import Path
from itertools import product
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.RunRegistry import RunRegistry


def _deep_update(d: dict, u: collections.abc.Mapping) -> dict:
    """Rekurencyjnie scal `u` w `d`: skalary nadpisują, słowniki schodzą głębiej.

    Args:
        d: Słownik docelowy, modyfikowany w miejscu.
        u: Słownik z wartościami nadpisującymi.

    Returns:
        Ten sam `d` po scaleniu (zwracany dla wygody łańcuchowania wywołań).
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _deep_update(d.get(k, {}) or {}, v)
        else:
            d[k] = v
    return d


# Prefiks dla proxy YAML-i — flat, bez slasha. Slash w referencji
# (poprzednio `tmp/<name>`) powodował zagnieżdżenie w `results/<exp>/tmp/...`,
# bo Hydra w MULTIRUN używa `${hydra:runtime.choices.optimizer}` w sweep.subdir
# template; `/` w wartości tworzył zagnieżdżony katalog. ETL
# (`list_run_directories` iterdir top-level) nie podchwytywał takich runów.
PROXY_PREFIX = "_proxy_"


def _write_proxy_yaml(
    target_dir: Path,
    exp_id: str,
    base_name: str,
    idx: int,
    suffix: Optional[str],
    base_content: dict,
    overrides: collections.abc.Mapping,
) -> str:
    """Zapisz plik proxy łączący `base_content` z `overrides`.

    Plik trafia do `configs/optimizer/` z przedrostkiem `_proxy_<exp_id>_...`,
    aby odwołanie Hydra (`optimizer=<nazwa>`) nie zawierało ukośnika
    (ukośnik w nazwie grupy konfiguracyjnej powodowałby zagnieżdżenie wyników).

    Args:
        target_dir: Katalog docelowy (`configs/optimizer/`).
        exp_id: Identyfikator eksperymentu — wchodzi w nazwę pliku.
        base_name: Nazwa bazowa optymalizatora (np. `"msffoa"`).
        idx: Indeks w `optimizers_list` zapewniający unikalność.
        suffix: Przyrostek środowiska (np. `"forest"`); `None` ⇒ jeden
            wspólny plik proxy dla wszystkich środowisk.
        base_content: Wczytana zawartość bazowego pliku YAML optymalizatora.
        overrides: Mapa nadpisań do scalenia z `base_content`.

    Returns:
        Nazwa pliku proxy bez rozszerzenia (np.
        `"_proxy_exp_xxx_msffoa_forest_0"`), gotowa jako wartość grupy
        konfiguracyjnej Hydra.
    """
    base_copy = copy.deepcopy(base_content)
    magic = {k: base_copy.pop(k) for k in ("_target_", "_partial_", "_convert_") if k in base_copy}

    merged = _deep_update(base_copy, overrides) if overrides else base_copy
    proxy_content = {**magic, **merged}

    if suffix is None:
        proxy_name = f"{PROXY_PREFIX}{exp_id}_{base_name}_{idx}"
    else:
        proxy_name = f"{PROXY_PREFIX}{exp_id}_{base_name}_{suffix}_{idx}"

    proxy_file = target_dir / f"{proxy_name}.yaml"
    with open(proxy_file, "w", encoding="utf-8") as f:
        f.write(f"# WYGENEROWANO AUTOMATYCZNIE - PROXY DLA {exp_id}\n")
        if suffix is not None:
            f.write(f"# Per-environment overrides: env={suffix}\n")
        yaml.dump(proxy_content, f, default_flow_style=False, sort_keys=False)
    return proxy_name


def expand_optimizers_for_environments(
    optimizers_list: list,
    environments: list,
    exp_id: str,
    configs_dir: Path,
) -> tuple[list[dict], list[str]]:
    """Rozwiń pary optymalizator × środowisko w macierz zadań, tworząc pliki proxy.

    Trzy dopuszczalne formaty wpisu w `optimizers_list`:
    - `"foo"` — bez nadpisań; jedna nazwa dla wszystkich środowisk.
    - `{name, overrides}` — wspólny plik proxy z polem `overrides`.
    - `{name, env_overrides: {env: {...}}}` — osobny plik proxy
      dla każdego środowiska.

    Args:
        optimizers_list: Wpisy z `parameters_grid.optimizers` — łańcuchy
            znaków albo słowniki w jednym z trzech formatów wyżej.
        environments: Lista nazw środowisk (`parameters_grid.environments`).
        exp_id: Identyfikator eksperymentu — wchodzi w nazwy plików proxy.
        configs_dir: Korzeń `configs/` projektu (zapis trafia do `configs/optimizer/`).

    Returns:
        Para `(job_matrix, base_names)`:
        - `job_matrix`: lista słowników
          `{"optimizer": <ref>, "environment": <env>, "base_name": <orig>}`.
          `<ref>` to nazwa pliku proxy (gdy są nadpisania) lub nazwa bazowa.
          `base_name` zachowuje zgodność z `parse_run_dir_name`.
        - `base_names`: lista nazw bazowych w kolejności zgłoszenia,
          z duplikatami.

    Raises:
        ValueError: Gdy wpis nie jest łańcuchem znaków ani słownikiem
            z kluczem `name`.
        FileNotFoundError: Gdy bazowy `configs/optimizer/<name>.yaml` nie istnieje.
    """
    # Proxy YAML-e lądują flat w `configs/optimizer/` z prefiksem `_proxy_`.
    # Wcześniej w `tmp/`, ale slash w referencji Hydra-config-group powodował
    # zagnieżdżenie wyników w `results/<exp>/tmp/...` (patrz `_write_proxy_yaml`).
    source_dir = configs_dir / "optimizer"
    target_dir = source_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    job_matrix: list[dict] = []
    base_names: list[str] = []

    for idx, opt in enumerate(optimizers_list):
        # 1) String — brak overrides ani env_overrides.
        if isinstance(opt, str):
            base_names.append(opt)
            for env in environments:
                job_matrix.append({"optimizer": opt, "environment": env, "base_name": opt})
            continue

        if not isinstance(opt, dict) or "name" not in opt:
            raise ValueError(f"Nieprawidłowy format definicji algorytmu: {opt}")

        base_name = opt["name"]
        base_names.append(base_name)
        base_overrides = opt.get("overrides") or {}
        env_overrides_map = opt.get("env_overrides") or {}

        # 2) Brak overrides w ogóle — jak string.
        if not base_overrides and not env_overrides_map:
            for env in environments:
                job_matrix.append({"optimizer": base_name, "environment": env, "base_name": base_name})
            continue

        # Do scalania potrzebny base_content z dysku (raz na optimizer).
        base_file_path = source_dir / f"{base_name}.yaml"
        if not base_file_path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku bazowego: {base_file_path}")
        with open(base_file_path, "r", encoding="utf-8") as bf:
            base_content = yaml.safe_load(bf)

        # Cache pojedynczego bazowego proxy (gdy `overrides` istnieje, ale env
        # nie ma własnego nadpisania).
        base_proxy_ref: Optional[str] = None

        def _ensure_base_proxy() -> str:
            nonlocal base_proxy_ref
            if base_proxy_ref is None:
                proxy_name = _write_proxy_yaml(
                    target_dir, exp_id, base_name, idx,
                    suffix=None, base_content=base_content, overrides=base_overrides,
                )
                # `proxy_name` zawiera już prefiks `_proxy_` — ref bez slasha.
                base_proxy_ref = proxy_name
            return base_proxy_ref

        for env in environments:
            if env in env_overrides_map:
                # Scal: base_overrides + env_overrides[env] (env wins).
                merged = _deep_update(copy.deepcopy(base_overrides), env_overrides_map[env])
                proxy_name = _write_proxy_yaml(
                    target_dir, exp_id, base_name, idx,
                    suffix=env, base_content=base_content, overrides=merged,
                )
                job_matrix.append({
                    "optimizer": proxy_name,
                    "environment": env,
                    "base_name": base_name,
                })
            elif base_overrides:
                job_matrix.append({
                    "optimizer": _ensure_base_proxy(),
                    "environment": env,
                    "base_name": base_name,
                })
            else:
                job_matrix.append({
                    "optimizer": base_name,
                    "environment": env,
                    "base_name": base_name,
                })

    return job_matrix, base_names

SWEEP_KEYS = ("optimizers", "environments", "avoidances")


def _collect_static_overrides(definition: dict) -> dict:
    """Zbierz nadpisania wstrzykiwane na najwyższym poziomie manifestu (głębokie scalanie).

    Args:
        definition: Wczytana definicja eksperymentu (YAML).

    Returns:
        Słownik nadpisań scalonych z dwóch źródeł (drugie ma pierwszeństwo):
        1. `parameters_grid.<klucz>` dla kluczy spoza zbioru
           {`optimizers`, `environments`, `avoidances`}.
        2. `static_overrides` z poziomu najwyższego.
    """
    grid = definition.get("parameters_grid", {}) or {}
    overrides: dict = {}
    for key, value in grid.items():
        if key in SWEEP_KEYS:
            continue
        overrides[key] = value
    for key, value in (definition.get("static_overrides") or {}).items():
        overrides[key] = value
    return overrides


PAIRING_MODES = ("crossed", "paired_only")


def _resolve_pairing(definition: dict) -> str:
    """Zwróć tryb parowania z definicji (domyślnie `'crossed'`).

    Args:
        definition: Wczytana definicja eksperymentu (YAML).

    Returns:
        `'crossed'` — pełny iloczyn kartezjański
        optymalizator × środowisko × unik × ziarno.
        `'paired_only'` — filtr `optimizer == avoidance`.

    Raises:
        ValueError: Gdy wartość jest spoza `PAIRING_MODES`.
    """
    pairing = definition.get("pairing", "crossed")
    if pairing not in PAIRING_MODES:
        raise ValueError(
            f"`pairing` musi być jednym z {PAIRING_MODES}, "
            f"otrzymano: {pairing!r}"
        )
    return pairing


def generate_yaml_content(
    exp_id: str,
    definition: dict,
    job_matrix: list[dict],
) -> str:
    """Zbuduj treść `manifest.yaml` (oraz `experiment_generated/<exp>.yaml`).

    Pary (optymalizator × środowisko) trafiają na najwyższy poziom jako
    `job_matrix`; pozostałe wymiary (unik × ziarno) lądują w
    `hydra.sweeper.params` i są rozwijane przez `run_subprocess.py`.

    Args:
        exp_id: Identyfikator eksperymentu — używany w `experiment_meta.id`
            i `sweep.dir`.
        definition: Wczytana definicja eksperymentu (YAML).
        job_matrix: Wynik `expand_optimizers_for_environments`.

    Returns:
        Pełna treść manifestu YAML do zapisu na dysk.

    Raises:
        ValueError: Gdy `job_matrix` jest puste, brak `avoidances` w siatce
            albo tryb `paired_only` nie ma żadnej pasującej pary.
    """
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = definition.get("name", "unnamed_experiment")
    runs = definition.get("runs_per_configuration", 1)
    grid = definition.get("parameters_grid", {})
    pairing = _resolve_pairing(definition)

    avoidances = grid.get("avoidances", [])

    if not job_matrix:
        raise ValueError("`job_matrix` jest puste — brak par (optimizer, environment).")
    if not avoidances:
        raise ValueError("`parameters_grid.avoidances` nie może być puste!")

    seeds = ",".join(str(i) for i in range(1, runs + 1))
    avoid_str = ",".join(avoidances)

    if pairing == "paired_only":
        # paired_only: tylko pary, gdzie optimizer (po expansion: pair["base_name"])
        # ma odpowiednik w avoidances. Liczba runów = runs × |valid_pairs| × env_count
        # (env-count już wbudowany w job_matrix — każdy entry to (opt, env)).
        valid_pairs_count = sum(1 for jm in job_matrix if jm.get("base_name") in avoidances)
        if valid_pairs_count == 0:
            raise ValueError(
                "`pairing: paired_only` ale żaden optimizer nie ma odpowiednika "
                f"w avoidances. Optimizers w job_matrix: "
                f"{sorted({jm.get('base_name') for jm in job_matrix})}, "
                f"avoidances: {avoidances}"
            )
        total_runs = runs * valid_pairs_count
    else:
        total_runs = runs * len(job_matrix) * len(avoidances)

    # Serializujemy job_matrix przez yaml.dump dla czytelności (block style).
    # Wcięcie 2 spacje, by trafiło pod top-level klucz w finalnym pliku.
    job_matrix_yaml = yaml.dump(
        {"job_matrix": job_matrix},
        default_flow_style=False,
        sort_keys=False,
    ).rstrip()

    static_overrides = _collect_static_overrides(definition)
    if static_overrides:
        static_overrides_yaml = (
            "\n# Static overrides (parameters_grid.<non-sweep> + static_overrides)\n"
            + yaml.dump(static_overrides, default_flow_style=False, sort_keys=False).rstrip()
            + "\n"
        )
    else:
        static_overrides_yaml = ""

    return f"""# @package _global_
# AUTOMATICALLY GENERATED EXPERIMENT CONFIGURATION
# Date: {date_str}
# Experiment ID: {exp_id}
# Based on definition: {name}
#
# Format Option A (per-environment overrides):
#   `job_matrix` zawiera jawne pary (optimizer, environment). Generator
#   `prepare_experiment.py` tworzy proxy yamls per (optimizer × env)
#   gdy zdefiniowano `env_overrides`. Czytane przez
#   `experiments/run_subprocess.py`.

defaults:
  - override /hydra/launcher: joblib

experiment_meta:
  id: "{exp_id}"
  name: "{name}"
  total_runs: {total_runs}
  pairing: {pairing}

{job_matrix_yaml}
{static_overrides_yaml}
hydra:
  mode: MULTIRUN
  sweep:
    dir: results/${{experiment_meta.id}}
    subdir: ${{hydra:runtime.choices.optimizer}}_${{hydra:runtime.choices.environment}}_${{hydra:runtime.choices.avoidance}}_seed${{seed}}
  launcher:
    n_jobs: 6
    # backend `loky` (zamiast `multiprocessing`) używa fork-and-exec
    # (czysty Python interpreter per-worker), co eliminuje współdzielenie
    # globalnego stanu C++ PyBullet między workerami. `multiprocessing` z
    # default fork-share state powodował patologię „drony stoją w PyBullet
    # mimo poprawnej trajektorii" w drugim+ jobie multirun.
    backend: loky
    prefer: processes
  sweeper:
    params:
      avoidance: {avoid_str}
      seed: {seeds}
"""

def main():
    """Wczytaj definicję YAML i wygeneruj manifest, pliki proxy oraz `RunRegistry`.

    Efekty uboczne:
        - Tworzy pliki proxy w `configs/optimizer/_proxy_<exp_id>_*.yaml`.
        - Zapisuje `configs/experiment_generated/<exp_id>.yaml` (konfiguracja Hydra).
        - Zapisuje `results/<exp_id>/manifest.yaml` (kopię) oraz
          `results/<exp_id>/original_definition.yaml`.
        - Inicjalizuje `results/<exp_id>/registry.db` rekordami w stanie PENDING.

    Wyjścia:
        Kończy z kodem 1, gdy: brak pliku definicji, błąd parsowania YAML,
        niepełna siatka parametrów, nieznany tryb `pairing`, albo
        `paired_only` zawiera optymalizatory bez odpowiednika w `avoidances`.
    """
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
    
    grid = definition.get("parameters_grid", {})
    raw_optimizers = grid.get("optimizers", [])
    env_names = list(grid.get("environments", []))
    avoid_names = list(grid.get("avoidances", []))

    if not raw_optimizers or not env_names or not avoid_names:
        print("❌ `parameters_grid` musi zawierać niepuste optimizers/environments/avoidances.")
        sys.exit(1)

    try:
        pairing = _resolve_pairing(definition)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    try:
        job_matrix, base_optimizer_names = expand_optimizers_for_environments(
            raw_optimizers, env_names, exp_id, configs_dir,
        )
    except Exception as e:
        print(f"❌ Błąd przetwarzania algorytmów: {e}")
        sys.exit(1)

    # Walidacja paired_only: każdy bazowy optimizer MUSI mieć odpowiednik
    # w `avoidances`. Inaczej run zostanie cicho wyfiltrowany.
    if pairing == "paired_only":
        unmatched = [o for o in dict.fromkeys(base_optimizer_names) if o not in avoid_names]
        if unmatched:
            print(
                f"❌ `pairing: paired_only` wymaga, by każdy optimizer miał "
                f"odpowiednika w `avoidances`. Brakuje: {unmatched}. "
                f"Avoidances: {avoid_names}."
            )
            sys.exit(1)

    generated_configs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        yaml_content = generate_yaml_content(exp_id, definition, job_matrix)
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

    runs_count = int(definition.get("runs_per_configuration", 1))
    if pairing == "paired_only":
        valid_pairs_count = sum(1 for jm in job_matrix if jm.get("base_name") in avoid_names)
        total_runs = runs_count * valid_pairs_count
    else:
        total_runs = runs_count * len(job_matrix) * len(avoid_names)

    # Sweep params dla RunRegistry generowane DOKŁADNIE z definicji eksperymentu
    # (nie z hardcoded list) — liczba wpisów PENDING musi się zgadzać z liczbą
    # wystartowanych jobów, inaczej `mark_started/completed` (UPDATE w
    # RunRegistry) trafiają w nieistniejące wiersze i registry rozjeżdża się
    # z parquetem.
    runs = int(definition.get("runs_per_configuration", 1))

    # Bazowe nazwy optymalizatorów — spójne ze sposobem, w jaki
    # `main.py:_get_registry_job_key` ekstraktuje krótką nazwę z
    # `cfg.optimizer._target_`. Per-env proxy override'y (`tmp/{exp_id}_X_<env>_N`)
    # i bazowe proxy (`tmp/{exp_id}_X_N`) wszystkie wskazują tym samym
    # `_target_`, więc registry użyje wpisu bazowej nazwy.
    unique_optimizers = list(dict.fromkeys(base_optimizer_names))
    if len(unique_optimizers) != len(base_optimizer_names):
        print(
            f"⚠ Wykryto powtarzające się bazowe nazwy optymalizatorów: "
            f"{base_optimizer_names} — registry użyje unikatów {unique_optimizers}. "
            f"Warianty z overrides współdzielą wpisy."
        )

    seed_values = list(range(1, runs + 1))

    if pairing == "paired_only":
        # `paired_only`: zachowaj tylko kombinacje, gdzie online avoidance
        # = offline optimizer (baseline porównania ceteris-paribus po
        # algorytmie reaktywnym = algorytmie planującym).
        sweep_params = [
            {"optimizer": o, "environment": e, "avoidance": o, "seed": s}
            for o, e, s in product(unique_optimizers, env_names, seed_values)
            if o in avoid_names
        ]
    else:
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
    print(f"Wygenerowane pliki proxy: {configs_dir}/optimizer/_proxy_{exp_id}_*.yaml")
    print(f"Zaplanowane symulacje: {total_runs}")
    print("\nAby usunąć pliki po wykonaniu eksperymentu:")
    print(
        f"rm -f {configs_dir}/optimizer/_proxy_{exp_id}_*.yaml && "
        f"rm -f {configs_dir}/experiment_generated/{exp_id}.yaml"
    )
    print("\nAby uruchomić eksperyment z optymalizacją wieloprocesorową, wywołaj:\n")
    print(f"./run.sh {exp_id}")

if __name__ == "__main__":
    main()