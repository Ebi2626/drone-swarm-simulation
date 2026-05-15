# E_configs — Pliki Hydra definiujące eksperyment

## Cel

Pliki konfiguracyjne Hydra definiujące **dokładnie** parametry eksperymentu, którego wyniki są cytowane w pracy. Bez tych plików seeds same w sobie nie pozwalają na reprodukcję — Hydra config jest *definicją* eksperymentu (wagi, hyperparametry, parametry środowiska).

## Struktura

```
E_configs/
├── base_config.yaml                       # Hydra defaulty globalne (configs/config.yaml)
├── algorithms/                            # 4 algorytmy — base configs (FOREST)
│   ├── nsga-3.yaml
│   ├── msffoa.yaml
│   ├── ooa.yaml
│   └── ssa.yaml
├── environments/                          # 2 środowiska
│   ├── forest.yaml
│   └── urban.yaml
└── experiment_generated/                  # Output `prepare_experiment.py`
    ├── experiment_definition.yaml         # INPUT: definicja 240-runowego benchmarku
    ├── experiment_generated.yaml          # OUTPUT 1: job_matrix (8 par alg × env)
    └── proxy_overrides_urban/             # OUTPUT 2: per-env hyperparams dla urban
        ├── nsga-3_urban.yaml
        ├── msffoa_urban.yaml
        ├── ooa_urban.yaml
        └── ssa_urban.yaml
```

## Dwie warstwy konfiguracji

**Hydra używa dwóch warstw dla tego eksperymentu:**

1. **Bazowe configi w `algorithms/`** — defaulty (= hyperparams dla *forest*):
   - `pop_size: 1001`, `n_gen` / `epochs: 200`
   - `n_inner_waypoints: 11`
   - `obstacle_safety_margin: 1.0`, `preferred_height: 5.0`

2. **Proxy overrides w `experiment_generated/proxy_overrides_urban/`** — generowane przez `experiments/prepare_experiment.py` dla każdej pary (algorytm × urban):
   - `pop_size: 1500`, `n_gen` / `epochs: 300`
   - `n_inner_waypoints: 15`
   - `obstacle_safety_margin: 3.0`, `preferred_height: 10.0`
   - Wagi celu zmienione (threat weight 0.8 → 5.0)

**Bez proxy plików recenzent nie wie, że urban budget jest 3× większy niż forest** (33.75M vs 11.01M ewaluacji składowych). Patrz [B_statistical_tests/budget_table.md](../B_statistical_tests/budget_table.md).

## Pipeline generowania konfiguracji

```
experiment_definition.yaml             # input z motywacją (Demšar 2006, runs_per_config=30)
        │
        ▼
experiments/prepare_experiment.py
        │
        ├──► experiment_generated.yaml         # job_matrix (8 par optimizer × env)
        │
        └──► proxy_overrides_urban/*.yaml      # 4 proxy yamls dla urban
                │
                ▼
        configs/optimizer/_proxy_*.yaml         # Hydra czyta te przy `environment=urban`
```

W trakcie wykonywania eksperymentu, dla każdego z 240 runów Hydra zapisuje swoją **finalną** (po overrides) konfigurację do `.hydra/config.yaml` w katalogu runu — patrz [G_per_run_seeds/<run_id>/.hydra/](../G_per_run_seeds/) (źródło prawdy dla konkretnego runu, w tym seed).

## Reprodukcja eksperymentu z tych plików

```bash
# 1. Sklonuj repo na referencyjnym commicie
git clone https://github.com/Ebi2626/drone-swarm-simulation.git
cd drone-swarm-simulation
git checkout cdca9524f58f54b5da720e80fcbd239595f4ea16

# 2. Skopiuj definicję z appendix do experiments/definitions/
cp appendix/E_configs/experiment_generated/experiment_definition.yaml \
   experiments/definitions/bio_inspired_benchmark.yaml

# 3. Wygeneruj job matrix + proxy overrides
python experiments/prepare_experiment.py --definition bio_inspired_benchmark

# 4. Uruchom 240 runów (UWAGA: ~120 h GPU)
python experiments/run_subprocess.py --experiment-id exp_<new_id>_bio_inspired_benchmark

# 5. Agregacja → analysis.db
python -c "from src.analysis.ExperimentAggregator import ExperimentAggregator; \
           ExperimentAggregator().aggregate('./results/exp_<new_id>.../')"
```
