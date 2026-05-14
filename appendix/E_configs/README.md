# E_configs — Pliki Hydra definiujące eksperyment

## Cel

Pliki konfiguracyjne Hydra definiujące **dokładnie** parametry eksperymentu, którego wyniki są cytowane w pracy. Bez tych plików seeds same w sobie nie pozwalają na reprodukcję — Hydra config jest *definicją* eksperymentu (wagi, hyperparametry, parametry środowiska).

## Struktura

```
E_configs/
├── base_config.yaml       # configs/config.yaml — defaulty globalne
├── algorithms/            # 4 algorytmy
│   ├── nsga3.yaml
│   ├── msfoa.yaml
│   ├── ooa.yaml
│   └── ssa.yaml
└── environments/          # 2 środowiska cytowane w pracy
    ├── forest.yaml
    └── urban.yaml
```

## Pliki do skopiowania

**Źródło:** [configs/](../../configs/)

| Plik docelowy | Źródło |
|---|---|
| `base_config.yaml` | `configs/config.yaml` |
| `algorithms/nsga3.yaml` | `configs/optimizer/nsga3.yaml` |
| `algorithms/msfoa.yaml` | `configs/optimizer/msfoa.yaml` (lub `msffoa.yaml`) |
| `algorithms/ooa.yaml` | `configs/optimizer/ooa.yaml` |
| `algorithms/ssa.yaml` | `configs/optimizer/ssa.yaml` |
| `environments/forest.yaml` | `configs/environment/forest.yaml` |
| `environments/urban.yaml` | `configs/environment/urban.yaml` |

## Komenda kopiowania

```bash
mkdir -p appendix/E_configs/{algorithms,environments}
cp configs/config.yaml appendix/E_configs/base_config.yaml
cp configs/optimizer/{nsga3,msfoa,ooa,ssa}.yaml appendix/E_configs/algorithms/
cp configs/environment/{forest,urban}.yaml appendix/E_configs/environments/
```

**Uwaga:** dokładne nazwy plików w `configs/optimizer/` mogą się różnić (np. `msffoa.yaml` zamiast `msfoa.yaml`). Sprawdź `ls configs/optimizer/` przed kopiowaniem.

## Per-run config snapshots

Pliki w tym katalogu to **default configs**. Każdy z 240 runów zapisuje również swoją finalną (po overrides) konfigurację w `.hydra/config.yaml` w katalogu runu — te pliki są w [G_per_run_seeds/<run_id>/.hydra/](../G_per_run_seeds/) i to one są pełnym źródłem prawdy o tym, co faktycznie wykonano w danym runie (z konkretnym seedem).

