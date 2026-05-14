# F_environment — Specyfikacja środowiska wykonawczego

## Cel

Pełna specyfikacja wersji Python + zależności, żeby recenzent mógł odtworzyć identyczne środowisko obliczeniowe. Wymóg z **López-Ibáñez et al. (2021)** *Reproducibility in evolutionary computation* — bez dokładnych wersji pakietów (np. pymoo, mealpy) wyniki metaheurystyk mogą się delikatnie różnić nawet przy identycznych seedach.

## Pliki

| Plik | Zawartość |
|---|---|
| `environment.yaml` | Wysokopoziomowa specyfikacja conda + pip (do reprodukcji środowiska na czystej maszynie) |
| `conda_env_export.yaml` | Pełny snapshot **wszystkich** zainstalowanych pakietów z dokładnymi wersjami i hash'ami |

## Pliki do skopiowania

| Plik docelowy | Źródło | Komenda |
|---|---|---|
| `environment.yaml` | [environment.yaml](../../environment.yaml) | `cp environment.yaml appendix/F_environment/` |

## Reprodukcja: jak zostały wygenerowane

| Plik docelowy | Komenda |
|---|---|
| `conda_env_export.yaml` | `conda activate drone-swarm-env && conda env export > appendix/F_environment/conda_env_export.yaml` |

## Reprodukcja środowiska

```bash
# Wariant 1: minimalny (z environment.yaml)
conda env create -f appendix/F_environment/environment.yaml
conda activate drone-swarm-env

# Wariant 2: bit-exact (z conda_env_export.yaml)
conda env create -f appendix/F_environment/conda_env_export.yaml -n drone-swarm-env-thesis
conda activate drone-swarm-env-thesis
```

Wariant 2 wymaga identycznej platformy (zgodnie z [CLAUDE.local.md](../../CLAUDE.local.md): Fedora 43, x86_64). Na innych systemach (Ubuntu, macOS) niektóre hash'e nie pasują — w takim wypadku użyj wariantu 1.

