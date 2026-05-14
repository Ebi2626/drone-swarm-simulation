# Citation & Source Code

## Repozytorium źródłowe

**GitHub:** https://github.com/Ebi2626/drone-swarm-simulation
**Commit referencyjny:** `cdca9524f58f54b5da720e80fcbd239595f4ea16`
**Branch:** `main`
**Tag thesis (do utworzenia przed obroną):** `v1.0-thesis`

### Klonowanie na stanie referencyjnym

```bash
git clone https://github.com/Ebi2626/drone-swarm-simulation.git
cd drone-swarm-simulation
git checkout cdca9524f58f54b5da720e80fcbd239595f4ea16
```

### Zalecane cytowanie

```bibtex
@mastersthesis{harmata2026_dronswarm,
  author  = {Harmata, Edwin},
  title   = {Por\'{o}wnanie bio-inspirowanych metaheurystyk (MSFOA, OOA, SSA)
             z klasycznym NSGA-III w problemie planowania trajektorii roju UAV},
  school  = {[Nazwa uczelni \-- uzupe{\l}ni{\'c}]},
  year    = {2026},
  note    = {Kod {\'z}r{\'o}d{\l}owy: https://github.com/Ebi2626/drone-swarm-simulation
             commit cdca9524f58f54b5da720e80fcbd239595f4ea16}
}
```

## DOI (Zenodo)

**Status:** *do utworzenia po obronie pracy.*

Procedura archiwizacji:
1. Utwórz tag GitHub: `git tag v1.0-thesis cdca9524 && git push origin v1.0-thesis`
2. Utwórz GitHub Release z tego tagu.
3. Połącz repo z Zenodo: https://zenodo.org/account/settings/github/ — kolejne GitHub Release automatycznie wygeneruje snapshot ZIP z DOI.
4. Po nadaniu DOI, uzupełnij to pole oraz tag w pliku [README.md](README.md).

Standard archiwizacji oprogramowania badawczego wg **Katz et al. (2021)** *Recognizing the value of software: a software citation guide*, F1000Research.

## Kluczowe zależności (snapshot z commitu)

Pełna lista w [F_environment/environment.yaml](F_environment/environment.yaml) i `conda_env_export.yaml`. Wybrane wersje:

- Python 3.10
- pymoo (NSGA-III)
- mealpy (SSA, OOA)
- PyBullet (fizyka)
- pandas, scipy, scikit-learn, statsmodels (analiza)
- matplotlib, seaborn (wykresy)
- Hydra (zarządzanie konfiguracjami)
- SQLite (analysis.db)

## Licencja

Licencja repozytorium: patrz plik `LICENSE` w korzeniu repo.

## Kontakt

Edwin Harmata — edwin.harmata@gmail.com
