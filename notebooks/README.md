# `notebooks/` — Notebooki demonstracyjne pracy magisterskiej

Katalog zawiera **sześć** notebooków Jupytera, które pokazują wybrane fragmenty
pipeline'u badawczego pracy magisterskiej *"Porównanie bio-inspirowanych
metaheurystyk (MSFOA, OOA, SSA) z klasycznym NSGA-III w problemie planowania
trajektorii roju UAV"*. Każdy notebook jest samowyjaśniający — sekcje
parametryzacji są wyraźnie oznaczone, a kluczowe decyzje metodologiczne
opatrzone komentarzami.

## Struktura katalogu

```
notebooks/
├── prepare_notebook.py            # Helper sys.path (importowany przez każdy nb)
├── 01_world_generation.ipynb      # Generowanie świata + wizualizacja 3D
├── 02_offline_optimization.ipynb  # Optymalizacja offline (MSFOA/OOA/SSA/NSGA-III)
├── 03_online_collision_scenario.ipynb  # Scenariusz head-on z dynamicznymi
│                                       # przeszkodami i online avoidance
├── 04_statistical_analysis.ipynb  # Testy statystyczne na metrykach M1–M13
├── 05_thesis_reproduction.ipynb   # Przegląd wszystkich tabel/wykresów cytowanych
│                                   # w pracy
└── 06_data_reproducibility.ipynb  # Schemat DB, manifest, surowe pliki per-run,
                                   # snapshot środowiska, ścieżka rekonstrukcji
```

## Opis notebooków

### 01. Generowanie świata i wizualizacja 3D

Wykorzystuje moduły `src.environments.abstraction.*` do zbudowania granic
świata (`WorldData`) i wygenerowania przeszkód (`ObstaclesData`) wybraną
strategią rozmieszczenia (`strategy_random_uniform`, `strategy_grid_jitter`,
`strategy_empty`). Wynik renderowany jest w 3D z **proporcjonalnym
skalowaniem osi** — dla wydłużonych korytarzy (forest 60×600×11 m,
urban 300×1000×20 m) najdłuższa oś jest *miękko* ściskana, by widok pozostał
czytelny.

**Edytowalne parametry:** `CONFIG_NAME`, `MASTER_SEED`, `STRATEGY_OVERRIDE`,
`MAX_DISPLAY_ASPECT`.

### 02. Optymalizacja offline (planowanie waypointów)

Uruchamia pełen pipeline `count_trajectories` z wybranym algorytmem
(`msffoa`, `ooa`, `ssa`, `nsga-3`) na domyślnym świecie zdefiniowanym przez
plik konfiguracyjny. Notebook poprzez `hydra.utils.instantiate` wczytuje
strategię z `configs/optimizer/<name>.yaml` i wywołuje ją bezpośrednio
(z minimalnym stubem `HydraConfig`, by `OptimizationHistoryWriter` zapisywał
historię do katalogu tymczasowego notebooka).

W komórce końcowej rysowana jest *krzywa zbieżności* z
`optimization_history.h5` — sumaryczna wartość feasible best-so-far per
generacja.

**Edytowalne parametry:** `OPTIMIZER_NAME`, `ENVIRONMENT_NAME`,
`MASTER_SEED`, `FAST_PARAMS`, `NUMBER_OF_WAYPOINTS`.

**Uwaga.** `FAST_PARAMS=True` (domyślnie) drastycznie redukuje budżet
obliczeniowy do ~30 s dla dry-runa. Wyniki nie są reprezentatywne dla
finalnych eksperymentów — pełne wartości `pop_size`/`epochs` znajdują się
w `configs/optimizer/*.yaml` (zob. notatkę w notebooku).

### 03. Scenariusz online (head-on)

Notebook ilustruje fazę *online* — reaktywne unikanie czołowo nadlatujących
dynamicznych przeszkód. Główny rój leci prostoliniowo (środowisko `empty`,
2 drony), a `simulation.dynamic_obstacles=true` aktywuje 2 przeszkody
startujące w punktach docelowych dronów i lecące do ich pozycji startowych.

Notebook wywołuje `python main.py …` przez `subprocess.run` w trybie
headless. Pełen pipeline (PyBullet + LiDAR + `SwarmFlightController`)
zapisuje wyniki do `results/<data>/<godzina>_empty_<opt>/`; notebook parsuje
najnowszy katalog i rysuje trajektorie dronów oraz zdarzenia uniku
(`evasion_events.csv`).

**Edytowalne parametry:** `AVOIDANCE_NAME`, `OPTIMIZER_NAME`,
`DURATION_SEC`, `MASTER_SEED`, `DRONE_TO_PLOT`.

**Uwaga.** Pierwszy run kompiluje funkcje `@njit` (Numba) — ~10–30 s
narzutu kompilacji.

### 04. Analiza statystyczna metryk

Powtarza testy statystyczne z pracy na *lokalnym* subsetcie
`appendix/A_metrics/`:

* Statystyki opisowe (n, mean, std, min, max, mediana, IQR) per (środowisko,
  algorytm).
* **Test Friedmana** + critical difference Nemenyiego (Demšar 2006).
* **A12 Vargha–Delaney** dla każdej pary algorytmów.
* **Wilson 95% CI** dla wskaźników binomialnych.

Pokrywa metryki M1, M2, M4, M6, M7, M9, M11, M13 z `INDEX.md`. Krzywe
konwergencji offline (M10) i online (M12) — mediana ± IQR po runach.

**Edytowalne parametry:** `METRICS_TO_ANALYZE`, `ENVIRONMENTS`, `ALPHA`.

Notebook pokrywa też M3 (gładkość F2+F4) i M5 (offline failure rate)
wczytując gotowe agregaty z `appendix/B_statistical_tests/` — kolumny
źródłowe (`F[1]`, `tracking_phase_collisions`) nie są w
`run_metrics_subset.csv`, więc bezpośrednie wyliczenie nie jest możliwe.

### 05. Reprodukcja tabel i wykresów

Centralna prezentacja **wszystkich** artefaktów cytowanych w pracy:

* T1 — tabela budżetu obliczeniowego (`budget_table.md`).
* T2–T21 — statystyki opisowe, Friedman+A12, Wilson 95% CI (CSV z
  `appendix/B_statistical_tests/`).
* W1–W18 — wykresy bar/box/konwergencja (PNG z `appendix/C_plots/`).
* Panele zbiorcze `thesis_stat_tables/` używane bezpośrednio w pracy.

Notebook nie liczy niczego od nowa — pełni rolę przeglądarki gotowych
artefaktów dla recenzenta.

### 06. Reprodukowalność danych

Domyka materiał z sekcji `appendix/`, których nie pokrywają notebooki
01–05:

* **D — schemat bazy `analysis.db`**: pełna lista 21 tabel + 5 widoków,
  diagram ERD (Mermaid), opis widoków analitycznych.
* **F — środowisko wykonawcze**: porównanie wersji krytycznych pakietów
  (numpy, pymoo, mealpy, pybullet, numba) z `environment.yaml` vs lokalnie
  zainstalowane.
* **G — surowe pliki per-run**: ilustracja struktury katalogu wybranego
  runa + bezpośrednia rekonstrukcja krzywej konwergencji M10 z
  `optimization_history.h5` i porównanie z agregatem z `A_metrics/`.
* **H — manifest 240 runów**: sanity-check kompletności (4 alg. ×
  2 środ. × 30 ziaren) + heatmapa.
* **`CITATION.md`** — pełna informacja o commicie referencyjnym.

Sekcja końcowa pokazuje procedurę pełnej rekonstrukcji eksperymentu
(odtworzenie środowiska conda + Hydra multi-run + agregacja w
`analysis.db`).

**Edytowalne parametry:** `EXAMPLE_RUN` (nazwa katalogu w `G_per_run_seeds/`),
`CRITICAL` (lista pakietów do porównania).

## Uruchamianie

1. Aktywuj środowisko conda:
   ```bash
   conda activate drone-swarm-env
   ```
2. Uruchom Jupyter w katalogu projektu lub w `notebooks/`:
   ```bash
   jupyter lab notebooks/
   ```
3. Otwórz wybrany notebook. Pierwsza komórka (`import prepare_notebook`)
   dodaje korzeń repozytorium do `sys.path`, co pozwala importować moduły
   z `src/`.

## Powiązania z resztą repozytorium

| Notebook | Wykorzystywane moduły / pliki |
|----------|-------------------------------|
| 01 | `src.environments.abstraction.*`, `src.utils.SeedRegistry`, `configs/environment/*.yaml` |
| 02 | `src.algorithms.abstraction.count_trajectories`, `configs/optimizer/*.yaml`, strategie `*_strategy.py` |
| 03 | `main.py` (subprocess), `configs/avoidance/*.yaml`, `results/` |
| 04 | `appendix/A_metrics/*.csv`, `appendix/B_statistical_tests/{summary,friedman,a12,wilson}/*` |
| 05 | `appendix/B_statistical_tests/*`, `appendix/C_plots/*`, `appendix/INDEX.md` |
| 06 | `appendix/{CITATION.md, D_database_schema/*, F_environment/*, G_per_run_seeds/*, H_run_manifest.csv}` |

## Konwencje

* Wszystkie notebooki **muszą** używać `prepare_notebook.project_root` jako
  bazy ścieżek — nie hardkoduj `/home/edwinh/...`.
* Każdy notebook ma sekcję *PARAMETRY DO EDYCJI* na początku (komentarz
  blokowy nad zmiennymi).
* Determinizm: ustawiamy `MASTER_SEED` i delegujemy sub-seedy przez
  `SeedRegistry`. Wartość domyślna **43** zgodna z `configs/config.yaml`
  (zob. `CLAUDE.md`).
* Dla operacji długotrwałych (notebook 02, 03) używamy zredukowanych
  parametrów (`FAST_PARAMS` / krótkie `DURATION_SEC`), zgodnie z regułą
  *dry-run* z `CLAUDE.md`.
