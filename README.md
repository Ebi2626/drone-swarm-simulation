# drone-swarm-simulation

Symulacja planowania tras roju dronów z wykorzystaniem nowoczesnych algorytmów heurystycznych
inspirowanych biologicznie (OOA, SSA, MSFFOA), porównywanych z klasycznym NSGA-III
w środowiskach fizycznych opartych na PyBullet.

## Struktura projektu

```
drone-swarm-simulation/
├── appendix/                   # Załącznik cyfrowy pracy magisterskiej
│   ├── INDEX.md                # Indeks: metryki M1–M13, tabele T1–T21, wykresy W1–W18
│   ├── A_metrics/              # Subset CSV z analysis.db (13 metryk × 240 runów)
│   ├── B_statistical_tests/    # Friedman, A12, Wilson CI (CSV + TeX + panele PNG)
│   ├── C_plots/                # 18 wykresów cytowanych w pracy (PDF + PNG)
│   ├── D_database_schema/      # Schema SQL + ERD bazy analytycznej
│   ├── E_configs/              # Snapshot konfiguracji Hydry dla eksperymentu
│   ├── F_environment/          # environment.yaml + pełny snapshot conda
│   ├── G_per_run_seeds/        # Surowe pliki per-run (240 katalogów)
│   └── H_run_manifest.csv      # Manifest 240 runów (run_id, alg, env, seed, status)
├── configs/                    # Modularna konfiguracja Hydra 1.3
│   ├── config.yaml             # Globalne parametry eksperymentu
│   ├── avoidance/              # Online avoidance: msffoa, nsga-3, ooa, ssa, none
│   ├── environment/            # Konfiguracje światów (empty, forest, urban) + strategie rozmieszczenia
│   └── optimizer/              # Konfiguracje optymalizatorów offline (msffoa, nsga-3, ooa, ssa)
├── experiments/                # Definicje i runner eksperymentów (prepare_experiment + run_subprocess)
├── notebooks/                  # 6 demonstracyjnych notebooków (zob. notebooks/README.md)
├── results/                    # Logi z konkretnych uruchomień (gitignored)
├── src/                        # Główna logika aplikacji
│   ├── algorithms/             # Logika optymalizacji (planowanie offline + unikanie online)
│   │   ├── abstraction/        # Strategy Pattern dla offline (MSFOA, NSGA-III, OOA, SSA)
│   │   └── avoidance/          # GenericOptimizingAvoidance + 4 sub-strategie
│   │                           # (IObstaclePredictor, IPathRepresentation,
│   │                           #  IFitnessEvaluator, IPathOptimizer)
│   ├── analysis/               # ETL i analiza statystyczna eksperymentów
│   │   ├── ExperimentAggregator.py  # Agregacja CSV/HDF5 z runów do analysis.db (SQLite)
│   │   ├── analyzer/                # Analiza statystyczna i generowanie raportów
│   │   │   ├── ExperimentAnalyzer.py    # Orkiestrator: testy statystyczne → wykresy → raport
│   │   │   ├── report_generator.py      # Generator raportów (Markdown + PDF via matplotlib PdfPages)
│   │   │   ├── report_template.md.j2    # Szablon Jinja2 raportu Markdown
│   │   │   └── plots/                   # Generatory wykresów (boxploty, konwergencja, CD diagramy, ...)
│   │   │       └── PLOTS_LEGEND.md      # Legenda: jak interpretować wykresy i diagramy
│   │   └── db/                  # Populacja bazy SQLite (schemat, metryki offline/online)
│   ├── environments/           # Środowiska symulacji 3D
│   │   ├── abstraction/        # Generatory przeszkód i granic świata
│   │   ├── obstacles/          # Enum typów przeszkód (BOX, CYLINDER)
│   │   ├── EmptyWorld.py       # Puste środowisko (sanity-check)
│   │   ├── ForestWorld.py      # Las — 13 cylindrów (60×600×11 m)
│   │   ├── SwarmBaseWorld.py   # Bazowa klasa abstrakcyjna środowiska
│   │   └── UrbanWorld.py       # Miasto — 27 boxów (300×1000×20 m)
│   ├── runner/                 # Strategie uruchamiania eksperymentów (Strategy Pattern)
│   │   ├── ExperimentDataStrategy.py   # Abstrakcyjny interfejs bazowy
│   │   ├── GenerationDataStrategy.py   # Nowy eksperyment: generowanie świata + optymalizacja offline
│   │   └── ReplayDataStrategy.py       # Odtwarzanie z zarchiwizowanych wyników CSV
│   ├── sensors/                # Symulacja sensorów UAV
│   │   ├── LidarSensor.py      # LiDAR 3D — 123 promienie, FOV 30°, zasięg 100m (PyBullet rayTestBatch)
│   │   └── lidar_visualization.py  # Wizualizacja debugowa / demo
│   ├── trajectory/             # Generowanie gładkich trajektorii 3D
│   │   ├── BSplineTrajectory.py       # Cubic B-Spline (ciągłość C², scipy.interpolate)
│   │   ├── ConstantSpeedProfile.py    # Stała prędkość — faza lokalnego unikania
│   │   └── TrapezoidalProfile.py      # Profil trapezowy — faza globalnej misji
│   └── utils/                  # Infrastruktura i funkcje pomocnicze
│       ├── SimulationLogger.py             # Centralny logger (bufor RAM → CSV)
│       ├── optimization_history_writer.py  # Asynchroniczny zapis HDF5 (historia frontów Pareto)
│       ├── config_parser.py                # Hydra config → struktury Python
│       ├── ValidationMessage.py            # Enum komunikatów błędów
│       └── pybullet_utils.py               # Wrappery PyBullet
├── tests/                      # Testy jednostkowe (odzwierciedla strukturę /src)
├── main.py                     # Runner z dwoma trybami uruchomienia (zawiera klasę ExperimentRunner):
│                               #   default: python main.py — nowa symulacja z dynamicznym światem
│                               #   replay:  python main.py --replay /results/2026-04-13/
├── run_etl.py                  # ETL pipeline: agregacja wyników + analiza statystyczna + raport
├── run.sh                      # Wrapper uruchamiający pełną sesję eksperymentów
├── clean-{all,experiments,optimizers,results}.sh   # Skrypty czyszczące katalogi wyjściowe
├── turn-off-after-experiment.sh  # Bezpieczne wyłączenie maszyny po zakończonym sweep
├── environment.yaml            # Zależności Conda
└── mypy.ini                    # Konfiguracja sprawdzania typów (mypy)
```

## Główne założenia

- Eksperymenty są **w pełni reprodukowalne** — stan świata, przeszkody i trajektorie
  archiwizowane do CSV/HDF5; ziarna zarządzane przez `SeedRegistry`.
- Macierzowy plan eksperymentów (final benchmark, `exp_20260508_…`):
  - **2 środowiska** finalne: `forest` (13 cylindrów, 60×600×11 m) i `urban`
    (27 boxów, 300×1000×20 m); `empty` służy jako sanity-check.
  - **4 algorytmy**: MSFOA, SSA, OOA (bio-inspirowane) + NSGA-III (punkt odniesienia).
  - **30 ziaren** per kombinacja ⇒ **240 runów** (4 × 2 × 30).
- **Dwufazowe planowanie**:
  - *Offline*: optymalizacja globalna waypointów na pełnym horyzoncie misji
    (wspólny `VectorizedEvaluator` — 5 obiektywów, 3 ograniczenia).
  - *Online*: reaktywne unikanie `GenericOptimizingAvoidance` — kompozycja
    4 sub-strategii (predictor / path / fitness / optimizer) z budżetem
    czasowym 0.5 s.
- **Reprezentacja trajektorii**: Cubic B-Spline (ciągłość C²) z trapezowym
  lub stałym profilem prędkości.
- **Miary oceny** (M1–M13, pełna lista w
  [`appendix/INDEX.md`](appendix/INDEX.md)):
  bezpieczeństwo trajektorii (F3+F5), długość (F1), gładkość (F2+F4),
  spójność roju, odsetek kolizji, długość krzywej unikowej, rejoin quality,
  wartość i tempo optymalizacji (offline + online), SP1.

## Przegląd architektury

```
CLI / konfiguracja Hydra
          │
          ▼
   ExperimentRunner (main.py)
          │
  ┌───────┴──────────────────┐
  │                          │
GenerationStrategy      ReplayStrategy
  │                          │
  ▼                          ▼
algorithms/abstraction/     archiwa CSV
 (MSFOA/NSGA-III/           (świat, przeszkody,
  OOA/SSA — offline)         trajektorie)
  │
  ▼
trajectory/ (B-Spline + profil prędkości)
  │
  ▼
environments/ (fizyka PyBullet)
  │
  ├── sensors/ (LiDAR 3D, batch ray test)
  │       │
  │       ▼
  │   algorithms/avoidance/ (GenericOptimizingAvoidance, FSM TRACK ⇄ EVADE ⇄ REJOIN)
  │
  └── utils/ (SimulationLogger → CSV/HDF5 → results/)
                                      │
                                      ▼
                           run_etl.py → analysis.db
                                      │
                                      ▼
                           tables/ + plots/ + report/
```

## Technologie

- **Python 3.10** — wersja wyznaczona przez `gym-pybullet-drones`
- **gym-pybullet-drones** — główny framework symulacji UAV
- **numpy** — wektoryzowane obliczenia numeryczne
- **scipy** — interpolacja B-Spline, transformacje przestrzenne
- **pandas** — analiza danych i operacje na CSV
- **matplotlib** — wykresy i wizualizacje
- **hydra-core** — hierarchiczna konfiguracja eksperymentów + grid search multirun
- **h5py** — zapis HDF5 dla historii optymalizacji
- **jupyter** — interaktywne notebooki
- **pytest** — testy jednostkowe
- **mypy** — statyczne sprawdzanie typów

## Instalacja

1. Sklonuj to repozytorium
2. Upewnij się, że masz zainstalowane [Conda](https://docs.conda.io/)
3. Przejdź do katalogu projektu: `cd ~/drone-swarm-simulation`
4. Utwórz środowisko: `conda env create -f environment.yaml`

## Uruchamianie symulacji

```bash
conda activate drone-swarm-env

# Pojedynczy eksperyment
python main.py environment=forest optimizer=msffoa

# Przeszukiwanie siatki (Hydra multirun)
python main.py environment=urban,forest optimizer=msffoa,nsga-3

# Odtworzenie zarchiwizowanego eksperymentu
python main.py --replay ./results/2026-04-13/12-30-urban_msffoa/
```

Wyniki zapisywane są w `/results/{data}/{czas}_{nazwa-świata}/`.

## Analiza wyników (ETL pipeline)

Po zakończeniu eksperymentu (lub zestawu eksperymentów) uruchom pipeline ETL,
który agreguje surowe wyniki do bazy SQLite, wykonuje testy statystyczne
(Friedman, Wilcoxon z korektą Holma-Bonferroniego, Vargha-Delaney A12),
generuje wykresy (boxploty, krzywe zbieżności, diagramy CD, heatmapy rang)
oraz kompiluje raport zbiorczy w formatach Markdown i PDF.

```bash
# Analiza wybranego eksperymentu
python run_etl.py results/exp_20260506_377919c3_per_env_test
```

Wyniki analizy zapisywane są w podkatalogu `analysis_output/` eksperymentu:

```
results/<experiment>/analysis_output/
├── analysis.db              # Baza SQLite z zagregowanymi metrykami
├── tables/                  # Tabele CSV (summary, Friedman, Wilcoxon, A12)
├── plots/                   # Wykresy (PDF + PNG): boxploty, konwergencja, CD, rankingi
│   └── PLOTS_LEGEND.md      # Legenda — jak interpretować wykresy i diagramy
└── report/
    ├── experiment_report.md  # Raport Markdown z tabelami i odwołaniami do wykresów
    └── experiment_report.pdf # Raport PDF (strony tytułowe, tabele, wykresy, wnioski)
```

Interpretację poszczególnych typów wykresów (diagramy CD, boxploty, krzywe
zbieżności itd.) opisuje plik
[`PLOTS_LEGEND.md`](src/analysis/analyzer/plots/PLOTS_LEGEND.md).

## Załącznik cyfrowy pracy magisterskiej (`appendix/`)

Po zakończeniu pracy magisterskiej kompletny zestaw artefaktów cytowanych
w tekście został wyodrębniony do katalogu [`appendix/`](appendix/).
Zawiera m.in.:

* **`INDEX.md`** — kompletna mapa 13 metryk (M1–M13), 21 tabel (T1–T21)
  i 18 wykresów (W1–W18) cytowanych w pracy.
* **`A_metrics/`, `B_statistical_tests/`, `C_plots/`** — gotowe agregaty
  CSV/TeX/PDF/PNG do reprodukcji wszystkich wyników.
* **`D_database_schema/`** — schemat SQLite + ERD + dokumentacja widoków
  analitycznych.
* **`E_configs/` + `F_environment/`** — snapshot konfiguracji Hydry
  i pełny export środowiska conda do bit-exact reprodukcji.
* **`G_per_run_seeds/`** — surowe pliki z 240 runów (CSV, HDF5).
* **`H_run_manifest.csv`** — manifest runów z statusem agregacji.
* **`CITATION.md`** — commit referencyjny `cdca9524…` + szablon BibTeX.

Notebooki **04**, **05** i **06** w `notebooks/` korzystają wprost z
zawartości tego katalogu jako autorytatywnego źródła wyników.

## Uruchamianie testów

Testy odzwierciedlają strukturę katalogów `/src`. Aby uruchomić wszystkie testy jednostkowe:

```bash
python -m pytest
```

## Notebooki

Aktywuj środowisko conda, a następnie uruchom serwer:

```bash
conda activate drone-swarm-env
jupyter lab notebooks/
```

W `/notebooks/` znajduje się 6 samowyjaśniających notebooków
(pełen opis: [`notebooks/README.md`](notebooks/README.md)):

| Plik | Tematyka |
|------|----------|
| `01_world_generation.ipynb` | Generowanie świata + przeszkód i wizualizacja 3D z proporcjonalnym skalowaniem osi |
| `02_offline_optimization.ipynb` | Pełen pipeline optymalizacji offline (MSFOA / OOA / SSA / NSGA-III) z krzywą zbieżności |
| `03_online_collision_scenario.ipynb` | Scenariusz head-on z dynamicznymi przeszkodami i wybranym algorytmem fazy online |
| `04_statistical_analysis.ipynb` | Testy Friedmana + Nemenyiego, A12 Vargha–Delaney i Wilson 95% CI na metrykach M1–M13 |
| `05_thesis_reproduction.ipynb` | Centralny przegląd wszystkich tabel T1–T21 i wykresów W1–W18 cytowanych w pracy |
| `06_data_reproducibility.ipynb` | Domknięcie reprodukowalności: schemat DB, manifest 240 runów, snapshot środowiska, rekonstrukcja z plików per-run |

## Dokumentacja modułów

Szczegółowe opisy działania poszczególnych komponentów systemu (założenia teoretyczne, algorytmy ewolucyjne, modele matematyczne trajektorii i fizyka sensorów) zostały wydzielone do dedykowanych plików markdown w głównym katalogu projektu:

| Moduł projektu | Plik z dokumentacją | Opis rozszerzony |
|----------------|---------------------|------------------|
| `src/algorithms/` | [`ALGORITHMS.md`](src/algorithms/ALGORITHMS.md) | Implementacje 4 algorytmów meta-heurystycznych (MSFOA, NSGA-III, OOA, SSA), Strategy Pattern dla fazy offline (wspólny `VectorizedEvaluator` z 5 obiektywami + 3 ograniczeniami) oraz online avoidance jako kompozycja 4 sub-strategii (`GenericOptimizingAvoidance`). |
| `configs/` | [`CONFIGS.md`](configs/CONFIGS.md) | Hierarchia konfiguracji Hydra 1.3, parametry środowisk i optymalizatorów, instrukcje dotyczące przeszukiwania siatki (grid search) oraz schemat dispatchu. |
| `src/environments/` | [`ENVIRONMENTS.md`](src/environments/ENVIRONMENTS.md) | Opis środowisk 3D (ForestWorld, UrbanWorld, EmptyWorld), generatorów przeszkód BOX/CYLINDER, strategii rozmieszczania (`random_uniform`, `grid_jitter`) oraz zapewnienia reprodukowalności (seed). |
| `src/runner/` | [`RUNNER.md`](src/runner/RUNNER.md) | Obsługa logiki uruchomieniowej za pomocą Strategy Pattern: `GenerationDataStrategy` (nowy eksperyment) i `ReplayDataStrategy` (odtwarzanie z CSV), z opisem 4-etapowego pipeline'u. |
| `src/sensors/` | [`SENSORS.md`](src/sensors/SENSORS.md) | Symulacja detekcji LiDAR 3D (stożkowy FOV 30°, 123 promienie zorganizowane w 7 pierścieni). Optymalizacja operacji przez batch ray casting dla roju 5 UAV oraz integracja z modułem unikania online. |
| `src/trajectory/` | [`TRAJECTORY.md`](src/trajectory/TRAJECTORY.md) | Generator gładkiej trajektorii B-Spline zapewniający ciągłość kinematyczną (C²), opisy profili prędkości (`TrapezoidalProfile` dla globalnych misji, `ConstantSpeedProfile` dla lokalnego unikania) oraz integracja z regulatorem PID w silniku PyBullet. |
| `src/utils/` | [`UTILS.md`](src/utils/UTILS.md) | Przegląd narzędzi pomocniczych: centralny system logowania `SimulationLogger`, asynchroniczny zapis przestrzeni decyzyjnej optymalizacji do formatu HDF5, parsowanie Hydry oraz mechanizmy walidacji środowiska. |
| `src/analysis/analyzer/plots/` | [`PLOTS_LEGEND.md`](src/analysis/analyzer/plots/PLOTS_LEGEND.md) | Legenda i przewodnik interpretacji wykresów: diagramy CD (Demšar 2006), boxploty, krzywe zbieżności, wykresy słupkowe, heatmapy rang, scatter ploty, projekcje frontu Pareto. Słowniczek metryk (HV, IGD+, GD) i testów statystycznych (Friedman, Nemenyi, Wilcoxon, A12). |

## Moja konfiguracja sprzętowa

- **Procesor**: AMD Ryzen 7 7700
- **RAM**: DDR5 64 GB / 6000 MHz (2×32 GB) CL30
- **Karta graficzna**: NVIDIA GIGABYTE RTX 4060 Ti
- **Dysk**: SSD UD85 2 TB PCIe M.2 NVMe Gen 4×4 (3600/2800 MB/s)
- **System operacyjny**: Fedora 43