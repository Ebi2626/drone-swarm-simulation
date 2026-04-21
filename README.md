# drone-swarm-simulation

Symulacja planowania tras roju dronów z wykorzystaniem nowoczesnych algorytmów heurystycznych
inspirowanych biologicznie (OOA, SSA, MSFFOA), porównywanych z klasycznym NSGA-III
w środowiskach fizycznych opartych na PyBullet.

## Struktura projektu

```
drone-swarm-simulation/
├── configs/                    # Modularna konfiguracja Hydra 1.3
│   ├── config.yaml             # Globalne parametry eksperymentu
│   ├── avoidance/              # Strategie unikania online (A*, none, hybrydy algorytmów)
│   ├── environment/            # Konfiguracje światów (empty, forest, urban) + strategie rozmieszczenia
│   └── optimizer/              # Konfiguracje optymalizatorów offline (msffoa, nsga-3, ooa, ssa)
├── experiments/                # Skrypty uruchomieniowe eksperymentów
├── notebooks/                  # Notebooki wyjaśniające bloki konstrukcyjne projektu
├── results/                    # Logi z konkretnych uruchomień
├── src/                        # Główna logika aplikacji
│   ├── algorithms/             # Logika optymalizacji (planowanie offline + unikanie online)
│   │   ├── abstraction/        # Implementacje Strategy Pattern (MSFOA, NSGA-III, OOA, SSA)
│   │   └── avoidance/          # Reaktywne unikanie lokalne (siatka A* 3D)
│   ├── environments/           # Środowiska symulacji 3D
│   │   ├── abstraction/        # Generatory przeszkód i granic świata
│   │   ├── obstacles/          # Enum typów przeszkód (BOX, CYLINDER)
│   │   ├── EmptyWorld.py       # Puste środowisko (benchmarki)
│   │   ├── ForestWorld.py      # Las — 17 przeszkód cylindrycznych
│   │   ├── SwarmBaseWorld.py   # Bazowa klasa abstrakcyjna środowiska
│   │   └── UrbanWorld.py       # Miasto — 27 przeszkód prostopadłościennych
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
├── ExperimentRunner.py         # Punkt wejścia dla eksperymentów konfigurowanych przez Hydra
├── main.py                     # Runner z dwoma trybami uruchomienia:
│                               #   default: python main.py — nowa symulacja z dynamicznym światem
│                               #   replay:  python main.py --replay /results/2026-04-13/
├── environment.yaml            # Zależności Conda
└── mypy.ini                    # Konfiguracja sprawdzania typów (mypy)
```

## Główne założenia

- Eksperymenty są **w pełni reprodukowalne** — stan świata, przeszkody i trajektorie archiwizowane do CSV
- Macierzowy plan eksperymentów:
  - **3 środowiska** o rosnącej gęstości przeszkód: `empty`, `forest` (17 cylindrów), `urban` (27 boxów)
  - **4 algorytmy**: MSFOA, SSA, OOA (inspirowane biologicznie) + NSGA-III (punkt odniesienia)
  - **2 warianty** każdego środowiska: przeszkody statyczne i dynamiczne
- **Dwufazowe planowanie**: optymalizacja globalna offline (waypoints) + reaktywne unikanie online (LiDAR + A*)
- **Reprezentacja trajektorii**: Cubic B-Spline (ciągłość C²) z trapezowym lub stałym profilem prędkości
- Miary oceny algorytmów:
  - Czas dotarcia do celu
  - Liczba kolizji (w tym kolizje między dronami w roju)
  - Gładkość trajektorii
  - Zachowanie w momencie pojawienia się dynamicznej przeszkody

## Przegląd architektury

```
CLI / konfiguracja Hydra
          │
          ▼
   ExperimentRunner
          │
  ┌───────┴──────────────────┐
  │                          │
GenerationStrategy      ReplayStrategy
  │                          │
  ▼                          ▼
algorithms/             archiwa CSV
 (MSFOA/NSGA-III/        (świat, przeszkody,
  OOA/SSA)                trajektorie)
  │
  ▼
trajectory/ (B-Spline)
  │
  ▼
environments/ (fizyka PyBullet)
  │
  ├── sensors/ (LiDAR → avoidance/)
  └── utils/ (SimulationLogger → results/)
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

## Uruchamianie testów

Testy odzwierciedlają strukturę katalogów `/src`. Aby uruchomić wszystkie testy jednostkowe:

```bash
python -m pytest
```

## Notebooki

Aktywuj środowisko conda, a następnie uruchom serwer:

```bash
jupyter notebook
```

Dostępne notebooki w `/notebooks/`:
- `world_generation.ipynb` — generowanie losowych światów z konfigurowalnymi parametrami
- `draw_trajectory.ipynb` — wizualizacja przykładowych trajektorii wszystkich algorytmów w abstrakcyjnej przestrzeni 2D/3D

## Dokumentacja modułów

Szczegółowe opisy działania poszczególnych komponentów systemu (założenia teoretyczne, algorytmy ewolucyjne, modele matematyczne trajektorii i fizyka sensorów) zostały wydzielone do dedykowanych plików markdown w głównym katalogu projektu:

| Moduł projektu | Plik z dokumentacją | Opis rozszerzony |
|----------------|---------------------|------------------|
| `src/algorithms/` | [`ALGORITHMS.md`](src/algorithms/ALGORITHMS.md) | Implementacje algorytmów ewolucyjnych (MSFOA, NSGA-III, OOA, SSA), mechanizm Strategy Pattern, wielokryterialne funkcje celu (długość trasy, ryzyko, energia) oraz reaktywne unikanie przeszkód A*. |
| `configs/` | [`CONFIGS.md`](configs/CONFIGS.md) | Hierarchia konfiguracji Hydra 1.3, parametry środowisk i optymalizatorów, instrukcje dotyczące przeszukiwania siatki (grid search) oraz schemat dispatchu. |
| `src/environments/` | [`ENVIRONMENTS.md`](src/environments/ENVIRONMENTS.md) | Opis środowisk 3D (ForestWorld, UrbanWorld, EmptyWorld), generatorów przeszkód BOX/CYLINDER, strategii rozmieszczania (`random_uniform`, `grid_jitter`) oraz zapewnienia reprodukowalności (seed). |
| `src/runner/` | [`RUNNER.md`](src/runner/RUNNER.md) | Obsługa logiki uruchomieniowej za pomocą Strategy Pattern: `GenerationDataStrategy` (nowy eksperyment) i `ReplayDataStrategy` (odtwarzanie z CSV), z opisem 4-etapowego pipeline'u. |
| `src/sensors/` | [`SENSORS.md`](src/sensors/SENSORS.md) | Symulacja detekcji LiDAR 3D (stożkowy FOV 30°, 123 promienie zorganizowane w 7 pierścieni). Optymalizacja operacji przez batch ray casting dla roju 5 UAV oraz integracja z modułem unikania online. |
| `src/trajectory/` | [`TRAJECTORY.md`](src/trajectory/TRAJECTORY.md) | Generator gładkiej trajektorii B-Spline zapewniający ciągłość kinematyczną (C²), opisy profili prędkości (`TrapezoidalProfile` dla globalnych misji, `ConstantSpeedProfile` dla lokalnego unikania) oraz integracja z regulatorem PID w silniku PyBullet. |
| `src/utils/` | [`UTILS.md`](src/utils/UTILS.md) | Przegląd narzędzi pomocniczych: centralny system logowania `SimulationLogger`, asynchroniczny zapis przestrzeni decyzyjnej optymalizacji do formatu HDF5, parsowanie Hydry oraz mechanizmy walidacji środowiska. |

## Moja konfiguracja sprzętowa

- **Procesor**: AMD Ryzen 7 7700
- **RAM**: DDR5 64 GB / 6000 MHz (2×32 GB) CL30
- **Karta graficzna**: NVIDIA GIGABYTE RTX 4060 Ti
- **Dysk**: SSD UD85 2 TB PCIe M.2 NVMe Gen 4×4 (3600/2800 MB/s)
- **System operacyjny**: Fedora 43