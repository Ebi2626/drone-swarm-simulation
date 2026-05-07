# MSFFOA — Multiple Swarm Fruit Fly Optimization Algorithm

## Referencja naukowa

Shi, K., Zhang, X., & Xia, S. (2020). *Multiple Swarm Fruit Fly Optimization
Algorithm Based Path Planning Method for Multi-UAVs.* Applied Sciences, 10(8), 2822.

## Cel algorytmu

Planowanie ścieżek dla wielu UAV w trójwymiarowej przestrzeni z zastosowaniem
wielorojowej metaheurystyki inspirowanej zachowaniem muszek owocówek (*Drosophila*).

## Struktura algorytmu

### Parametry konfiguracyjne

| Parametr | Paper | Implementacja | Uwagi |
|---|---|---|---|
| `pop_size` | M — łączna populacja | `pop_size` | Musi być podzielny przez `n_swarms` |
| `n_swarms` (G) | G — liczba podrojów | `n_swarms` (default 5) | Paper Sec. 1 |
| `coe1`, `coe2` | Wagi krzyżowania (coe1 + coe2 = 1) | `coe1=0.8`, `coe2=0.2` | Eq. 14, Sec. 4 |
| `threshold` | Próg fazy globalnej/lokalnej | Dynamiczny: `initial_fitness × threshold_ratio` | **Adaptacja** — paper podaje stałą |
| `max_generations` (NC) | Liczba iteracji | `max_generations` (default 500) | — |
| R (step size) | Skalar | Anizotropowy wektor `step_global_frac`, `step_local_frac` | **Adaptacja** — per-oś |

### Fazy algorytmu (per generacja)

#### Faza 1: Multi-Swarm with Multi-Tasks Searching Strategy (Sec. 2)

Każdy z G podrojów realizuje przeszukiwanie wokół swojego lidera `x_{g,best}`:

- **Faza globalna** (gdy `fitness_lidera > threshold`):

  Eq. 7–8: `X_{g,i} = x_{g,best} + step_global ⊙ sin(2 · U(-1, 1))`

- **Faza lokalna** (gdy `fitness_lidera ≤ threshold`):

  Eq. 9–10: `X_{g,i} = x_{g,best} + step_local ⊙ U(-1, 1)`

Generuje `P = pop_size / G` kandydatów na rój; najlepszy staje się `old_best`.

#### Faza 2: Competitive Strategies of Offspring (Sec. 4, Eq. 14)

Krzyżowanie międzyrojowe — potomek powstaje jako kombinacja wypukła losowego lidera
i aktualnego kandydata:

`new_pop = coe1 · x_{rand_swarm,best} + coe2 · old_pop`

Najlepszy z potomków staje się `new_best`.

#### Faza 3: Competition & Update (Sec. 4, Eq. 18–19)

Turniejowa selekcja: porównanie `old_best` vs `new_best` → `winner`.
Elityzm lokalny: lider roju jest zastępowany przez `winner` **wyłącznie gdy**
`winner_fit < swarm_best_fit` (monotoniczne ulepszanie historycznie najlepszego).

Aktualizacja globalnego optimum: `global_best ← min(swarm_best_fit)`.

## Reprezentacja rozwiązania

- Wektor decyzyjny: tensor `(N_drones, N_inner, 3)` — wewnętrzne punkty kontrolne
  krzywej B-Spline w przestrzeni fizycznej (metry).
- Punkty start/target doklejane przed ewaluacją, tworząc wielobok kontrolny.
- Post-processing: `generate_bspline_batch` → gęsta trajektoria.

## Funkcja fitness (SOO via TrajectorySOOAdapter)

Skalaryzacja wielocelowej ewaluacji `VectorizedEvaluator`:

`fitness = (F / F_ref) @ weights + penalty_weight × max(0, G)`

Gdzie:
- `F`: [f1_trajectory_cost, f2_height_angle, f3_threat, f4_turn, f5_coordination]
- `G`: [obstacle_collisions, swarm_collisions, kinematic_penalty]
- `F_ref`: normalizacja na bazie trajektorii prostoliniowej (straight-line reference)

## Pragmatyczne adaptacje vs paper

Pełna lista adaptacji udokumentowana w `core_msffoa.py` (linie 1–38):

1. **Pominięcie smell-space (Sec. 3)** — paper transformuje pozycje (X,Y) na fizyczne
   współrzędne przez `S = 1/√(X² + Y²)`. Implementacja operuje bezpośrednio na
   fizycznych punktach kontrolnych B-Spline. Skalowanie `step_global`/`step_local`
   rekompensuje brak tego mapowania.

2. **Anizotropowy step-size (R)** — paper definiuje R jako skalar. Implementacja
   stosuje wektor 3D `[R_x, R_y, R_z]` z uwagi na asymetrię świata
   (np. forest: 60×600×11 m).

3. **Dynamiczny threshold** — paper podaje threshold jako parametr stały.
   Implementacja kalibruje go dynamicznie: `initial_fitness × threshold_ratio`
   z podłogą `THRESHOLD_FLOOR = 0.01`.

4. **Zewnętrzna populacja inicjalna** — `initial_population` z `StraightLineNoiseSampling`
   zamiast wewnętrznej inicjalizacji, dla zachowania warunku ceteris paribus
   z NSGA-III i OOA.
