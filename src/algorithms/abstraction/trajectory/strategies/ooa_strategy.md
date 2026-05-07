# OOA — Osprey Optimization Algorithm

## Referencja naukowa

Trojovský, P., & Dehghani, M. (2023). *Osprey Optimization Algorithm: A new
bio-inspired metaheuristic algorithm for solving engineering optimization problems.*
Frontiers in Mechanical Engineering, 8, 136.

## Cel algorytmu

Optymalizacja trajektorii roju dronów z wykorzystaniem metaheurystyki inspirowanej
zachowaniem rybołowów polujących na ryby. Algorytm realizuje eksplorację
(identyfikacja i pogoń za zdobyczą) oraz eksploatację (przenoszenie zdobyczy
do optymalnej pozycji).

## Struktura algorytmu

### Parametry konfiguracyjne

| Parametr | Paper | Implementacja | Uwagi |
|---|---|---|---|
| `pop_size` | N — rozmiar populacji | `pop_size` (default 200) | — |
| `epoch` | T — maksymalna liczba iteracji | `epochs` / `n_gen` (default 500) | — |

OOA jest algorytmem **bezparametrowym** w sensie paperu — nie posiada
hiper-parametrów specyficznych dla algorytmu (brak odpowiedników ST, PD, SD z SSA).
Jedyne konfiguracje to `pop_size` i `epoch`.

### Fazy algorytmu (per generacja, per osobnik)

#### Faza 1: Identyfikacja pozycji i polowanie na rybę (Eksploracja, Eq. 5)

Dla każdego osobnika `x_i`:
1. Identyfikacja zbioru osobników o lepszym fitness niż `x_i`.
2. Z prawdopodobieństwem 50%: wybór `x_fish` = `x_best` (globalny lider)
   lub losowy osobnik z lepszych.
3. Aktualizacja pozycji:

   `x_new = x_i + N(0,1) × (x_fish − r₁ × x_i)`

   gdzie `r₁ ∈ {1, 2, 3}` (losowa liczba całkowita).

Greedy acceptance: `x_i ← x_new` tylko gdy `fitness(x_new) < fitness(x_i)`.

#### Faza 2: Przenoszenie ryby do odpowiedniej pozycji (Eksploatacja, Eq. 7)

Generacja pozycji losowej w przestrzeni poszukiwań:

`x_new = x_i + lb + U(0,1) × (ub − lb)`

Greedy acceptance: analogicznie jak w Fazie 1.

### Sekwencyjność aktualizacji

OOA aktualizuje osobniki **sekwencyjnie** (nie równolegle) — każdy osobnik
widzi zaktualizowane pozycje poprzedników w tej samej generacji.
W mealpy: `is_parallelizable = False`.

## Reprezentacja rozwiązania

- Wektor decyzyjny: spłaszczony tensor `(N_drones × N_inner × 3,)` —
  wewnętrzne punkty kontrolne krzywej B-Spline.
- Dekodowanie w `OOAProblemAdapter._decode_inner()` do `(Pop, N_drones, N_inner, 3)`.
- Post-processing: `generate_bspline_batch` → gęsta trajektoria.

## Funkcja fitness (SOO via TrajectorySOOAdapter)

Identyczna skalaryzacja jak w MSFFOA i SSA:

`fitness = (F / F_ref) @ weights + penalty_weight × max(0, G)`

## Mechanizm wymuszania granic (Hard Clipping)

OOA bywa numerycznie niestabilne — Eq. 5 zawiera mnożnik `N(0,1)` na dużych
różnicach pozycji, co może generować wartości ±∞. Implementacja wymusza granice
na **trzech poziomach**:

1. `OOAProblemAdapter.amend_position()` — wywoływane przez mealpy po każdym ruchu
   (`np.clip` + `np.nan_to_num`).
2. `obj_func()` — fail-safe clipping na wejściu do ewaluacji.
3. `LoggedOriginalOOA.evolve()` — clipping przed zapisem historii.

## Pragmatyczne adaptacje vs paper

1. **Framework mealpy** — algorytm nie jest implementowany od zera, lecz używa
   `mealpy.swarm_based.OOA.OriginalOOA`. Logika ewolucyjna pochodzi z biblioteki,
   a projekt dodaje warstwę adaptera (`OOAProblemAdapter`) i loggera
   (`LoggedOriginalOOA`).

2. **Wspólna inicjalizacja populacji** — `StraightLineNoiseSampling` zamiast
   domyślnej losowej inicjalizacji mealpy, dla zachowania ceteris paribus.

3. **Wspólne granice (xl/xu)** — z `SwarmOptimizationProblem`, identyczne
   co w NSGA-III, MSFFOA i SSA.

4. **Trójpoziomowy hard clipping** — rozszerzenie wobec paperu, konieczne
   z uwagi na niestabilność numeryczną w kontekście 3D trajektorii.
