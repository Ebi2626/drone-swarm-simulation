# NSGA-III — Non-dominated Sorting Genetic Algorithm III

## Referencja naukowa

Deb, K., & Jain, H. (2014). *An Evolutionary Many-Objective Optimization Algorithm
Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems
With Box Constraints.* IEEE Transactions on Evolutionary Computation, 18(4), 577–601.

## Cel algorytmu

Wielocelowa optymalizacja trajektorii roju dronów z jednoczesną minimalizacją
pięciu celów (koszt przebiegu, wysokość/kąt, zagrożenie, zakręty, koordynacja roju)
przy trzech ograniczeniach nierównościowych.

## Struktura algorytmu

### Parametry konfiguracyjne

| Parametr | Paper | Implementacja | Uwagi |
|---|---|---|---|
| `pop_size` | N — rozmiar populacji | `pop_size` (default 100) | Dostosowany do ref_dirs |
| `n_gen` | Liczba generacji | `n_gen` (default 100) | — |
| `eta_c` | Indeks dystrybucji SBX | `eta_c` (default 15) | Crossover |
| `eta_m` | Indeks dystrybucji PM | `eta_m` (default 20) | Mutacja |
| `crossover_prob` | Prawdopodobieństwo krzyżowania | `crossover_prob` (default 0.9) | — |
| `mutation_prob` | Prawdopodobieństwo mutacji | `mutation_prob` (default 0.1) | — |
| `n_partitions` | Das-Dennis partycje | Obliczane automatycznie | `calculate_n_partitions()` |
| `decision_mode` | — | `knee_point` / `safety` / `equal` | Selekcja końcowa |

### Mechanizm algorytmu

NSGA-III realizuje wielocelową optymalizację ewolucyjną z sortowaniem
niezdominowanym i punktami referencyjnymi Das-Dennis:

#### 1. Inicjalizacja

Generacja `pop_size` osobników z `StraightLineNoiseSampling` — zaszumiona
linia prosta między punktami start↔target, z anizotropowym szumem
(`noise_std_xy=2.0`, `noise_std_z=0.3`).

#### 2. Operatory genetyczne

- **Krzyżowanie SBX** (Simulated Binary Crossover):
  Deb & Agrawal (1995). `eta_c` kontroluje rozproszenie potomków.

- **Mutacja PM** (Polynomial Mutation):
  Deb & Goyal (1996). `eta_m` kontroluje zasięg perturbacji.

#### 3. Selekcja środowiskowa

1. **Sortowanie niezdominowane**: populacja dzielona na fronty F₁, F₂, ...
2. **Punkty referencyjne Das-Dennis**: równomiernie rozłożone na simpleksie
   w przestrzeni celów.
3. **Niche-count selection**: przy wypełnianiu ostatniego frontu preferowane są
   rozwiązania bliskie niedoreprezentowanym punktom referencyjnym.

#### 4. Kryterium stopu

Stała liczba generacji `n_gen` — identycznie jak `epoch` w SSA/OOA i
`max_generations` w MSFFOA. Zapewnia sprawiedliwy budżet obliczeniowy.

#### 5. Wybór rozwiązania końcowego (Decision Making)

Z frontu Pareto wybierane jest jedno rozwiązanie wg strategii:

- `KneePointDecision` (default) — punkt kolana na froncie.
- `SafetyPriorityDecision` — priorytet minimalizacji kolizji.
- `EqualWeightsDecision` — równe wagi normalizowanych celów.

## Reprezentacja rozwiązania

- Wektor decyzyjny: spłaszczony `(N_drones × N_inner × 3,)`.
- Dekodowanie w `SwarmOptimizationProblem._evaluate()`.
- Post-processing: `generate_bspline_batch` → gęsta trajektoria.

## Funkcja fitness (MOO — VectorizedEvaluator)

NSGA-III operuje **bezpośrednio** na 5 celach i 3 ograniczeniach:

**Cele** (minimalizowane):
- `f₁` — koszt przebiegu (długość + odchylenie od prostej)
- `f₂` — koszt wysokości (odchylenie od H_pref + kąt wznoszenia)
- `f₃` — koszt zagrożenia (penetracja stref przeszkód)
- `f₄` — koszt zakrętów (kąty w płaszczyźnie XY)
- `f₅` — koszt koordynacji (kara wykładnicza za zbliżenie dronów)

**Ograniczenia** (G ≤ 0 = feasible):
- `g₁` — kolizje z przeszkodami
- `g₂` — kolizje wewnątrzrojowe
- `g₃` — kara kinematyczna (max dystans między węzłami, max przyspieszenie)

## Kluczowe różnice wobec SOO algorytmów (MSFFOA, OOA, SSA)

| Aspekt | NSGA-III | MSFFOA / OOA / SSA |
|---|---|---|
| Typ optymalizacji | Wielocelowy (MOO) | Jednocelowy (SOO) |
| Cele | 5 celów natywnie | Skalaryzacja `(F/F_ref) @ w` |
| Ograniczenia | 3 ograniczenia natywnie | `penalty_weight × max(0, G)` |
| Framework | pymoo | mealpy (OOA, SSA) / custom (MSFFOA) |
| Selekcja | Niezdominowane sortowanie | Greedy / elityzm |
| Wynik | Front Pareto + decision maker | Jedno najlepsze rozwiązanie |

## Pragmatyczne adaptacje vs paper

1. **Dynamiczne Das-Dennis partycje** — `calculate_n_partitions()` automatycznie
   dobiera liczbę partycji do `pop_size` i `n_obj=5`, zamiast wymagać
   ręcznego obliczania C(n_obj + p - 1, p).

2. **Stałe kryterium stopu** — `get_termination("n_gen", n_gen)` zamiast
   wielowarunkowego terminatora, dla sprawiedliwego porównania budżetu
   obliczeniowego z SOO algorytmami.

3. **Anizotropowy szum inicjalizacji** — `noise_std_z=0.3` vs `noise_std_xy=2.0`,
   bo zakres Z jest typowo rzędu kilku metrów.

4. **Wspólna infrastruktura ceteris paribus** — `StraightLineNoiseSampling`,
   `SwarmOptimizationProblem`, `VectorizedEvaluator`, `generate_bspline_batch`.
