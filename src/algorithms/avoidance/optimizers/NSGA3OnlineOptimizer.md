# NSGA3OnlineOptimizer — Online NSGA-III dla Reactive Avoidance

## Referencja naukowa

Deb, K., & Jain, H. (2014). *An Evolutionary Many-Objective Optimization Algorithm
Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems
With Box Constraints.* IEEE Transactions on Evolutionary Computation, 18(4), 577–601.

## Cel

Reaktywna optymalizacja trajektorii unikowej z wykorzystaniem NSGA-III
w kontekście ograniczonego budżetu czasu (~1 s). Mimo natywnej wielocelowości
NSGA-III, optimizer operuje na **jednym skalarnym celu** (weighted-sum)
z przyczyn uczciwości porównawczej z SSA/OOA/MSFOA (fairness fix 2026-05-03).

## Kontekst architektoniczny

```
GenericOptimizingAvoidance  (sampling_seed → _sampling_rng)
  ├── IObstaclePredictor   → ConstantVelocityPredictor
  ├── IPathRepresentation  → SingleArcDeflection (gene_dim=2)
  ├── IFitnessEvaluator    → WeightedSumFitness
  ├── IPathOptimizer       → NSGA3OnlineOptimizer  ← TEN PLIK
  └── pre-generacja populacji → PathProblem.initial_population
```

## Parametry konfiguracyjne (YAML/Hydra)

| Parametr | Default | Paper | Uwagi |
|---|---|---|---|
| `epoch` | 10 | — | Terminacja: `("n_gen", epoch)` |
| `pop_size` | 20 | N | Mały ze względu na budżet ~1 s |
| `n_partitions` | 4 | Das-Dennis p | Irrelevant dla n_obj=1 (→ 1 ref_dir) |
| `decision_mode` | "safety" | — | Legacy; nieużywany po fairness fix |
| `min_compute_time_s` | 0.05 | — | Poniżej → natychmiastowy `timed_out` |
| `rng` | None | — | Seed dla reproducibility |

## Mechanizm algorytmu

### 1. Definicja problemu pymoo

`_build_pymoo_problem()` tworzy `pymoo.Problem` z:
- `n_var = gene_dim` (2 z `SingleArcDeflection`)
- `n_obj = 1` — skalarny weighted-sum fitness (fairness fix)
- `xl, xu` z `SingleArcDeflection.gene_bounds()`
- `_evaluate(X, out)`: per-osobnik `decode_genes → BSpline → WeightedSumFitness.evaluate()`

### 2. NSGA-III z 1D objective

```python
ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=1)
# → 1 ref_dir; NSGA-III niching zachowane w decision space,
#   ale obj space 1D więc niching nie wpływa na selekcję
```

NSGA-III mechanika (niezdominowane sortowanie + reference-point niching)
jest nadal aktywna, ale z 1 celem redukuje się do zwykłego sortowania
po fitness (analogicznie jak GA z turnijem).

### 3. Selekcja końcowa

`argmin(F[:, 0])` — najniższy skalarny fitness z feasible populacji.

Historyczne tryby (`_select_from_pareto` z `decision_mode`):
- `"safety"` — lexsort po `c_safety` (legacy, pre-fairness fix)
- `"weighted"` — weighted-sum z wagami `WeightedSumFitness`
- `"knee_point"` — geometryczny knee w 4D Pareto

Te tryby są zachowane w kodzie, ale nieaktywne po fairness fix (n_obj=1).

## Reprezentacja rozwiązania

Identyczna jak w pozostałych online optimizerach:
- Gene space: 2D `[magnitude, peak_position]` z `SingleArcDeflection`
- Dekodowanie: `decode_genes → 5-waypoint BSpline`

## Funkcja fitness (WeightedSumFitness — scalar)

```
cost = w_safety × C_safety + w_energy × C_energy
     + w_jerk × C_jerk + w_symmetry × C_symmetry
```

**Fairness fix (2026-05-03, Krok 1)**: NSGA-III dotychczas używał
`evaluate_components` (4D Pareto) + `decision_mode="safety"` (lexsort
po c_safety, ignorując c_energy/jerk/symmetry). Pozostałe 3 algorytmy
używały scalar weighted-sum. Różne funkcje celu → nieporównywalność.
Ujednolicono: NSGA-III też scalar weighted-sum.

## Mechanizm budżetu czasu

1. **Cooperative**: `_BudgetCallback.__call__()` woła `budget.check_or_raise()`
   po każdej generacji → `BudgetExceeded` propaguje przez `pymoo.minimize()`
2. **Best-so-far capture**: callback snapshot'uje `algorithm.pop` PRZED
   rzuceniem `BudgetExceeded` → recovery z najlepszego feasible osobnika
3. **Hard deadline (SIGALRM)**: zewnętrzny circuit breaker

### Convergence trace

Per-gen minimum feasible fitness (< 1e8 sentinel). Jeśli cała populacja
infeasible → `inf`.

## Pragmatyczne adaptacje vs paper (Deb & Jain 2014)

1. **n_obj=1 zamiast many-objective** — fundamentalna rozbieżność. NSGA-III
   jest zaprojektowany dla M ≥ 3 celów z reference-point niching. Przy n_obj=1:
   - Sortowanie niezdominowane → zwykłe sortowanie po fitness
   - Das-Dennis ref_dirs → 1 punkt (trivialny)
   - Niching → nieaktywny
   - Efektywnie: NSGA-III zachowuje się jak µ+λ ES z crossoverem SBX i mutacją PM
   
   **Uzasadnienie**: warunek ceteris paribus — porównywanie algorytmów wymaga
   identycznej funkcji celu. Multi-obj NSGA-III vs scalar SOO to porównywanie
   jabłek z gruszkami.

2. **pop_size=20, epoch=10** — paper Deb & Jain testuje na pop_size=92–212,
   generations=400–750. Online budżet ~1 s na 2D problem nie pozwala na
   więcej. Przy gene_dim=2 i pop_size=20, NSGA-III ma nadmiar mocy
   (20 osobników na 2D → 10/wymiar) — konwergencja powinna nastąpić
   w <5 generacjach.

3. **Brak `eliminate_duplicates`** — domyślna pymoo NSGA3 konfiguracja.
   Online z 2D gene space i pop=20 duplikaty rzadkie.

4. **Best-so-far recovery przy BudgetExceeded** — paper nie adresuje
   kwestii budżetu czasu (NSGA-III projektowane dla batch offline).
   `_BudgetCallback` z capture'em populacji to pragmatyczny dodatek.

5. **Zewnętrzna inicjalizacja ceteris paribus (c.p. fix 2026-05-07)** —
   populacja początkowa jest pre-generowana w `GenericOptimizingAvoidance`
   ze wspólnego `np.random.default_rng(sampling_seed)` i wstrzykiwana
   przez `PathProblem.initial_population`. NSGA3OnlineOptimizer przekazuje
   ją jako `sampling` kwarg do pymoo `NSGA3`:
   ```python
   nsga3_kwargs = dict(ref_dirs=ref_dirs, pop_size=self.pop_size)
   if problem.initial_population is not None
       and problem.initial_population.shape[0] == self.pop_size:
       nsga3_kwargs["sampling"] = np.clip(problem.initial_population, lb, ub)
   algorithm = NSGA3(**nsga3_kwargs)
   ```
   **Ważne**: `sampling=None` NIE jest przekazywane explicite — pymoo
   traktuje `None` jako nadpisanie defaultowego `FloatRandomSampling`,
   co powoduje `TypeError`. Dlatego klucz `sampling` jest dodawany do
   `nsga3_kwargs` warunkowo, tylko gdy `initial_population` jest dostępne.
   Gdy brak — pymoo stosuje domyślny `FloatRandomSampling` (MT19937).

6. **Sentinel filter (1e8)** — identycznie jak w pozostałych online optimizerach.
