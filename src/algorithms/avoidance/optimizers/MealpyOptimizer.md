# MealpyOptimizer — Generyczny wrapper SSA/OOA dla Reactive Avoidance

## Referencje naukowe

- **SSA**: Xue, J., & Shen, B. (2020). *A novel swarm intelligence optimization
  approach: sparrow search algorithm.* Systems Science & Control Engineering, 8(1), 22–34.
- **OOA**: Trojovský, P., & Dehghani, M. (2023). *Osprey Optimization Algorithm.*
  Frontiers in Mechanical Engineering, 8, 136.
- **mealpy**: Thieu, N. V., & Mirjalili, S. (2023). *MEALPY: An open-source library
  for latest meta-heuristic algorithms in Python.* Journal of Systems Architecture, 139, 102871.

## Cel

Generyczny adapter `IPathOptimizer` na dowolny algorytm mealpy (SSA, OOA, etc.)
do reaktywnej optymalizacji trajektorii unikowej w czasie rzeczywistym.
Jedyna różnica między SSA a OOA to `algorithm_factory` w konfiguracji YAML:

```yaml
# SSA:
algorithm_factory:
  _target_: mealpy.swarm_based.SSA.OriginalSSA
  _partial_: true
algorithm_kwargs: {ST: 0.8, PD: 0.2, SD: 0.1}

# OOA:
algorithm_factory:
  _target_: mealpy.swarm_based.OOA.OriginalOOA
  _partial_: true
```

## Kontekst architektoniczny

```
GenericOptimizingAvoidance  (sampling_seed → _sampling_rng)
  ├── IObstaclePredictor   → ConstantVelocityPredictor
  ├── IPathRepresentation  → SingleArcDeflection (gene_dim=2)
  ├── IFitnessEvaluator    → WeightedSumFitness
  ├── IPathOptimizer       → MealpyOptimizer  ← TEN PLIK
  │                            └── algorithm_factory: OriginalSSA | OriginalOOA
  └── pre-generacja populacji → PathProblem.initial_population
```

## Parametry konfiguracyjne (YAML/Hydra)

| Parametr | Default | Uwagi |
|---|---|---|
| `algorithm_factory` | — | `_partial_: true` Hydra callable → `mealpy.Optimizer` |
| `epoch` | 10 | Górny cap — natywne `max_time` przerywa wcześniej |
| `pop_size` | 20 | Mały ze względu na budżet ~1 s |
| `min_compute_time_s` | 0.05 | Poniżej → natychmiastowy `timed_out` |
| `rng` | None | Seed dla reproducibility |
| `algorithm_kwargs` | {} | Przekazane do `algorithm_factory` (np. SSA: ST, PD, SD) |

### Parametry SSA-specific (Xue & Shen 2020)

| Parametr | Default | Paper | Opis |
|---|---|---|---|
| `ST` | 0.8 | Safety Threshold ∈ [0.5, 1.0] | Próg przełączania producent → follower |
| `PD` | 0.2 | Producer Density ∈ (0, 1) | Udział producentów |
| `SD` | 0.1 | Sentinel Density ∈ (0, 1) | Udział strażników |

### Parametry OOA-specific

OOA jest algorytmem bezparametrowym — brak dodatkowych `algorithm_kwargs`.

## Mechanizm działania

### 1. Budowa problemu mealpy

`_make_mealpy_problem()` tworzy closure-based `MealpyProblem`:
- `bounds = FloatVar(lb, ub)` z `SingleArcDeflection.gene_bounds()`
- `obj_func(x)` = `decode_genes(x) → BSpline → WeightedSumFitness.evaluate()`
- Niezdekodowalne geny → `1e9` (sentinel penalty)

### 2. Terminacja czasowa (natywna mealpy)

```python
termination = {
    "max_epoch": self.epoch,
    "max_time": max(min_compute_time_s, budget.remaining),
}
```

Mealpy honoruje `max_time` deterministycznie po każdej generacji — primary
defense line. **Nie** używamy `budget.check_or_raise()` w hot-loopie (mealpy
ma własny mechanizm).

### 3. Convergence trace

`algo.history.list_global_best_fit` → lista per-gen best fitness.
Robustna ekstrakcja: `Target.fitness` lub plain float.

## Reprezentacja rozwiązania

Identyczna jak w `MSFFOAOnlineOptimizer`:
- Gene space: 2D `[magnitude, peak_position]` z `SingleArcDeflection`
- Dekodowanie: `decode_genes → 5-waypoint BSpline`

## Funkcja fitness (WeightedSumFitness)

Identyczna jak w `MSFFOAOnlineOptimizer`:
```
cost = w_safety × C_safety + w_energy × C_energy
     + w_jerk × C_jerk + w_symmetry × C_symmetry
```

## Mechanizm budżetu czasu

1. **Primary**: mealpy natywne `max_time` w `termination` dict
2. **Backup**: SIGALRM `hard_deadline()` w `GenericOptimizingAvoidance`

Różnica vs `MSFFOAOnlineOptimizer`: MealpyOptimizer **nie** woła
`budget.check_or_raise()` — polega na natywnym mealpy `max_time`.
Jest to jedyna defense line wewnątrz optimizera; external SIGALRM chroni
przed bugami w mealpy.

## Pragmatyczne adaptacje vs papery

### Wspólne dla SSA i OOA

1. **Framework mealpy** — logika ewolucyjna z biblioteki. Brak modyfikacji
   `evolve()`. Adapter dodaje: problem wrapping, terminację czasową, sentinel
   filter, convergence trace extraction.

2. **2D gene space** — paper SSA i OOA pracują na benchmarkach o wymiarze
   10–1000. Tu operujemy na 2D `[magnitude, peak_position]`, co zmienia
   proporcje populacja/wymiarowość (pop=20 na D=2 → 10 osobników/wymiar).

3. **Sentinel filter (1e8)** — identycznie jak w MSFFOA online.

4. **Zewnętrzna inicjalizacja ceteris paribus (c.p. fix 2026-05-07)** —
   populacja początkowa jest pre-generowana w `GenericOptimizingAvoidance`
   ze wspólnego `np.random.default_rng(sampling_seed)` jako `U(lb, ub)`
   i wstrzykiwana przez `PathProblem.initial_population`. MealpyOptimizer
   konwertuje ją na mealpy `starting_solutions`:
   ```python
   if problem.initial_population is not None
       and problem.initial_population.shape[0] == self.pop_size:
       starting_solutions = [np.clip(sol, lb, ub) for sol in problem.initial_population]
   algo.solve(…, starting_solutions=starting_solutions)
   ```
   Gdy `initial_population` jest `None` (np. stub optimizer w testach),
   mealpy stosuje domyślną inicjalizację wewnętrzną (uniform random).
   Gwarantuje to identyczną sekwencję punktów startowych we wszystkich
   4 optimizerach niezależnie od wewnętrznego PRNG (mealpy PCG64 vs
   pymoo MT19937 vs custom MSFOA PCG64).

### SSA-specific

5. **Niestabilność Eq. 4 (naśladowcy)** — `exp((x_worst - x) / i²)` może
   generować ±∞. W 2D gene space problem jest mniej dotkliwy (różnice pozycji
   rzędu kilku metrów), ale `nan_to_num` w mealpy jest nadal aktywny.

### OOA-specific

6. **Bezparametrowość** — OOA nie ma hiperparametrów do tuning. Jedyne
   konfiguracje to `pop_size` i `epoch` — prostota jest zaletą w online,
   gdzie czas na tuning jest zerowy.
