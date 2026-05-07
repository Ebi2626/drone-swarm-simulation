# MSFFOAOnlineOptimizer — Online MSFOA dla Reactive Avoidance

## Referencja naukowa

Shi, K., Zhang, X., & Xia, S. (2020). *Multiple Swarm Fruit Fly Optimization
Algorithm Based Path Planning Method for Multi-UAVs.* Applied Sciences, 10(8), 2822.

## Cel

Reaktywna optymalizacja trajektorii unikowej w czasie rzeczywistym (budżet ~1 s).
Algorytm MSFOA pracuje na płaskim wektorze YZ-genów (delty deflection)
generowanym przez `SingleArcDeflection`, produkując krzywe B-Spline unikowe.

## Kontekst architektoniczny

```
GenericOptimizingAvoidance  (sampling_seed → _sampling_rng)
  ├── IObstaclePredictor   → ConstantVelocityPredictor
  ├── IPathRepresentation  → SingleArcDeflection (gene_dim=2)
  ├── IFitnessEvaluator    → WeightedSumFitness
  ├── IPathOptimizer       → MSFFOAOnlineOptimizer  ← TEN PLIK
  └── pre-generacja populacji → PathProblem.initial_population
```

## Parametry konfiguracyjne (YAML/Hydra)

| Parametr | Default | Paper Sec. | Uwagi |
|---|---|---|---|
| `pop_size` | 20 | Sec. 1 (M) | Drastycznie mniejszy niż offline (200) |
| `n_swarms` (G) | 4 | Sec. 1 | P = pop_size/G = 5 fly/swarm |
| `max_generations` | 10 | — | Cooperative budget zwykle przerwie wcześniej |
| `coe1` / `coe2` | 0.8 / 0.2 | Eq. 14 | coe1 + coe2 = 1 (walidowane) |
| `threshold_ratio` | 0.5 | Sec. 1 (adaptacja) | `threshold = global_best_fit × 0.5` |
| `step_global_frac` | 0.10 | Eq. 7 (adaptacja) | 10% zakresu genów (~3× większy niż offline) |
| `step_local_frac` | 0.03 | Eq. 9 (adaptacja) | 3% zakresu genów |
| `min_compute_time_s` | 0.05 | — | Poniżej → natychmiastowy `timed_out` |

## Fazy algorytmu (per generacja)

Identyczne z offline `core_msffoa.py`, ale na wektorze (K,) zamiast tensora
(N_drones, N_inner, 3):

### Faza 1: Multi-Swarm Searching (Eq. 7–8)

- **Globalna** (swarm_best_fit > threshold):
  `x_new = x_best + step_global × sin(2 × U(-1,1))`

- **Lokalna** (swarm_best_fit ≤ threshold):
  `x_new = x_best + step_local × U(-1,1)`

### Faza 2: Cross-Swarm Crossover (Eq. 14)

`new_pop = coe1 × x_{rand_leader} + coe2 × old_pop`

### Faza 3: Competition & Elitism (Eq. 18–19)

Lider roju → aktualizacja tylko gdy `winner_fit < swarm_best_fit` (monotoniczny).

## Reprezentacja rozwiązania

- **Gene space**: 2D wektor `[magnitude, peak_position]` z `SingleArcDeflection`
- `magnitude` ∈ [0.8, 4.0] m — odchylenie od osi bazowej
- `peak_position` ∈ [0.3, 0.7] — pozycja szczytu na odcinku start→rejoin
- Dekodowanie: `decode_genes → 5-waypoint BSpline`

## Funkcja fitness (WeightedSumFitness)

```
cost = w_safety × C_safety + w_energy × C_energy
     + w_jerk × C_jerk + w_symmetry × C_symmetry
```

Gdzie:
- `C_safety` — quadratic hinge penalty za zbliżenie do predykowanej pozycji przeszkody
- `C_energy` — ∫|∂²p/∂u²|² du (proxy energii sterowania)
- `C_jerk` — max|∂³p/∂u³| (peak control effort)
- `C_symmetry` — odchylenie od osi sticky-axis (Fiorini-Shiller anti-flip-flop)

## Mechanizm budżetu czasu

1. **Cooperative**: `budget.check_or_raise()` na początku każdej generacji →
   `BudgetExceeded` → zwrot `best_so_far` jeśli feasible
2. **Hard deadline (SIGALRM)**: zewnętrzny circuit breaker w
   `GenericOptimizingAvoidance` (`time_budget_s × hard_kill_factor`)

## Pragmatyczne adaptacje vs paper

1. **Flat vector zamiast 3D tensora** — online operuje na 2D gene space
   (magnitude, peak_position), nie na (N_drones, N_inner, 3). Dla 1 drona ×
   2 geny, wielorojowość jest zachowana (G=4, P=5) ale operuje na wektorach
   o wymiarze K=2.

2. **Populacja 20 zamiast 200** — online budżet ~1 s wymusza drastyczną
   redukcję. Konwergencja wspierana przez:
   - mniejszy search space (2D vs ~150D offline),
   - większe step fractions (0.10 vs 0.01 offline).

3. **Zewnętrzna inicjalizacja ceteris paribus (c.p. fix 2026-05-07)** —
   populacja początkowa jest pre-generowana w `GenericOptimizingAvoidance`
   ze wspólnego `np.random.default_rng(sampling_seed)` jako `U(lb, ub)`
   i wstrzykiwana przez `PathProblem.initial_population`. Optimizer
   przyjmuje ją warunkowo:
   ```python
   if problem.initial_population is not None
       and problem.initial_population.shape == (self.pop_size, K):
       pop = np.clip(problem.initial_population, lb, ub)
   else:
       pop = lb + rng.uniform(…) * (ub - lb)  # fallback wewnętrzny
   ```
   Gwarantuje to identyczną sekwencję punktów startowych we wszystkich
   4 optimizerach (MSFOA, SSA, OOA, NSGA-III) niezależnie od ich
   wewnętrznego PRNG. Offline analog: `StraightLineNoiseSampling` —
   online nie stosuje szumu wokół prostej (brak globalnej trasy do
   zaszumienia), lecz uniform w gene bounds.

4. **Sentinel filter (1e8)** — osobniki z `decode_genes → None` dostają
   fitness 1e9. Filtr post-optymalizacyjny odrzuca wynik jeśli
   `global_best_fit > 1e8` → `status="failed"`.

5. **Best-so-far recovery** — przy `BudgetExceeded` zwracamy najlepszego
   feasible osobnika zamiast `None`, co było regresją w starszych wersjach.
