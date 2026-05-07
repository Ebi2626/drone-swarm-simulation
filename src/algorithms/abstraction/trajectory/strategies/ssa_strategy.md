# SSA — Sparrow Search Algorithm

## Referencja naukowa

Xue, J., & Shen, B. (2020). *A novel swarm intelligence optimization approach:
sparrow search algorithm.* Systems Science & Control Engineering, 8(1), 22–34.

## Cel algorytmu

Optymalizacja trajektorii roju dronów z wykorzystaniem metaheurystyki inspirowanej
zachowaniem wróbli — podziałem populacji na producentów (poszukiwaczy pożywienia),
naśladowców (followerów) i strażników (alarm sparrows).

## Struktura algorytmu

### Parametry konfiguracyjne

| Parametr | Paper | Implementacja | Uwagi |
|---|---|---|---|
| `pop_size` | N — rozmiar populacji | `pop_size` (default 200) | — |
| `epoch` | T_max — maks. iteracji | `epochs` / `n_gen` (default 500) | — |
| `ST` | Safety Threshold ∈ [0.5, 1.0] | `ST` (default 0.8) | Paper Eq. 3 — próg bezpieczeństwa |
| `PD` | Producer Density ∈ (0, 1) | `PD` (default 0.2) | Udział producentów w populacji |
| `SD` | Sentinel Density ∈ (0, 1) | `SD` (default 0.1) | Udział strażników (alarm sparrows) |

### Fazy algorytmu (per generacja)

Populacja jest posortowana malejąco wg fitness. Indeks `n1 = ⌊PD × N⌋`,
`n2 = ⌊SD × N⌋`.

#### Faza 1: Aktualizacja producentów (Eq. 3, idx < n1)

Producenci (najlepsi osobnicy) eksplorują przestrzeń:

- Jeśli `R₂ < ST` (brak zagrożenia):

  `x_new = x × exp(−(idx+1) / (U × epoch + ε))`

  gdzie `U ∈ U(0.5, 1)` — losowy współczynnik.
  Jeżeli wykładnik > 5, zastępowany jest wartością `U(0,1)`.

- Jeśli `R₂ ≥ ST` (zagrożenie):

  `x_new = x + N(0, 1)` — losowy spacer gaussowski.

#### Faza 2: Aktualizacja naśladowców (Eq. 4, idx ≥ n1)

- **Naśladowcy bliscy liderowi** (idx ≤ N/2):

  `x_new = x_best + |x − x_best| × A₁ᵀ`

  gdzie `A = sign(U(-1,1))`, `A₁ = Aᵀ(AAᵀ)⁻¹L`, `L = 𝟏(1, D)`.
  Realizuje ruch w kierunku lidera z macierzą pseudoodwrotną.

- **Naśladowcy odlegli** (idx > N/2):

  `x_new = N(0,1) × exp((x_worst − x) / (idx+1)²)`

  Drgania skrzydeł — mechanizm ucieczki od najgorszych pozycji.
  **Uwaga**: wyrażenie `exp(...)` jest źródłem niestabilności numerycznej
  przy dużych różnicach pozycji.

#### Faza 3: Aktualizacja strażników — Alarm Value (Eq. 5, losowe n2 osobników)

- Jeśli `fitness(x_i) > fitness(x_best)`:

  `x_new = x_best + N(0,1) × |x − x_best|`

- Jeśli `fitness(x_i) ≤ fitness(x_best)` (osobnik bliski optimum):

  `x_new = x + U(-1,1) × |x − x_worst| / (f(x) − f(x_worst) + ε)`

### Selekcja

Greedy acceptance: każda aktualizacja pozycji jest akceptowana tylko gdy nowy
fitness jest lepszy od starego.

## Reprezentacja rozwiązania

- Wektor decyzyjny: spłaszczony tensor `(N_drones × N_inner × 3,)`.
- Dekodowanie w `SSAProblemAdapter._decode_inner()`.
- Post-processing: `generate_bspline_batch` → gęsta trajektoria.

## Funkcja fitness (SOO via TrajectorySOOAdapter)

Identyczna skalaryzacja jak w MSFFOA i OOA:

`fitness = (F / F_ref) @ weights + penalty_weight × max(0, G)`

## Mechanizm wymuszania granic (Hard Clipping)

Analogicznie jak OOA — trójpoziomowy clipping:

1. `SSAProblemAdapter.amend_position()` — `np.clip` + `np.nan_to_num`.
2. `obj_func()` — fail-safe na wejściu.
3. `LoggedOriginalSSA.evolve()` — przed zapisem historii.

SSA jest **szczególnie podatna** na eksplozję numeryczną — Eq. 4
zawiera `exp((x_worst − x_i) / i²)`, co dla dużych różnic pozycji
generuje wartości ±∞. Stąd `nan_to_num` jest krytyczny.

## Pragmatyczne adaptacje vs paper

1. **Framework mealpy** — `mealpy.swarm_based.SSA.OriginalSSA` dostarcza
   logikę ewolucyjną. Projekt dodaje `SSAProblemAdapter` (hard clipping)
   i `LoggedOriginalSSA` (logowanie per-generacja).

2. **Wspólna inicjalizacja populacji** — `StraightLineNoiseSampling` zamiast
   domyślnej losowej inicjalizacji mealpy.

3. **Wspólne granice (xl/xu)** — z `SwarmOptimizationProblem`.

4. **Trójpoziomowy hard clipping** — rozszerzenie wobec paperu, krytyczne
   z uwagi na niestabilność numeryczną Eq. 4.
