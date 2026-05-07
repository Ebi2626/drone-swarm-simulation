# ETL Pipeline — Dokumentacja tabel

Progressywna dokumentacja tabel `analysis.db`, gromadzona w trakcie
naprawy ETL pipeline (Kamień 1 z `plan.md`). Każda sekcja per-tabela
opisuje:
- **Cel** — co reprezentuje (pojedynczy wpis = co fizycznie znaczy?)
- **Źródło danych** — CSV/h5/derived
- **Populator** — moduł `src/analysis/db/populate_*.py` zapisujący
- **Konsumenci** — co czyta (widoki SQL, populatory derived, analyzer)
- **Klucze + relacje** — PK/FK + cardinality
- **Znane ograniczenia** — pola intencjonalnie NULL, decyzje projektowe
- **Historia napraw** — co i kiedy poprawione

## Architektura ETL — ogólny przepływ

```
results/exp_<id>/<run_dir>/        ← surowe artefakty per run
    ├── trajectories.csv
    ├── collisions.csv
    ├── evasion_events.csv
    ├── generated_obstacles.csv
    ├── world_boundaries.csv
    ├── online_optimization.csv
    ├── convergence_traces.csv
    ├── optimization_timings.csv
    ├── counted_trajectories.csv
    └── optimization_history/optimization_history.h5
            │
            │ (1) initialize_database  ← schema.sql
            │ (2) populate_database    ← per-run loader
            │ (3) build_reference_pareto_sets ← cross-run
            │ (4) backfill_moo_quality_with_reference
            ↓
results/exp_<id>/analysis.db
            ↓
    (5) ExperimentAnalyzer.analyze
            ↓
results/exp_<id>/analysis_output/
    ├── tables/   (CSV + LaTeX)
    └── plots/    (PDF)
```

**Kolejność populacji per run** (z `populate_database.py`):
1. `_register_run_files` → `run_files`
2. `_load_optimization_timings` → `optimization_timings`
3. `_load_collisions` → `collisions`
4. `_load_evasion_events` → `evasion_events`
5. `_load_world_boundaries` → `world_boundaries`
6. `_load_generated_obstacles` → `generated_obstacles`
7. `_load_counted_trajectories` → `counted_trajectory_points`
8. `_load_trajectories` → `trajectory_samples`
9. `_load_optimization_history` → `optimization_generation_stats`
10. `populate_online_metrics` → `online_optimization_tasks`,
    `online_convergence_traces`
11. `populate_moo_quality` → dodatkowe wpisy w `optimization_generation_stats`
    (spread/spacing/r2/gd/igd+/hv per gen)
12. `populate_trajectory_metrics` → `trajectory_metrics`
13. `populate_iteration_metrics` → `iteration_metrics` (z generation_stats)
14. `populate_uav_metrics` → `uav_metrics`
15. `populate_online_safety_metrics` → `uav_online_metrics`
16. `populate_run_metrics` → `run_metrics` (agregat z uav_metrics +
    uav_online_metrics + others)
17. `populate_offline_objectives` → `run_metrics` UPDATE z h5 best-feasible
    F-vector

**Po wszystkich runach** (cross-run):
1. `build_reference_pareto_sets` → `reference_pareto_sets`,
   `reference_points`
2. `backfill_moo_quality_with_reference` → re-run `populate_moo_quality`
   z R+r* dla każdego runa, potem re-run `populate_iteration_metrics`
   + `populate_run_metrics`

## Status naprawy tabel

| # | Tabela | Status | Sekcja |
|---|--------|--------|--------|
| 1 | `evasion_events` | ✅ done (2026-05-07) | [↓](#evasion_events) |
| 2 | `generated_obstacles` | ✅ done (2026-05-07) | [↓](#generated_obstacles) |
| 3 | `metric_definitions` | ✅ done — DROP (2026-05-07) | [↓](#metric_definitions) |
| 4 | `online_optimization_tasks` | ✅ done (2026-05-07) | [↓](#online_optimization_tasks) |
| 5 | `pareto_run_metrics` | ✅ done — DROP (2026-05-07) | [↓](#pareto_run_metrics) |
| 6 | `reference_pareto_sets` | ✅ done — keep, doc-only (2026-05-07) | [↓](#reference_pareto_sets) |
| 7 | `reference_points` | ✅ done — keep, doc-only (2026-05-07) | [↓](#reference_points) |
| 8 | `run_files` | ✅ done (2026-05-07) | [↓](#run_files) |
| 9 | `run_metrics` | ✅ done (2026-05-07) | [↓](#run_metrics) |
| 10 | `runs` | ✅ done (2026-05-07) | [↓](#runs) |
| 11 | `uav_metrics` | ✅ done (2026-05-07) | [↓](#uav_metrics) |

---

## Per-table

### evasion_events

**Cel:** lifecycle-log fazy reaktywnego unikania per-drone. Pojedynczy
wpis = jedno zdarzenie (`trigger`, `plan_built`, `no_plan`,
`cooldown_skip`, `rejoin`, `collision`, `cooldown_skip`).

**Źródło danych:** `<run_dir>/evasion_events.csv` zapisany przez
`SimulationLogger.log_evasion_event` w trakcie symulacji (callsity:
[SwarmFlightController.py:756, 935, 1066, 1244, 1283](src/algorithms/SwarmFlightController.py)).

**Populator:** `_load_evasion_events` w
[populate_database.py:266](src/analysis/db/populate_database.py#L266).

**Konsumenci:**
- `run_metrics.evasion_event_count` (agregat COUNT przez
  `populate_run_metrics`)
- `vw_run_summary` / `vw_run_online_summary` (widoki SQL)
- Bezpośrednio w analizatorze: brak (przyszły wykres distribution
  preferred_axis vs avoidance algo)

**Klucze:**
- PK `event_id` (autoincrement)
- UNIQUE `(run_id, event_index)` — porządek per run
- UNIQUE `(run_id, sim_time, drone_id, event_type, event_index)` —
  zapobiega duplikatom przy re-load
- FK `run_id → runs(run_id) ON DELETE CASCADE`

**Schema (po refaktorze 2026-05-07):**

| Kolumna | Typ | Opis |
|---------|-----|------|
| `event_id` | INTEGER PK | auto |
| `run_id` | TEXT NOT NULL | FK → runs |
| `event_index` | INTEGER | porządek w runie |
| `sim_time` | REAL | wallclock symulacji [s] |
| `drone_id` | INTEGER | indeks drona w roju |
| `event_type` | TEXT | trigger / plan_built / no_plan / cooldown_skip / rejoin |
| `mode` | INTEGER | flight mode w momencie zdarzenia (`MODE_*` w SwarmFlightController) |
| `ttc` | REAL | time-to-collision [s]; NaN dla zdarzeń bez analizy zagrożenia |
| `ttc_source` | TEXT | `'oracle_discrete'` lub `'continuous'`; NULL gdy brak ttc |
| `dist_to_threat` | REAL | bieżąca odległość do zagrożenia [m] |
| `threat_x/y/z` | REAL | pozycja zagrożenia w momencie zdarzenia |
| `threat_vx/vy/vz` | REAL | prędkość zagrożenia |
| `rejoin_x/y/z`, `rejoin_arc` | REAL | punkt powrotu (NaN dla event_type=trigger/cooldown) |
| `preferred_axis` | TEXT | `'X'/'Y'/'Z'` lub NULL |
| `fallback_used` | INTEGER (0/1) | czy planner skorzystał z fallbacku |
| `pos_error_at_rejoin`, `vel_error_at_rejoin` | REAL | błędy w punkcie rejoin |
| `planning_wall_time_s` | REAL | czas planowania ewazji |
| `notes` | TEXT | dowolny komentarz; po refaktorze 2026-05-07 NIE zawiera "axis=..." |

**Znane ograniczenia / decyzje:**

1. **`ttc` ≠ funkcja `dist_to_threat`** — w 31% wpisów `ttc=3.4`
   niezależnie od dystansu. To NIE bug: `ttc` z `ttc_source='oracle_discrete'`
   to wynik dyskretyzowanej predykcji po splajnach
   (`SwarmFlightController._oracle_threat_lookahead`,
   `np.arange(0, 4.5, dt)`). Opisuje *przyszłą* kolizję, nie obecną odległość.
2. **`preferred_axis` często NULL** — znany bug: online optimizers
   ([MealpyOptimizer](src/algorithms/avoidance/optimizers/MealpyOptimizer.py),
   [NSGA3OnlineOptimizer](src/algorithms/avoidance/optimizers/NSGA3OnlineOptimizer.py),
   [MSFFOAOnlineOptimizer](src/algorithms/avoidance/optimizers/MSFFOAOnlineOptimizer.py))
   nie wstawiają `axis_chosen` do `OptimizationResult.extra`. Stąd
   `GenericOptimizingAvoidance.py:143` zawsze odczytuje `None`. Tu
   ETL prawidłowo to odzwierciedla NULL-em zamiast magic-stringu
   `"unknown"`. Naprawa po stronie online optimizers — Kamień 2.

**Historia napraw:**

- **2026-05-07**: usunięta kolumna `astar_success` (algorytm A* wycofany;
  pole było zawsze `NOT fallback_used` — semantycznie redundantne).
- **2026-05-07**: dodana kolumna `ttc_source` żeby jednoznacznie odróżnić
  oracle-dyskretyzowane TTC od continuous `dist/closing_speed`.
- **2026-05-07**: dodana kolumna `preferred_axis` (X/Y/Z|NULL) wyodrębniona
  z poprzedniego anti-patternu `notes="axis=..."`. Parser ETL obsługuje
  zarówno nowe CSV (osobna kolumna) jak i stare (parsowanie `notes`).
- **2026-05-07**: `SimulationLogger.log_evasion_event` przyjmuje teraz
  `ttc_source` i `preferred_axis` jako kwargs; backward-compat alias
  `astar_success` → konwersja na `fallback_used = NOT astar_success`.

**Testy regresyjne:**
[tests/analysis/db/test_evasion_events_load.py](tests/analysis/db/test_evasion_events_load.py)
(5 testów: schema shape, legacy CSV compat, new CSV format, invalid axis).

### generated_obstacles

**Cel:** statyczne przeszkody scenariusza per run. Pojedynczy wpis = jedna
przeszkoda (cylinder w forest, box w urban). Pozycja = środek bryły.

**Źródło danych:** `<run_dir>/generated_obstacles.csv` zapisany przez
[`SimulationLogger.log_obstacles`](src/utils/SimulationLogger.py).
- Cylinder CSV (forest): 5 kolumn `[x, y, z, radius, height]`
- Box CSV (urban): 6 kolumn `[x, y, z, length, width, height]`

`SimulationLogger._obstacles_to_dataframe` drop'uje `unused_dim` ze
źródłowej `ObstaclesData.data` (która kanonicznie ma shape (N, 6) — patrz
`generate_obstacles.ObstaclesData.__doc__`) przed zapisem dla cylindra.

**Populator:** [`_load_generated_obstacles`](src/analysis/db/populate_database.py).

**Konsumenci:**
- `run_metrics.obstacle_count` (`COUNT(*)` w `populate_run_metrics`)
- Nie ma jeszcze konsumenta w analizatorze — przyszłe wykorzystanie:
  density-aware difficulty per scenario, cross-env normalizacja.

**Klucze:**
- PK `obstacle_id` (autoincrement)
- UNIQUE `(run_id, obstacle_index)`
- FK `run_id → runs(run_id) ON DELETE CASCADE`

**Schema (po refaktorze 2026-05-07):**

| Kolumna | Typ | Cylinder | Box | Opis |
|---------|-----|----------|-----|------|
| `obstacle_id` | INT PK | - | - | autoinc |
| `run_id` | TEXT | wymagane | wymagane | FK → runs |
| `obstacle_index` | INT | wymagane | wymagane | porządek w runie |
| `x, y, z` | REAL | wymagane | wymagane | środek bryły [m] |
| `shape_type` | TEXT | `'cylinder'` | `'box'` | dyskryminator |
| `radius` | REAL | NOT NULL | **NULL** | promień cylindra [m] |
| `length` | REAL | **NULL** | NOT NULL | dłuższy bok box-a (X) [m] |
| `width` | REAL | **NULL** | NOT NULL | krótszy bok box-a (Y) [m] |
| `height` | REAL | NOT NULL | NOT NULL | wysokość Z [m] |

**CHECK constraint:**
```sql
(shape_type='cylinder' AND radius IS NOT NULL AND length IS NULL AND width IS NULL)
OR
(shape_type='box' AND radius IS NULL AND length IS NOT NULL AND width IS NOT NULL)
```

**Znane ograniczenia / decyzje:**

1. **Backward-compat** dla starych CSV cylindra (z 6. kolumną `unused_dim`):
   populator po prostu jej nie czyta (ignore via `csv.DictReader`).
2. `ObstaclesData.data` kanonicznie ma shape (N, 6). Replay (`ReplayDataStrategy`)
   pad'uje 5-kolumnowy CSV cylindra z `0.0` w 6. kolumnie żeby zachować ten
   kontrakt (in-memory tensor pozostaje niezmieniony, tylko CSV ma 5 kol).

**Historia napraw:**

- **2026-05-07**: KRYTYCZNY BUG — w `_load_generated_obstacles` dla BOX
  zapisywano `length → kolumna 'radius'` i `width → 'unused_dim'`. Stąd
  analizator urban miał **kolumna `radius=15` dla boxów** zamiast NULL,
  fałszując semantykę.
- **2026-05-07**: rozdzielenie semantyki przez `shape_type` + dedykowane
  kolumny `radius/length/width/height` z CHECK constraint. Drop
  `unused_dim`. Backward-compat dla starych CSV.
- **2026-05-07**: pre-existing bug w `ReplayDataStrategy` (oczekiwał kol.
  `unused_dim` w CSV cylindra, którego logger nie pisał) — naprawiony przy
  okazji. Replay pad'uje matrix do (N, 6).

**Testy regresyjne:**
[tests/analysis/db/test_generated_obstacles_load.py](tests/analysis/db/test_generated_obstacles_load.py)
(7 testów: schema shape, cylinder/box load, legacy CSV compat, regresja
na bug `length → radius`, CHECK constraint integrity).

### metric_definitions

**Status: USUNIĘTA 2026-05-07.**

**Cel (historyczny):** self-documenting metadata o metrykach
(`metric_name`, `scope`, `comparable_across_algorithms`, `description`).

**Decyzja:** DROP. Argumenty:
1. **Zero usage** w >6 miesięcy istnienia — żaden populator jej nie zapełniał,
   żaden konsumer nie czytał.
2. Pole `comparable_across_algorithms BOOLEAN` było **fundamentalnie
   niedostateczne** — np. HV jest cross-comparable ALE TYLKO przy tym samym
   `r*` w `(env, n_obj)`. Tej zależności nie da się zakodować boolean'em.
3. Semantyka metryk udokumentowana w 2 lepszych miejscach:
   - Komentarze SQL przy `iteration_metrics` / `run_metrics` w
     [schema.sql](src/analysis/db/schema.sql)
   - Ten plik (`ETL_TABLES.md`) z linkami do papers (Demšar 2006,
     Riquelme 2015, Ishibuchi 2018, etc.)

**Testy regresyjne:**
[tests/analysis/db/test_metric_definitions_removed.py](tests/analysis/db/test_metric_definitions_removed.py)
(1 test: tabela nie istnieje po `initialize_database`).

### online_optimization_tasks

**Cel:** per-trigger summary online avoidance optimization. Pojedynczy
wpis = jedno wywołanie `compute_evasion_plan` per drone per
`trigger_time`. Lifecycle: status (ok/timed_out/failed) z planu →
outcome (rejoined_ok/collided_*/never_rejoined/pending) z wykonania.

**Źródło danych:** `<run_dir>/online_optimization.csv` zapisany przez
`SimulationLogger.log_online_optimization_trigger`. Generowany z
`OnlineOptimizationRecord` w
[GenericOptimizingAvoidance.py:236-252](src/algorithms/avoidance/GenericOptimizingAvoidance.py#L236).

**Populator:** [`populate_online_metrics`](src/analysis/db/populate_online_metrics.py).

**Konsumenci:**
- `vw_run_online_summary` (widok SQL agregujący per run)
- `run_metrics.evasion_event_count`, `total_evasion_triggers`
- ExperimentAnalyzer: bar plots `evasion_event_count` per (alg, env)

**Klucze:**
- PK `(run_id, drone_id, trigger_time)`
- FK `run_id → runs(run_id) ON DELETE CASCADE`

**Schema (po refaktorze 2026-05-07):**

| Kolumna | Typ | Opis |
|---------|-----|------|
| `run_id` | TEXT | FK → runs |
| `drone_id` | INT | indeks drona |
| `trigger_time` | REAL | sim time triggera [s] |
| `algorithm` | TEXT | nazwa avoidance algorytmu (np. "MSFOA", "NSGA3") |
| `status` | TEXT | `ok` / `timed_out` / `failed` |
| `reason` | TEXT | szczegóły status (gdy nie ok) |
| `best_fitness`, `evaluations_completed`, `generations_completed` | metryki optimizera |
| `wallclock_s` | REAL | czas wykonania optimizera |
| `time_budget_s` | REAL | dostępny budżet [s] |
| `chosen_axis` | TEXT | `'X'/'Y'/'Z'` lub NULL (CHECK constraint) |
| `plan_waypoints_json`, `plan_total_duration_s`, `plan_arc_length_m` | metadata planu |
| `outcome` | TEXT | `rejoined_ok` / `collided_*` / `never_rejoined` / `pending` |
| `pos_err_at_rejoin_m`, `vel_err_at_rejoin_mps`, `time_to_rejoin_s` | metryki rejoin |

**CHECK constraint:** `chosen_axis IS NULL OR chosen_axis IN ('X', 'Y', 'Z')`.

**Znane ograniczenia:**

1. **`chosen_axis` często NULL** — ten sam bug co w `evasion_events.preferred_axis`:
   online optimizers ([MealpyOptimizer](src/algorithms/avoidance/optimizers/MealpyOptimizer.py),
   [NSGA3OnlineOptimizer](src/algorithms/avoidance/optimizers/NSGA3OnlineOptimizer.py),
   [MSFFOAOnlineOptimizer](src/algorithms/avoidance/optimizers/MSFFOAOnlineOptimizer.py))
   nie wstawiają `axis_chosen` do `OptimizationResult.extra`. Naprawa po
   stronie online — Kamień 2.
2. **`outcome` może być `'pending'`** dla niezakończonych triggerów (np.
   gdy symulacja kończy się przed BLEND_END dla danego planu). To NIE
   jest bug — outcome jest aktualizowany lifecycle'm w
   `update_online_optimization_outcome`.

**Historia napraw:**

- **2026-05-07**: dodany CHECK constraint na `chosen_axis ∈ ('X','Y','Z')`.
- **2026-05-07**: `GenericOptimizingAvoidance.py:222` — gdy
  `plan.preferred_axis = "unknown"` (znany bug), normalizujemy do `""`
  (puste = NULL po stronie ETL) zamiast magic stringu.
- **2026-05-07**: populator normalizuje legacy "unknown" → NULL na
  load (tolerancja dla istniejących CSV).

**Testy regresyjne:**
[tests/analysis/db/test_online_optimization_tasks_load.py](tests/analysis/db/test_online_optimization_tasks_load.py)
(5 testów: unknown→NULL, ""→NULL, valid X/Y/Z, invalid 'left'/'up'→NULL,
direct INSERT z 'unknown' rejected by CHECK).

### pareto_run_metrics

**Status: USUNIĘTA 2026-05-07.**

**Cel (historyczny):** osobna tabela MOO per-run dla NSGA-III
(decision_mode, selected_solution_index, nondominated_count,
feasible_nondominated_count, hypervolume, igd_plus, epsilon_indicator,
reference_point_json, reference_set_id, indicator_config_json).

**Decyzja: DROP.**

**Argumenty:**

1. **Zero usage** w >6 miesięcy istnienia.
2. **7 z 10 kolumn duplikowało `run_metrics`** (decision_mode,
   selected_solution_index, nondominated_count, feasible_nondominated_count,
   hypervolume, igd_plus, reference_point_json — wszystkie aktywnie
   wypełniane w `run_metrics`).
3. Pozostałe 3 unikalne pola to YAGNI:
   - `epsilon_indicator` — nieliczone (mamy GD/IGD+/HV/spread/spacing/R2)
   - `reference_set_id` — niepotrzebne (1 ref set per (env, n_obj))
   - `indicator_config_json` — niepotrzebne (config globalny, nie per-run)
4. Single source of truth dla MOO per-run = `run_metrics`. Dodanie
   nowych metryk MOO → kolumny w `run_metrics`, nie osobna tabela.

**Testy regresyjne:**
[tests/analysis/db/test_metric_definitions_removed.py](tests/analysis/db/test_metric_definitions_removed.py)
(test `test_pareto_run_metrics_table_removed`).

### reference_pareto_sets

**Cel:** merged Pareto reference set R per `(environment, n_obj)` — punkty
non-dominated zebrane z last-gen feasible fronts WSZYSTKICH runów dla
danego (env, n_obj). Używane jako "ground truth" dla GD/IGD+
(Riquelme et al. 2015 §4).

**Źródło danych:** generated cross-run przez
[`build_reference_pareto_sets`](src/analysis/db/build_reference_pareto.py)
po fazie populate_database. Czyta last-gen feasible-ND fronty z
`optimization_history.h5` każdego runu.

**Populator:** `build_reference_pareto_sets` w
[build_reference_pareto.py](src/analysis/db/build_reference_pareto.py).

**Konsumenci:**
- `backfill_moo_quality_with_reference` — wczytuje R przez
  `load_reference_set` i podaje do `populate_moo_quality(reference_set=R)`
  do liczenia GD/IGD+

**Klucze:**
- PK `(environment, n_obj, point_idx, objective_j)` — long-form storage
- INDEX `(environment, n_obj)` dla szybkiego loadu R

**Schema:**

| Kolumna | Typ | Opis |
|---------|-----|------|
| `environment` | TEXT NOT NULL | nazwa env (forest/urban) |
| `n_obj` | INT NOT NULL | wymiarowość objective space (5) |
| `point_idx` | INT NOT NULL | indeks punktu w R |
| `objective_j` | INT NOT NULL | indeks objective (0..n_obj-1) |
| `value` | REAL NOT NULL | wartość objective_j dla point_idx |

**Charakterystyka wartości** (przykład z `exp_20260506_377919c3`,
32 runów per env):

| env | obj_j (znaczenie) | n_rows | min | max | uwagi |
|-----|-------------------|--------|-----|-----|-------|
| forest | 0 (f1 trajectory_cost) | 1124 | 2495 | 3115 | trasa ~600m |
| forest | 1 (f2 height_cost) | 1124 | 60 | 123 | normalne |
| forest | 2 (f3 threat_cost) | 1124 | **0** | **0** | wszystkie feasible-ND mają threat=0 |
| forest | 3 (f4 turn_cost) | 1124 | 0.015 | 0.164 | smoothness |
| forest | 4 (f5 coord_cost) | 1124 | 0 | 2.4 | sparse outliers |
| urban | 0 (f1) | 4600 | 4402 | 5139 | trasa ~1000m |
| urban | 1 (f2) | 4600 | 118 | 393 | szerszy zakres |
| urban | 2 (f3) | 4600 | 0 | 1.67 | gęste budynki ⇒ niezerowe |
| urban | 3 (f4) | 4600 | 0.014 | 2.03 | duża wariacja |
| urban | 4 (f5) | 4600 | 0 | 3.02 | normalne |

**Znane semantyczne właściwości (NIE bugi):**

1. **forest f3 (j=2) zawsze 0** — cylindry forest są sparse (13 na
   600×60m = 1 cylinder per 2700 m²), feasible trajectories trywialnie
   unikają wszystkich → `threat_cost=0` dla każdego feasible-ND point.
   To **prawidłowe zachowanie**, nie bug.
2. **Spread `value` `[0, 5139]` widoczny przy braku filtra** — wynika
   z mieszania różnych skal: forest f1 ~2500-3000, urban f1 ~4400-5100,
   ALE forest f3 = 0. Per `(env, j)` wartości są wewnętrznie spójne.
3. **|R| zależy od liczby runów + diversity:** dla 32 runów per env
   średnio ~35 ND-points per run forest (|R|=1124) i ~144 urban
   (|R|=4600 = większa różnorodność trajektorii w gęstym urban). To
   konsekwencja eksploracji algorytmów MOO — większy front = lepsza
   eksploracja Pareto.

**Historia napraw:**

- **2026-05-07**: weryfikacja semantyki — bez zmian w kodzie/schema.
  Tylko dokumentacja w tym pliku wyjaśniająca dlaczego forest j=2 ma
  same zera + dlaczego spread `value` jest tak szeroki cross-env.

### reference_points

**Cel:** referencyjny punkt r* (nadir worst-case) per `(env, n_obj)`,
używany do liczenia hypervolume. Stored long-form: jeden wiersz per
komponent r*.

**Źródło danych:** generated cross-run przez
[`build_reference_pareto_sets`](src/analysis/db/build_reference_pareto.py)
PO zbudowaniu R (reference Pareto set). r* = `nadir + ε·(nadir−ideal)`
z merged feasible-ND set (Ishibuchi, Imada, Setoguchi & Nojima 2018,
"How to Specify a Reference Point in Hypervolume Calculation",
Evolutionary Computation 26(3):411–440), domyślnie ε=0.1.

**Populator:** `build_reference_pareto_sets` (drugi krok, po
`reference_pareto_sets`).

**Konsumenci:**
- `backfill_moo_quality_with_reference` — `load_reference_point` →
  pass jako `reference_point=r*` do `populate_moo_quality` dla HV
- HV jest zdefiniowane TYLKO względem r*, więc `reference_points` jest
  niezbędne do interpretacji `iteration_metrics.hypervolume`

**Klucze:**
- PK `(environment, n_obj, objective_j)` — long-form per komponent
- INDEX `(environment, n_obj)` dla szybkiego loadu

**Schema:**

| Kolumna | Typ | Opis |
|---------|-----|------|
| `environment` | TEXT NOT NULL | nazwa env |
| `n_obj` | INT NOT NULL | wymiarowość obj space |
| `objective_j` | INT NOT NULL | indeks komponentu (0..n_obj-1) |
| `value` | REAL NOT NULL | wartość r*[j] |
| `margin` | REAL NOT NULL DEFAULT 0.1 | ε z formuły |
| `method` | TEXT NOT NULL DEFAULT 'nadir_plus_eps_range' | metoda obliczenia r* |

**Cardinality:** `n_envs × n_obj` wierszy łącznie. Dla naszego
eksperymentu: 2 envy × 5 obj = **10 wierszy** — to **kanoniczna,
oczekiwana liczba**, nie bug.

**Przykład wartości** (z `exp_20260506_377919c3`):

| env | r* (5D wektor) |
|-----|----------------|
| forest | `[3177.08, 129.67, 0.10, 0.18, 2.65]` |
| urban | `[5213.60, 420.12, 1.84, 2.23, 3.32]` |

Każdy komponent r*[j] przekracza max(R[:,j]) o `ε·range[j]` (Ishibuchi
2018) — gwarantuje że każdy feasible-ND punkt frontu jest zdominowany
przez r*, warunek konieczny dla `HV > 0`.

**Znane ograniczenia:**

1. **`method='nadir_plus_eps_range'`** to obecnie jedyna implementacja.
   Schema dopuszcza inne metody (np. `'fixed_global'` z hardcoded r*),
   ale aktualny populator nie korzysta. Future-proof.
2. **Edge case: degenerate R (1 punkt)** → `range=0` → fallback w
   `build_reference_pareto.py` używa `max(|nadir|, 1.0)` zamiast
   `range`, żeby uniknąć dzielenia/HV=0. Patrz comment w kodzie.

**Historia napraw:**

- **2026-05-06**: dodana w sesji HV backfill (poprzedni plan, sesja D).
- **2026-05-07**: weryfikacja cardinality — 10 wierszy = 2×5 jest
  oczekiwane. Bez zmian w kodzie/schema.

### run_metrics

**Cel:** agregat per-run wszystkich istotnych metryk: trajektoria offline,
collision/evasion counts, MOO indicators (HV/IGD+/GD/spread/spacing/R2/AUC/
convergence_speed), online inter-UAV safety/energy/smoothness.

**Źródło danych (multi-source):**
- `uav_metrics` — agregat success + path length per drone
- `trajectory_metrics` — total path lengths
- `collisions`, `evasion_events`, `generated_obstacles` — counts
- `optimization_generation_stats` — MOO metrics last-gen
- `iteration_metrics` — gd_final/spread_final/spacing_final/r2_final
- `uav_online_metrics` — inter-UAV safety, energy, smoothness
- **F-vector z h5** (przez `populate_offline_objectives`) — final_objective,
  final_objective_f1_trajectory, final_objective_f2_height_angle,
  total_threat_cost, total_turn_penalty, total_coordination_cost,
  final_objectives_json

**Populator:** [`populate_run_metrics`](src/analysis/db/populate_run_metrics.py)
+ [`populate_offline_objectives`](src/analysis/db/populate_offline_objectives.py).

**Konsumenci:**
- ExperimentAnalyzer — wszystkie tabele/wykresy summary z analysis_output/
- `vw_run_summary`, `vw_seed_summary`, `vw_global_summary` (widoki SQL)
- `MetricExtractor.run_summary` — podstawowy DataFrame dla analizatora

**Klucze:** PK `run_id`, FK → runs.

**Schema (po refaktorze 2026-05-07, 32 kolumn vs poprzednio 42):**

**ZACHOWANE / NAPRAWIONE:**
- `final_objective` ← F[0] z h5 (poprzednio NULL — overwrite bug)
- `total_threat_cost` ← F[2] z h5 (poprzednio NULL)
- `total_turn_penalty` ← F[3] z h5 (poprzednio NULL)
- `total_coordination_cost` ← F[4] z h5 (już wcześniej działało)
- `final_objective_f1_trajectory` ← F[0] z h5
- `final_objective_f2_height_angle` ← F[1] z h5
- `final_objectives_json` ← cały F z h5
- `total_path_length_2d/3d` ← agregat trajectory_metrics
- `collision_count`, `evasion_event_count`, `obstacle_count`
- `nondominated_count`, `hypervolume`, `igd_plus`
- `min/mean_inter_uav_distance_m`, `total_inter_uav_safety_violations`,
  `mean_energy_indicator`, `mean_smoothness_indicator`
- `gd_final`, `spread_final`, `spacing_final`, `r2_final`,
  `convergence_speed_gen`, `auc_best_so_far`
- `success`, `drone_count`, `best_iteration`
- `summary_json`

**DROPPED 2026-05-07:**

*Legacy 8-component costs* (żaden z 4 algorytmów nie produkuje):
- `total_energy_cost`
- `total_smoothness_cost`
- `total_altitude_cost`
- `total_terrain_penalty`
- `total_climb_penalty`
- `total_collision_penalty`

*Never-populated metadata*:
- `decision_mode` — był w NSGA-III config, nikt nie zapisywał do `meta`
- `selected_solution_index` — decision-making runtime nie zapisywał
- `feasible_nondominated_count` — nigdy nieliczone
  (`_load_optimization_history` liczy nondominated_count po wszystkich,
  nie po feasible-only)
- `reference_point_json` — single source of truth = `reference_points` table

**Krytyczne historie napraw:**

1. **2026-05-07 OVERWRITE BUG fix:**
   - **Symptom:** `final_objective`, `total_threat_cost`, `total_turn_penalty`
     zawsze NULL na końcu pipeline'u, mimo że `populate_offline_objectives`
     je UPDATE'uje z F-vectora h5.
   - **Root cause:** `backfill_moo_quality_with_reference` wywołuje
     `populate_run_metrics` po `populate_offline_objectives`, a
     `populate_run_metrics` ON CONFLICT SET nadpisywało te kolumny
     z `excluded.<col>` = NULL (bo `uav_metrics` ich nie zawiera).
   - **Fix:** (a) usunięte te 3 kolumny z `ON CONFLICT SET` w
     populate_run_metrics; (b) dodane wywołanie `populate_offline_objectives`
     w `backfill_moo_quality_with_reference` po populate_run_metrics,
     dla pewności że F-vector jest ostatnim słowem.
   - **Walidacja:** dryrun forest+urban → `final_objective` filled w 100%
     wierszy (poprzednio 100% NULL).

2. **2026-05-07 schema cleanup:** 10 dead columns → DROP. Schema z 42 →
   32 kolumn.

**Uwaga dla `total_inter_uav_safety_violations` zawsze 0:** to NIE
schema bug — to logic bug w `populate_online_safety_metrics`. Naprawa
przy `uav_metrics` (Tabela 11/11).

**Testy regresyjne:** pełen suite `tests/analysis/db/` + dryrun end-to-end.

### run_files

**Cel:** rejestr plików źródłowych runu z metadanymi (size, mtime, row count).
Pojedynczy wpis = jeden plik per (run_id, file_role).

**Źródło danych:** generated przez `_register_run_files` per run jako
pierwszy krok populate_database. Nie wymaga osobnego CSV — czyta
metadata z systemu plików (`Path.stat()`).

**Populator:** [`_register_run_files`](src/analysis/db/populate_database.py).

**Konsumenci:**
- `tests/analysis/db/test_pipeline_integration.py::test_run_files_registers_online_csvs`
  — weryfikuje że oczekiwane CSVs są zarejestrowane
- Future: data provenance audit, broken-file detection w experiment_dir

**Klucze:**
- PK `(run_id, file_role)` — 12 file_role per run = 12 wierszy
- FK `run_id → runs(run_id) ON DELETE CASCADE`

**Schema (po refaktorze 2026-05-07):**

| Kolumna | Typ | Wypełniana | Opis |
|---------|-----|------------|------|
| `run_id` | TEXT | ✅ | FK → runs |
| `file_role` | TEXT | ✅ | klucz semantyczny (np. `trajectories_csv`) |
| `relative_path` | TEXT | ✅ | ścieżka względem run_dir |
| `file_format` | TEXT | ✅ | rozszerzenie (csv/h5/log) |
| `exists_flag` | INT 0/1 | ✅ | czy plik fizycznie istnieje |
| `size_bytes` | INT | ✅ | rozmiar [B] (NULL gdy !exists) |
| `row_count` | INT | ✅ tylko CSV | n_lines − 1 (header) |
| `modified_at` | TEXT (ISO 8601) | ✅ | mtime UTC |

**Plików rejestrowanych** (12 file_role):
`collisions_csv, counted_trajectories_csv, evasion_events_csv,
generated_obstacles_csv, lidar_hits_h5, main_log,
optimization_history_h5, optimization_timings_csv, trajectories_csv,
world_boundaries_csv, online_optimization_csv, convergence_traces_csv`.

**Decyzje projektowe:**

1. **`row_count` tylko dla CSV.** h5 ma własną granulację (multiple
   datasets, każdy z własnym shape) — pojęcie "row count" niejednoznaczne.
   Log ma niespecyficzne linie (komunikaty diagnostyczne, nie dane).
2. **`modified_at` w ISO 8601 UTC** — `datetime.fromtimestamp(..., timezone.utc)
   .isoformat(timespec='seconds')`. Kompatybilne z SQL `datetime()` queries.
3. **`exists_flag=0`** dla nieobecnych plików (np. `lidar_hits.h5` gdy
   `log_lidar_hits=False`) — ETL to znormalizowane.

**Historia napraw:**

- **2026-05-07**: usunięte kolumny `checksum` (SHA256 nigdy nieużyte —
  YAGNI) i `extra_json` (niesprecyzowane future-proof).
- **2026-05-07**: wypełniane `modified_at` (ISO 8601 z mtime) i
  `row_count` (CSV line count − 1). Wcześniej oba były 100% NULL.

**Testy regresyjne:**
[tests/analysis/db/test_run_files_register.py](tests/analysis/db/test_run_files_register.py)
(7 testów: schema cleanup, row_count CSV, modified_at ISO, NULLs dla
nieobecnych/h5/log).

### runs

**Cel:** rejestr runów eksperymentu — jeden wpis = jeden katalog runa
(`<exp_dir>/<run_dir_name>/`). Zawiera identyfikację (`optimizer_algo`,
`avoidance_algo`, `environment`, `seed`, `algorithm_pair`) + status
agregacji ETL.

**Źródło danych:** `populate_database` parsuje `run_dir.name` przez
`parse_run_dir_name` (regex `^opt_(forest|urban)_avoid_seedN$`).

**Populator:** `populate_database` (INSERT/UPDATE per-run, nie osobny
populator).

**Konsumenci:**
- WSZYSTKIE inne populator-y (FK constraint)
- `vw_run_summary`, `vw_seed_summary`, `vw_global_summary`, etc.
- ExperimentAnalyzer.MetricExtractor.run_summary

**Klucze:**
- PK `run_id` (= `run_dir.name`)
- UNIQUE `(optimizer_algo, environment, avoidance_algo, seed)` — gwarantuje
  spójność z `experiments/definitions/<...>.yaml`
- INDEX po `(environment, seed)`, `(optimizer_algo, avoidance_algo)`,
  `(algorithm_pair, seed)` dla szybkich filtrów

**Schema (po refaktorze 2026-05-07, 12 kolumn vs poprzednio 16):**

**Identyfikacja:**
- `run_id`, `run_dir_name`, `source_path`
- `optimizer_algo`, `avoidance_algo`, `environment`, `seed`, `algorithm_pair`

**Lifecycle agregacji:**
- `aggregation_status` ∈ {'discovered', 'aggregated', 'failed', 'partial'}
- `aggregation_error` (TEXT) — komunikat błędu gdy `status='failed'`
- `discovered_at` — timestamp INSERT
- `aggregated_at` — timestamp completion (NULL gdy status != 'aggregated')

**DROPPED 2026-05-07:**

- `decision_mode` — duplikat (też DROPPED z run_metrics; pole z config
  Hydra, nie metadata runa)
- `notes` — YAGNI (dowolny komentarz, nigdy nieużywany)
- `run_config_json` — Hydra config jest już w `<run_dir>/.hydra/config.yaml`,
  duplikacja niepotrzebna

**Lifecycle:**

```
populate_database per-run:
  1. INSERT runs (status='discovered', aggregation_error=NULL)
  2. try:
        ... wszystkie populatory ...
        UPDATE runs status='aggregated', aggregated_at=NOW
     except Exception as e:
        UPDATE runs status='failed', aggregation_error=str(e)
        # KONTYNUUJ z kolejnymi runami (pojedynczy fail nie blokuje)
```

**Historia napraw:**

- **2026-05-07**: usunięte 3 never-populated kolumny (decision_mode, notes,
  run_config_json). Schema 16 → 12 kolumn.
- **2026-05-07**: dodany try/except w populate_database — błędy per-run
  są teraz zapisywane do `aggregation_error` zamiast wywalać cały ETL.
  Pojedynczy fail nie blokuje agregacji innych runów.

### uav_metrics

**Cel:** metryki agregowane per-drone w runie. Pojedynczy wpis = jeden
dron (typowo 5 wpisów per run dla swarm of 5).

**Źródło danych:** derived z `trajectory_metrics`, `trajectory_samples`,
`counted_trajectory_points`, `collisions`, `evasion_events`.

**Populator:** [`populate_uav_metrics`](src/analysis/db/populate_uav_metrics.py).

**Konsumenci:** `populate_run_metrics` (agregat per-run z UAV).

**Klucze:** PK `(run_id, uav_id)`, FK → runs.

**Schema (po refaktorze 2026-05-07, 8 kolumn vs poprzednio 17):**

- `run_id`, `uav_id` (PK)
- `success` — czy drone osiągnął cel bez kolizji
- `path_length_2d`, `path_length_3d` — długość trajektorii
- `collision_count`, `evasion_event_count` — counts per drone
- `extra_json` — JSON z `actual_*`/`planned_*` altitude statistics

**DROPPED 2026-05-07** (9 kolumn — wszystkie zawsze NULL w populatorze):

- `final_objective` — F-vector w h5 jest *zbiorczy per swarm*, nie
  rozbity per drone, więc per-UAV final_objective nie istnieje
- `energy_cost`, `smoothness_cost`, `threat_cost`, `altitude_cost`,
  `terrain_penalty`, `turn_penalty`, `climb_penalty`, `collision_penalty`
  — legacy 8-component objective, żaden z 4 algorytmów ich per-UAV nie
  produkuje. Aktualny `VectorizedEvaluator` ma 5-obj F-vector aplikowane
  do całego swarmu

**Pokrewna: `uav_online_metrics`** — bug naprawiony 2026-05-07:

- **Symptom:** `total_inter_uav_safety_violations` zawsze 0 w `run_metrics`.
- **Root cause:** `DEFAULT_INTER_UAV_SAFETY_THRESHOLD_M = 1.0 m` — wartość
  ustalona historycznie dla `collision_radius=0.4` (wzór `0.4 × 2.5 ≈ 1.0`).
  Aktualne configi mają `collision_radius=2.0`, więc realistyczny próg to
  `2 × collision_radius = 4.0 m`. Stary próg 1.0 był ZAWSZE niższy niż
  obserwowany min_dist (~1.9 m+), stąd 100% wpisów = 0 violations.
- **Fix:** `DEFAULT_INTER_UAV_SAFETY_THRESHOLD_M = 4.0`.
- **Walidacja:** dryrun forest+urban → 21 violations detected
  (forest=10, urban=11) zamiast 0.

**Historia napraw:**

- **2026-05-07**: usunięte 9 dead cost columns. Schema 17 → 8 kolumn.
- **2026-05-07**: naprawiony bug w `populate_online_safety_metrics` —
  threshold 1.0 → 4.0 m. Bug objawiał się jako `total_inter_uav_safety_violations`
  zawsze 0 w `run_metrics`.


