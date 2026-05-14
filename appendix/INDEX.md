# INDEX — Załącznik cyfrowy do pracy magisterskiej

**Praca:** *Porównanie bio-inspirowanych metaheurystyk (MSFOA, OOA, SSA) z klasycznym NSGA-III w problemie planowania trajektorii roju UAV*

Plik źródłowy: [praca/src/Praca magisterska-final.md](../praca/src/Praca magisterska-final.md)
Eksperyment źródłowy: `results/exp_20260508_f3f718f8_bio_inspired_benchmark/` (4 algorytmy × 2 środowiska × 30 seeds × N_avoidance = 240 runów).
Commit referencyjny: `cdca9524f58f54b5da720e80fcbd239595f4ea16` (branch `main`).

Niniejszy załącznik zawiera **wyłącznie** artefakty cytowane w pracy. Pełne wyniki pipeline'u analizy (266 plików w `tables/`, 6 podkatalogów `plots/`) znajdują się w repozytorium źródłowym pod ścieżką podaną w [CITATION.md](CITATION.md).

---

## Sekcja A — Metryki wykorzystane w pracy

Tabela zbiorcza 13 metryk cytowanych w rozdziale 3 pracy. Direction: ↓ = niżej lepiej, ↑ = wyżej lepiej.

| # | Nazwa w pracy | Kolumna DB (`run_metrics` / inna) | Rozdz. | Dir. | Surowy plik źródłowy | Populator ETL |
|---|---|---|---|---|---|---|
| M1 | Bezpieczeństwo trajektorii (F3+F5) | `trajectory_safety_f3_f5` *(derived: F[2]+F[4])* | 3.2.1.1 | ↓ | `optimization_history/optimization_history.h5:objectives_matrix` | [populate_offline_objectives.py](../src/analysis/db/populate_offline_objectives.py) + [metric_extractor.py](../src/analysis/analyzer/metric_extractor.py) |
| M2 | Długość trajektorii (F1) | `final_objective_f1_trajectory` *(F[0])* | 3.2.1.1 | ↓ | `optimization_history/optimization_history.h5:objectives_matrix` | [populate_offline_objectives.py](../src/analysis/db/populate_offline_objectives.py) |
| M3 | Gładkość trajektorii (F2+F4) | `trajectory_smoothness_f2_f4` *(derived: F[1]+F[3])* | 3.2.1.1 | ↓ | `optimization_history/optimization_history.h5:objectives_matrix` | [populate_offline_objectives.py](../src/analysis/db/populate_offline_objectives.py) + [metric_extractor.py:146-157](../src/analysis/analyzer/metric_extractor.py#L146-L157) |
| M4 | Spójność roju | `swarm_cohesion_deviation` *(derived: `\|min_inter_uav − 5\| + \|max_inter_uav − 5\|`)* | 3.2.1.1 | ↓ | `trajectories.csv` → `uav_online_metrics` | [populate_online_safety_metrics.py](../src/analysis/db/populate_online_safety_metrics.py) + [metric_extractor.py:146-157](../src/analysis/analyzer/metric_extractor.py#L146-L157) |
| M5 | Odsetek trajektorii kolizyjnych | `is_offline_failure` *(`tracking_phase_collisions > 0`)* | 3.2.1.1 | ↓ | `collisions.csv` + `evasion_events.csv` | [populate_run_metrics.py:52-75](../src/analysis/db/populate_run_metrics.py#L52-L75) |
| M6 | Długość krzywej unikowej | `mean_evasion_arc_length_m` | 3.2.1.2 | ↓ | `online_optimization.csv:plan_arc_length_m` | [populate_online_metrics.py](../src/analysis/db/populate_online_metrics.py) + [populate_run_metrics.py](../src/analysis/db/populate_run_metrics.py) |
| M7 | Skuteczność powrotu na trajektorię nominalną | `rejoin_quality` *(TOPSIS composite z 3 błędów)* | 3.2.1.2 | ↓ | `online_optimization.csv:{pos_err_at_rejoin_m, vel_err_at_rejoin_mps, time_to_rejoin_s}` | [populate_rejoin_quality.py](../src/analysis/db/populate_rejoin_quality.py) |
| M8 | Stosunek udanych uników | `1 − is_online_failure` *(`evasion_phase_collisions == 0`)* | 3.2.1.2 | ↑ | `collisions.csv` + `evasion_events.csv` | [populate_run_metrics.py:52-75](../src/analysis/db/populate_run_metrics.py#L52-L75) |
| M9 | Wartość optymalizacji offline | `final_objective` *(weighted normalized sum, Hwang & Yoon 1981)* | 3.2.2.1 | ↓ | `optimization_history.h5` + `offline_objective_normalization` | [populate_final_objective_aggregated.py](../src/analysis/db/populate_final_objective_aggregated.py) |
| M10 | Tempo optymalizacji offline | `best_so_far` per iteration | 3.2.2.1 | ↓ | `optimization_history/optimization_history.h5:objectives_matrix` | [populate_iteration_metrics.py](../src/analysis/db/populate_iteration_metrics.py) |
| M11 | Wartość optymalizacji online | `mean_online_best_fitness` | 3.2.2.2 | ↓ | `online_optimization.csv:best_fitness` | [populate_run_metrics.py:152-154](../src/analysis/db/populate_run_metrics.py#L152-L154) |
| M12 | Tempo optymalizacji online | `best_fitness` per generation | 3.2.2.2 | ↓ | `convergence_traces.csv` → `online_convergence_traces` | [populate_online_metrics.py](../src/analysis/db/populate_online_metrics.py) |
| M13 | SP1 online | `online_sp1 = avg_evals_ok / success_rate` *(Auger & Hansen 2005)* | 3.2.2.2 | ↓ | `online_optimization.csv` | [populate_run_metrics.py:202-207](../src/analysis/db/populate_run_metrics.py#L202-L207) |

### Definicje i odniesienia literaturowe

- **M1 — Bezpieczeństwo trajektorii (F3+F5)** = suma kosztu zagrożeń (`total_threat_cost` = F[2]) i kosztu koordynacji/zachowania separacji (`total_coordination_cost` = F[4]). Sformułowanie wieloobiektowe w [src/algorithms/abstraction/trajectory/objective_constrains.py](../src/algorithms/abstraction/trajectory/objective_constrains.py).
- **M2 — Długość trajektorii (F1)** = `final_objective_f1_trajectory` (F[0] z `objectives_matrix` h5), best feasible solution z ostatniej generacji.
- **M3 — Gładkość trajektorii (F2+F4)** = `final_objective_f2_height_angle` (F[1], kara za zmiany wysokości i kąta) + `total_turn_penalty` (F[3], kara za zakręty).
- **M4 — Spójność roju** = odchylenie od docelowego dystansu między dronami 5 m. Mierzone jako `|min_inter_uav − 5| + |max_inter_uav − 5|`, gdzie min/max są agregatami po wszystkich krokach symulacji. Niskie wartości → ścisła formacja blisko 5 m spacing; wysokie → kompresja lub rozpraszanie.
- **M5 — Odsetek trajektorii kolizyjnych** = procent runów z `tracking_phase_collisions > 0` (kolizja w fazie wykonywania planu offline, *przed* aktywacją online avoidance). Definicja per [failure_success_methodology.md §1](../reports/failure_success_methodology.md).
- **M6 — Długość krzywej unikowej** = średnia długość łuku planu unikowego (B-spline) wygenerowanego przez optymalizator online w trakcie reaktywnego unikania. Proxy dla zużycia energii podczas manewru.
- **M7 — Rejoin quality** = TOPSIS-based composite z trzech błędów powrotu: pozycyjnego (m), prędkościowego (m/s), czasowego (s), normalizowanych per środowisko. Niższe wartości → szybszy i dokładniejszy powrót na trajektorię nominalną. Definicja [src/analysis/db/populate_rejoin_quality.py](../src/analysis/db/populate_rejoin_quality.py).
- **M8 — Stosunek udanych uników** = proporcja runów z `evasion_phase_collisions == 0` (system avoidance zapobiegł kolizji). Przedział ufności Wilsona (Wilson 1927; Newcombe 1998).
- **M9 — Wartość optymalizacji offline** = `Σ_{i=1..5} w_i · F_best[i] / F_ref_env[i]`, gdzie F_ref_env to per-environment median F_best (Hwang & Yoon 1981 — weighted-sum MCDM). Wagi `w_i` z `.hydra/config.yaml`; canonical fallback dla NSGA-III: `[0.05, 0.5, 0.8, 1.0, 0.25]`.
- **M10 — Tempo optymalizacji offline** = krzywa zbieżności best-so-far per generacja, agregowana medianą ±IQR po seedach.
- **M11 — Wartość optymalizacji online** = średnia z `best_fitness` po wszystkich wywołaniach optymalizatora online w danym runie (w warunkach budżetu czasowego ~0.5 s).
- **M12 — Tempo optymalizacji online** = krzywa zbieżności best_fitness per generacja w online avoidance, agregowana medianą ±IQR po trygerach.
- **M13 — SP1 (Success Performance 1)** = `evals_when_ok / online_success_rate` (Auger & Hansen 2005 — *Performance Evaluation of an Advanced Local Search Evolutionary Algorithm*). Łączy efektywność (liczba ewaluacji) z prawdopodobieństwem sukcesu (proporcja `status='ok'`).

### Surowe pliki per-run wykorzystywane przez te metryki

Z katalogu `results/<run_id>/`, do appendiksu **kopiujemy** wyłącznie:

| Plik | Format | Zasila metryki | Rozmiar (typ.) |
|---|---|---|---|
| `evasion_events.csv` | CSV | M5, M8 (klasyfikacja faz kolizji) | < 100 KB |
| `online_optimization.csv` | CSV | M6, M7, M11, M13 | < 500 KB |
| `convergence_traces.csv` | CSV | M12 | 100 KB – 5 MB |
| `optimization_history/optimization_history.h5` | HDF5 | M1, M2, M3, M9, M10 | 1 MB – 50 MB |
| `.hydra/config.yaml` | YAML | wagi M9 + seeds + parametry algorytmu | < 20 KB |

**Pomijamy** (zbyt duże, dane redundantne z agregatami w DB):
- `trajectories.csv` (40k+ wierszy/run) — surowe samples z PyBullet; M4 jest już zagregowane w `uav_online_metrics` (subset w `A_metrics/`).
- `lidar_hits.h5` — surowe odczyty LiDAR; nie cytowane bezpośrednio w pracy.
- `collisions.csv` — wszystkie kolizje są już zliczone w `run_metrics.tracking_phase_collisions` i `evasion_phase_collisions`.
- `counted_trajectories.csv`, `world_boundaries.csv`, `generated_obstacles.csv`, `optimization_timings.csv` — metadane środowiska/runu, nie cytowane bezpośrednio.

---

## Sekcja B — Mapowanie Wykres N → plik

18 wykresów cytowanych w spisie pracy (linie 1506–1525 [praca/src/Praca magisterska-final.md](../praca/src/Praca magisterska-final.md)). Wszystkie istnieją w `results/exp_20260508_f3f718f8_bio_inspired_benchmark/analysis_output/plots/` w wariantach `.pdf` (do druku) i `.png` (do raportu cyfrowego).

| # | Rozdz. | Opis wykresu | Plik źródłowy (PDF + PNG) | Metryka | Generator |
|---|---|---|---|---|---|
| W1 | 3.2.1.1 | Bar: odsetek trajektorii kolizyjnych, forest | `plots/bar/bar_forest_failure_rate_offline.{pdf,png}` | M5 | [bar_plots.py](../src/analysis/analyzer/plots/bar_plots.py) |
| W2 | 3.2.1.1 | Bar: odsetek trajektorii kolizyjnych, urban | `plots/bar/bar_urban_failure_rate_offline.{pdf,png}` | M5 | [bar_plots.py](../src/analysis/analyzer/plots/bar_plots.py) |
| W3 | 3.2.1.1 | Box: długość trajektorii F1, forest | `plots/boxplots/boxplot_forest_trajectory_length_f1.{pdf,png}` | M2 | [box_plots.py](../src/analysis/analyzer/plots/box_plots.py) |
| W4 | 3.2.1.1 | Box: długość trajektorii F1, urban | `plots/boxplots/boxplot_urban_trajectory_length_f1.{pdf,png}` | M2 | [box_plots.py](../src/analysis/analyzer/plots/box_plots.py) |
| W5 | 3.2.1.1 | Box: gładkość F2+F4, forest | `plots/boxplots/boxplot_forest_trajectory_smoothness_f2_f4.{pdf,png}` | M3 | [box_plots.py](../src/analysis/analyzer/plots/box_plots.py) |
| W6 | 3.2.1.1 | Box: gładkość F2+F4, urban | `plots/boxplots/boxplot_urban_trajectory_smoothness_f2_f4.{pdf,png}` | M3 | [box_plots.py](../src/analysis/analyzer/plots/box_plots.py) |
| W7 | 3.2.1.1 | Box: spójność roju, forest | `plots/boxplots/boxplot_forest_swarm_cohesion_deviation.{pdf,png}` | M4 | [box_plots.py](../src/analysis/analyzer/plots/box_plots.py) |
| W8 | 3.2.1.1 | Box: spójność roju, urban | `plots/boxplots/boxplot_urban_swarm_cohesion_deviation.{pdf,png}` | M4 | [box_plots.py](../src/analysis/analyzer/plots/box_plots.py) |
| W9 | 3.2.1.2 | Box: długość krzywej unikowej, forest | `plots/boxplots/boxplot_forest_mean_evasion_arc_length_m.{pdf,png}` | M6 | [box_plots.py](../src/analysis/analyzer/plots/box_plots.py) |
| W10 | 3.2.1.2 | Box: długość krzywej unikowej, urban | `plots/boxplots/boxplot_urban_mean_evasion_arc_length_m.{pdf,png}` | M6 | [box_plots.py](../src/analysis/analyzer/plots/box_plots.py) |
| W11 | 3.2.1.2 | Box: skuteczność powrotu na trajektorię, forest | `plots/boxplots/boxplot_forest_rejoin_quality.{pdf,png}` | M7 | [box_plots.py](../src/analysis/analyzer/plots/box_plots.py) |
| W12 | 3.2.1.2 | Box: skuteczność powrotu na trajektorię, urban | `plots/boxplots/boxplot_urban_rejoin_quality.{pdf,png}` | M7 | [box_plots.py](../src/analysis/analyzer/plots/box_plots.py) |
| W13 | 3.2.2.1 | Box: wartość optymalizacji, forest | `plots/boxplots/boxplot_forest_final_objective.{pdf,png}` | M9 | [box_plots.py](../src/analysis/analyzer/plots/box_plots.py) |
| W14 | 3.2.2.1 | Box: wartość optymalizacji, urban | `plots/boxplots/boxplot_urban_final_objective.{pdf,png}` | M9 | [box_plots.py](../src/analysis/analyzer/plots/box_plots.py) |
| W15 | 3.2.2.1 | Line: tempo optymalizacji offline, forest | `plots/convergence/convergence_forest_best_so_far.{pdf,png}` | M10 | [convergence_plots.py](../src/analysis/analyzer/plots/convergence_plots.py) |
| W16 | 3.2.2.1 | Line: tempo optymalizacji offline, urban | `plots/convergence/convergence_urban_best_so_far.{pdf,png}` | M10 | [convergence_plots.py](../src/analysis/analyzer/plots/convergence_plots.py) |
| W17 | 3.2.2.2 | Line: tempo optymalizacji online, forest | `plots/convergence/online_convergence_forest.{pdf,png}` | M12 | [convergence_plots.py](../src/analysis/analyzer/plots/convergence_plots.py) |
| W18 | 3.2.2.2 | Line: tempo optymalizacji online, urban | `plots/convergence/online_convergence_urban.{pdf,png}` | M12 | [convergence_plots.py](../src/analysis/analyzer/plots/convergence_plots.py) |

**Uwaga merytoryczna:** w obecnej wersji [praca/src/Praca magisterska-final.md](../praca/src/Praca magisterska-final.md) (linie 1248, 1252) podpisy Wykres 11 i 12 zawierają błąd („długości krzywej unikowej" zamiast „skuteczności powrotu na trajektorię nominalną") — odnotowane w [praca/src/spisy-rysunkow-tabel-wykresow.md](../praca/src/spisy-rysunkow-tabel-wykresow.md). W tabeli powyżej zastosowano poprawną wersję merytoryczną (wariant B).

---

## Sekcja C — Mapowanie Tabela N → plik(i)

21 tabel cytowanych w pracy. **Trzy klasy** tabel:

1. **Statystyki opisowe** (`summary_{metric}.csv` + `.tex`): n, mean, std, min, max, median, q25, q75, low_power_warning. Cross-environment, jeden wiersz per (env, optimizer).
2. **Testy Friedmana + A12** (`{env}_friedman_{metric}.csv` + `{env}_a12_{metric}.csv`): per-environment.
3. **Testy Wilsona** (`failure_rate_*.csv` + `.tex`): proporcje binomialne z 95% CI.

Bonus dla recenzenta: PNG w `appendix/B_statistical_tests/thesis_stat_tables/` (z [praca/chapter-3/stat_tables/](../praca/chapter-3/stat_tables/)) to **gotowe panele** wykorzystane w pracy magisterskiej (Friedman + A12 + side-by-side forest/urban), generowane przez [scripts/generate_thesis_stat_tables.py](../scripts/generate_thesis_stat_tables.py).

| # | Rozdz. | Opis | Plik(i) źródłowy(e) | Metryka | Thesis PNG (panel) |
|---|---|---|---|---|---|
| T1 | 2.3 | Budżet obliczeniowy offline (pop_size, n_gen, K=11, D=5 → budget) | [B_statistical_tests/budget_table.md](B_statistical_tests/budget_table.md) | — | — |
| T2 | 3.2.1.1 | Statystyki — bezpieczeństwo trajektorii (F3+F5) | `tables/summary_trajectory_safety_f3_f5.{csv,tex}` | M1 | `stat_3211_trajectory_safety.png` |
| T3 | 3.2.1.1 | Friedman + A12 — bezpieczeństwo trajektorii | `tables/{forest,urban}_friedman_trajectory_safety_f3_f5.csv` + `_a12_` warianty | M1 | `stat_3211_trajectory_safety.png` |
| T4 | 3.2.1.1 | Statystyki — długość trajektorii (F1) | `tables/summary_trajectory_length_f1.{csv,tex}` | M2 | `stat_3211_trajectory_length.png` |
| T5 | 3.2.1.1 | Friedman + A12 — długość trajektorii | `tables/{forest,urban}_friedman_trajectory_length_f1.csv` + `_a12_` warianty | M2 | `stat_3211_trajectory_length.png` |
| T6 | 3.2.1.1 | Statystyki — gładkość (F2+F4) | `tables/summary_trajectory_smoothness_f2_f4.{csv,tex}` | M3 | `stat_3211_trajectory_smoothness.png` |
| T7 | 3.2.1.1 | Friedman + A12 — gładkość | `tables/{forest,urban}_friedman_trajectory_smoothness_f2_f4.csv` + `_a12_` warianty | M3 | `stat_3211_trajectory_smoothness.png` |
| T8 | 3.2.1.1 | Statystyki — spójność roju | `tables/summary_swarm_cohesion_deviation.{csv,tex}` | M4 | `stat_3211_swarm_cohesion.png` |
| T9 | 3.2.1.1 | Friedman + A12 — spójność roju | `tables/{forest,urban}_friedman_swarm_cohesion_deviation.csv` + `_a12_` warianty | M4 | `stat_3211_swarm_cohesion.png` |
| T10 | 3.2.1.2 | Statystyki — długość krzywej unikowej | `tables/summary_mean_evasion_arc_length_m.{csv,tex}` | M6 | `stat_3212_evasion_arc_length.png` |
| T11 | 3.2.1.2 | Friedman + A12 — długość krzywej unikowej | `tables/{forest,urban}_friedman_mean_evasion_arc_length_m.csv` + `_a12_` warianty | M6 | `stat_3212_evasion_arc_length.png` |
| T12 | 3.2.1.2 | Statystyki — skuteczność powrotu na trajektorię | `tables/summary_rejoin_quality.{csv,tex}` | M7 | `stat_3212_rejoin_quality.png` |
| T13 | 3.2.1.2 | Friedman + A12 — skuteczność powrotu | `tables/{forest,urban}_friedman_rejoin_quality.csv` + `_a12_` warianty | M7 | `stat_3212_rejoin_quality.png` |
| T14 | 3.2.1.2 | Statystyki — stosunek udanych uników | [B_statistical_tests/wilson/evasion_success_rate.csv](B_statistical_tests/wilson/evasion_success_rate.csv) (derived: `1 − failure_rate_online`) | M8 | `stat_3212_online_safety.png` |
| T15 | 3.2.1.2 | Wilson 95% CI — bezpieczeństwo uników | `tables/failure_rate_online.{csv,tex}` (+ `failure_rate_offline` jako kontekst) | M8 | `stat_3212_online_safety.png` |
| T16 | 3.2.2.1 | Statystyki — wartość optymalizacji offline | `tables/summary_final_objective.{csv,tex}` | M9 | `stat_3221_final_objective.png` |
| T17 | 3.2.2.1 | Friedman + A12 — skuteczność optymalizacji | `tables/{forest,urban}_friedman_final_objective.csv` + `_a12_` warianty | M9 | `stat_3221_final_objective.png` |
| T18 | 3.2.2.1 | Friedman + A12 — szybkość optymalizacji (AUC) | `tables/{forest,urban}_friedman_auc_best_so_far.csv` + `_a12_` warianty | M10 | `stat_3221_auc_best_so_far.png` |
| T19 | 3.2.2.2 | Statystyki — wartość optymalizacji online | `tables/summary_mean_online_best_fitness.{csv,tex}` | M11 | `stat_3222_online_best_fitness.png` |
| T20 | 3.2.2.2 | Friedman + A12 — optymalizacja online | `tables/{forest,urban}_friedman_mean_online_best_fitness.csv` + `_a12_` warianty | M11 | `stat_3222_online_best_fitness.png` |
| T21 | 3.2.2.2 | Friedman + A12 — SP1 online | `tables/{forest,urban}_friedman_online_sp1.csv` + `_a12_` warianty | M13 | `stat_3222_online_sp1.png` |

---

## Sekcja D — Reprodukcja artefaktów pochodnych

Następujące artefakty wymagają jednorazowego eksportu/agregacji w fazie kopiowania:

| # | Cel | Lokalizacja docelowa | Komenda/skrypt |
|---|---|---|---|
| D1 | Subset `run_metrics` (13 kolumn z thesis × 240 wierszy) | `A_metrics/run_metrics_subset.csv` | `sqlite3 analysis.db "SELECT run_id, optimizer, environment, avoidance, seed, final_objective_f1_trajectory, total_threat_cost, total_turn_penalty, total_coordination_cost, final_objective, mean_evasion_arc_length_m, rejoin_quality, mean_online_best_fitness, online_sp1, tracking_phase_collisions, evasion_phase_collisions, min_inter_uav_distance_m, max_inter_uav_distance_m FROM vw_run_summary;"` |
| D2 | Subset `iteration_metrics` (best_so_far per gen) | `A_metrics/iteration_metrics_subset.csv` | `sqlite3 analysis.db "SELECT run_id, iteration, best_so_far, hypervolume, feasible_ratio FROM iteration_metrics;"` |
| D3 | Subset `online_convergence_traces` | `A_metrics/online_convergence_subset.csv` | `sqlite3 analysis.db "SELECT run_id, drone_id, trigger_time, generation, best_fitness FROM online_convergence_traces;"` |
| D4 | Run manifest (240 wierszy z statusem) | `H_run_manifest.csv` | `sqlite3 analysis.db "SELECT run_id, optimizer, environment, avoidance, seed, aggregation_status FROM runs;"` |
| D5 | Conda env snapshot | `F_environment/conda_env_export.yaml` | `conda env export -n drone-swarm-env > F_environment/conda_env_export.yaml` |
| D6 | Tabela 1 — budżet obliczeniowy | `B_statistical_tests/budget_table.{csv,md}` | Ręcznie z [configs/optimizer/](../configs/optimizer/) — 4 wiersze × 4 kolumny |
| D7 | Tabela 14 — evasion success rate | `B_statistical_tests/wilson/evasion_success_rate.csv` | Derived: `1 − failure_rate_online` |

---

## Sekcja E — Struktura katalogu `appendix/`

```
appendix/
├── INDEX.md                          # ← ten plik
├── CITATION.md                       # GitHub URL + commit hash
├── README.md                         # Instrukcja dla recenzenta
├── A_metrics/                        # Zagregowane CSV per metryka
├── B_statistical_tests/              # Friedman, A12, Wilson + thesis_stat_tables PNG
├── C_plots/                          # 18 wykresów (PDF + PNG)
├── D_database_schema/                # Schema SQL + ERD
├── E_configs/                        # Hydra configs (definicja eksperymentu)
├── F_environment/                    # environment.yaml + conda env snapshot
├── G_per_run_seeds/                  # 240 katalogów per-run (subset plików)
└── H_run_manifest.csv                # Manifest z seedami (240 wierszy)
```

Każdy podkatalog zawiera własny `README.md` z: celem, listą plików, ścieżką źródłową, komendą do skopiowania/wygenerowania.

---

## Sekcja F — Bibliografia metodologiczna (cytowana w pracy)

Pełne odniesienia dla metryk i testów statystycznych:

- **Auger, A. & Hansen, N.** (2005). Performance Evaluation of an Advanced Local Search Evolutionary Algorithm. *IEEE CEC 2005*. *(SP1 metric)*
- **Demšar, J.** (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. *JMLR* 7:1–30. *(Friedman + Nemenyi)*
- **Hansen, N. & Jaszkiewicz, A.** (1998). Evaluating the quality of approximations to the non-dominated set. *Tech. Rep. IMM-REP-1998-7, Technical University of Denmark*. *(R2 indicator)*
- **Hwang, C.L. & Yoon, K.** (1981). *Multiple Attribute Decision Making: Methods and Applications.* Springer §4.2. *(weighted normalized sum, TOPSIS)*
- **Ishibuchi, H., Imada, R., Setoguchi, Y. & Nojima, Y.** (2018). How to Specify a Reference Point in Hypervolume Calculation for Fair Performance Comparison. *Evolutionary Computation* 26(3):411–440.
- **Ishibuchi, H., Masuda, H., Tanigaki, Y. & Nojima, Y.** (2015). Modified Distance Calculation in Generational Distance and Inverted Generational Distance. *EMO 2015*. *(IGD+)*
- **López-Ibáñez, M. et al.** (2021). Reproducibility in evolutionary computation. *ACM TELO* 1(4). *(rationale dla zawartości załącznika)*
- **Newcombe, R.G.** (1998). Two-sided confidence intervals for the single proportion. *Statistics in Medicine* 17(8):857–872. *(Wilson CI)*
- **Riquelme, N., Von Lücken, C. & Barán, B.** (2015). Performance metrics in multi-objective optimization. *CLEI Electronic Journal* 18(1). *(GD, HV normalization)*
- **Schott, J.R.** (1995). Fault Tolerant Design Using Single and Multicriteria Genetic Algorithm Optimization. *MSc Thesis, MIT*. *(Spacing)*
- **Vargha, A. & Delaney, H.D.** (2000). A critique and improvement of the CL common language effect size statistics. *J. Educ. Behav. Stat.* 25(2):101–132. *(A12)*
- **Wilson, E.B.** (1927). Probable inference, the law of succession, and statistical inference. *JASA* 22(158):209–212.
