# D_database_schema — Schemat bazy `analysis.db`

## Cel

Dokumentacja schematu SQLite, którą wykorzystuje pipeline ETL do agregacji wyników eksperymentu. Sama `analysis.db` (861 MB) **nie jest dołączona** do załącznika — recenzent może ją zrekonstruować z surowych CSV w [G_per_run_seeds/](../G_per_run_seeds/) poprzez `ExperimentAggregator`.

## Zawartość

| Plik | Cel | Źródło |
|---|---|---|
| [`schema.sql`](schema.sql) | Pełna definicja SQL DDL (21 tabel, 5 widoków, 22 indeksy) | [`src/analysis/db/schema.sql`](../../src/analysis/db/schema.sql) |
| [`ERD.md`](ERD.md) | Diagram Mermaid relacji między 11 tabelami cytowanymi w pracy | napisany ręcznie |
| [`views.md`](views.md) | Dokumentacja 5 widoków analitycznych (`vw_run_summary`, `vw_seed_summary`, `vw_global_summary`, `vw_run_online_summary`, `vw_algo_cross_sim_comparison`) | napisany ręcznie |

## Tabele relevantne dla pracy (subset z 17)

Tylko tabele, których kolumny są cytowane w pracy lub są wymagane do zrozumienia struktury danych:

| Tabela | Klucz | Zasilane przez | Cytowane metryki |
|---|---|---|---|
| `runs` | `run_id` | dir basename | metadata (optimizer, environment, avoidance, seed) |
| `run_metrics` | `run_id` | populate_run_metrics + populate_offline_objectives | M1, M2, M3, M5, M6, M7, M8, M9, M11, M13 |
| `iteration_metrics` | `(run_id, iteration)` | populate_iteration_metrics | M10 (`best_so_far`) |
| `uav_online_metrics` | `(run_id, uav_id)` | populate_online_safety_metrics | M4 components |
| `online_optimization_tasks` | `(run_id, drone_id, trigger_time)` | populate_online_metrics | M6, M7 raw |
| `online_convergence_traces` | `(run_id, drone_id, trigger_time, gen)` | populate_online_metrics | M12 |
| `collisions` | `(run_id, ...)` | populate_database | M5, M8 |
| `evasion_events` | `(run_id, event_index)` | populate_database | M5, M8 (faza klasyfikacja) |
| `offline_objective_normalization` | `(environment, f_idx)` | populate_final_objective_aggregated | M9 (F_ref_env median) |
| `reference_pareto_sets` | `(env, n_obj, point_idx, obj_j)` | build_reference_pareto_sets | konteks dla MOO indicators |
| `reference_points` | `(env, n_obj, obj_j)` | build_reference_pareto_sets | r\* dla HV |

Pełna dokumentacja: [src/analysis/ANALYSIS.md](../../src/analysis/ANALYSIS.md) i [src/analysis/ETL_TABLES.md](../../src/analysis/ETL_TABLES.md).

