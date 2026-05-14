# A_metrics — Zagregowane dane per metryka

## Cel

Subset zawartości `analysis.db` ograniczony do **13 metryk cytowanych w pracy** (patrz [INDEX.md §A](../INDEX.md#sekcja-a--metryki-wykorzystane-w-pracy)). Pełna `analysis.db` (861 MB) nie jest dołączona do załącznika.

## Reprodukcja: jak zostały wygenerowane

| Plik docelowy | Komenda |
|---|---|
| `run_metrics_subset.csv` | `sqlite3 results/exp_20260508_f3f718f8_bio_inspired_benchmark/analysis.db -header -csv "SELECT run_id, optimizer, environment, avoidance, seed, final_objective_f1_trajectory, total_threat_cost, total_turn_penalty, total_coordination_cost, final_objective, mean_evasion_arc_length_m, rejoin_quality, mean_online_best_fitness, online_sp1, tracking_phase_collisions, evasion_phase_collisions, min_inter_uav_distance_m, max_inter_uav_distance_m FROM vw_run_summary;" > appendix/A_metrics/run_metrics_subset.csv` |
| `iteration_metrics_subset.csv` | `sqlite3 ...analysis.db -header -csv "SELECT run_id, iteration, best_so_far, hypervolume, feasible_ratio FROM iteration_metrics;" > appendix/A_metrics/iteration_metrics_subset.csv` |
| `online_convergence_subset.csv` | `sqlite3 ...analysis.db -header -csv "SELECT run_id, drone_id, trigger_time, generation, best_fitness FROM online_convergence_traces;" > appendix/A_metrics/online_convergence_subset.csv` |

## Mapping kolumn

Pełna lista kolumn `run_metrics_subset.csv` (z mapowaniem na metryki w pracy):

| Kolumna CSV | Metryka w pracy | Direction |
|---|---|---|
| `final_objective_f1_trajectory` | M2: Długość trajektorii (F1) | ↓ |
| `total_threat_cost` | M1 component (F[2]) | ↓ |
| `total_turn_penalty` | M3 component (F[3]) | ↓ |
| `total_coordination_cost` | M1 component (F[4]) | ↓ |
| `final_objective` | M9: Wartość optymalizacji offline | ↓ |
| `mean_evasion_arc_length_m` | M6: Długość krzywej unikowej | ↓ |
| `rejoin_quality` | M7: Skuteczność powrotu | ↓ |
| `mean_online_best_fitness` | M11: Wartość optymalizacji online | ↓ |
| `online_sp1` | M13: SP1 online | ↓ |
| `tracking_phase_collisions` | bazowa dla M5 | — |
| `evasion_phase_collisions` | bazowa dla M8 | — |
| `min/max_inter_uav_distance_m` | komponenty M4 (`swarm_cohesion_deviation`) | — |

## Uwaga: metryki *derived*

Trzy metryki w pracy są derived w runtime przez [src/analysis/analyzer/metric_extractor.py:146-157](../../src/analysis/analyzer/metric_extractor.py#L146-L157) — nie są osobnymi kolumnami w DB. Wzory:

```python
# M1: trajectory_safety_f3_f5
trajectory_safety_f3_f5 = total_threat_cost + total_coordination_cost

# M3: trajectory_smoothness_f2_f4
trajectory_smoothness_f2_f4 = final_objective_f2_height_angle + total_turn_penalty

# M4: swarm_cohesion_deviation
swarm_cohesion_deviation = abs(min_inter_uav_distance_m - 5.0) \
                         + abs(max_inter_uav_distance_m - 5.0)
```

Wartości można odtworzyć w pandas po wczytaniu `run_metrics_subset.csv`.

