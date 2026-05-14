# Widoki SQL w `analysis.db`

5 widoków zdefiniowanych w [schema.sql](schema.sql), używanych przez analyzer i opcjonalnie przez recenzenta do ad-hoc zapytań. Definicje wyekstraktowane ze schema.sql linie 784–1090.

## 1. `vw_run_summary` (linia 784)

**Cel:** kompletny zbiór per-run = `runs LEFT JOIN run_metrics`. Główny widok analityczny — używany przez `MetricExtractor.run_summary()` w [src/analysis/analyzer/metric_extractor.py](../../src/analysis/analyzer/metric_extractor.py).

**Kolumny:** wszystkie z `runs` (metadata) + wszystkie z `run_metrics` (32 metryki) + flag `has_metrics`.

**Cytowane metryki:** M1–M9, M11, M13 (offline + online agregaty per-run).

**Przykład:**
```sql
SELECT optimizer_algo, environment, AVG(final_objective) AS mean_fo
FROM vw_run_summary
GROUP BY optimizer_algo, environment;
```

## 2. `vw_seed_summary` (linia 843)

**Cel:** agregacja per (environment, optimizer_algo, avoidance_algo, seed) — zwykle 1 wiersz per kombinacja, ale grupuje jeśli istnieją duplicate seeds.

**Kolumny:** environment, optimizer_algo, avoidance_algo, seed, n_runs, agregaty (MEDIAN/AVG/STDDEV) z metryk.

**Zastosowanie:** weryfikacja determinizmu (powinno być 1 run per seed) i debugging.

## 3. `vw_global_summary` (linia 862)

**Cel:** najwyższy poziom agregacji per (environment, optimizer_algo, avoidance_algo) — `n_seeds = 30` w każdym wierszu.

**Kolumny:** environment, optimizer_algo, avoidance_algo, n_seeds, n_successful_runs, AVG/STDDEV/MEDIAN/MIN/MAX z `final_objective`, `final_objective_f1_trajectory`, `total_threat_cost`, `total_turn_penalty`, itp.

**Zastosowanie:** szybki przegląd wyników bez przechodzenia przez `summary_*.csv` z analyzera. Bezpośrednio porównywalne z Tabelami T2–T8 w pracy.

**Przykład:**
```sql
SELECT environment, optimizer_algo, n_successful_runs, mean_final_objective
FROM vw_global_summary
ORDER BY environment, mean_final_objective;
```

## 4. `vw_run_online_summary` (linia 996)

**Cel:** per-run agregaty online performance z `online_optimization_tasks` i `online_convergence_traces`. Zasila Tabele T10, T12, T19 (online statistics).

**Kolumny:** run_id, n_triggers, mean/median_best_fitness, mean_pos_err, mean_vel_err, mean_time_to_rejoin, rejoin_success_rate, budget_violation_rate, online_sp1.

**Cytowane metryki:** M6 (`mean_evasion_arc_length_m`), M7 (`rejoin_quality`), M11 (`mean_online_best_fitness`), M13 (`online_sp1`).

## 5. `vw_algo_cross_sim_comparison` (linia 1044)

**Cel:** porównanie cross-simulation algorytmów online (SWaP, rejoin success rate, safety) per (environment, optimizer_algo).

**Kolumny:** environment, optimizer_algo, avoidance_algo, n_runs, agregaty z `vw_run_online_summary`.

**Zastosowanie:** widok wykorzystywany w raportach analytycznych zewnętrznych względem standardowego pipelinu `ExperimentAnalyzer`.

## Reprodukcja widoków

Wszystkie widoki są tworzone z `IF NOT EXISTS` przez ETL aggregator. Po zrekonstruowaniu `analysis.db` z `G_per_run_seeds/` poprzez:

```python
from src.analysis.ExperimentAggregator import ExperimentAggregator
ExperimentAggregator().aggregate('./results/<reconstructed_exp>/')
```

— widoki będą dostępne automatycznie.

## Indeksy wspierające widoki

Schema.sql definiuje **22 indeksy OLAP** dla optymalizacji zapytań analitycznych (linie 1095–1139 schema.sql). Najważniejsze:

- `idx_runs_optimizer_env` — wspiera GROUP BY w `vw_global_summary`
- `idx_run_metrics_final_objective` — wspiera ORDER BY w rankingach
- `idx_iteration_metrics_run_iter` — wspiera krzywe konwergencji M10
- `idx_online_convergence_run_trigger` — wspiera krzywe konwergencji M12
