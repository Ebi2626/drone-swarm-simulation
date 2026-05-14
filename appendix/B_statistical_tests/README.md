# B_statistical_tests — Testy statystyczne cytowane w pracy

## Cel

Wyłącznie testy statystyczne odpowiadające tabelom **T2–T21** z pracy. Cały `analysis_output/tables/` zawiera 266 plików (Friedman/A12/Wilson dla wszystkich 32 kolumn `run_metrics`); tutaj wybrane są tylko te 13 metryk faktycznie cytowanych.

## Struktura

```
B_statistical_tests/
├── summary/                # Statystyki opisowe (Tabele T2, T4, T6, T8, T10, T12, T14, T16, T19)
├── friedman/               # Friedman + Nemenyi (per environment)
├── a12/                    # Vargha-Delaney A12 effect size (per environment)
├── wilson/                 # Wilson 95% CI dla proporcji (Tabela T15)
└── thesis_stat_tables/     # Gotowe PNG złożone z forest/urban side-by-side
```

## Pliki do skopiowania

**Źródło:** `results/exp_20260508_f3f718f8_bio_inspired_benchmark/analysis_output/tables/`

### `summary/` (9 plików × 2 formaty)

| Plik | Tabela w pracy |
|---|---|
| `summary_trajectory_safety_f3_f5.{csv,tex}` | T2 |
| `summary_trajectory_length_f1.{csv,tex}` | T4 |
| `summary_trajectory_smoothness_f2_f4.{csv,tex}` | T6 |
| `summary_swarm_cohesion_deviation.{csv,tex}` | T8 |
| `summary_mean_evasion_arc_length_m.{csv,tex}` | T10 |
| `summary_rejoin_quality.{csv,tex}` | T12 |
| `summary_final_objective.{csv,tex}` | T16 |
| `summary_mean_online_best_fitness.{csv,tex}` | T19 |

### `friedman/` (10 metryk × 2 środowiska = 20 plików)

Pattern: `{env}_friedman_{metric}.csv` gdzie `env ∈ {forest, urban}`.

Metryki: `trajectory_safety_f3_f5`, `trajectory_length_f1`, `trajectory_smoothness_f2_f4`, `swarm_cohesion_deviation`, `mean_evasion_arc_length_m`, `rejoin_quality`, `final_objective`, `auc_best_so_far`, `mean_online_best_fitness`, `online_sp1`.

### `a12/` (10 metryk × 2 środowiska = 20 plików)

Pattern: `{env}_a12_{metric}.csv`. Te same metryki jak Friedman.

### `wilson/` (3 pliki)

| Plik | Cel |
|---|---|
| `failure_rate_offline.{csv,tex}` | M5 — Odsetek trajektorii kolizyjnych offline |
| `failure_rate_online.{csv,tex}` | T15 — Wilson CI dla bezpieczeństwa uników |
| `evasion_success_rate.csv` | T14 — derived: `1 − failure_rate_online` (Wilson CI: granice odbite) |

### `thesis_stat_tables/` (14 PNG)

**Źródło:** [praca/chapter-3/stat_tables/](../../praca/chapter-3/stat_tables/)

Gotowe panele składające forest+urban side-by-side, generowane przez [scripts/generate_thesis_stat_tables.py](../../scripts/generate_thesis_stat_tables.py):

- `stat_3211_trajectory_safety.png` — T2+T3
- `stat_3211_trajectory_length.png` — T4+T5
- `stat_3211_trajectory_smoothness.png` — T6+T7
- `stat_3211_swarm_cohesion.png` — T8+T9
- `stat_3211_offline_failure.png` — M5 panel
- `stat_3212_evasion_arc_length.png` — T10+T11
- `stat_3212_rejoin_quality.png` — T12+T13
- `stat_3212_online_safety.png` — T14+T15
- `stat_3221_final_objective.png` — T16+T17
- `stat_3221_auc_best_so_far.png` — T18
- `stat_3221_wallclock_offline.png` — kontekst dla rozdz. 3.2.2.1
- `stat_3222_online_best_fitness.png` — T19+T20
- `stat_3222_online_sp1.png` — T21
- `stat_3222_online_auc.png` — kontekst dla rozdz. 3.2.2.2

## Reprodukcja: komenda generująca

```bash
SRC="results/exp_20260508_f3f718f8_bio_inspired_benchmark/analysis_output/tables"
DST="appendix/B_statistical_tests"

# summary
for m in trajectory_safety_f3_f5 trajectory_length_f1 trajectory_smoothness_f2_f4 swarm_cohesion_deviation mean_evasion_arc_length_m rejoin_quality final_objective mean_online_best_fitness; do
  cp $SRC/summary_${m}.{csv,tex} $DST/summary/ 2>/dev/null
done

# friedman + a12 per env
for m in trajectory_safety_f3_f5 trajectory_length_f1 trajectory_smoothness_f2_f4 swarm_cohesion_deviation mean_evasion_arc_length_m rejoin_quality final_objective auc_best_so_far mean_online_best_fitness online_sp1; do
  for env in forest urban; do
    cp $SRC/${env}_friedman_${m}.csv $DST/friedman/ 2>/dev/null
    cp $SRC/${env}_a12_${m}.csv $DST/a12/ 2>/dev/null
  done
done

# wilson
cp $SRC/failure_rate_{offline,online}.{csv,tex} $DST/wilson/

# thesis stat tables PNG
cp praca/chapter-3/stat_tables/*.png $DST/thesis_stat_tables/

# Tabela 1 (budżet) — ręczna, patrz INDEX.md §D D6
```

## Tabela T1 — budżet obliczeniowy

Pełna tabela odpowiadająca Tabeli 1 w pracy: [budget_table.md](budget_table.md).

Skrócone podsumowanie (wartości z [E_configs/algorithms/](../E_configs/algorithms/) i `.hydra/config.yaml` per-run):

- **Forest:** P=1001, G=200, K=11, D=5 → budżet **11 011 000** ewaluacji składowych
- **Urban:** P=1500, G=300, K=11, D=5 → budżet **24 750 000** ewaluacji składowych

Wzór: P × G × K × D (z s. 73 pracy).

