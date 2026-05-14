# C_plots — Wykresy cytowane w pracy (18 sztuk)

## Cel

Wyłącznie 18 wykresów (W1–W18) cytowanych w pracy. Pełny `analysis_output/plots/` zawiera dziesiątki wykresów diagnostycznych (`pareto/`, `cd_diagrams/`, `rankings/`, boxploty dla 30+ metryk), które NIE są dołączone do załącznika.

Każdy wykres dostępny w dwóch wariantach: **PDF** (do druku w wersji papierowej pracy, jakość wektorowa) i **PNG** (do raportu cyfrowego/elektronicznej wersji pracy, 150 dpi).

## Struktura i mapowanie

```
C_plots/
├── bar/                # W1, W2 — odsetek trajektorii kolizyjnych
├── boxplots/           # W3–W14 — boxploty per metryka per środowisko
└── convergence/        # W15–W18 — krzywe zbieżności offline/online
```

### `bar/`

| Wykres | Plik | Metryka |
|---|---|---|
| W1 | `bar_forest_failure_rate_offline.pdf` + `.png` | M5 |
| W2 | `bar_urban_failure_rate_offline.pdf` + `.png` | M5 |

### `boxplots/`

| Wykres | Plik | Metryka |
|---|---|---|
| W3 | `boxplot_forest_trajectory_length_f1.pdf` + `.png` | M2 |
| W4 | `boxplot_urban_trajectory_length_f1.pdf` + `.png` | M2 |
| W5 | `boxplot_forest_trajectory_smoothness_f2_f4.pdf` + `.png` | M3 |
| W6 | `boxplot_urban_trajectory_smoothness_f2_f4.pdf` + `.png` | M3 |
| W7 | `boxplot_forest_swarm_cohesion_deviation.pdf` + `.png` | M4 |
| W8 | `boxplot_urban_swarm_cohesion_deviation.pdf` + `.png` | M4 |
| W9 | `boxplot_forest_mean_evasion_arc_length_m.pdf` + `.png` | M6 |
| W10 | `boxplot_urban_mean_evasion_arc_length_m.pdf` + `.png` | M6 |
| W11 | `boxplot_forest_rejoin_quality.pdf` + `.png` | M7 |
| W12 | `boxplot_urban_rejoin_quality.pdf` + `.png` | M7 |
| W13 | `boxplot_forest_final_objective.pdf` + `.png` | M9 |
| W14 | `boxplot_urban_final_objective.pdf` + `.png` | M9 |

### `convergence/`

| Wykres | Plik | Metryka |
|---|---|---|
| W15 | `convergence_forest_best_so_far.pdf` + `.png` | M10 (offline, agreg. mean±std) |
| W16 | `convergence_urban_best_so_far.pdf` + `.png` | M10 |
| W17 | `online_convergence_forest.pdf` + `.png` | M12 (online, agreg. median±IQR) |
| W18 | `online_convergence_urban.pdf` + `.png` | M12 |

## Reprodukcja: komenda generująca

```bash
SRC="results/exp_20260508_f3f718f8_bio_inspired_benchmark/analysis_output/plots"
DST="appendix/C_plots"

# Bar (W1-W2)
cp $SRC/bar/bar_{forest,urban}_failure_rate_offline.{pdf,png} $DST/bar/

# Boxploty (W3-W14)
for m in trajectory_length_f1 trajectory_smoothness_f2_f4 swarm_cohesion_deviation \
         mean_evasion_arc_length_m rejoin_quality final_objective; do
  for env in forest urban; do
    cp $SRC/boxplots/boxplot_${env}_${m}.{pdf,png} $DST/boxplots/
  done
done

# Convergence (W15-W16 offline + W17-W18 online)
for env in forest urban; do
  cp $SRC/convergence/convergence_${env}_best_so_far.{pdf,png} $DST/convergence/
  cp $SRC/convergence/online_convergence_${env}.{pdf,png} $DST/convergence/
done
```

Razem: **36 plików** (18 PDF + 18 PNG).

