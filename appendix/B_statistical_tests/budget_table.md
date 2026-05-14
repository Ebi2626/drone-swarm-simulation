# Tabela 1 — Budżet obliczeniowy optymalizacji offline

Tabela odpowiadająca **Tabeli 1** w pracy. Liczby ekstraktowane z faktycznych plików Hydra konfiguracyjnych eksperymentu `exp_20260508_f3f718f8_bio_inspired_benchmark` (commit `cdca9524`).

## Wzór budżetu (cytat z pracy, s. 73)

```
Budżet obliczeniowy = P × G × K × D
```

gdzie:
- **P** — rozmiar populacji (`pop_size`)
- **G** — liczba generacji/epok (`n_gen` lub `epochs`)
- **K** — liczba wewnętrznych punktów trajektorii drona (`n_inner_waypoints`)
- **D** — liczba dronów w roju (`num_drones`)

## Parametry wspólne dla wszystkich algorytmów

| Parametr | Wartość | Źródło |
|---|---|---|
| K (`n_inner_waypoints`) | 11 | [E_configs/algorithms/](../E_configs/algorithms/) — wszystkie 4 algorytmy |
| D (`num_drones`) | 5 | [E_configs/environments/{forest,urban}.yaml](../E_configs/environments/) |

## Budżet per środowisko

### Środowisko *forest*

| Algorytm | P (pop_size) | G (n_gen / epochs) | P × G | Budżet = P × G × K × D |
|---|---:|---:|---:|---:|
| NSGA-III | 1001 | 200 | 200 200 | **11 011 000** |
| MSFFOA | 1001 | ~200* | 200 200 | **11 011 000** |
| OOA | 1001 | 200 | 200 200 | **11 011 000** |
| SSA | 1001 | ~200* | 200 200 | **11 011 000** |

\* MSFFOA i SSA definiują `max_generations: 10000` jako twardy limit, ale są ograniczone budżetem ewaluacji równym pozostałym algorytmom (`pop_size × n_gen` z `nsga-3.yaml`/`ooa.yaml`).

### Środowisko *urban*

| Algorytm | P (pop_size) | G (n_gen / epochs) | P × G | Budżet = P × G × K × D |
|---|---:|---:|---:|---:|
| NSGA-III | 1500 | 300 | 450 000 | **24 750 000** |
| MSFFOA | 1500 | ~300* | 450 000 | **24 750 000** |
| OOA | 1500 | 300 | 450 000 | **24 750 000** |
| SSA | 1500 | ~300* | 450 000 | **24 750 000** |

## Skalibrowanie

Rozmiar populacji został dobrany empirycznie podczas testowych symulacji (cytat z s. 73 pracy):

> "Wielkość populacji została dobrana empirycznie w czasie testowych symulacji, aby odpowiadała złożoności problemu. Środowisko miejskie ze względu na większy obszar oraz większą liczbę przeszkód o większych wymiarach otrzymało większy budżet wynikający z tej różnicy."

Konkretne uzasadnienie dla NSGA-III (komentarz z [E_configs/algorithms/nsga-3.yaml](../E_configs/algorithms/nsga-3.yaml)):

> "n_gen skalibrowane empirycznie (exp_20260506): Q75 konwergencji przy iteracji 69 (~69k NFE). n_gen=100 daje ~100k NFE = 1.4× margines."

## Budżet czasowy online

Faza online operuje pod innym typem budżetu (cytat z s. 86 pracy, linia 935):

> "`time_budget_s` — budżet czasowy na optymalizację — **0.5 s**"

Online pop_size jest stałe = 12 (`pop_size: 12` w sekcji `online_optimizer`). Liczba generacji online jest zmienna i zależy od czasu wallclock — patrz [A_metrics/online_convergence_subset.csv](../A_metrics/) (kolumna `generation` per `trigger_time`).

## Źródła

- Wzór: praca, s. 73, linia 855–858
- P, G dla forest: [E_configs/algorithms/{nsga-3,ooa,msffoa,ssa}.yaml](../E_configs/algorithms/)
- P, G dla urban: z `.hydra/config.yaml` per-run (overrides Hydra dla środowiska urban) — przykłady w [G_per_run_seeds/<run_id>/.hydra/config.yaml](../G_per_run_seeds/)
- K, D: jak wyżej w sekcji "Parametry wspólne"
