# Załącznik cyfrowy — Praca magisterska

## Co znajduje się w tym katalogu

Załącznik cyfrowy do pracy magisterskiej *Porównanie bio-inspirowanych metaheurystyk (MSFOA, OOA, SSA) z klasycznym NSGA-III w problemie planowania trajektorii roju UAV*.

Zawartość ograniczona jest **wyłącznie do artefaktów cytowanych bezpośrednio w pracy**. Pełne wyniki pipeline'u analizy (266 plików w `tables/`, dziesiątki wykresów diagnostycznych w `plots/cd_diagrams/`, `plots/pareto/`, `plots/rankings/`) NIE są dołączone ze względu na rozmiar (59,2GB).

## Jak czytać ten załącznik

1. **Zacznij od [INDEX.md](INDEX.md)** — kompletny indeks wszystkich metryk (M1–M13), wykresów (W1–W18) i tabel (T1–T21) z pracy. Każdy element zmapowany na konkretny plik w tym katalogu.
2. **[CITATION.md](CITATION.md)** — link do repozytorium GitHub + commit hash + DOI (jeśli aktualne).
3. **Podkatalogi A–H** — pogrupowane artefakty:
   - `A_metrics/` — wyfiltrowane CSV z `analysis.db` (13 metryk × 240 runów)
   - `B_statistical_tests/` — Friedman, A12, Wilson CI (CSV + LaTeX + gotowe panele PNG)
   - `C_plots/` — 18 wykresów cytowanych w pracy (PDF do druku + PNG do raportu cyfrowego)
   - `D_database_schema/` — schemat SQL bazy + ERD (do zrozumienia struktury danych)
   - `E_configs/` — pliki Hydra definiujące eksperyment
   - `F_environment/` — `environment.yaml` + pełny conda export (reproducibility środowiska)
   - `G_per_run_seeds/` — surowe pliki per-run (240 katalogów; tylko te potrzebne do reprodukcji metryk z pracy)
4. **[H_run_manifest.csv](H_run_manifest.csv)** — manifest 240 runów: (run_id, optimizer, environment, avoidance, seed, status).

Każdy podkatalog zawiera własny `README.md` z opisem zawartości i odniesieniami do oryginalnych ścieżek w repo.

## Jak reprodukować wyniki z pracy

**Wariant minimalny** (na podstawie zagregowanych CSV w tym załączniku):

```bash
# 1. Wczytaj statystyki opisowe i testy
python -c "import pandas as pd; print(pd.read_csv('appendix/A_metrics/run_metrics_subset.csv').describe())"

# 2. Zweryfikuj testy Friedmana
ls appendix/B_statistical_tests/friedman/
```

**Wariant pełny** (od surowych danych do publikacji):

```bash
# 1. Sklonuj repo na commicie referencyjnym (patrz CITATION.md)
git clone git@github.com:Ebi2626/drone-swarm-simulation.git
cd drone-swarm-simulation
git checkout cdca9524f58f54b5da720e80fcbd239595f4ea16

# 2. Zbuduj środowisko conda
conda env create -f appendix/F_environment/environment.yaml
conda activate drone-swarm-env

# 3. Uruchom eksperyment (UWAGA: 240 runów × ~30 min/run ≈ 120 h GPU)
python main.py --multirun \
  optimizer=nsga3,msfoa,ooa,ssa \
  environment=forest,urban \
  seed=range(0,30)

# 4. Agregacja ETL → analysis.db
python -c "from src.analysis.ExperimentAggregator import ExperimentAggregator; \
           ExperimentAggregator().aggregate('./results/<your_run>/')"

# 5. Analiza porównawcza → tables/ + plots/
python run_etl.py ./results/<your_run>/
```

## Ograniczenia załącznika

1. **Brak `analysis.db`** (861 MB) — można ją zrekonstruować ją z surowych CSV w `G_per_run_seeds/` poprzez `ExperimentAggregator`. Subsety w `A_metrics/` są wystarczające do weryfikacji testów statystycznych z `B_statistical_tests/`.
2. **Brak `trajectories.csv`** (40k+ wierszy/run × 240 runów ≈ 9.6 mln wierszy) — kinematic safety metrics są już zagregowane w `A_metrics/run_metrics_subset.csv` (kolumny `min/max/mean_inter_uav_distance_m`).
3. **Brak `lidar_hits.h5`** — pomocnicze, nie cytowane w pracy.
4. **Kod źródłowy** nie jest dołączony — patrz [CITATION.md](CITATION.md) dla linku do snapshot na GitHubie pod tagiem `v1.0-thesis` (commit `cdca9524`).

## Kontakt

W przypadku pytań co do reprodukcji: edwin.harmata@gmail.com.
