# ERD — Entity Relationship Diagram

Uproszczony diagram pokazujący **11 tabel cytowanych w pracy** (z 21 obecnych w pełnym schemacie SQLite). Pełna definicja w [schema.sql](schema.sql).

## Diagram

```mermaid
erDiagram
    runs ||--o| run_metrics : "1:1"
    runs ||--o{ iteration_metrics : "1:N (gen)"
    runs ||--o{ uav_online_metrics : "1:N (uav_id)"
    runs ||--o{ collisions : "1:N"
    runs ||--o{ evasion_events : "1:N"
    runs ||--o{ online_optimization_tasks : "1:N"
    runs ||--o{ online_convergence_traces : "1:N (per trigger)"
    runs ||--o{ trajectory_samples : "1:N (timesteps)"

    reference_pareto_sets ||--|| reference_points : "(env, n_obj)"
    offline_objective_normalization }o--|| runs : "via environment"

    runs {
        TEXT run_id PK
        TEXT optimizer_algo
        TEXT avoidance_algo
        TEXT environment
        INT seed
        TEXT aggregation_status
    }

    run_metrics {
        TEXT run_id PK_FK
        REAL final_objective "M9 weighted sum (Hwang Yoon 1981)"
        REAL final_objective_f1_trajectory "M2 F[0]"
        REAL total_threat_cost "F[2] z M1"
        REAL total_turn_penalty "F[3] z M3"
        REAL total_coordination_cost "F[4] z M1"
        REAL mean_evasion_arc_length_m "M6"
        REAL rejoin_quality "M7 TOPSIS"
        REAL mean_online_best_fitness "M11"
        REAL online_sp1 "M13 (Auger Hansen 2005)"
        INT tracking_phase_collisions "baza dla M5"
        INT evasion_phase_collisions "baza dla M8"
        REAL min_inter_uav_distance_m "M4 component"
        REAL max_inter_uav_distance_m "M4 component"
    }

    iteration_metrics {
        TEXT run_id FK
        INT iteration PK
        REAL best_so_far "M10 krzywa offline"
        REAL hypervolume
        REAL feasible_ratio
    }

    uav_online_metrics {
        TEXT run_id FK
        INT uav_id PK
        REAL min_inter_uav_distance_m
        REAL max_inter_uav_distance_m
        REAL mean_inter_uav_distance_m
        REAL energy_indicator
        REAL smoothness_indicator
    }

    collisions {
        TEXT run_id FK
        REAL sim_time
        INT drone_id
        INT other_body_id
    }

    evasion_events {
        TEXT run_id FK
        REAL time
        INT drone_id
        TEXT event_type "trigger | rejoin"
        REAL ttc
        REAL dist_to_threat
    }

    online_optimization_tasks {
        TEXT run_id FK
        INT drone_id PK
        REAL trigger_time PK
        REAL best_fitness "→ M11"
        REAL plan_arc_length_m "→ M6"
        REAL pos_err_at_rejoin_m "→ M7"
        REAL vel_err_at_rejoin_mps "→ M7"
        REAL time_to_rejoin_s "→ M7"
        TEXT status "ok | budget_violation"
    }

    online_convergence_traces {
        TEXT run_id FK
        INT drone_id PK
        REAL trigger_time PK
        INT generation PK
        REAL best_fitness "→ M12 krzywa online"
    }

    trajectory_samples {
        TEXT run_id FK
        INT sample_index PK
        REAL sim_time
        INT drone_id
        REAL x
        REAL y
        REAL z
    }

    reference_pareto_sets {
        TEXT environment PK
        INT n_obj PK
        INT point_idx PK
        INT objective_j PK
        REAL value
    }

    reference_points {
        TEXT environment PK
        INT n_obj PK
        INT objective_j PK
        REAL value "r*"
        REAL ideal_value "z*"
    }

    offline_objective_normalization {
        TEXT environment PK
        INT f_idx PK
        REAL f_ref_median "→ M9 normalizacja"
        INT n_runs
    }
```

## Kluczowe relacje

1. **`runs` → `run_metrics`** (1:1) — każdy z 240 runów ma jeden rekord agregatów per-run.
2. **`runs` → `iteration_metrics`** (1:N) — N = liczba generacji (`forest`: 200, `urban`: 300 dla offline) **× 4 algorytmy × 30 seeds**. Cytowane jako M10.
3. **`runs` → `online_optimization_tasks`** (1:N) — N = liczba triggerów online avoidance per run (zmienne, zależy od `evasion_event_count`). Cytowane przez M6, M7, M11, M13.
4. **`runs` → `online_convergence_traces`** (1:N) — N = liczba (trigger_time × generation) per run. Cytowane przez M12 (krzywe online).
5. **`reference_pareto_sets` ↔ `reference_points`** — Pareto reference (R) i punkt referencyjny (r\*) per (env, n_obj), używane do MOO indicators (HV, IGD+ — nie cytowane bezpośrednio w pracy, ale wymagane dla MOO quality).
6. **`offline_objective_normalization`** — per-environment median F_best, używana w `final_objective` (M9) jako mianownik weighted normalized sum.

## Tabele pomijane w tym diagramie (pełne 21 tabel w schema.sql)

Następujące tabele istnieją w `analysis.db`, ale nie są bezpośrednio cytowane w pracy:

- `meta`, `run_files` — metadane pipelinu ETL
- `uav_metrics` — per-UAV path length (zagregowane do `run_metrics`)
- `optimization_generation_stats` — surowy long-form per-generation (pivotowany do `iteration_metrics`)
- `optimization_timings`, `world_boundaries`, `generated_obstacles` — metadane środowiska
- `counted_trajectory_points` — planowane waypoints
- `trajectory_metrics` — per-source path length agregaty
- `evasion_events` — pomocnicze (główna funkcja: klasyfikacja faz kolizji)

## Renderowanie

Mermaid renderuje się automatycznie w GitHub, GitLab, większości edytorów markdown (Obsidian, VSCode z extension). Do PDF/druku: skopiuj kod między `mermaid` blokami i wklej do https://mermaid.live aby wygenerować SVG/PNG.
