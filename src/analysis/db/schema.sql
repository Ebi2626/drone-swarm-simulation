BEGIN;

-- =========================================================
-- 1. Metadane bazy / eksperymentu
-- =========================================================
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- =========================================================
-- 2. Runy
-- Jeden katalog runu = jeden rekord
-- =========================================================
-- Refaktor 2026-05-07: usunięte never-populated kolumny `decision_mode`
-- (duplikat z run_metrics, też tam DROPPED), `notes` (YAGNI, dowolny
-- komentarz nigdy nieużywany), `run_config_json` (Hydra config jest
-- już w `<run_dir>/.hydra/config.yaml` — duplikacja niepotrzebna).
-- `aggregation_error` zachowane — wypełniane przy try/except per-run
-- (status='failed' + error message).
CREATE TABLE IF NOT EXISTS runs (
    run_id               TEXT PRIMARY KEY,
    run_dir_name         TEXT NOT NULL UNIQUE,
    source_path          TEXT NOT NULL,
    optimizer_algo       TEXT NOT NULL,
    avoidance_algo       TEXT NOT NULL,
    environment          TEXT NOT NULL,
    seed                 INTEGER NOT NULL CHECK (seed >= 0),
    algorithm_pair       TEXT NOT NULL,
    aggregation_status   TEXT NOT NULL DEFAULT 'discovered'
                         CHECK (aggregation_status IN ('discovered', 'aggregated', 'failed', 'partial')),
    aggregation_error    TEXT,
    discovered_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    aggregated_at        TEXT,
    UNIQUE (optimizer_algo, environment, avoidance_algo, seed)
);

-- =========================================================
-- 3. Rejestr plików źródłowych znalezionych w runie
-- =========================================================
-- Rejestr plików źródłowych per run. Po refaktorze 2026-05-07:
-- usunięte pola `checksum` i `extra_json` (YAGNI, nigdy nieużywane).
-- Dodane wypełnianie `modified_at` (mtime ISO 8601) i `row_count` (CSV linie - 1).
CREATE TABLE IF NOT EXISTS run_files (
    run_id           TEXT NOT NULL,
    file_role        TEXT NOT NULL,
    relative_path    TEXT NOT NULL,
    file_format      TEXT,
    exists_flag      INTEGER NOT NULL DEFAULT 1 CHECK (exists_flag IN (0,1)),
    size_bytes       INTEGER CHECK (size_bytes IS NULL OR size_bytes >= 0),
    row_count        INTEGER CHECK (row_count IS NULL OR row_count >= 0),
    modified_at      TEXT,
    PRIMARY KEY (run_id, file_role),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- =========================================================
-- 4. Metryki końcowe per run
-- Refaktor 2026-05-07:
-- * Usunięte legacy 8-component costs (`total_energy_cost`,
--   `total_smoothness_cost`, `total_altitude_cost`,
--   `total_terrain_penalty`, `total_climb_penalty`,
--   `total_collision_penalty`) — żaden z 4 algorytmów
--   (msffoa/ooa/ssa/nsga-3) ich nie produkuje. Aktualny
--   `VectorizedEvaluator` ma 5-obj F-vector mapowane na
--   `final_objective_f1_trajectory/f2_height_angle/total_threat_cost/
--   total_turn_penalty/total_coordination_cost`.
-- * Usunięte never-populated: `decision_mode` (w Hydra config, nie
--   zapisywane do meta), `selected_solution_index` (decision-making
--   runtime nie zapisuje), `feasible_nondominated_count` (nigdy
--   nieliczone), `reference_point_json` (mamy `reference_points` table).
-- =========================================================
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id                      TEXT PRIMARY KEY,
    drone_count                 INTEGER CHECK (drone_count IS NULL OR drone_count >= 0),
    success                     INTEGER NOT NULL DEFAULT 1 CHECK (success IN (0,1)),

    -- Końcowa wybrana trajektoria / rozwiązanie
    final_objective             REAL,
    total_path_length_2d        REAL,
    total_path_length_3d        REAL,
    -- F[2] z 5-obj F-vector (zob. populate_offline_objectives)
    total_threat_cost           REAL,
    -- F[3] z 5-obj F-vector
    total_turn_penalty          REAL,
    collision_count             INTEGER NOT NULL DEFAULT 0 CHECK (collision_count >= 0),
    evasion_event_count         INTEGER NOT NULL DEFAULT 0 CHECK (evasion_event_count >= 0),
    obstacle_count              INTEGER NOT NULL DEFAULT 0 CHECK (obstacle_count >= 0),
    best_iteration              INTEGER CHECK (best_iteration IS NULL OR best_iteration >= 0),

    -- Metryki zbioru Pareto / MOO
    nondominated_count          INTEGER CHECK (nondominated_count IS NULL OR nondominated_count >= 0),
    hypervolume                 REAL,
    igd_plus                    REAL,

    -- Diagnostyki MOO last-gen (Kamień 2 — 2026-05-07).
    --   front_size_last_gen — |F_feas ∩ ND| z ostatniej generacji. SOO
    --     ze skalaryzacją zwykle daje 1 (wszystko zbiega do jednego
    --     rozwiązania) → HV trywialne, GD≈0. Diagnostyka anomalii OOA.
    --   hypervolume_normalized — HV / Π(r* − ideal) ∈ [0,1].
    --     Riquelme 2015 §3.6: cross-env porównywalne (skala każdego env
    --     znormalizowana przez full bounding box obj-space).
    front_size_last_gen         INTEGER CHECK (
        front_size_last_gen IS NULL OR front_size_last_gen >= 0
    ),
    hypervolume_normalized      REAL CHECK (
        hypervolume_normalized IS NULL OR hypervolume_normalized >= 0.0
    ),

    -- Online safety / energy / smoothness aggregates (notatki.md, 2026-05-04)
    -- Liczone w `populate_online_safety_metrics` z `trajectory_samples` i
    -- agregowane do `run_metrics` przez `populate_run_metrics`.
    min_inter_uav_distance_m            REAL,
    mean_inter_uav_distance_m           REAL,
    total_inter_uav_safety_violations   INTEGER CHECK (
        total_inter_uav_safety_violations IS NULL OR
        total_inter_uav_safety_violations >= 0
    ),
    mean_energy_indicator               REAL,
    mean_smoothness_indicator           REAL,

    -- Offline F-vector — best feasible solution z optimization_history.h5
    -- last generation (2026-05-05). Mapowanie:
    --   F[0] = f1 trajectory_cost (length + shape) → final_objective_f1_trajectory
    --   F[1] = f2 height_angle_cost              → final_objective_f2_height_angle
    --   F[2] = f3 threat_cost                    → total_threat_cost (re-use)
    --   F[3] = f4 turn_cost                      → total_turn_penalty (re-use)
    --   F[4] = f5 coordination_cost              → total_coordination_cost
    -- Schema patrz `objective_constrains.VectorizedEvaluator`.
    final_objective_f1_trajectory       REAL,
    final_objective_f2_height_angle     REAL,
    total_coordination_cost             REAL,
    final_objectives_json               TEXT,

    -- MOO quality indicators last-gen (2026-05-06).
    -- Patrz tabela `iteration_metrics` po opisy + literaturę.
    --   gd_final / spread_final / spacing_final / r2_final — wartość
    --     wskaźnika z ostatniej generacji (snapshot ostatecznej jakości
    --     fronta).
    --   convergence_speed_gen — pierwsza generacja, w której HV ≥
    --     0.9 · HV(last_gen) (proxy szybkości zbieżności). NULL gdy HV
    --     niedostępne lub HV ostatecznie < 0.9·max.
    --   auc_best_so_far — pole pod krzywą best_so_far(g), znormalizowane
    --     przez liczbę generacji; lower=lepiej, integruje całą historię
    --     zbieżności w skalar.
    gd_final                            REAL,
    spread_final                        REAL,
    spacing_final                       REAL,
    r2_final                            REAL,
    convergence_speed_gen               INTEGER CHECK (
        convergence_speed_gen IS NULL OR convergence_speed_gen >= 0
    ),
    auc_best_so_far                     REAL,

    objective_components_json   TEXT,
    summary_json                TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- =========================================================
-- 5. Metryki per UAV
-- Refaktor 2026-05-07: usunięte 9 legacy cost columns, których
-- `populate_uav_metrics` zawsze wstawiał jako NULL (`final_objective`,
-- `energy_cost`, `smoothness_cost`, `threat_cost`, `altitude_cost`,
-- `terrain_penalty`, `turn_penalty`, `climb_penalty`, `collision_penalty`)
-- — żaden z 4 algorytmów (msffoa/ooa/ssa/nsga-3) ich per-UAV nie produkuje.
-- F-vector w h5 jest zbiorczy (per swarm), nie rozbity per drone, więc
-- per-UAV wartości nie istnieją w danych źródłowych. Per-run F-vector
-- mapowany jest do `run_metrics` przez `populate_offline_objectives`.
-- =========================================================
CREATE TABLE IF NOT EXISTS uav_metrics (
    run_id                      TEXT NOT NULL,
    uav_id                      INTEGER NOT NULL CHECK (uav_id >= 0),
    success                     INTEGER CHECK (success IN (0,1)),
    path_length_2d              REAL,
    path_length_3d              REAL,
    collision_count             INTEGER NOT NULL DEFAULT 0 CHECK (collision_count >= 0),
    evasion_event_count         INTEGER NOT NULL DEFAULT 0 CHECK (evasion_event_count >= 0),
    extra_json                  TEXT,
    PRIMARY KEY (run_id, uav_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- =========================================================
-- 5b. Metryki online per UAV — Inter-UAV safety, energy, smoothness
--     (notatki.md, 2026-05-04). Liczone post-hoc z trajectory_samples
--     przez `populate_online_safety_metrics`.
-- =========================================================
CREATE TABLE IF NOT EXISTS uav_online_metrics (
    run_id                            TEXT NOT NULL,
    uav_id                            INTEGER NOT NULL CHECK (uav_id >= 0),

    -- Inter-UAV safety (notatki.md #1)
    -- Min/mean odległości tego UAVa od najbliższego sąsiada w roju w czasie.
    -- `inter_uav_safety_violation_count` = liczba próbek gdzie min-pairwise
    -- distance dla tego UAVa < `inter_uav_safety_threshold_m`.
    min_inter_uav_distance_m          REAL,
    mean_inter_uav_distance_m         REAL,
    inter_uav_safety_violation_count  INTEGER CHECK (
        inter_uav_safety_violation_count IS NULL OR
        inter_uav_safety_violation_count >= 0
    ),
    inter_uav_safety_threshold_m      REAL CHECK (
        inter_uav_safety_threshold_m IS NULL OR
        inter_uav_safety_threshold_m >= 0
    ),

    -- Energy efficiency proxy (notatki.md #2)
    -- ∫ ‖v‖² dt / total_path_length_3d  → m·s⁻²·m / m = m/s² (mniej = wydajniej).
    -- Standard w UAV literature: F_drag ∝ v², więc ∫ v² dt to proxy total drag work.
    -- Reference: McAllister et al. (2017), "Quantifying energy efficiency of
    -- multirotor UAV trajectories."
    energy_indicator                  REAL,
    speed_squared_integral            REAL,
    mean_speed_mps                    REAL,
    max_speed_mps                     REAL,

    -- Smoothness (notatki.md #3)
    -- ∫ ‖a‖² dt — przybliżenie z różnicowania prędkości po próbkach.
    -- Niższa wartość = gładsza trasa. Standard w trajectory smoothness
    -- literature: minimum-acceleration cost (Hauser & Hubicki 2007).
    smoothness_indicator              REAL,
    accel_squared_integral            REAL,
    mean_accel_mps2                   REAL,
    max_accel_mps2                    REAL,

    sample_count                      INTEGER CHECK (
        sample_count IS NULL OR sample_count >= 0
    ),
    duration_s                        REAL CHECK (
        duration_s IS NULL OR duration_s >= 0
    ),
    extra_json                        TEXT,

    PRIMARY KEY (run_id, uav_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- =========================================================
-- 6. Historia optymalizacji / zbieżność
-- =========================================================
CREATE TABLE IF NOT EXISTS iteration_metrics (
    run_id                           TEXT NOT NULL,
    iteration                        INTEGER NOT NULL CHECK (iteration >= 0),

    -- Wspólne metryki procesu
    population_size                  INTEGER CHECK (population_size IS NULL OR population_size >= 0),
    feasible_solutions               INTEGER CHECK (feasible_solutions IS NULL OR feasible_solutions >= 0),
    feasible_ratio                   REAL CHECK (
                                         feasible_ratio IS NULL OR
                                         (feasible_ratio >= 0.0 AND feasible_ratio <= 1.0)
                                     ),
    diversity_metric                 REAL,
    elapsed_s                        REAL CHECK (elapsed_s IS NULL OR elapsed_s >= 0),
    eval_count_cumulative            INTEGER CHECK (
                                         eval_count_cumulative IS NULL OR
                                         eval_count_cumulative >= 0
                                     ),

    -- Naruszenia ograniczeń: wspólne dla constrained SOO i MOO
    constraint_violation_best        REAL,
    constraint_violation_mean        REAL,
    constraint_violation_worst       REAL,

    -- Metryki skalarnego przebiegu:
    -- dla SOO = rzeczywisty scalar objective,
    -- dla NSGA-III = wyłącznie auxiliary progress score
    best_so_far                      REAL,
    current_best                     REAL,
    current_mean                     REAL,
    current_std                      REAL,
    current_worst                    REAL,

    -- Metryki specyficzne dla populacji Pareto / NSGA-III
    nondominated_solutions           INTEGER CHECK (
                                         nondominated_solutions IS NULL OR
                                         nondominated_solutions >= 0
                                     ),
    nondominated_ratio               REAL CHECK (
                                         nondominated_ratio IS NULL OR
                                         (nondominated_ratio >= 0.0 AND nondominated_ratio <= 1.0)
                                     ),
    hypervolume                      REAL,

    -- MOO quality indicators (2026-05-06).
    -- Reference: Riquelme, Lücken & Baran (2015) "Performance metrics in
    -- multi-objective optimization", CLEI Electronic Journal 18(1).
    --   gd            — Generational Distance (Van Veldhuizen 1999): mean
    --                   distance fronta do reference set R. Niższe = lepiej.
    --   igd_plus      — IGD+ (Ishibuchi et al. 2015): Pareto-compliant
    --                   wariant IGD; d^+(r,f)=max(0,f-r) component-wise.
    --   spread        — Δ-metric (Deb et al. 2002): równomierność rozkładu
    --                   na froncie; Δ=0 idealnie równomierne.
    --   spacing       — Spacing S (Schott 1995): std odl. do najbliższego
    --                   sąsiada (lower=better).
    --   r2_indicator  — R2 (Hansen & Jaszkiewicz 1998): mean min Tchebycheff'a
    --                   po wektorach wag; odporne na monotoniczne transf.
    igd_plus                         REAL,
    gd                               REAL,
    spread                           REAL,
    spacing                          REAL,
    r2_indicator                     REAL,

    -- Diagnostyki MOO per gen (Kamień 2 — 2026-05-07).
    --   front_size — |F_feas ∩ ND| w tej generacji. Pozwala śledzić
    --     czy front degeneruje (SOO zbiega do 1 punktu) lub puchnie
    --     (NSGA-III).
    --   hypervolume_normalized — HV znormalizowane przez Π(r* − ideal)
    --     z `reference_points`. Cross-env porównywalne (Riquelme 2015 §3.6).
    front_size                       INTEGER CHECK (
                                         front_size IS NULL OR front_size >= 0
                                     ),
    hypervolume_normalized           REAL,

    -- Pole na dodatkowe rzeczy typu updated_swarms_count, swarms_in_global_phase,
    -- nd_rank summary, niche occupancy, itp.
    extra_json                       TEXT,

    PRIMARY KEY (run_id, iteration),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- =========================================================
-- 7. Timingi etapów optymalizacji
-- =========================================================
CREATE TABLE IF NOT EXISTS optimization_timings (
    run_id           TEXT NOT NULL,
    algorithm_name   TEXT,
    stage_name       TEXT NOT NULL,
    wall_time_s      REAL NOT NULL CHECK (wall_time_s >= 0),
    cpu_time_s       REAL CHECK (cpu_time_s IS NULL OR cpu_time_s >= 0),
    timestamp_utc    TEXT,
    success          INTEGER CHECK (success IN (0,1)),
    PRIMARY KEY (run_id, stage_name),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- =========================================================
-- 8. Granice świata / scenariusza
-- =========================================================
CREATE TABLE IF NOT EXISTS world_boundaries (
    run_id           TEXT PRIMARY KEY,
    x_dimension      REAL,
    x_min_bound      REAL,
    x_max_bound      REAL,
    x_center         REAL,
    y_dimension      REAL,
    y_min_bound      REAL,
    y_max_bound      REAL,
    y_center         REAL,
    z_dimension      REAL,
    z_min_bound      REAL,
    z_max_bound      REAL,
    z_center         REAL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- =========================================================
-- 9. Przeszkody wygenerowane dla runu
-- Pojedynczy wpis = jedna przeszkoda statyczna w scenariuszu (cylinder
-- w forest / box w urban). Pozycja = środek bryły.
-- Źródło: `<run_dir>/generated_obstacles.csv` (`SimulationLogger.log_obstacles`).
--
-- Refaktor 2026-05-07: rozdzielona semantyka per shape_type.
-- Wcześniejsza schema (`radius, height, unused_dim`) była CYLINDER-centric;
-- BOX hackował przez przepisanie length→radius, width→unused_dim. Stąd
-- analyzer odczytując "radius" dla urban dostawał length budynku (15m),
-- a "unused_dim" zawierał width — całkowicie mylące.
-- =========================================================
CREATE TABLE IF NOT EXISTS generated_obstacles (
    obstacle_id       INTEGER PRIMARY KEY,
    run_id            TEXT NOT NULL,
    obstacle_index    INTEGER NOT NULL CHECK (obstacle_index >= 0),
    -- Pozycja środka [m] (wspólne dla obu kształtów)
    x                 REAL NOT NULL,
    y                 REAL NOT NULL,
    z                 REAL NOT NULL,
    -- Wymiary (semantyka zależna od `shape_type`):
    --   cylinder  → radius, height; length, width = NULL
    --   box       → length, width, height; radius = NULL
    shape_type        TEXT NOT NULL CHECK (shape_type IN ('cylinder', 'box')),
    radius            REAL CHECK (radius IS NULL OR radius >= 0),
    length            REAL CHECK (length IS NULL OR length >= 0),
    width             REAL CHECK (width IS NULL OR width >= 0),
    height            REAL CHECK (height IS NULL OR height >= 0),
    -- Constraint integralności: cylinder MUSI mieć radius, box MUSI mieć
    -- length+width; nigdy nie mieszamy.
    CHECK (
        (shape_type = 'cylinder' AND radius IS NOT NULL AND length IS NULL AND width IS NULL)
        OR
        (shape_type = 'box' AND radius IS NULL AND length IS NOT NULL AND width IS NOT NULL)
    ),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    UNIQUE (run_id, obstacle_index)
);

-- =========================================================
-- 10. Zdarzenia kolizji
-- =========================================================
CREATE TABLE IF NOT EXISTS collisions (
    collision_id      INTEGER PRIMARY KEY,
    run_id            TEXT NOT NULL,
    event_index       INTEGER NOT NULL CHECK (event_index >= 0),
    sim_time          REAL NOT NULL CHECK (sim_time >= 0),
    drone_id          INTEGER NOT NULL CHECK (drone_id >= 0),
    other_body_id     INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    UNIQUE (run_id, event_index),
    UNIQUE (run_id, sim_time, drone_id, other_body_id)
);

-- =========================================================
-- 11. Zdarzenia unikowe
-- Pojedynczy wpis = jedno per-drone zdarzenie reaktywnego unikania
-- (trigger / plan_built / no_plan / cooldown_skip / rejoin / collision).
-- Źródło: `evasion_events.csv` (`SimulationLogger.log_evasion_event`).
--
-- UWAGI SEMANTYCZNE:
-- * `ttc` może być z dwóch źródeł (patrz `ttc_source`):
--     - 'oracle_discrete': time-to-collision z deterministycznej predykcji
--       po splajnach (`SwarmFlightController._oracle_threat_lookahead`),
--       dyskretyzowane do kroku `dt` (lookahead `np.arange(0, 4.5, dt)`).
--       Stąd częste wartości "okrągłe" jak 3.4, 2.9 — to nie cap, tylko
--       granularność dyskretyzacji. NIE jest funkcją `dist_to_threat` —
--       opisuje *przyszłą* kolizję, nie odległość bieżącą.
--     - 'continuous': klasyczne `dist / closing_speed` (fallback gdy
--       oracle niedostępne). Liniowo zależne od `dist_to_threat`.
-- * `dist_to_threat` to **bieżąca** odległość w momencie zdarzenia,
--   `ttc` to **przyszły** czas do kolizji — nie są wprost skorelowane.
-- * `preferred_axis` (right/left/up/down lub NULL): oś wybrana przez
--   `AxisChooser` dla SingleArcDeflection. NULL gdy avoidance nie zwrócił
--   `axis_chosen` w `OptimizationResult.extra`. Notacja kierunkowa:
--   'right'/'left' to lateral perpendicular do drone forward XY (+/- 90°),
--   'up'/'down' to z-axis (+/-).
-- * Kolumna `astar_success` została **wycofana 2026-05-07** (algorytm A*
--   już nie istnieje; pole było semantycznie redundantne z `fallback_used`,
--   zawsze `astar_success = NOT fallback_used`).
-- =========================================================
CREATE TABLE IF NOT EXISTS evasion_events (
    event_id                 INTEGER PRIMARY KEY,
    run_id                   TEXT NOT NULL,
    event_index              INTEGER NOT NULL CHECK (event_index >= 0),
    sim_time                 REAL NOT NULL CHECK (sim_time >= 0),
    drone_id                 INTEGER NOT NULL CHECK (drone_id >= 0),
    event_type               TEXT NOT NULL,
    mode                     INTEGER,
    ttc                      REAL,
    ttc_source               TEXT CHECK (
        ttc_source IS NULL OR ttc_source IN ('oracle_discrete', 'continuous')
    ),
    dist_to_threat           REAL,
    threat_x                 REAL,
    threat_y                 REAL,
    threat_z                 REAL,
    threat_vx                REAL,
    threat_vy                REAL,
    threat_vz                REAL,
    rejoin_x                 REAL,
    rejoin_y                 REAL,
    rejoin_z                 REAL,
    rejoin_arc               REAL,
    preferred_axis           TEXT CHECK (
        preferred_axis IS NULL OR preferred_axis IN ('right', 'left', 'up', 'down')
    ),
    fallback_used            INTEGER CHECK (fallback_used IN (0,1)),
    pos_error_at_rejoin      REAL,
    vel_error_at_rejoin      REAL,
    planning_wall_time_s     REAL CHECK (planning_wall_time_s IS NULL OR planning_wall_time_s >= 0),
    notes                    TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    UNIQUE (run_id, event_index),
    UNIQUE (run_id, sim_time, drone_id, event_type, event_index)
);

-- =========================================================
-- 12. Punkty trajektorii offline
-- =========================================================
CREATE TABLE IF NOT EXISTS counted_trajectory_points (
    run_id          TEXT NOT NULL,
    drone_id        INTEGER NOT NULL CHECK (drone_id >= 0),
    waypoint_id     INTEGER NOT NULL CHECK (waypoint_id >= 0),
    x               REAL NOT NULL,
    y               REAL NOT NULL,
    z               REAL NOT NULL,
    PRIMARY KEY (run_id, drone_id, waypoint_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- =========================================================
-- 13. Próbki trajektorii rzeczywistej
-- =========================================================
CREATE TABLE IF NOT EXISTS trajectory_samples (
    run_id       TEXT NOT NULL,
    sample_index INTEGER NOT NULL CHECK (sample_index >= 0),
    sim_time     REAL NOT NULL CHECK (sim_time >= 0),
    drone_id     INTEGER NOT NULL CHECK (drone_id >= 0),
    x            REAL NOT NULL,
    y            REAL NOT NULL,
    z            REAL NOT NULL,
    roll         REAL,
    pitch        REAL,
    yaw          REAL,
    vx           REAL,
    vy           REAL,
    vz           REAL,
    PRIMARY KEY (run_id, drone_id, sample_index),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    UNIQUE (run_id, sim_time, drone_id)
);

-- =========================================================
-- 14. Metryki optymalizacji z optimization_history.h5
-- =========================================================
CREATE TABLE IF NOT EXISTS optimization_generation_stats (
    run_id           TEXT NOT NULL,
    generation       INTEGER NOT NULL CHECK (generation >= 0),
    source_name      TEXT NOT NULL,
    metric_name      TEXT NOT NULL,
    metric_value     REAL NOT NULL,
    PRIMARY KEY (run_id, generation, source_name, metric_name),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- =========================================================
-- 15. Metryki trajektorii
-- =========================================================
CREATE TABLE IF NOT EXISTS trajectory_metrics (
    run_id                      TEXT NOT NULL,
    source_name                 TEXT NOT NULL,
    uav_id                      INTEGER NOT NULL CHECK (uav_id >= 0),
    point_count                 INTEGER CHECK (point_count IS NULL OR point_count >= 0),
    path_length_2d              REAL,
    path_length_3d              REAL,
    min_altitude                REAL,
    max_altitude                REAL,
    mean_altitude               REAL,
    extra_json                  TEXT,
    PRIMARY KEY (run_id, source_name, uav_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- =========================================================
-- 16. (USUNIĘTE 2026-05-07) tabela `pareto_run_metrics`
-- 7 z 10 kolumn duplikowało `run_metrics` (decision_mode,
-- selected_solution_index, nondominated_count, feasible_nondominated_count,
-- hypervolume, igd_plus, reference_point_json). Pozostałe 3 pola
-- (epsilon_indicator, reference_set_id, indicator_config_json) to
-- YAGNI — przyszłe potencjalne pola, nigdy nieużywane.
-- Single source of truth dla MOO per-run = `run_metrics`.
-- =========================================================

-- =========================================================
-- 17. (USUNIĘTE 2026-05-07) tabela `metric_definitions`
-- Self-documenting metadata które nigdy nie były wypełniane ani
-- konsumowane. Semantyka metryk udokumentowana w `src/analysis/ETL_TABLES.md`
-- + komentarzach do `iteration_metrics` / `run_metrics` w tej schemie.
-- =========================================================

-- =========================================================
-- 18. Widoki analityczne
-- =========================================================
CREATE VIEW IF NOT EXISTS vw_run_summary AS
SELECT
    r.run_id,
    r.run_dir_name,
    r.source_path,
    r.optimizer_algo,
    r.avoidance_algo,
    r.environment,
    r.seed,
    r.algorithm_pair,
    r.aggregation_status,
    r.discovered_at,
    r.aggregated_at,
    CASE WHEN m.run_id IS NULL THEN 0 ELSE 1 END AS has_metrics,
    m.drone_count,
    m.success,
    m.final_objective,
    m.total_path_length_2d,
    m.total_path_length_3d,
    m.total_threat_cost,
    m.total_turn_penalty,
    m.collision_count,
    m.evasion_event_count,
    m.obstacle_count,
    m.best_iteration,
    -- Online metryki agregowane (notatki.md, 2026-05-04)
    m.min_inter_uav_distance_m,
    m.mean_inter_uav_distance_m,
    m.total_inter_uav_safety_violations,
    m.mean_energy_indicator,
    m.mean_smoothness_indicator,
    -- Offline F-vector best feasible (2026-05-05)
    m.final_objective_f1_trajectory,
    m.final_objective_f2_height_angle,
    m.total_coordination_cost
FROM runs r
LEFT JOIN run_metrics m
    ON m.run_id = r.run_id;

CREATE VIEW IF NOT EXISTS vw_seed_summary AS
SELECT
    environment,
    optimizer_algo,
    avoidance_algo,
    algorithm_pair,
    seed,
    COUNT(run_id)               AS run_count,
    AVG(final_objective)        AS mean_final_objective,
    MIN(final_objective)        AS best_final_objective,
    MAX(final_objective)        AS worst_final_objective,
    AVG(total_path_length_3d)   AS mean_path_length_3d,
    AVG(total_threat_cost)      AS mean_threat_cost,
    AVG(collision_count)        AS mean_collision_count,
    AVG(success)                AS success_rate
FROM vw_run_summary
WHERE has_metrics = 1
GROUP BY environment, optimizer_algo, avoidance_algo, algorithm_pair, seed;

CREATE VIEW IF NOT EXISTS vw_global_summary AS
SELECT
    environment,
    optimizer_algo,
    avoidance_algo,
    algorithm_pair,
    COUNT(run_id)               AS run_count,
    AVG(final_objective)        AS mean_final_objective,
    MIN(final_objective)        AS best_final_objective,
    MAX(final_objective)        AS worst_final_objective,
    AVG(total_path_length_3d)   AS mean_path_length_3d,
    AVG(total_threat_cost)      AS mean_threat_cost,
    AVG(collision_count)        AS mean_collision_count,
    AVG(success)                AS success_rate
FROM vw_run_summary
WHERE has_metrics = 1
GROUP BY environment, optimizer_algo, avoidance_algo, algorithm_pair;

-- =========================================================
-- 19. Indeksy
-- =========================================================
CREATE INDEX IF NOT EXISTS idx_runs_environment_seed
    ON runs(environment, seed);

CREATE INDEX IF NOT EXISTS idx_runs_optimizer_avoidance
    ON runs(optimizer_algo, avoidance_algo);

CREATE INDEX IF NOT EXISTS idx_runs_algorithm_pair_seed
    ON runs(algorithm_pair, seed);

CREATE INDEX IF NOT EXISTS idx_run_metrics_success
    ON run_metrics(success);

CREATE INDEX IF NOT EXISTS idx_iteration_metrics_run_iteration
    ON iteration_metrics(run_id, iteration);

CREATE INDEX IF NOT EXISTS idx_iteration_metrics_iteration
    ON iteration_metrics(iteration);

CREATE INDEX IF NOT EXISTS idx_optimization_timings_run
    ON optimization_timings(run_id);

CREATE INDEX IF NOT EXISTS idx_generated_obstacles_run
    ON generated_obstacles(run_id);

CREATE INDEX IF NOT EXISTS idx_collisions_run
    ON collisions(run_id);

CREATE INDEX IF NOT EXISTS idx_collisions_run_time
    ON collisions(run_id, sim_time);

CREATE INDEX IF NOT EXISTS idx_evasion_events_run
    ON evasion_events(run_id);

CREATE INDEX IF NOT EXISTS idx_evasion_events_run_time
    ON evasion_events(run_id, sim_time);

CREATE INDEX IF NOT EXISTS idx_evasion_events_run_drone_type
    ON evasion_events(run_id, drone_id, event_type);

CREATE INDEX IF NOT EXISTS idx_counted_trajectory_points_run_drone
    ON counted_trajectory_points(run_id, drone_id);

CREATE INDEX IF NOT EXISTS idx_trajectory_samples_run_drone_time
    ON trajectory_samples(run_id, drone_id, sim_time);

CREATE INDEX IF NOT EXISTS idx_opt_gen_stats_run_gen
    ON optimization_generation_stats(run_id, generation);

CREATE INDEX IF NOT EXISTS idx_opt_gen_stats_run_metric
    ON optimization_generation_stats(run_id, source_name, metric_name);

CREATE INDEX IF NOT EXISTS idx_trajectory_metrics_run_uav
    ON trajectory_metrics(run_id, uav_id);


-- =========================================================
-- 20. Surowe dane z online_optimization.csv
-- Reprezentuje pojedyncze wywołanie optymalizatora online
-- w celu uniknięcia przeszkody.
-- =========================================================
CREATE TABLE IF NOT EXISTS online_optimization_tasks (
    run_id                  TEXT NOT NULL,
    drone_id                INTEGER NOT NULL CHECK (drone_id >= 0),
    trigger_time            REAL NOT NULL CHECK (trigger_time >= 0),
    algorithm               TEXT NOT NULL,
    status                  TEXT NOT NULL,
    reason                  TEXT,
    best_fitness            REAL,
    evaluations_completed   INTEGER CHECK (evaluations_completed >= 0),
    generations_completed   INTEGER CHECK (generations_completed >= 0),
    wallclock_s             REAL NOT NULL CHECK (wallclock_s >= 0),
    time_budget_s           REAL NOT NULL CHECK (time_budget_s >= 0),
    -- `chosen_axis` ∈ {'right','left','up','down'} albo NULL gdy plan=None.
    -- Notacja kierunkowa zgodna z `AxisChooser._choose` w SingleArcDeflection:
    -- 'right'/'left' to lateral perpendicular do drone forward XY,
    -- 'up'/'down' to z-axis. NULL ≠ magic string "unknown".
    chosen_axis             TEXT CHECK (
        chosen_axis IS NULL OR chosen_axis IN ('right', 'left', 'up', 'down')
    ),
    plan_waypoints_json     TEXT,
    plan_total_duration_s   REAL,
    plan_arc_length_m       REAL,
    outcome                 TEXT,
    pos_err_at_rejoin_m     REAL,
    vel_err_at_rejoin_mps   REAL,
    time_to_rejoin_s        REAL,

    PRIMARY KEY (run_id, drone_id, trigger_time),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- =========================================================
-- 21. Surowe dane z convergance_traces.csv
-- Śledzenie zbieżności optymalizatora w czasie rzeczywistym.
-- =========================================================
CREATE TABLE IF NOT EXISTS online_convergence_traces (
    run_id          TEXT NOT NULL,
    drone_id        INTEGER NOT NULL CHECK (drone_id >= 0),
    trigger_time    REAL NOT NULL CHECK (trigger_time >= 0),
    algorithm       TEXT NOT NULL,
    generation      INTEGER NOT NULL CHECK (generation >= 0),
    best_fitness    REAL NOT NULL,
    
    PRIMARY KEY (run_id, drone_id, trigger_time, generation),
    FOREIGN KEY (run_id, drone_id, trigger_time) 
        REFERENCES online_optimization_tasks(run_id, drone_id, trigger_time) ON DELETE CASCADE
);

-- =========================================================
-- 22. Widok: Agregacja wydajności algorytmów online na poziomie RUNU
-- Pozwala sprawdzić, jak dany algorytm unikania poradził 
-- sobie w pojedynczej pełnej symulacji.
-- =========================================================
CREATE VIEW IF NOT EXISTS vw_run_online_summary AS
SELECT
    o.run_id,
    o.algorithm,
    r.environment,
    r.seed,
    COUNT(*) AS total_evasion_triggers,

    -- Statystyki czasu wykonania (Wallclock)
    AVG(o.wallclock_s) AS avg_wallclock_s,
    MAX(o.wallclock_s) AS max_wallclock_s,

    -- Wskaźniki niezawodności czasu rzeczywistego
    SUM(CASE WHEN o.reason = 'budget_exceeded_returned_best_so_far' THEN 1 ELSE 0 END) AS budget_exceeded_count,
    CAST(SUM(CASE WHEN o.reason = 'budget_exceeded_returned_best_so_far' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS budget_violation_rate,

    -- Statystyki procesu ewolucyjnego
    AVG(o.generations_completed) AS avg_generations_completed,
    AVG(o.evaluations_completed) AS avg_evaluations_completed,
    AVG(o.best_fitness) AS avg_best_fitness,

    -- Statystyki jakości powrotu na trajektorię
    SUM(CASE WHEN o.outcome = 'rejoined_ok' THEN 1 ELSE 0 END) AS successful_rejoins,
    AVG(o.pos_err_at_rejoin_m) AS avg_pos_err_m,
    AVG(o.vel_err_at_rejoin_mps) AS avg_vel_err_mps,
    AVG(o.time_to_rejoin_s) AS avg_time_to_rejoin_s,

    -- Inter-UAV / energy / smoothness (notatki.md, 2026-05-04). Każda
    -- z tych metryk jest per-run (nie zależy od `algorithm`), ale dorzucone
    -- do tej widoku, by mieć "wszystko o online performance" w 1 query.
    rm.min_inter_uav_distance_m,
    rm.mean_inter_uav_distance_m,
    rm.total_inter_uav_safety_violations,
    rm.mean_energy_indicator,
    rm.mean_smoothness_indicator

FROM online_optimization_tasks o
JOIN runs r ON o.run_id = r.run_id
LEFT JOIN run_metrics rm ON rm.run_id = o.run_id
GROUP BY o.run_id, o.algorithm, r.environment, r.seed;

-- =========================================================
-- 23. Widok: Globalna agregacja krzyżowa (Cross-Simulation)
-- Idealna do weryfikacji hipotez badawczych i tworzenia
-- tabel do publikacji. Agreguje wyniki różnych seedów dla 
-- pary (środowisko, algorytm).
-- =========================================================
CREATE VIEW IF NOT EXISTS vw_algo_cross_sim_comparison AS
SELECT 
    environment,
    algorithm,
    COUNT(DISTINCT run_id) AS runs_count,
    SUM(total_evasion_triggers) AS total_evasion_triggers_all_runs,
    
    -- Średni czas obliczeń napotkany przez algorytm (SWaP)
    AVG(avg_wallclock_s) AS mean_of_avg_wallclock_s,
    MAX(max_wallclock_s) AS absolute_max_wallclock_s,
    
    -- Jaki odsetek wywołań przekracza budżet czasu?
    AVG(budget_violation_rate) AS mean_budget_violation_rate,
    
    -- Skuteczność ewolucyjna
    AVG(avg_generations_completed) AS mean_generations_completed,
    AVG(avg_best_fitness) AS mean_best_fitness,
    
    -- Jakość lotu / dynamiki
    AVG(avg_pos_err_m) AS mean_pos_error_at_rejoin,
    AVG(avg_vel_err_mps) AS mean_vel_error_at_rejoin,
    
    -- Niezawodność (np. stosunek udanych powrotów do wszystkich prób)
    CAST(SUM(successful_rejoins) AS REAL) / SUM(total_evasion_triggers) AS overall_rejoin_success_rate,

    -- Inter-UAV / energy / smoothness (notatki.md, 2026-05-04)
    AVG(min_inter_uav_distance_m) AS mean_min_inter_uav_distance_m,
    SUM(total_inter_uav_safety_violations) AS total_inter_uav_safety_violations,
    AVG(mean_energy_indicator) AS mean_energy_indicator,
    AVG(mean_smoothness_indicator) AS mean_smoothness_indicator

FROM vw_run_online_summary
GROUP BY environment, algorithm;

-- =========================================================
-- Indeksy dla optymalizacji zapytań OLAP
-- =========================================================
CREATE INDEX IF NOT EXISTS idx_online_tasks_run_algo 
    ON online_optimization_tasks(run_id, algorithm);

CREATE INDEX IF NOT EXISTS idx_online_convergence_run_drone
    ON online_convergence_traces(run_id, drone_id, trigger_time);

CREATE INDEX IF NOT EXISTS idx_uav_online_metrics_run
    ON uav_online_metrics(run_id);

-- =========================================================
-- 24. Reference Pareto sets (cross-run merged fronts)
-- Per (environment, n_obj) zbiorczy non-dominated front zbudowany
-- z last-gen fronts wszystkich runów (algorytm × seed). Używany do
-- liczenia GD i IGD+ (oba wymagają reference set R).
-- Reference: Riquelme, Lücken & Baran (2015); Ishibuchi et al. (2015).
-- =========================================================
CREATE TABLE IF NOT EXISTS reference_pareto_sets (
    environment   TEXT NOT NULL,
    n_obj         INTEGER NOT NULL CHECK (n_obj > 0),
    point_idx     INTEGER NOT NULL CHECK (point_idx >= 0),
    objective_j   INTEGER NOT NULL CHECK (objective_j >= 0),
    value         REAL NOT NULL,
    PRIMARY KEY (environment, n_obj, point_idx, objective_j)
);

CREATE INDEX IF NOT EXISTS idx_reference_pareto_env_nobj
    ON reference_pareto_sets(environment, n_obj);

-- =========================================================
-- 25. Reference points (for HV computation; per (env, n_obj))
-- Reference point r* dla hypervolume — *worst-case* punkt w obj-space,
-- liczony jako nadir + ε·(nadir − ideal) z merged Pareto set R.
-- Reference: Ishibuchi, Imada, Setoguchi & Nojima (2018) "How to Specify
-- a Reference Point in Hypervolume Calculation for Fair Performance
-- Comparison", Evolutionary Computation 26(3):411–440.
--
-- Domyślnie ε=0.1; dla SOO algorytmów (msffoa/ooa/ssa) z weighted-sum
-- skalaryzacją populacja zwykle skupia się w wąskim regionie obj-space,
-- stąd HV(SOO) << HV(NSGA-III) — to oczekiwany efekt do raportowania.
-- =========================================================
-- Refaktor 2026-05-07 (Kamień 2): dodana kolumna `ideal_value` =
-- min(R, axis=0) per j; potrzebna do `hypervolume_normalized = HV / Π(r* − ideal)`
-- (Riquelme 2015 §3.6) bez ponownego skanowania `reference_pareto_sets`.
CREATE TABLE IF NOT EXISTS reference_points (
    environment   TEXT NOT NULL,
    n_obj         INTEGER NOT NULL CHECK (n_obj > 0),
    objective_j   INTEGER NOT NULL CHECK (objective_j >= 0),
    value         REAL NOT NULL,
    ideal_value   REAL,
    margin        REAL NOT NULL DEFAULT 0.1
                  CHECK (margin >= 0),
    method        TEXT NOT NULL DEFAULT 'nadir_plus_eps_range',
    PRIMARY KEY (environment, n_obj, objective_j)
);

CREATE INDEX IF NOT EXISTS idx_reference_points_env_nobj
    ON reference_points(environment, n_obj);


COMMIT;