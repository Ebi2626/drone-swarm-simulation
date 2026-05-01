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
CREATE TABLE IF NOT EXISTS runs (
    run_id               TEXT PRIMARY KEY,
    run_dir_name         TEXT NOT NULL UNIQUE,
    source_path          TEXT NOT NULL,
    optimizer_algo       TEXT NOT NULL,
    avoidance_algo       TEXT NOT NULL,
    environment          TEXT NOT NULL,
    seed                 INTEGER NOT NULL,
    algorithm_pair       TEXT NOT NULL,
    aggregation_status   TEXT NOT NULL DEFAULT 'discovered'
                         CHECK (aggregation_status IN ('discovered', 'aggregated', 'failed', 'partial')),
    aggregation_error    TEXT,
    discovered_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    aggregated_at        TEXT,
    notes                TEXT,
    run_config_json      TEXT,
    UNIQUE (optimizer_algo, environment, avoidance_algo, seed)
);

-- =========================================================
-- 3. Rejestr plików źródłowych znalezionych w runie
-- =========================================================
CREATE TABLE IF NOT EXISTS run_files (
    run_id           TEXT NOT NULL,
    file_role        TEXT NOT NULL,
    relative_path    TEXT NOT NULL,
    file_format      TEXT,
    exists_flag      INTEGER NOT NULL DEFAULT 1 CHECK (exists_flag IN (0,1)),
    size_bytes       INTEGER,
    row_count        INTEGER,
    checksum         TEXT,
    modified_at      TEXT,
    extra_json       TEXT,
    PRIMARY KEY (run_id, file_role),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- =========================================================
-- 4. Metryki końcowe per run
-- Tu trafiają dane do porównań algorytmów
-- =========================================================
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id                      TEXT PRIMARY KEY,
    drone_count                 INTEGER,
    success                     INTEGER NOT NULL DEFAULT 1 CHECK (success IN (0,1)),
    final_objective             REAL,
    total_path_length_2d        REAL,
    total_path_length_3d        REAL,
    total_energy_cost           REAL,
    total_smoothness_cost       REAL,
    total_threat_cost           REAL,
    total_altitude_cost         REAL,
    total_terrain_penalty       REAL,
    total_turn_penalty          REAL,
    total_climb_penalty         REAL,
    total_collision_penalty     REAL,
    collision_count             INTEGER NOT NULL DEFAULT 0,
    evasion_event_count         INTEGER NOT NULL DEFAULT 0,
    obstacle_count              INTEGER NOT NULL DEFAULT 0,
    best_iteration              INTEGER,
    objective_components_json   TEXT,
    summary_json                TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- =========================================================
-- 5. Metryki per UAV w obrębie runu
-- =========================================================
CREATE TABLE IF NOT EXISTS uav_metrics (
    run_id                      TEXT NOT NULL,
    uav_id                      INTEGER NOT NULL,
    success                     INTEGER CHECK (success IN (0,1)),
    final_objective             REAL,
    path_length_2d              REAL,
    path_length_3d              REAL,
    energy_cost                 REAL,
    smoothness_cost             REAL,
    threat_cost                 REAL,
    altitude_cost               REAL,
    terrain_penalty             REAL,
    turn_penalty                REAL,
    climb_penalty               REAL,
    collision_penalty           REAL,
    collision_count             INTEGER NOT NULL DEFAULT 0,
    evasion_event_count         INTEGER NOT NULL DEFAULT 0,
    extra_json                  TEXT,
    PRIMARY KEY (run_id, uav_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- =========================================================
-- 6. Historia optymalizacji / zbieżność
-- =========================================================
CREATE TABLE IF NOT EXISTS iteration_metrics (
    run_id                      TEXT NOT NULL,
    iteration                   INTEGER NOT NULL,
    best_so_far                 REAL,
    current_best                REAL,
    current_mean                REAL,
    current_std                 REAL,
    current_worst               REAL,
    feasible_solutions          INTEGER,
    diversity_metric            REAL,
    elapsed_s                   REAL,
    extra_json                  TEXT,
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
    wall_time_s      REAL NOT NULL,
    cpu_time_s       REAL,
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
-- =========================================================
CREATE TABLE IF NOT EXISTS generated_obstacles (
    obstacle_id       INTEGER PRIMARY KEY,
    run_id            TEXT NOT NULL,
    obstacle_index    INTEGER NOT NULL,
    x                 REAL NOT NULL,
    y                 REAL NOT NULL,
    z                 REAL NOT NULL,
    radius            REAL,
    height            REAL,
    unused_dim        REAL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    UNIQUE (run_id, obstacle_index)
);

-- =========================================================
-- 10. Zdarzenia kolizji
-- =========================================================
CREATE TABLE IF NOT EXISTS collisions (
    collision_id      INTEGER PRIMARY KEY,
    run_id            TEXT NOT NULL,
    event_index       INTEGER,
    sim_time          REAL NOT NULL,
    drone_id          INTEGER NOT NULL,
    other_body_id     INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    UNIQUE (run_id, sim_time, drone_id, other_body_id)
);
-- =========================================================
-- 11. Zdarzenia unikowe
-- =========================================================
CREATE TABLE IF NOT EXISTS evasion_events (
    event_id                 INTEGER PRIMARY KEY,
    run_id                   TEXT NOT NULL,
    event_index              INTEGER,

    sim_time                 REAL NOT NULL,
    drone_id                 INTEGER NOT NULL,
    event_type               TEXT NOT NULL,
    mode                     INTEGER,

    ttc                      REAL,
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

    astar_success            INTEGER CHECK (astar_success IN (0,1)),
    fallback_used            INTEGER CHECK (fallback_used IN (0,1)),

    pos_error_at_rejoin      REAL,
    vel_error_at_rejoin      REAL,
    planning_wall_time_s     REAL,
    notes                    TEXT,

    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    UNIQUE (run_id, sim_time, drone_id, event_type, event_index)
);

-- =========================================================
-- 12. Punkty trajektorii
-- Umożliwia późniejsze przeliczenia metryk i wykresy tras
-- =========================================================
CREATE TABLE IF NOT EXISTS counted_trajectory_points (
    run_id          TEXT NOT NULL,
    drone_id        INTEGER NOT NULL,
    waypoint_id     INTEGER NOT NULL,
    x               REAL NOT NULL,
    y               REAL NOT NULL,
    z               REAL NOT NULL,
    PRIMARY KEY (run_id, drone_id, waypoint_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- =========================================================
-- 13. Rzeczywiste punkty trajektorii
-- Pobierane z silnika symulacji w określonym interwale
-- =========================================================
CREATE TABLE IF NOT EXISTS trajectory_samples (
    run_id       TEXT NOT NULL,
    sim_time     REAL NOT NULL,
    drone_id     INTEGER NOT NULL,
    x            REAL NOT NULL,
    y            REAL NOT NULL,
    z            REAL NOT NULL,
    roll         REAL,
    pitch        REAL,
    yaw          REAL,
    vx           REAL,
    vy           REAL,
    vz           REAL,
    PRIMARY KEY (run_id, sim_time, drone_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);



CREATE TABLE IF NOT EXISTS trajectory_metrics (
    run_id                      TEXT NOT NULL,
    source_name                 TEXT NOT NULL,
    uav_id                      INTEGER NOT NULL,
    point_count                 INTEGER,
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
-- 14. Widoki analityczne
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
    m.drone_count,
    m.success,
    m.final_objective,
    m.total_path_length_2d,
    m.total_path_length_3d,
    m.total_energy_cost,
    m.total_smoothness_cost,
    m.total_threat_cost,
    m.total_altitude_cost,
    m.total_terrain_penalty,
    m.total_turn_penalty,
    m.total_climb_penalty,
    m.total_collision_penalty,
    m.collision_count,
    m.evasion_event_count,
    m.obstacle_count,
    m.best_iteration
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
    COUNT(*)                    AS run_count,
    AVG(final_objective)        AS mean_final_objective,
    MIN(final_objective)        AS best_final_objective,
    MAX(final_objective)        AS worst_final_objective,
    AVG(total_path_length_3d)   AS mean_path_length_3d,
    AVG(total_energy_cost)      AS mean_energy_cost,
    AVG(total_collision_penalty) AS mean_collision_penalty,
    AVG(collision_count)        AS mean_collision_count,
    AVG(success)                AS success_rate
FROM vw_run_summary
GROUP BY environment, optimizer_algo, avoidance_algo, algorithm_pair, seed;

CREATE VIEW IF NOT EXISTS vw_global_summary AS
SELECT
    environment,
    optimizer_algo,
    avoidance_algo,
    algorithm_pair,
    COUNT(*)                    AS run_count,
    AVG(final_objective)        AS mean_final_objective,
    MIN(final_objective)        AS best_final_objective,
    MAX(final_objective)        AS worst_final_objective,
    AVG(total_path_length_3d)   AS mean_path_length_3d,
    AVG(total_energy_cost)      AS mean_energy_cost,
    AVG(total_collision_penalty) AS mean_collision_penalty,
    AVG(collision_count)        AS mean_collision_count,
    AVG(success)                AS success_rate
FROM vw_run_summary
GROUP BY environment, optimizer_algo, avoidance_algo, algorithm_pair;

-- =========================================================
-- 15. Indeksy
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

CREATE INDEX IF NOT EXISTS idx_evasion_events_run
    ON evasion_events(run_id);

CREATE INDEX IF NOT EXISTS idx_evasion_events_run_time
    ON evasion_events(run_id, sim_time);

CREATE INDEX IF NOT EXISTS idx_evasion_events_run_drone_type
    ON evasion_events(run_id, drone_id, event_type);


CREATE INDEX IF NOT EXISTS idx_trajectory_metrics_run_uav
    ON trajectory_metrics(run_id, uav_id);

CREATE INDEX IF NOT EXISTS idx_counted_trajectory_points_run_drone
    ON counted_trajectory_points(run_id, drone_id);

CREATE INDEX IF NOT EXISTS idx_collisions_run_time
    ON collisions(run_id, sim_time);

CREATE INDEX IF NOT EXISTS idx_trajectory_samples_run_drone_time
    ON trajectory_samples(run_id, drone_id, sim_time);

COMMIT;