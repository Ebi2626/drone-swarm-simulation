import logging
from pathlib import Path

import h5py
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)

class ExperimentAggregator:
    """
    Agreguje wyniki z drzewa eksperymentu utworzonego przez prepare_experiment.py
    oraz kompiluje dodatkowe metryki takie jak działanie LiDARu oraz spójność roju.
    """
    
    def __init__(self, experiment_root: str | Path):
        self.experiment_root = Path(experiment_root)
        if not self.experiment_root.exists():
            raise FileNotFoundError(f"Katalog eksperymentu nie istnieje: {self.experiment_root}")

    def _discover_run_dirs(self) -> list[Path]:
        run_dirs = []
        search_paths = [self.experiment_root, self.experiment_root / "tmp"]
        for search_path in search_paths:
            if not search_path.exists():
                continue
            for path in search_path.iterdir():
                if path.is_dir() and "_seed" in path.name:
                    run_dirs.append(path)
        logger.info(f"Znaleziono {len(run_dirs)} folderów z pojedynczymi runami.")
        return sorted(run_dirs)

    def _parse_run_meta(self, run_dir: Path) -> dict:
        exp_id = self.experiment_root.name
        dirname = run_dir.name
        
        prefix = f"{exp_id}_"
        if dirname.startswith(prefix):
            remainder = dirname[len(prefix):]
        else:
            remainder = dirname
            
        parts = remainder.split("_")
        if len(parts) < 4 or not parts[-1].startswith("seed"):
            raise ValueError(f"Nie można poprawnie sparsować nazwy: {dirname}")
            
        seed_str = parts[-1].replace("seed", "")
        avoidance = parts[-2]
        environment = parts[-3]
        optimizer = "_".join(parts[:-3]) 
        
        return {
            "experiment_id": exp_id,
            "optimizer": optimizer,
            "environment": environment,
            "avoidance": avoidance,
            "seed": int(seed_str),
            "run_dir": str(run_dir)
        }

    def _compute_path_lengths(self, traj: pd.DataFrame) -> tuple[float, float]:
        """Oblicza sumaryczną i średnią długość ścieżek z szeregów czasowych trajektorii."""
        total = 0.0
        per_drone = []

        for _, g in traj.sort_values(["drone_id", "time"]).groupby("drone_id"):
            xyz = g[["x", "y", "z"]].to_numpy()
            if len(xyz) < 2:
                per_drone.append(0.0)
                continue
            seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
            L = float(seg.sum())
            per_drone.append(L)
            total += L

        mean_per_drone = float(np.mean(per_drone)) if per_drone else 0.0
        return total, mean_per_drone

    def _compute_inter_drone_distance(self, traj: pd.DataFrame) -> float:
        """
        Oblicza średni dystans Euklidesowy pomiędzy wszystkimi parami dronów
        uśredniony w czasie trwania całego lotu. (Wymiar: Odległość)
        """
        if traj.empty or traj["drone_id"].nunique() < 2:
            return np.nan
        
        distances_per_timestep = []
        for time_val, group in traj.groupby("time"):
            xyz = group[["x", "y", "z"]].to_numpy()
            if len(xyz) < 2:
                continue
            # pdist zwraca płaską tablicę odległości pomiędzy każdą unikalną parą punktów
            distances = pdist(xyz, metric='euclidean')
            distances_per_timestep.append(np.mean(distances))
            
        if not distances_per_timestep:
            return np.nan
            
        return float(np.mean(distances_per_timestep))

    def _extract_lidar_metrics(self, h5_path: Path) -> dict:
        """
        Pobiera zagregowane dane statystyczne ze śladów czujnika LiDAR.
        """
        if not h5_path.exists():
            return {"lidar_total_hits": 0, "lidar_mean_distance": np.nan}
        
        try:
            with h5py.File(h5_path, "r") as f:
                # To jest część wymagająca dopasowania - zakłada, 
                # że istnieje zbiór np. f['hits'] lub skanowanie po pierwszym kluczu.
                dataset_key = list(f.keys())[0]  
                data = f[dataset_key][:]
                
                # Przykład: Zliczenie niezerowych odczytów jako trafień. 
                hits = int(np.count_nonzero(data))
                # Przykład: Średnia wartość trafień (jeśli HDF5 przechowuje odległości do uderzenia)
                mean_dist = float(np.mean(data[data > 0])) if hits > 0 else np.nan
                
                return {
                    "lidar_total_hits": hits,
                    "lidar_mean_distance": mean_dist
                }
        except Exception as e:
            logger.warning(f"Błąd odczytu LiDAR dla {h5_path}: {e}")
            return {"lidar_total_hits": 0, "lidar_mean_distance": np.nan}

    def _extract_scalar_row(self, run_dir: Path) -> dict:
        meta = self._parse_run_meta(run_dir)

        p_collisions = run_dir / "collisions.csv"
        p_timings = run_dir / "optimization_timings.csv"
        p_traj = run_dir / "trajectories.csv"
        p_evasion = run_dir / "evasion_events.csv"
        p_obstacles = run_dir / "generated_obstacles.csv"
        p_lidar = run_dir / "lidar_hits.h5"

        collisions = pd.read_csv(p_collisions) if p_collisions.exists() else pd.DataFrame()
        timings = pd.read_csv(p_timings) if p_timings.exists() else pd.DataFrame()
        traj = pd.read_csv(p_traj) if p_traj.exists() else pd.DataFrame()
        evasion = pd.read_csv(p_evasion) if p_evasion.exists() else pd.DataFrame()
        obstacles = pd.read_csv(p_obstacles) if p_obstacles.exists() else pd.DataFrame()

        total_opt_wall = np.nan
        total_opt_cpu = np.nan
        optimization_success = False
        
        if not timings.empty and "stage_name" in timings.columns:
            total_opt = timings[timings["stage_name"] == "total_optimization"]
            if not total_opt.empty:
                total_opt_wall = float(total_opt["wall_time_s"].iloc[0])
                total_opt_cpu = float(total_opt["cpu_time_s"].iloc[0])
                optimization_success = bool(timings["success"].astype(str).eq("True").all())

        # --- Evasion events metrics ---
        # Obsługa różnych modeli algorytmów uniku w tym `avoidance="none"`
        avoidance_sr, fallback_r, mean_plan_time, max_plan_time, plans_count = np.nan, np.nan, np.nan, np.nan, 0
        rejoin_count = 0
        
        if not evasion.empty and "event_type" in evasion.columns:
            plans = evasion[evasion["event_type"] == "plan_built"].copy()
            rejoin_count = int((evasion["event_type"] == "rejoin").sum())
            if not plans.empty:
                plans_count = len(plans)
                
                # Zamiast "astar_success", sprawdzamy "success" ogólnie, 
                # albo fallbackujemy do astar_success ze względu na strukturę pliku
                success_col = "astar_success" if "astar_success" in plans.columns else "success"
                
                if success_col in plans.columns:
                    plans[success_col] = plans[success_col].astype(str).map({"True": True, "False": False})
                    avoidance_sr = float(plans[success_col].mean())
                
                if "fallback_used" in plans.columns:
                    plans["fallback_used"] = plans["fallback_used"].astype(str).map({"True": True, "False": False})
                    fallback_r = float(plans["fallback_used"].mean())
                    
                mean_plan_time = float(plans["planning_wall_time_s"].mean())
                max_plan_time = float(plans["planning_wall_time_s"].max())

        # Trajektorie
        total_path_length, mean_path_length = 0.0, 0.0
        inter_drone_distance = np.nan
        mission_end_time = np.nan
        drone_count = 0
        
        if not traj.empty:
            total_path_length, mean_path_length = self._compute_path_lengths(traj)
            inter_drone_distance = self._compute_inter_drone_distance(traj)
            mission_end_time = float(traj["time"].max())
            drone_count = int(traj["drone_id"].nunique())

        lidar_metrics = self._extract_lidar_metrics(p_lidar)

        row = {
            **meta,
            "collision_count": int(len(collisions)),
            "collision_drone_count": int(collisions["drone_id"].nunique()) if not collisions.empty else 0,
            "optimization_wall_time_s": total_opt_wall,
            "optimization_cpu_time_s": total_opt_cpu,
            "optimization_success": optimization_success,
            "mission_end_time_s": mission_end_time,
            "drone_count": drone_count,
            "trajectory_samples": int(len(traj)),
            "total_path_length": total_path_length,
            "mean_path_length_per_drone": mean_path_length,
            "mean_inter_drone_distance": inter_drone_distance,
            "planning_events": plans_count,
            "avoidance_success_rate": avoidance_sr,
            "fallback_rate": fallback_r,
            "mean_planning_wall_time_s": mean_plan_time,
            "max_planning_wall_time_s": max_plan_time,
            "rejoin_events": rejoin_count,
            "obstacle_count": int(len(obstacles)),
            **lidar_metrics
        }
        return row

    def build_master_metrics(self) -> pd.DataFrame:
        rows = []
        run_dirs = self._discover_run_dirs()
        
        if not run_dirs:
            logger.warning(f"Brak runów w strukturze eksperymentu: {self.experiment_root}")
            return pd.DataFrame()

        for run_dir in run_dirs:
            try:
                rows.append(self._extract_scalar_row(run_dir))
            except Exception as e:
                import traceback
                logger.error(f"Błąd ETL dla katalogu {run_dir.name}: {str(e)}\n{traceback.format_exc()}")
                rows.append({
                    "run_dir": str(run_dir),
                    "etl_error": str(e),
                })

        df = pd.DataFrame(rows)
        
        for col in ["optimizer", "environment", "avoidance"]:
            if col in df.columns:
                df[col] = df[col].astype("category")
                
        out = self.experiment_root / "master_metrics.parquet"
        df.to_parquet(out, index=False, compression="snappy")
        logger.info(f"Zapisano dane skalarne do {out} (zliczono {len(df)} uruchomień)")
        return df