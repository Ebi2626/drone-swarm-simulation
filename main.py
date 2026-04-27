# flake8: noqa: E402
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import sys
import time
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from src.runner.ExperimentDataStrategy import ExperimentDataStrategy
from src.runner.GenerationDataStrategy import GenerationDataStrategy
from src.runner.ReplayDataStrategy import ReplayDataStrategy

from src.utils.input_utils import InputHandler, CommandType
from src.utils.SimulationLogger import SimulationLogger
from src.utils.RunRegistry import RunRegistry
from src.utils.pybullet_utils import update_camera_position
from src.algorithms.TrajectoryFollowingAlgorithm import TrajectoryFollowingAlgorithm
from src.environments.obstacles.ObstacleShape import ObstacleShape

import pybullet as p


class ExperimentRunner:
    def __init__(self, cfg: DictConfig, data_strategy: ExperimentDataStrategy):
        self.cfg = cfg
        self.data_strategy = data_strategy

        # Parametry symulacji i drona
        self.drone_model = cfg.simulation.get("drone_model", "CF2X")
        self.phyics = cfg.simulation.get("physics", "PYB")
        self.num_drones = cfg.environment.params.get("num_drones")
        self.ctrl_freq = cfg.simulation.get("ctrl_freq", 48)
        self.pyb_freq = cfg.simulation.get("pyb_freq", 240)
        self.sim_speed_multiplier = cfg.simulation.get("sim_speed_multiplier", 5.0)

        # --- Dynamiczne przeszkody ---
        self.use_dynamic_obstacles = cfg.simulation.get("dynamic_obstacles", False)
        self.num_dynamic_obstacles = (
            self.num_drones if self.use_dynamic_obstacles else 0
        )
        self.total_agents = self.num_drones + self.num_dynamic_obstacles

        # Parametry wizualizacji
        self.tracked_drone_id = cfg.visualization.get("tracked_drone_id", 0)
        self.show_lidar_rays = cfg.visualization.get("show_lidar_rays", False)
        self.lidar_draw_interval = cfg.visualization.get("lidar_draw_interval", 5)
        if self.tracked_drone_id >= self.num_drones:
            self.tracked_drone_id = 0

        # Zmienne środowiskowe (nadpisane przez strategię lub initialize_world)
        self.environemnt = None
        self.world_data = None
        self.obstacles_data = None
        self.drones_trajectories = None
        self.trajectory_controller = None
        self.dynamic_obstacle_trajectory_controller = None
        self.logger = None
        self.input_handler = None

        # Parametry środowiska
        self.initial_rpys = cfg.environment.get("initial_rpys")
        self.start_positions = np.array(
            cfg.environment.get("initial_xyzs"), dtype=np.float64
        )
        self.end_positions = np.array(cfg.environment.get("end_xyzs"), dtype=np.float64)
        self.placement_strategy_name = cfg.environment.params.get("placement_strategy")
        self.ground_position = cfg.environment.params.get("ground_position")
        self.track_length = cfg.environment.params.get("track_length")
        self.track_width = cfg.environment.params.get("track_width")
        self.track_height = cfg.environment.params.get("track_height")
        self.shape_type: ObstacleShape = cfg.environment.params.get("shape_type")
        self.obstacles_number = cfg.environment.params.get("obstacles_number")
        self.obstacle_width = cfg.environment.params.get("obstacle_width")
        self.obstacle_height = cfg.environment.params.get("obstacle_height")
        self.obstacle_length = cfg.environment.params.get("obstacle_length")
        self.safe_radius = cfg.environment.params.get("safe_radius")

        # Liczba próbek wynikowej dense trajektorii (B-Spline po
        # post-processingu) — NIEZALEŻNA od `n_inner_waypoints`, czyli liczby
        # wewnętrznych węzłów kontrolnych optymalizowanych przez metaheurystykę.
        # Wcześniej oba pola dzieliły wartość, więc kontroler dostawał ~28
        # punktów na trajektorię ~600m → ~21 m między próbkami, za rzadko dla
        # PID 48 Hz (patrz plan.md, Krok 4). Default 200 daje ~3 m / próbkę.
        self.number_of_waypoints = cfg.simulation.get("dense_samples", 200)

    # =========================================================================
    # PRZYGOTOWANIE EKSPERYMENTU
    # =========================================================================

    def prepare_experiment(self):
        """Przygotowuje wszystkie wymagane dane i komponenty dla symulacji."""
        if self.cfg.simulation.gui:
            self.input_handler = InputHandler(self.num_drones)

        if self.cfg.logging.enabled:
            output_dir = self.cfg.logging.get("output_dir")
            if output_dir is None:
                output_dir = HydraConfig.get().runtime.output_dir
            self.logger = SimulationLogger(
                output_dir=output_dir,
                log_freq=self.cfg.logging.log_freq,
                ctrl_freq=self.ctrl_freq,
                num_drones=self.num_drones,
            )

        # Delegacja przygotowania danych (generacja lub wczytanie z CSV)
        self.data_strategy.prepare_data(self)

        self._init_trajectory_following_algorithm()

    def _init_trajectory_following_algorithm(self):
        """Inicjalizuje kontrolery trajektorii dla dronów i (opcjonalnie) przeszkód."""
        shared_params = {
            "ctrl_freq": self.ctrl_freq,
            "collision_radius": 0.5,
            "hover_duration": self.cfg.simulation.get("hover_duration", 3.0),
            "cruise_speed": self.cfg.optimizer.algorithm_params.get(
                "cruise_speed", 8.0
            ),
            "max_accel": self.cfg.optimizer.algorithm_params.get("max_accel", 2.0),
        }

        avoidance_algo = None
        if "avoidance" in self.cfg and self.cfg.avoidance.get("enable", False):
            avoidance_algo = instantiate(self.cfg.avoidance)
            print(f"[INFO] Aktywowano algorytm omijania: {avoidance_algo.name}")

        # Kontroler dla głównego roju
        self.trajectory_controller = TrajectoryFollowingAlgorithm(
            parent=self,
            num_drones=self.num_drones,
            is_obstacle=False,
            avoidance_algorithm=avoidance_algo,
            params={
                **shared_params,
                "acceptance_radius": 0.2,
                "enable_avoidance": avoidance_algo is not None,
            },
        )

        # Kontroler dla dynamicznych przeszkód (jeśli włączone w konfiguracji)
        if self.use_dynamic_obstacles:
            self.dynamic_obstacle_trajectory_controller = TrajectoryFollowingAlgorithm(
                parent=self,
                num_drones=self.num_dynamic_obstacles,
                is_obstacle=True,
                avoidance_algorithm=None,
                params={
                    **shared_params,
                    "acceptance_radius": 0.5,
                    "enable_avoidance": False,
                },
            )

    # =========================================================================
    # INICJALIZACJA ŚWIATA
    # =========================================================================

    def initialize_world(self):
        """Tworzy środowisko PyBullet z odpowiednią liczbą agentów."""
        if self.use_dynamic_obstacles:
            all_initial_xyzs = np.vstack((self.start_positions, self.end_positions))
            all_end_xyzs = np.vstack((self.end_positions, self.start_positions))
            if self.initial_rpys is not None:
                all_initial_rpys = np.vstack((self.initial_rpys, self.initial_rpys))
            else:
                all_initial_rpys = None
        else:
            all_initial_xyzs = self.start_positions
            all_end_xyzs = self.end_positions
            all_initial_rpys = self.initial_rpys

        self.environemnt = instantiate(
            self.cfg.environment,
            world_data=self.world_data,
            obstacles_data=self.obstacles_data,
            drone_model=self.drone_model,
            physics=self.phyics,
            num_drones=self.total_agents,
            initial_xyzs=all_initial_xyzs,
            end_xyzs=all_end_xyzs,
            initial_rpys=all_initial_rpys,
            gui=self.cfg.simulation.gui,
            ctrl_freq=self.ctrl_freq,
            pyb_freq=self.pyb_freq,
            primary_num_drones=self.num_drones,
            dynamic_obstacles_enabled=self.use_dynamic_obstacles,
            num_dynamic_obstacles=self.num_dynamic_obstacles,
        )

    # =========================================================================
    # POMOCNICZE METODY PĘTLI
    # =========================================================================

    def _update_camera(self, drone_states: list):
        """Aktualizuje pozycję kamery podążając za wybranym dronem."""
        if not self.cfg.visualization.camera_follow:
            return
        update_camera_position(
            drone_state=drone_states[self.tracked_drone_id],
            distance=self.cfg.visualization.camera_distance,
            yaw_offset=self.cfg.visualization.camera_yaw,
            pitch=self.cfg.visualization.camera_pitch,
        )

    def _init_active_drones(self):
        """Inicjalizuje zbiór aktywnych dronów głównego roju."""
        self.active_drones = set(range(self.num_drones))
        self.acceptance_radius = self.trajectory_controller.params.get(
            "acceptance_radius", 0.2
        )

    def _process_collisions(self, sim_time: float, current_step: int):
        """Przetwarza kolizje — loguje i usuwa z aktywnych tylko drony głównego roju."""
        for d_id, o_id in self.environemnt.get_detailed_collisions():
            if d_id >= self.num_drones:
                continue
            if self.logger:
                self.logger.log_collision(sim_time, d_id, o_id)
            if d_id in self.active_drones:
                self.active_drones.remove(d_id)
                print(
                    f"[INFO] Dron {d_id} uległ kolizji w czasie {sim_time:.2f}s (krok {current_step})."
                )

    def _process_arrivals(self, drone_states: list, sim_time: float):
        """Sprawdza czy drony głównego roju dotarły do celu."""
        try:
            radius = float(np.squeeze(self.acceptance_radius))
        except (TypeError, ValueError):
            raise ValueError(f"Błędny promień akceptacji: {self.acceptance_radius}")

        for d_id in list(self.active_drones):
            pos = np.array(drone_states[d_id][0:3]).flatten()
            target = np.array(self.end_positions[d_id]).flatten()

            if pos.size == 0 or target.size == 0:
                continue

            if float(np.linalg.norm(pos - target)) <= radius:
                self.active_drones.remove(d_id)
                print(f"[INFO] Dron {d_id} osiągnął cel w czasie {sim_time:.2f}s.")

    def _get_all_drone_states(self) -> list:
        """Zwraca stany wszystkich total_agents agentów ze środowiska."""
        return [
            self.environemnt._getDroneStateVector(d) for d in range(self.total_agents)
        ]

    def _split_states(self, all_states: list) -> tuple[list, list]:
        """Rozdziela stany wszystkich agentów na drony i przeszkody."""
        drone_states = all_states[: self.num_drones]
        obstacle_states = (
            all_states[self.num_drones :] if self.use_dynamic_obstacles else []
        )
        return drone_states, obstacle_states

    def _merge_actions(
        self, drone_actions: np.ndarray, obstacle_actions: np.ndarray | None
    ) -> np.ndarray:
        """Scala akcje dronów i przeszkód w jedną tablicę dla env.step()."""
        if obstacle_actions is None:
            return drone_actions
        return np.vstack((drone_actions, obstacle_actions))

    # =========================================================================
    # GŁÓWNA PĘTLA SYMULACJI
    # =========================================================================

    def run(self):
        print("Running experiment...")
        self.initialize_world()

        self.trajectory_controller.init_lidars(self.environemnt.CLIENT)

        is_running = not self.cfg.simulation.gui
        current_step = 0
        max_steps = int(self.cfg.simulation.duration_sec * self.ctrl_freq)
        progress_interval = max(1, max_steps // 10)

        self._init_active_drones()
        start_real_time = time.time()
        print(
            f"[DEBUG] Start symulacji na {max_steps} kroków "
            f"({'z' if self.use_dynamic_obstacles else 'bez'} dynamicznych przeszkód)."
        )

        while current_step < max_steps:
            loop_start = time.time()

            if self.cfg.simulation.gui:
                cmd = self.input_handler.get_command()
                if cmd:
                    if cmd.type == CommandType.TOGGLE_SIMULATION:
                        is_running = not is_running
                    elif cmd.type == CommandType.SWITCH_DRONE_CAMERA:
                        self.tracked_drone_id = cmd.payload
                    elif cmd.type == CommandType.TOGGLE_LIDAR_RAYS:
                        self.show_lidar_rays = not self.show_lidar_rays
                        if not self.show_lidar_rays and hasattr(
                            self.trajectory_controller, "clear_lidar_rays"
                        ):
                            self.trajectory_controller.clear_lidar_rays()

            if is_running:
                sim_time = current_step / self.ctrl_freq

                all_pre_states = self._get_all_drone_states()
                drone_pre_states, obstacle_pre_states = self._split_states(
                    all_pre_states
                )

                drone_actions = self.trajectory_controller.compute_actions(
                    drone_pre_states, current_time=sim_time
                )

                obstacle_actions = None
                if self.use_dynamic_obstacles:
                    obstacle_actions = (
                        self.dynamic_obstacle_trajectory_controller.compute_actions(
                            obstacle_pre_states, current_time=sim_time
                        )
                    )

                if self.show_lidar_rays and self.cfg.simulation.gui:
                    if current_step % self.lidar_draw_interval == 0:
                        self.trajectory_controller.draw_lidar_rays(
                            drone_pre_states, self.tracked_drone_id
                        )

                self.environemnt.step(
                    self._merge_actions(drone_actions, obstacle_actions)
                )

                all_post_states = self._get_all_drone_states()
                drone_post_states, _ = self._split_states(all_post_states)

                if self.logger:
                    self.logger.log_step(current_step, sim_time, drone_post_states)

                self._process_collisions(sim_time, current_step)
                self._process_arrivals(drone_post_states, sim_time)

                if not self.active_drones:
                    print(
                        f"[DEBUG] Wszystkie drony zakończyły lot. Przerwano w kroku {current_step}."
                    )
                    break

                current_step += 1

                if (
                    not self.cfg.simulation.gui
                    and current_step % progress_interval == 0
                ):
                    pct = 100 * current_step / max_steps
                    print(f"[INFO] Postęp: {current_step}/{max_steps} ({pct:.0f}%)")

            if self.cfg.simulation.gui:
                drone_states_for_cam, _ = self._split_states(
                    self._get_all_drone_states()
                )
                self._update_camera(drone_states_for_cam)
                elapsed = time.time() - loop_start
                target_period = (1.0 / self.ctrl_freq) / self.sim_speed_multiplier
                if elapsed < target_period:
                    time.sleep(target_period - elapsed)
                if not p.isConnected():
                    break

        duration = time.time() - start_real_time
        print(f"[DEBUG] Koniec symulacji. Czas: {duration:.2f}s ---")
        if self.logger:
            self.logger.save()
        self.environemnt.close()


# =============================================================================
# PUNKTY WEJŚCIA
# =============================================================================


def _get_registry_job_key(cfg: DictConfig) -> dict | None:
    """Wyciąga z konfiguracji Hydry klucz i upewnia się, że pasuje do formatu w bazie."""
    if "experiment_meta" not in cfg or "id" not in cfg.experiment_meta:
        return None

    # 1. OPTYMALIZATOR
    # Skrypt prepare_experiment.py zapisuje krótkie nazwy: 'msffoa', 'ssa', 'ooa', 'nsga-3'
    # Musimy zmapować pełną ścieżkę z powrotem na tę krótką nazwę.
    opt_target = str(cfg.get("optimizer", {}).get("_target_", "")).lower()
    if "msffoa" in opt_target:
        optimizer_name = "msffoa"
    elif "ssa" in opt_target:
        optimizer_name = "ssa"
    elif "ooa" in opt_target:
        optimizer_name = "ooa"
    elif "nsga" in opt_target:
        optimizer_name = "nsga-3"
    else:
        optimizer_name = "unknown"

    # 2. ŚRODOWISKO
    env_name = str(cfg.get("environment", {}).get("name", "unknown")).lower()

    # 3. AVOIDANCE
    # W bazie jest 'none' z małej litery, w cfg często 'None'
    avoidance_name = str(cfg.get("avoidance", {}).get("name", "none")).lower()

    # 4. SEED
    seed_val = int(cfg.get("seed", 0))

    return {
        "exp_id": cfg.experiment_meta.id,
        "optimizer": optimizer_name,
        "environment": env_name,
        "avoidance": avoidance_name,
        "seed": seed_val,
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main_generate(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    # --- RunRegistry: guard wznawiania ---
    registry = None
    job_key = None
    job_meta = _get_registry_job_key(cfg)

    if job_meta:
        # Wyciągamy exp_id i usuwamy go z dict, żeby reszta posłużyła jako kwargs do bazy
        exp_id = job_meta.pop("exp_id")
        job_key = job_meta  # teraz zawiera tylko 4 klucze: optimizer, env, avoid, seed

        db_path = Path("results") / exp_id / "registry.db"

        # Tworzymy folder tylko jeśli nie istnieje (Hydra Joblib robi to z opóźnieniem)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        registry = RunRegistry(db_path)

        if not registry.should_run(**job_key):
            print(
                f"[SKIP] Zadanie już ukończone: "
                f"{job_key['optimizer']}/{job_key['environment']}/"
                f"avoidance={job_key['avoidance']}/seed={job_key['seed']}"
            )
            return

        registry.mark_started(**job_key)
        print(
            f"[REGISTRY] Start: {job_key['optimizer']}/{job_key['environment']}/"
            f"avoidance={job_key['avoidance']}/seed={job_key['seed']}"
        )

    # --- Właściwe uruchomienie eksperymentu ---
    try:
        runner = ExperimentRunner(cfg, GenerationDataStrategy())
        runner.prepare_experiment()
        runner.run()

        if registry and job_key:
            registry.mark_completed(**job_key)
            print(f"[REGISTRY] Ukończono: seed={job_key['seed']}")

    except BaseException as exc:
        if registry and job_key:
            error_message = (
                "Przerwano przez użytkownika (CTRL+C)"
                if isinstance(exc, KeyboardInterrupt)
                else str(exc)
            )
            registry.mark_failed(**job_key, error_msg=error_message)
            print(f"[REGISTRY] Błąd: seed={job_key['seed']} — {exc}")
        raise
    finally:
        # Gwarantowane ubicie wątków fizyki w C++ po zakończeniu
        try:
            if p.isConnected():
                p.disconnect()
        except Exception:
            pass


def main_replay(results_dir: str, headless: bool = False):
    results_path = Path(results_dir)
    cfg_path = results_path / ".hydra" / "config.yaml"

    if not cfg_path.exists():
        print(f"[ERROR] Nie znaleziono pliku konfiguracyjnego w {cfg_path}")
        sys.exit(1)

    cfg = OmegaConf.load(cfg_path)
    replay_output_dir = results_path / "replay"
    replay_output_dir.mkdir(exist_ok=True)
    OmegaConf.update(cfg, "logging.enabled", False)
    OmegaConf.update(cfg, "logging.output_dir", str(replay_output_dir))
    if headless:
        OmegaConf.update(cfg, "simulation.gui", False)

    runner = ExperimentRunner(cfg, ReplayDataStrategy(results_path))
    runner.prepare_experiment()
    runner.run()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--replay":
        if len(sys.argv) < 3:
            print(
                "Użycie: python main.py --replay <ścieżka_do_katalogu_wyników> [--headless]"
            )
            sys.exit(1)
        replay_path = sys.argv[2]
        headless = "--headless" in sys.argv[3:]
        sys.argv = [sys.argv[0]]
        main_replay(replay_path, headless)
    else:
        if "--headless" in sys.argv:
            sys.argv.remove("--headless")
            sys.argv.append("simulation.gui=false")
        main_generate()
