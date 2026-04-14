import sys
import time
import hydra
import numpy as np
import pybullet as p
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from src.runner.ExperimentDataStrategy import ExperimentDataStrategy
from src.runner.GenerationDataStrategy import GenerationDataStrategy
from src.runner.ReplayDataStrategy import ReplayDataStrategy

from src.utils.input_utils import InputHandler, CommandType
from src.utils.SimulationLogger import SimulationLogger
from src.utils.pybullet_utils import update_camera_position 
from src.algorithms.TrajectoryFollowingAlgorithm import TrajectoryFollowingAlgorithm
from src.environments.obstacles.ObstacleShape import ObstacleShape

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
        
        # Parametry wizualizacji
        self.tracked_drone_id = cfg.visualization.get("tracked_drone_id", 0)
        self.show_lidar_rays = cfg.visualization.get("show_lidar_rays", False)
        self.lidar_draw_interval = cfg.visualization.get("lidar_draw_interval", 5)
        if self.tracked_drone_id >= self.num_drones:
            self.tracked_drone_id = 0

        # Zmienne środowiskowe (zostaną nadpisane przez strategię)
        self.environemnt = None
        self.world_data = None
        self.obstacles_data = None
        self.trajectories = None
        self.logger = None
        self.input_handler = None
        
        # Parametry środowiska zdefiniowane w konfiguracji
        self.initial_rpys = cfg.environment.get("initial_rpys")
        self.start_positions = np.array(cfg.environment.get("initial_xyzs"), dtype=np.float64)
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

        # Ustawienia optymalizatora
        self.number_of_waypoints = cfg.optimizer.algorithm_params.get("n_inner_waypoints")

    def prepare_experiment(self):
        """Metoda przygotowująca wszystkie wymagane dane i komponenty dla symulacji"""
        
        # Inicjalizacja GUI i kontrolerów wejścia
        if self.cfg.simulation.gui:
            self.input_handler = InputHandler(self.num_drones)

        # Inicjalizacja loggera (przydatne zarówno dla generacji jak i powtórek)
        if self.cfg.logging.enabled:
            output_dir = self.cfg.logging.get("output_dir")
            if output_dir is None:
                output_dir = HydraConfig.get().runtime.output_dir
            self.logger = SimulationLogger(
                output_dir=output_dir, log_freq=self.cfg.logging.log_freq,
                ctrl_freq=self.ctrl_freq, num_drones=self.num_drones
            )

        # Delegacja przygotowania środowiska (Offline Path-Planning lub Wczytywanie z CSV)
        self.data_strategy.prepare_data(self)
        
        # Inicjalizacja algorytmu sterującego
        self._init_trajectory_following_algorithm()

    def _init_trajectory_following_algorithm(self):
        self.trajectory_controller = TrajectoryFollowingAlgorithm(
            parent=self,
            num_drones=self.num_drones,
            params={
                "acceptance_radius": 0.2,
                "ctrl_freq": self.ctrl_freq,
            }
        )

    def initialize_world(self):
        self.environemnt = instantiate(
            self.cfg.environment,
            world_data=self.world_data,
            obstacles_data=self.obstacles_data,
            drone_model=self.drone_model,
            physics=self.phyics,
            initial_xyzs=self.start_positions,
            end_xyzs=self.end_positions,
            gui=self.cfg.simulation.gui,
            ctrl_freq=self.ctrl_freq,
            pyb_freq=self.pyb_freq,
        )

    def _update_camera(self, all_states: list):
        if not self.cfg.visualization.camera_follow:
            return
        target_state = all_states[self.tracked_drone_id]
        update_camera_position(
            drone_state=target_state, distance=self.cfg.visualization.camera_distance,
            yaw_offset=self.cfg.visualization.camera_yaw, pitch=self.cfg.visualization.camera_pitch
        )
    
    def _init_active_drones(self):
        self.active_drones = set(range(self.num_drones))
        self.acceptance_radius = self.trajectory_controller.params.get("acceptance_radius", 0.2)

    def _process_collisions(self, sim_time: float, current_step: int):
        for d_id, o_id in self.environemnt.get_detailed_collisions():
            if self.logger:
                self.logger.log_collision(sim_time, d_id, o_id)
            if d_id in self.active_drones:
                self.active_drones.remove(d_id)
                print(f"[INFO] Dron {d_id} uległ kolizji w czasie {sim_time:.2f}s (krok {current_step}).")

    def _process_arrivals(self, all_states: list, sim_time: float):
        try:
            radius = float(np.squeeze(self.acceptance_radius))
        except (TypeError, ValueError):
            raise ValueError(f"Błędny promień akceptacji: {self.acceptance_radius}")

        for d_id in list(self.active_drones):
            pos = np.array(all_states[d_id][0:3]).flatten()
            target = np.array(self.end_positions[d_id]).flatten()
            
            if pos.size == 0 or target.size == 0:
                continue
                
            dist = float(np.linalg.norm(pos - target))
            if dist <= radius:
                self.active_drones.remove(d_id)
                print(f"[INFO] Dron {d_id} osiągnął cel w czasie {sim_time:.2f}s.")

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
        print(f"[DEBUG] Start symulacji na {max_steps} kroków.")

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
                        if not self.show_lidar_rays and hasattr(self.trajectory_controller, "clear_lidar_rays"):
                            self.trajectory_controller.clear_lidar_rays()

            if is_running:
                # Odczyt stanu przed krokiem — używany do obliczenia akcji
                pre_states = [self.environemnt._getDroneStateVector(d) for d in range(self.num_drones)]
                sim_time = current_step / self.ctrl_freq
                actions = self.trajectory_controller.compute_actions(pre_states, current_time=sim_time)

                if self.show_lidar_rays and self.cfg.simulation.gui:
                    if current_step % self.lidar_draw_interval == 0:
                        self.trajectory_controller.draw_lidar_rays(pre_states, self.tracked_drone_id)

                self.environemnt.step(actions)

                # Odczyt stanu po kroku — używany do logowania i detekcji zdarzeń
                all_states = [self.environemnt._getDroneStateVector(d) for d in range(self.num_drones)]

                if self.logger:
                    self.logger.log_step(current_step, sim_time, all_states)

                self._process_collisions(sim_time, current_step)
                self._process_arrivals(all_states, sim_time)

                if not self.active_drones:
                    print(f"[DEBUG] Wszystkie drony zakończyły lot. Przerwano symulację w kroku {current_step}.")
                    break

                current_step += 1

                if not self.cfg.simulation.gui and current_step % progress_interval == 0:
                    pct = 100 * current_step / max_steps
                    print(f"[INFO] Postęp: {current_step}/{max_steps} ({pct:.0f}%)")

            if self.cfg.simulation.gui:
                all_states = [self.environemnt._getDroneStateVector(d) for d in range(self.num_drones)]
                self._update_camera(all_states)
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

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main_generate(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    runner = ExperimentRunner(cfg, GenerationDataStrategy())
    runner.prepare_experiment()
    runner.run()

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
    # Obsługa prostych flag CLI przed parsowaniem Hydry
    if len(sys.argv) > 1 and sys.argv[1] == "--replay":
        # python main.py --replay path/to/results [--headless]
        if len(sys.argv) < 3:
            print("Użycie: python main.py --replay <ścieżka_do_katalogu_wyników> [--headless]")
            sys.exit(1)
        replay_path = sys.argv[2]
        headless = "--headless" in sys.argv[3:]
        sys.argv = [sys.argv[0]]
        main_replay(replay_path, headless=headless)
    else:
        # Standardowe zachowanie Hydry dla tworzenia nowego eksperymentu
        # Flaga --headless jest zamieniana na override Hydry: simulation.gui=false
        # python main.py --headless [inne_overrides_hydry...]
        if "--headless" in sys.argv:
            sys.argv.remove("--headless")
            sys.argv.append("simulation.gui=false")
        main_generate()