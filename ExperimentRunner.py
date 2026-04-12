import functools
import os
import time
import hydra
import numpy as np
import pybullet as p
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from configs.environment.strategies.placement_strategies import get_placement_strategy
from src.environments.EmptyWorld import EmptyWorld
from src.environments.ForestWorld import ForestWorld
from src.environments.UrbanWorld import UrbanWorld
from src.environments.abstraction.generate_obstacles import ObstaclesData, PlacementStrategy, generate_obstacles
from src.environments.abstraction.generate_world_boundaries import WorldData, generate_world_boundaries
from src.environments.obstacles.ObstacleShape import ObstacleShape
from src.utils.SimulationLogger import SimulationLogger
from src.utils.pybullet_utils import update_camera_position 
from src.utils.input_utils import InputHandler, CommandType
from src.algorithms.TrajectoryFollowingAlgorithm import TrajectoryFollowingAlgorithm

class ExperimentRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.drone_model = cfg.simulation.get("drone_model", "CF2X")
        self.phyics = cfg.simulation.get("physics", "PYB")
        self.num_drones = cfg.environment.params.get("num_drones")
        self.ctrl_freq = cfg.simulation.get("ctrl_freq", 48)
        self.pyb_freq = cfg.simulation.get("pyb_freq", 240)
        self.tracked_drone_id = cfg.visualization.get("tracked_drone_id", 0)
        self.show_lidar_rays = cfg.visualization.get("show_lidar_rays", False)
        self.lidar_draw_interval = cfg.visualization.get("lidar_draw_interval", 5)
        self.environemnt: EmptyWorld | ForestWorld | UrbanWorld | None = None
        self.world_data: WorldData | None = None
        self.obstacles_data: ObstaclesData | None = None
        self.placement_strategy: PlacementStrategy | None = None
        self.counting_protocol = None
        self.logger = None
        self.input_handler = None
        self.nsga3_optimizer = None 
        self.trajectories = None
        self.number_of_waypoints = cfg.optimizer.algorithm_params.get("n_inner_waypoints")
        self.initial_rpys = cfg.environment.get("initial_rpys")
        self.start_positions = np.array(cfg.environment.get("initial_xyzs"), dtype=np.int64)
        self.end_positions = np.array(cfg.environment.get("end_xyzs"), dtype=np.int64)
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
        self.sim_speed_multiplier = cfg.simulation.get("speed_multiplier", 5.0)
        if self.tracked_drone_id >= self.num_drones:
            self.tracked_drone_id = 0

    def _init_components(self):
        # Components initialization

        # 1. World initialization (3d space, boundaries, etc.)
        self.world_data = generate_world_boundaries(
            width=self.track_width,
            length=self.track_length,
            height=self.track_height,
            ground_height=self.ground_position
        )

        # 2. Obstacles initialization
        if self.placement_strategy_name is not None:
            self.obstacles_data = generate_obstacles(
                self.world_data,
                n_obstacles=self.obstacles_number,
                shape_type=self.shape_type,
                placement_strategy=get_placement_strategy(self.placement_strategy_name),
                size_params={
                    'length': self.obstacle_length,
                    'width': self.obstacle_width,
                    'height': self.obstacle_height,
                },
                start_positions=self.start_positions,
                target_positions=self.end_positions
            )

        # 3. Algorithm initialization (optimization algorithm class)
        self.counting_strategy = instantiate(self.cfg.optimizer)
        self.counting_protocol = functools.partial(
            self.counting_strategy,
            start_positions = self.start_positions,
            target_positions = self.end_positions,
            obstacles_data = self.obstacles_data,
            world_data = self.world_data,
            number_of_waypoints = self.number_of_waypoints,
            drone_swarm_size = self.num_drones
        )# Count trajectories from optimize (algorithm params coming from hydra config)

        # 4. Logger initialization conditionally based on configuration
        if self.cfg.logging.enabled:
            try:
                output_dir = HydraConfig.get().runtime.output_dir
            except Exception:
                output_dir = os.getcwd()
            
            self.logger = SimulationLogger(
                output_dir=output_dir,
                log_freq=self.cfg.logging.log_freq,
                ctrl_freq=self.ctrl_freq,
                num_drones=self.num_drones
            )

        # 5. Input handler initialization (if GUI enabled)
        if self.cfg.simulation.gui:
            self.input_handler = InputHandler(self.num_drones)
    
    def _count_trajectories(self, counting_protocol) -> NDArray[np.float64]:
            print(f"\n🚀 Uruchamianie obliczeń (Offline Path-Planning)...")
            return counting_protocol()
    
    def _init_trajectory_following_algorithm(self):
        self.trajectory_controller = TrajectoryFollowingAlgorithm(
            parent=self,
            num_drones=self.num_drones,
            params={
                "acceptance_radius": 0.2,
                "ctrl_freq": self.ctrl_freq,
            }
        )

    def _init_world(self):
        self.environemnt = instantiate(
            self.cfg.environment,
            drone_model=self.drone_model,
            physics=self.phyics,
            initial_rpys=self.initial_rpys,
            initial_xyzs=self.start_positions.tolist(),
            end_xyzs=self.end_positions.tolist(),
            num_drones=self.num_drones,
            ground_position=self.ground_position,
            track_length=self.track_length,
            track_width=self.track_width,
            track_height=self.track_height,
            obstacles_number=self.obstacles_number,
            obstacle_width=self.obstacle_width,
            obstacle_length=self.obstacle_length,
            obstacle_height=self.obstacle_height,
            gui=self.cfg.simulation.gui,
        )

    def _update_camera(self, all_states: np.ndarray[tuple[int]]):
        """
        Updating camera position attached to choosen drone in the swarm

        Args:
            all_states (ndarray[tuple[int]]): all data about dron position, orientation, velocity etc.
        """
        if not self.cfg.visualization.camera_follow:
            return

        target_state = all_states[self.tracked_drone_id]
                
        update_camera_position(
            drone_state=target_state,
            distance=self.cfg.visualization.camera_distance,
            yaw_offset=self.cfg.visualization.camera_yaw,
            pitch=self.cfg.visualization.camera_pitch
        )
    
    def _init_active_drones(self):
        """Inicjalizacja zbioru aktywnych dronów oraz parametrów akceptacji."""
        self.active_drones = set(range(self.num_drones))
        self.acceptance_radius = self.trajectory_controller.params.get("acceptance_radius", 0.2)

    def _process_collisions(self, sim_time: float, current_step: int):
        """Wykrywanie kolizji, logowanie zdarzeń oraz usuwanie dronów rozbitych ze zbioru aktywnych."""
        for d_id, o_id in self.environemnt.get_detailed_collisions():
            if self.logger:
                self.logger.log_collision(sim_time, d_id, o_id)
            if d_id in self.active_drones:
                self.active_drones.remove(d_id)
                print(f"[INFO] Dron {d_id} uległ kolizji w czasie {sim_time:.2f}s (krok {current_step}).")

    def _process_arrivals(self, all_states: list, sim_time: float):
        """Weryfikacja dystansu do celu i usuwanie z aktywnych tych dronów, które osiągnęły zadany punkt."""
        
        # Upewniamy się, że acceptance_radius jest zrzutowany na zwykłą liczbę (skalar)
        # Jeśli na tym etapie poleci TypeError, to wiesz, że źle zainicjalizowano zmienną
        try:
            radius = float(np.squeeze(self.acceptance_radius))
        except (TypeError, ValueError):
            raise ValueError(f"Błędny promień akceptacji! Obecna wartość: {self.acceptance_radius}")

        for d_id in list(self.active_drones):
            # Flatten zapobiega problemom, gdyby tablice były zagnieżdżone np. [[x, y, z]]
            pos = np.array(all_states[d_id][0:3]).flatten()
            target = np.array(self.end_positions[d_id]).flatten()
            
            # Pomijanie pustych danych, jeśli framework nie zdążył ich wysłać
            if pos.size == 0 or target.size == 0:
                continue
                
            # Wyliczenie dystansu jako bezpieczny float
            dist = float(np.linalg.norm(pos - target))
            
            if dist <= radius:
                self.active_drones.remove(d_id)
                print(f"[INFO] Dron {d_id} osiągnął cel w czasie {sim_time:.2f}s.")
    
    def offilne_trajectory_counting(self):
        self._init_components() # Initialization world and algorithm classes based on hydra config
        self.trajectories = self._count_trajectories(self.counting_protocol) # counting trajectories based on optimization class from hydra config
        if self.logger is not None:
            self.logger.log_chosen_trajectories(self.trajectories)
            self.logger.log_world_dimensions(self.world_data)
            self.logger.log_obstacles(self.obstacles_data)
        self._init_trajectory_following_algorithm() # Initialize trajectory following algorithm

    def initialize_world(self):
        self.environemnt = instantiate(
            self.cfg.environment,
            world_data=self.world_data,           # już wygenerowane w _init_components()
            obstacles_data=self.obstacles_data,   # już wygenerowane w _init_components()
            drone_model=self.drone_model,
            physics=self.phyics,
            initial_xyzs=self.start_positions,
            end_xyzs=self.end_positions,
            gui=self.cfg.simulation.gui,
            ctrl_freq=self.ctrl_freq,
            pyb_freq=self.pyb_freq,
        )

    def run(self):
        print("Running experiment...")
        self.initialize_world()
        self.trajectory_controller.init_lidars(self.environemnt.CLIENT)

        is_running = False
        current_step = 0
        max_steps = int(self.cfg.simulation.duration_sec * self.ctrl_freq)
        
        if not self.cfg.simulation.gui:
            is_running = True

        self._init_active_drones()

        start_real_time = time.time()
        print(f"[DEBUG] Start symulacji na {max_steps} kroków. O godzinie: {start_real_time:.2f}s ---")
        breakpoint
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

            all_states = [self.environemnt._getDroneStateVector(d) for d in range(self.num_drones)]
            if is_running:
                sim_time = current_step / self.ctrl_freq
                
                # A. Decyzja algorytmu
                actions = self.trajectory_controller.compute_actions(all_states, current_time=sim_time)

                # A.1 Wizualizacja promieni lidaru (co K-tą klatkę)
                if self.show_lidar_rays and self.cfg.simulation.gui:
                    if current_step % self.lidar_draw_interval == 0:
                        self.trajectory_controller.draw_lidar_rays(all_states, self.tracked_drone_id)

                # B. Krok fizyki
                self.environemnt.step(actions)
                
                # C. Logowanie
                if self.logger:
                    self.logger.log_step(current_step, sim_time, all_states)
                    for d_id, o_id in self.environemnt.get_detailed_collisions():
                        self.logger.log_collision(sim_time, d_id, o_id)

                # D. Logika weryfikacji floty (Rozdzielone odpowiedzialności)
                self._process_collisions(sim_time, current_step)
                self._process_arrivals(all_states, sim_time)

                # E. Warunek wyjścia
                if not self.active_drones:
                    print(f"[DEBUG] Wszystkie drony zakończyły lot. Przerwano symulację w kroku {current_step}.")
                    break

                current_step += 1

            if self.cfg.simulation.gui:
                self._update_camera(all_states)
                # self.environemnt.render()
                
                # Utrzymanie stałego FPS
                elapsed = time.time() - loop_start
                
                # Dzielimy docelowy czas trwania kroku przez mnożnik
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
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    runner = ExperimentRunner(cfg)
    runner.offilne_trajectory_counting()
    runner.run()

if __name__ == "__main__":
    main()
