import os
import time
import hydra
import numpy as np
import pybullet as p
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src import get_environment, get_algorithm
from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.utils.SimulationLogger import SimulationLogger
from src.utils.pybullet_utils import update_camera_position 
from src.utils.input_utils import InputHandler, CommandType
from src.algorithms.NSGA3.SwarmDroneOptimizer import SwarmDroneOptimizer

class ExperimentRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.drone_model = cfg.simulation.get("drone_model", "CF2X")
        self.phyics = cfg.simulation.get("physics", "PYB")
        self.num_drones = cfg.num_drones
        self.ctrl_freq = cfg.simulation.get("ctrl_freq", 48)
        self.pyb_freq = cfg.simulation.get("pyb_freq", 240)
        self.tracked_drone_id = cfg.visualization.get("tracked_drone_id", 0)
        if self.tracked_drone_id >= self.num_drones:
            self.tracked_drone_id = 0
        self.world = None
        self.algorithm = None
        self.logger = None
        self.input_handler = None
        self.obstacles: ObstaclesData = None
        self.nsga3_optimizer = None 
        self.trajectories = None
        self.best_trajectories = None
        self.start_positions = np.array(cfg.environment.get("initial_xyzs"))
        self.end_positions = np.array(cfg.environment.get("end_xyzs"))
        self.ground_position = cfg.environment.get("ground_position")
        self.track_length = cfg.environment.get("track_length")
        self.track_width = cfg.environment.get("track_width")
        self.track_height = cfg.environment.get("track_height")
        self.obstacles_number = cfg.environment.get("obstacles_number")
        self.obstacle_width = cfg.environment.get("obstacle_width")
        self.obstacle_height = cfg.environment.get("obstacle_height")
        self.obstacle_height = cfg.environment.get("obstacle_height")

    def _init_components(self):
        WorldClass = get_environment(self.cfg.environment.name)
        self.world = WorldClass(
            drone_model = self.drone_model,
            physics = self.phyics,
            initial_xyzs = self.start_positions,
            end_xyzs = self.end_positions,
            ground_position = self.ground_position,
            track_length = self.cfg.track_length,
            track_width = self.track_width,
            track_height = self.track_height,
            obstacles_number = self.obstacles_number,
            obstacle_width = self.obstacle_width,
            obstacle_length = self.obstacle_length,
            obstacle_height = self.obstacle_height,
            gui=self.cfg.simulation.gui
        )

        alg_name = self.cfg.algorithm.class_name
        AlgClass = get_algorithm(alg_name)
        self.algorithm = AlgClass(
            parent=self,
            num_drones=self.num_drones, 
            params=self.cfg.algorithm.params,
        )

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

        if self.cfg.simulation.gui:
            self.input_handler = InputHandler(self.num_drones)

    def _update_camera(self, all_states):
        if not self.cfg.visualization.camera_follow:
            return

        target_state = all_states[self.tracked_drone_id]
                
        update_camera_position(
            drone_state=target_state,
            distance=self.cfg.visualization.camera_distance,
            yaw_offset=self.cfg.visualization.camera_yaw,
            pitch=self.cfg.visualization.camera_pitch
        )

    def _init_obstacles(self) -> None:
        self.obstacles = self.world.generate_obstacles()
    
    def _init_nsga3_optimizer(self):
        print("Initializing NSGA-3 optimizer...")
        print(self.cfg.environment)
        self.nsga3_optimizer = SwarmDroneOptimizer(
            space_limits=[
                self.cfg.environment.params.get("track_width"),
                self.cfg.environment.params.get("track_length"),
                self.cfg.environment.params.get("track_height")
            ],
            n_drones=self.num_drones,
            n_waypoints=20,
            start_positions=self.start_positions,
            end_positions=self.end_positions,
            obstacles=self.obstacles.data
        )

    def run(self):
        print("Running experiment...")
        self._init_components()
        self._init_obstacles()
        self._init_nsga3_optimizer()
        self.trajectories = self.nsga3_optimizer.run_optimization(
            pop_size=self.cfg.algorithm.params.get("pop_size", 1000),
            n_gen=self.cfg.algorithm.params.get("n_gen", 200)
        )
        self.best_trajectories = self.nsga3_optimizer.get_best_trajectories(
            self.trajectories, n=5
        )
        self.world.draw_obstacles(self.obstacles)
        self.world.reset()

        is_running = False
        current_step = 0
        max_steps = int(self.cfg.simulation.duration_sec * self.ctrl_freq)
        
        if not self.cfg.simulation.gui:
            is_running = True

        start_real_time = time.time()
        print(f"[DEBUG] Start symulacji na {max_steps} kroków. O godzinie: {start_real_time:.2f}s ---")
        while current_step < max_steps:
            loop_start = time.time()

            if self.cfg.simulation.gui:
                cmd = self.input_handler.get_command()
                if cmd:
                    if cmd.type == CommandType.TOGGLE_SIMULATION:
                        is_running = not is_running
                    
                    elif cmd.type == CommandType.SWITCH_DRONE_CAMERA:
                        self.tracked_drone_id = cmd.payload

            all_states = [self.world._getDroneStateVector(d) for d in range(self.num_drones)]
            if is_running:
                sim_time = current_step / self.ctrl_freq
                
                # A. Decyzja algorytmu
                actions = self.algorithm.compute_actions(all_states, current_time=sim_time)
                
                # B. Krok fizyki
                self.world.step(actions)
                
                # C. Logowanie
                if self.logger:
                    self.logger.log_step(current_step, sim_time, all_states)
                    for d_id, o_id in self.world.get_detailed_collisions():
                        self.logger.log_collision(sim_time, d_id, o_id)

                current_step += 1

            if self.cfg.simulation.gui:
                self._update_camera(all_states)
                self.world.render()
                
                # Utrzymanie stałego FPS
                elapsed = time.time() - loop_start
                target_period = 1.0 / self.ctrl_freq
                
                if elapsed < target_period:
                    time.sleep(target_period - elapsed)
                
                if not p.isConnected():
                    break

        duration = time.time() - start_real_time
        print(f"[DEBUG] Koniec symulacji. Czas: {duration:.2f}s ---")
        if self.logger:
            self.logger.save()
        self.world.close()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    runner = ExperimentRunner(cfg)
    runner.run()

if __name__ == "__main__":
    main()
