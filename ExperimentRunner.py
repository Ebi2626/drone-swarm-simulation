import os
import time
import hydra
import numpy as np
import pandas as pd
import pybullet as p
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src import get_environment, get_algorithm
from src.utils.SimulationLogger import SimulationLogger
from src.utils.pybullet_utils import update_camera_position 
from src.utils.input_utils import InputHandler, CommandType
from src.algorithms.NSGA3.SwarmDroneOptimizer import SwarmDroneOptimizer

class ExperimentRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.num_drones = cfg.num_drones
        self.ctrl_freq = cfg.simulation.get("ctrl_freq", 48)
        self.pyb_freq = cfg.simulation.get("pyb_freq", 240)
        self.tracked_drone_id = cfg.visualization.get("tracked_drone_id", 0)
        if self.tracked_drone_id >= self.num_drones:
            self.tracked_drone_id = 0
        self.env = None
        self.algorithm = None
        self.logger = None
        self.input_handler = None
        self.obstacles = None
        self.obstacles_matrix = None
        self.nsga3_optimizer = None 
        self.trajectories = None
        self.best_trajectories = None
        self.start_positions = np.array(self.cfg.initial_xyzs)
        self.end_positions = np.array(self.cfg.environment.get("target_position"))

    def _init_components(self):
        WorldClass = get_environment(self.cfg.environment.name)
        self.env = WorldClass(
            num_drones=self.num_drones,
            initial_xyzs=np.array(self.cfg.initial_xyzs),
            pyb_freq=self.pyb_freq,
            ctrl_freq=self.ctrl_freq,
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

    def _getObstacles(self):
        if( self.obstacles is not None):
            return self.obstacles
        self.obstacles = np.array(self.env.obstacles)
        return self.obstacles
    
    def _getObstaclesMatrix(self):
        if (self.obstacles_matrix is not None):
            return self.obstacles_matrix
        
        self.obstacles_matrix = np.zeros((len(self.obstacles), 4))
        for i in range(len(self.obstacles)):
            self.obstacles_matrix[i] = [
                self.obstacles[i].position[0],
                self.obstacles[i].position[1],
                self.obstacles[i].radius,
                self.obstacles[i].height
                ]
        return self.obstacles_matrix
        
    
    def _init_nsga3_optimizer(self):
        if self.obstacles is None:
            self._getObstacles()
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
            obstacles=self._getObstaclesMatrix()
        )

    def run(self):
        self._init_components()
        self.env.reset()
        self._init_nsga3_optimizer()
        self.trajectories = self.nsga3_optimizer.run_optimization(
            pop_size=self.cfg.algorithm.params.get("pop_size", 1000),
            n_gen=self.cfg.algorithm.params.get("n_gen", 200)
        )
        self.best_trajectories = self.nsga3_optimizer.get_best_trajectories(
            self.trajectories, n=5
        )

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

            all_states = [self.env._getDroneStateVector(d) for d in range(self.num_drones)]
            if is_running:
                sim_time = current_step / self.ctrl_freq
                
                # A. Decyzja algorytmu
                actions = self.algorithm.compute_actions(all_states, current_time=sim_time)
                
                # B. Krok fizyki
                self.env.step(actions)
                
                # C. Logowanie
                if self.logger:
                    self.logger.log_step(current_step, sim_time, all_states)
                    for d_id, o_id in self.env.get_detailed_collisions():
                        self.logger.log_collision(sim_time, d_id, o_id)

                current_step += 1

            if self.cfg.simulation.gui:
                self._update_camera(all_states)
                self.env.render()
                
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
        self.env.close()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    runner = ExperimentRunner(cfg)
    runner.run()

if __name__ == "__main__":
    main()
