import os
import time
from pathlib import Path

import hydra
import numpy as np
import pybullet as p
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src import get_environment
from src.algorithms.TrajectoryFollowingAlgorithm import TrajectoryFollowingAlgorithm
from src.algorithms.abstraction.count_trajectories import count_trajectories
from src.algorithms.abstraction.trajectory.strategies.nsga3_swarm_strategy import nsga3_swarm_strategy
from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.utils.DataPipelineReader import load_obstacles_csv, load_trajectories_csv
from src.utils.DataPipelineWriter import (
    save_obstacles_csv,
    save_start_end_positions_csv,
    save_trajectories_csv,
    save_world_data_csv,
)
from src.utils.SimulationLogger import SimulationLogger
from src.utils.pybullet_utils import update_camera_position
from src.utils.input_utils import InputHandler, CommandType

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
        self.start_positions = np.array(cfg.environment.get("initial_xyzs"), dtype=np.float64)
        self.end_positions = np.array(cfg.environment.get("end_xyzs"), dtype=np.float64)
        self.ground_position = cfg.environment.params.get("ground_position")
        self.track_length = cfg.environment.params.get("track_length")
        self.track_width = cfg.environment.params.get("track_width")
        self.track_height = cfg.environment.params.get("track_height")
        self.obstacles_number = cfg.environment.params.get("obstacles_number")
        self.obstacle_width = cfg.environment.params.get("obstacle_width")
        self.obstacle_height = cfg.environment.params.get("obstacle_height")
        self.obstacle_length = cfg.environment.params.get("obstacle_length")
        self.data_pipeline_cfg = cfg.get("data_pipeline")

    def _init_world(self):
        WorldClass = get_environment(self.cfg.environment.name)
        self.world = WorldClass(
            drone_model=self.drone_model,
            physics=self.phyics,
            initial_xyzs=self.start_positions,
            end_xyzs=self.end_positions,
            ground_position=self.ground_position,
            track_length=self.track_length,
            track_width=self.track_width,
            track_height=self.track_height,
            obstacles_number=self.obstacles_number,
            obstacle_width=self.obstacle_width,
            obstacle_length=self.obstacle_length,
            obstacle_height=self.obstacle_height,
            drone_number=self.num_drones,
            gui=self.cfg.simulation.gui
        )

    def _init_obstacles(self) -> None:
        self.obstacles = self.world.generate_obstacles()

    def _compute_trajectories(self):
        """Compute trajectories using count_trajectories with nsga3_swarm_strategy."""
        world_data = self.world._generate_world_def()
        algorithm_params = OmegaConf.to_container(self.cfg.algorithm.params, resolve=True)
        n_output_waypoints = algorithm_params.get("n_output_waypoints", 100)

        trajectories = count_trajectories(
            world_data=world_data,
            obstacles_data=self.obstacles,
            counting_protocol=nsga3_swarm_strategy,
            drone_swarm_size=self.num_drones,
            number_of_waypoints=n_output_waypoints,
            start_positions=self.start_positions.astype(np.float64),
            target_positions=self.end_positions.astype(np.float64),
            algorithm_params=algorithm_params
        )
        return world_data, trajectories

    def _get_pipeline_output_dir(self) -> Path:
        configured = self.data_pipeline_cfg.save.get("output_dir", None)
        if configured:
            return Path(configured)
        try:
            return Path(HydraConfig.get().runtime.output_dir) / "data"
        except Exception:
            return Path(os.getcwd()) / "data"

    def _maybe_save_pipeline_data(self, world_data, obstacles, trajectories) -> None:
        if not self.data_pipeline_cfg or not self.data_pipeline_cfg.save.enabled:
            return

        output_dir = self._get_pipeline_output_dir()
        if self.data_pipeline_cfg.save.get("world_data", True):
            save_world_data_csv(world_data, output_dir)
        if self.data_pipeline_cfg.save.get("obstacles", True):
            save_obstacles_csv(obstacles, output_dir)
        if self.data_pipeline_cfg.save.get("trajectories", True):
            save_trajectories_csv(trajectories, output_dir)
        if self.data_pipeline_cfg.save.get("positions", True):
            save_start_end_positions_csv(self.start_positions, self.end_positions, output_dir)

    def _maybe_load_pipeline_data(self):
        if not self.data_pipeline_cfg or not self.data_pipeline_cfg.load.enabled:
            return None, None

        input_dir = self.data_pipeline_cfg.load.get("input_dir")
        if not input_dir:
            raise ValueError("data_pipeline.load.enabled=true wymaga ustawienia data_pipeline.load.input_dir")

        obstacles = self.obstacles
        trajectories = None
        if self.data_pipeline_cfg.load.get("obstacles", True):
            obstacles = load_obstacles_csv(input_dir)
        if self.data_pipeline_cfg.load.get("trajectories", True):
            trajectories = load_trajectories_csv(input_dir, self.num_drones)
        return obstacles, trajectories

    def _init_follower(self, trajectories):
        """Create trajectory follower with pre-computed trajectories."""
        follower_params = OmegaConf.to_container(self.cfg.follower.params, resolve=True)
        self.algorithm = TrajectoryFollowingAlgorithm(
            num_drones=self.num_drones,
            trajectories=trajectories,
            params=follower_params
        )

    def _init_logger(self):
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

    def run(self):
        print("Running experiment...")

        # 1. Create world and prepare data
        self._init_world()
        self._init_obstacles()
        world_data = self.world._generate_world_def()

        loaded_obstacles, loaded_trajectories = self._maybe_load_pipeline_data()
        if loaded_obstacles is not None:
            self.obstacles = loaded_obstacles

        if loaded_trajectories is not None:
            trajectories = loaded_trajectories
            print(f"Loaded trajectories from CSV: shape {trajectories.shape}")
        else:
            print("Computing trajectories via NSGA-III...")
            world_data, trajectories = self._compute_trajectories()
            print(f"Trajectories computed: shape {trajectories.shape}")

        self._maybe_save_pipeline_data(world_data, self.obstacles, trajectories)

        # 2. Create follower with ready trajectories
        self._init_follower(trajectories)

        # 3. Reset simulation first, then render the generated world on top of it.
        # Calling reset() after draw_obstacles() recreates the default PyBullet scene
        # and visually overwrites the generated environment with the gray plane.
        self.world.reset()
        self.world.draw_obstacles(self.obstacles)
        if self.cfg.simulation.gui:
            self.world._finalize_render()

        # 4. Init logger and input handler
        self._init_logger()
        if self.cfg.simulation.gui:
            self.input_handler = InputHandler(self.num_drones)

        # 5. Simulation loop
        is_running = bool(self.cfg.simulation.get("autostart", True))
        current_step = 0
        max_steps = int(self.cfg.simulation.duration_sec * self.ctrl_freq)

        if not self.cfg.simulation.gui:
            is_running = True
        elif not is_running:
            print("[INFO] Simulation paused. Press SPACE to start.")

        start_real_time = time.time()
        print(f"[DEBUG] Starting simulation for {max_steps} steps.")
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

                actions = self.algorithm.compute_actions(all_states, current_time=sim_time)
                self.world.step(actions)

                if self.logger:
                    self.logger.log_step(current_step, sim_time, all_states)

                current_step += 1

            if self.cfg.simulation.gui:
                self._update_camera(all_states)
                self.world.render()

                elapsed = time.time() - loop_start
                target_period = 1.0 / self.ctrl_freq

                if elapsed < target_period:
                    time.sleep(target_period - elapsed)

                if not p.isConnected():
                    break

        duration = time.time() - start_real_time
        print(f"[DEBUG] Simulation finished. Duration: {duration:.2f}s")
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
