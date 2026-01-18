import os
import numpy as np
import pandas as pd
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from src.environments.SwarmBaseWorld import SwarmBaseWorld
from src.utils.config_parser import sanitize_init_params
from hydra.core.hydra_config import HydraConfig

class ForestWorld(SwarmBaseWorld):
    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        physics: Physics = Physics.PYB,
        initial_xyzs=None,
        initial_rpys=None,
        num_trees: int = 100,
        track_length: float = 1000.0,
        tree_height: float = 10.0,
        tree_radius: float = 1.0,
        track_height: float = 12.0,
        track_width: float = 50.0,
        **kwargs,
    ):
        drone_model, physics, initial_xyzs, initial_rpys = sanitize_init_params(
            drone_model, physics, initial_xyzs, initial_rpys
        )

        self.num_trees = num_trees
        self.track_length = track_length
        self.tree_height = tree_height
        self.track_height = track_height
        self.start_safe_zone = 20.0
        self.end_safe_zone = 20.0
        self.tree_radius = tree_radius
        self.track_width = track_width

        super().__init__(
            drone_model=drone_model,
            physics=physics,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            obstacles=True,
            **kwargs,
        )

    def _addObstacles(self):
        print(f"[DEBUG] Forrest generating: {self.num_trees} trees...")

        self._init_render_silently()

        self._setup_environment(
            self.track_length,
            self.track_width,
            self.track_height,
            ground_color=[0.3, 0.5, 0.3, 1.0],
        )
        self._createForest()
        print("[DEBUG] Forest generated succesfully.")

    def _createTree(self, x, y):
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.tree_radius + np.random.uniform(0, 0.5),
            height=self.tree_height,
        )

        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.tree_radius,
            length=self.tree_height,
            rgbaColor=[0.4, 0.25, 0.1, 1],
        )

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, self.tree_height / 2],
        )

    def _createForest(self):
        forest_start_x = self.start_safe_zone
        forest_end_x = self.track_length - self.end_safe_zone
        corridor_width = self.track_width - 20.0

        np.random.seed(42)

        obstacle_positions = []

        for _ in range(self.num_trees):
            x = np.random.uniform(forest_start_x, forest_end_x)
            y = np.random.uniform(-corridor_width / 2, corridor_width / 2)
            self._createTree(x, y)
            obstacle_positions.append([x, y, self.tree_radius, self.tree_height])

        df = pd.DataFrame(obstacle_positions, columns=['x', 'y', 'radius', 'height'])
        df.to_csv(os.path.join(HydraConfig.get().runtime.output_dir, 'obstacles.csv'), index=False)
        print(f"Zapisano pozycje {self.num_trees} drzew do obstacles.csv")
