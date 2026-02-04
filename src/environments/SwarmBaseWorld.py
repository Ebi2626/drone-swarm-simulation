
from abc import abstractmethod
from typing import List
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gymnasium import spaces
import numpy as np
import pandas as pd
import pybullet as p
import os
from hydra.core.hydra_config import HydraConfig

from src.environments.obstacles.Obstacle import Obstacle

class SwarmBaseWorld(BaseAviary):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obstacles = None
    
    # Common environment setup methods
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._finalize_render()
        return obs, info
    
    def _init_render_silently(self):
        if self.GUI:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    def _finalize_render(self):
        if self.GUI:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def _clear_default_plane(self):
        print("[DEBUG] Deleting default plane...")
        for i in range(p.getNumBodies()):
            if "plane" in p.getBodyInfo(i)[1].decode("utf-8"):
                p.removeBody(i)

    def _create_ground(self, length, width, color=[0.5, 0.5, 0.55, 1.0], thickness=0.1):
        print("[DEBUG] Creating ground...")

        half_len = length / 2 + 50.0 
        half_wid = width / 2 + 50.0 
        z_pos = 0.0 - (thickness / 2.0)

        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_len, half_wid, thickness/2])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_len, half_wid, thickness/2], rgbaColor=color)
        
        p.createMultiBody(0, col_id, vis_id, [length/2, 0, z_pos])
        print(f"[DEBUG] Ground created ({length}x{width})")
    
    def _create_ceiling(self, length, width, height):
        print("[DEBUG] Creating ceiling...")
        half_len = length / 2 + 50.0 
        half_wid = width / 2 + 50.0 
        
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_len, half_wid, 0.5])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_len, half_wid, 0.5], rgbaColor=[0, 0, 1, 0.1])
        
        p.createMultiBody(0, col_id, vis_id, [length/2, 0, height])
        print(f"[DEBUG] Created ceiling at height {height}m")

    def _setup_environment(self, length, width, track_height, ground_color):
        print("[DEBUG] Setting up environment (ground and ceiling)...")
        self._clear_default_plane()
        self._create_ground(length, width, color=ground_color)
        if track_height > 0:
            self._create_ceiling(length, width, track_height)

        print("[DEBUG] Environment setup complete.")

    def get_detailed_collisions(self):
        """
        Sprawdza kolizje dla każdego drona.
        Zwraca listę krotek: (drone_id, other_body_id)
        """
        collisions = []
        for drone_id in range(self.NUM_DRONES):
            # Pobieramy ID ciała fizycznego drona w PyBullet
            # W BaseAviary drony są trzymane w self.DRONE_IDS (lista)
            drone_body_id = self.DRONE_IDS[drone_id]
            
            # Zapytanie do silnika fizycznego o punkty kontaktu
            contact_points = p.getContactPoints(bodyA=drone_body_id)
            
            if contact_points:
                for contact in contact_points:
                    # contact[2] to bodyB (obiekt, w który uderzyliśmy)
                    other_body_id = contact[2]
                    # Unikamy logowania "kolizji" z samym sobą (rzadkie, ale możliwe w błędnych modelach)
                    if other_body_id != drone_body_id:
                        collisions.append((drone_id, other_body_id))
                        
        # Usuwamy duplikaty (np. wielokrotne punkty styku w jednej klatce)
        return list(set(collisions))
    
    def printObstaclesToFile(self, obstacleType: str):
        if(obstacleType == "CYLINDER"):
            columns = ['x', 'y', 'radius', 'height']
        if(obstacleType == "CUBOID"):
            columns = ['x', 'y', 'length', 'width', 'height']
        
        df = pd.DataFrame(self.obstacles, columns= columns)
        df.to_csv(os.path.join(HydraConfig.get().runtime.output_dir, 'obstacles.csv'), index=False)
        print(f"Zapisano pozycje {len(self.obstacles)} przeszkód typu {obstacleType} do obstacles.csv")

    # Implementations needed for BaseAviary / Gymnasium
    def _actionSpace(self):
        act_lower_bound = np.array([[0., 0., 0., 0.] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.NUM_DRONES, 20), dtype=np.float32)

    def _computeObs(self):

        return self._getDroneStateVector(0)
        
    def _preprocessAction(self, action):
        return action
    
    def _computeReward(self):
        return -1.0

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        return {"answer": 42}


# Abstract methods to implement in children
    @abstractmethod
    def generate_obstacles(self) -> List[Obstacle]:
        raise NotImplementedError("Subclasses must implement the generate_obstacles method.")
    
    @abstractmethod
    def draw_obstacles(self):
        raise NotImplementedError("Subclasses must implement the draw_obstacles method.")
