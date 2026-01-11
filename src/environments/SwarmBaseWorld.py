from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gymnasium import spaces
import numpy as np
import pybullet as p

class SwarmBaseWorld(BaseAviary):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    # Common environment setup methods
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

    def _setup_environment(self, length, width, ceiling_height, ground_color):
        print("[DEBUG] Setting up environment (ground and ceiling)...")
        self._clear_default_plane()
        self._create_ground(length, width, color=ground_color)
        if ceiling_height > 0:
            self._create_ceiling(length, width, ceiling_height)

        print("[DEBUG] Environment setup complete.")

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
