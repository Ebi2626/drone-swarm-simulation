import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from src.environments.SwarmBaseWorld import SwarmBaseWorld
from src.utils.config_parser import sanitize_init_params

class UrbanWorld(SwarmBaseWorld):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 physics: Physics = Physics.PYB,
                 initial_xyzs=None,
                 initial_rpys=None,
                 city_length: float = 1000.0,
                 city_width: float = 200.0,
                 block_size: float = 40.0,
                 street_width: float = 15.0,
                 min_height: float = 10.0,
                 max_height: float = 80.0,
                 skyscraper_prob: float = 0.1,
                 street_block_prob: float = 0.2,
                 ceiling_height: float = 120.0,
                 **kwargs):
        
        drone_model, physics, initial_xyzs, initial_rpys = sanitize_init_params(
            drone_model, physics, initial_xyzs, initial_rpys
        )
        
        self.city_length = city_length
        self.city_width = city_width
        self.block_size = block_size
        self.street_width = street_width
        self.min_height = min_height
        self.max_height = max_height
        self.skyscraper_prob = skyscraper_prob
        self.street_block_prob = street_block_prob
        self.ceiling_height = ceiling_height
        
        super().__init__(drone_model=drone_model,
                         physics=physics,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         obstacles=True,
                         **kwargs)

    def _addObstacles(self):
        print("[DEBUG] Generating urban environment...") 
        self._init_render_silently()
        self._setup_environment(self.city_length, self.city_width, self.ceiling_height, ground_color=[0.4, 0.4, 0.45, 1.0])
        self._createCity()
        print("[DEBUG] Urban environment generated.")

    def _create_building(self, x, y, len_x, len_y, is_blocker=False):
        shade = np.random.uniform(0.6, 0.9)

        # Blocker is a building crossing the street to avoid
        # easy passage for the drones in straight line
        if is_blocker:
            h = np.random.uniform(self.min_height, self.max_height * 0.7)
            color = [shade, shade, shade, 1]
        elif np.random.random() < self.skyscraper_prob:
            h = self.max_height * 1.5
            color = [0.2, 0.2, 0.3, 1]
        else:
            h = np.random.uniform(self.min_height, self.max_height)
            color = [shade, shade, shade, 1]

        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[len_x/2, len_y/2, h/2])
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[len_x/2, len_y/2, h/2], rgbaColor=color)
        
        p.createMultiBody(0, col_shape, vis_shape, [x, y, h/2])

    def _createCity(self):
        step = self.block_size + self.street_width
        start_x = 20.0
        end_x = self.city_length - 20.0
        start_y = -self.city_width / 2
        end_y = self.city_width / 2
        
        np.random.seed(101) 
        
        current_x = start_x
        while current_x < end_x:
            current_y = start_y
            while current_y < end_y:
                
                self._create_building(current_x, current_y, self.block_size, self.block_size)
                
                if np.random.random() < self.street_block_prob:
                    blocker_y = current_y + self.block_size/2 + self.street_width/2
                    
                    if blocker_y < end_y:
                        self._create_building(
                            x=current_x, 
                            y=blocker_y, 
                            len_x=self.block_size, 
                            len_y=self.street_width + 0.5, 
                            is_blocker=True
                        )
                current_y += step
            current_x += step
