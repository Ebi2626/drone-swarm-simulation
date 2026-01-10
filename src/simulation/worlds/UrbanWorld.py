import numpy as np
import pybullet as p
from src.simulation.worlds.SwarmBaseWorld import SwarmBaseWorld

class UrbanWorld(SwarmBaseWorld):
    def __init__(self,
                 city_length: float = 1000.0,
                 city_width: float = 200.0,
                 block_size: float = 40.0,    # Rozmiar podstawy budynku
                 street_width: float = 15.0,  # Szerokość ulicy
                 min_height: float = 10.0,    # Niskie kamienice
                 max_height: float = 80.0,    # Wieżowce
                 skyscraper_prob: float = 0.1, # 10% szans na bardzo wysoki budynek
                 **kwargs):
        
        self.city_length = city_length
        self.city_width = city_width
        self.block_size = block_size
        self.street_width = street_width
        self.min_height = min_height
        self.max_height = max_height
        self.skyscraper_prob = skyscraper_prob
        
        super().__init__(**kwargs)

    def _addObstacles(self):
        p.loadURDF("plane.urdf")
        
        # Obliczenie geometrii siatki miasta
        step = self.block_size + self.street_width
        
        # Generujemy budynki w pętlach (siatka ulic)
        # Zakres X: od początku strefy bezpiecznej (np. 50m) do końca
        start_x = 50.0
        end_x = self.city_length - 50.0
        
        # Zakres Y: symetrycznie wokół osi lotu
        start_y = -self.city_width / 2
        end_y = self.city_width / 2
        
        np.random.seed(101) # Inny seed niż w lesie dla różnorodności badań
        
        current_x = start_x
        while current_x < end_x:
            current_y = start_y
            while current_y < end_y:
                
                # Decyzja o wysokości:
                # Jeśli wylosujemy "skyscpraper", budynek jest wyższy niż max_height
                # To zmusza drony do ominięcia go bokiem, a nie górą.
                if np.random.random() < self.skyscraper_prob:
                    h = self.max_height * 1.5 # 120m - trudne do przelotu górą
                    color = [0.3, 0.3, 0.35, 1] # Ciemnoszary
                else:
                    h = np.random.uniform(self.min_height, self.max_height)
                    shade = np.random.uniform(0.5, 0.8)
                    color = [shade, shade, shade, 1] # Odcienie betonu
                
                # Tworzenie budynku (Box)
                # PyBullet używa halfExtents (połowa wymiaru)
                collision_shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[self.block_size/2, self.block_size/2, h/2]
                )
                
                visual_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[self.block_size/2, self.block_size/2, h/2],
                    rgbaColor=color
                )
                
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=visual_shape,
                    basePosition=[current_x, current_y, h/2]
                )
                
                current_y += step
            current_x += step

        # Opcjonalnie: Sufit ograniczający (jak w lesie), 
        # ale w mieście często chcemy pozwolić na lot "nad miastem" 
        # tylko jeśli koszt energetyczny wznoszenia jest wysoki.
        # W tym kodzie nie dodaję sufitu, zakładając że 120m wieżowce 
        # są wystarczającą barierą.
