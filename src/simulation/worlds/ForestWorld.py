import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics # <--- FIX 2: Importy Enumów
from src.simulation.worlds.SwarmBaseWorld import SwarmBaseWorld

class ForestWorld(SwarmBaseWorld):
    def __init__(self, 
                 drone_model: DroneModel = DroneModel.CF2X, # <--- FIX 3: Domyślne wartości
                 physics: Physics = Physics.PYB,
                 initial_xyzs=None,
                 initial_rpys=None,
                 num_trees: int = 100, 
                 track_length: float = 1000.0,
                 tree_height: float = 10.0,
                 ceiling_height: float = 12.0,
                 **kwargs):
        
        # --- FIX 4: Konwersja String -> Enum (dla Hydry) ---
        if isinstance(drone_model, str):
            try:
                drone_model = DroneModel[drone_model]
            except KeyError:
                drone_model = DroneModel.CF2X
                
        if isinstance(physics, str):
            try:
                physics = Physics[physics]
            except KeyError:
                physics = Physics.PYB

        # --- FIX 5: Konwersja ListConfig -> Numpy Array ---
        if initial_xyzs is not None:
            initial_xyzs = np.array(list(initial_xyzs))
        if initial_rpys is not None:
            initial_rpys = np.array(list(initial_rpys))
            

        if 'obstacles' in kwargs:
            del kwargs['obstacles']

        # Przypisanie parametrów lasu
        self.num_trees = num_trees
        self.track_length = track_length
        self.tree_height = tree_height
        self.ceiling_height = ceiling_height
        self.start_safe_zone = 20.0
        self.end_safe_zone = 20.0
        
        # Wywołanie konstruktora klasy bazowej z poprawionymi typami
        super().__init__(drone_model=drone_model,
                         physics=physics,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         obstacles=True,
                         **kwargs)

    def _addObstacles(self):
        print(f"[DEBUG] Generowanie lasu: {self.num_trees} drzew...") 
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        """Generuje las oraz sufit operacyjny."""
        
        # --- FIX 7: Załadowanie podłoża ---
        # FIX: Zamiast standardowego plane.urdf, tworzymy własne podłoże

        # 1 - usuwamy oryginalne podłoże:
        num_bodies = p.getNumBodies()
        for i in range(num_bodies):
            # Sprawdzamy czy ciało nie jest dronem (Twoje drony mają ID w self.DRONE_IDS)
            # Uwaga: self.DRONE_IDS może być jeszcze puste lub niepełne w tym momencie initu.
            # Bezpieczniejsza metoda: Sprawdzamy nazwę ciała.
            body_info = p.getBodyInfo(i)
            body_name = body_info[1].decode("utf-8")
            
            # Plane z pybullet_data nazywa się zazwyczaj "plane"
            if "plane" in body_name:
                p.removeBody(i)
                print("[DEBUG] Usunięto domyślną szachownicę (plane.urdf)")
        # 2 - tworzymy własne podłoże z boxów
        # Rozciągamy je na całą długość trasy (track_length) + marginesy
        # Połowa długości i szerokości (halfExtents)
        ground_thickness = 0.002  # Grubość podłoża 2mm
        air_gap = 0.01
        z_pos = air_gap + ground_thickness / 2  # Minimalne uniesienie nad zerem Z

        ground_half_length = self.track_length / 2 + 50.0 
        ground_half_width = 100.0 
        
        ground_col = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=[ground_half_length, ground_half_width, ground_thickness / 2.0]
        )
        
        ground_vis = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[ground_half_length, ground_half_width, ground_thickness / 2.0],
            rgbaColor=[0.3, 0.5, 0.3, 1.0] # Kolor: Ciemna zieleń (trawa)
        )
        
        p.createMultiBody(
            baseMass=0, # Statyczny obiekt
            baseCollisionShapeIndex=ground_col,
            baseVisualShapeIndex=ground_vis,
            basePosition=[self.track_length / 2, 0, z_pos]
        )

        # Logika generowania lasu (bez zmian)
        forest_start_x = self.start_safe_zone
        forest_end_x = self.track_length - self.end_safe_zone
        corridor_width = 40.0

        np.random.seed(42)

        for _ in range(self.num_trees):
            x = np.random.uniform(forest_start_x, forest_end_x)
            y = np.random.uniform(-corridor_width/2, corridor_width/2)
            
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.5 + np.random.uniform(0, 0.5),
                height=self.tree_height
            )
            
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.5,
                length=self.tree_height,
                rgbaColor=[0.4, 0.25, 0.1, 1]
            )

            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[x, y, self.tree_height / 2]
            )

        # Sufit (Geofencing)
        ceiling_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.track_length/2 + 50, 50, 0.5]
        )
        ceiling_visual = p.createVisualShape(
             p.GEOM_BOX,
             halfExtents=[self.track_length/2 + 50, 50, 0.5],
             rgbaColor=[0, 0, 1, 0.1]
        )
        
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ceiling_shape,
            baseVisualShapeIndex=ceiling_visual,
            basePosition=[self.track_length/2, 0, self.ceiling_height]
        )

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        print("[DEBUG] Las wygenerowany.")
