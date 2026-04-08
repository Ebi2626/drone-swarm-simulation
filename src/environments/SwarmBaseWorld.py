import inspect

from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.abstraction.generate_world_boundaries import WorldData
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from abc import abstractmethod
from gymnasium import spaces
import numpy as np
import pybullet as p


class SwarmBaseWorld(BaseAviary):
    def __init__(
        self,
        world_data: WorldData,
        obstacles_data: ObstaclesData,
        **kwargs
    ):
        self.bounds: WorldData = world_data
        self.obstacles: ObstaclesData = obstacles_data
        self.ground_position = world_data.min_bounds[2]  # ground_height zapisany w min_bounds[2]

        valid_keys = inspect.signature(BaseAviary.__init__).parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        super().__init__(**filtered_kwargs)
    # ------------------------------------------------------------------ #
    # Reset & render                                                       #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        print("[DEBUG]: SwarmBaseWorld Reset()")
        obs, info = super().reset(seed=seed, options=options)
        if self.GUI:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return obs, info

    def _init_render_silently(self):
        print("[DEBUG]: SwarmBaseWorld init_render_silently()")
        if self.GUI:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    def _finalize_render(self):
        print("finalize_render()")
        if self.GUI:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # ------------------------------------------------------------------ #
    # Building world geometry from world_data                             #
    # ------------------------------------------------------------------ #

    def _clear_default_plane(self):
        print("[DEBUG] Deleting default plane...")
        for i in range(p.getNumBodies()):
            if "plane" in p.getBodyInfo(i)[1].decode("utf-8"):
                p.removeBody(i)

    def _create_ground(self, color=[0.8, 0.5, 0.55, 1.0], thickness=0.1):
        print("_create_ground()")
        width  = self.bounds.dimensions[0]
        length = self.bounds.dimensions[1]

        # Zastosuj dodatkowy margines (np. 100) do całych wymiarów, a potem podziel na pół:
        half_len = (length + 100.0) / 2
        half_wid = (width + 100.0) / 2
        z_pos = self.ground_position - (thickness / 2.0)

        col_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half_wid, half_len, thickness / 2.0]
        )
        vis_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[half_wid, half_len, thickness / 2.0],
            rgbaColor=color
        )
        
        # Ważne: Środek prostokąta musi uwzględniać połowę obu osi, a rozmiary do halfExtents muszą być zmapowane X: width, Y: length
        p.createMultiBody(0, col_id, vis_id, [width / 2.0, length / 2.0, z_pos])
        print(f"[DEBUG] Ground created ({length}x{width})")

    def _create_ceiling(self):
        print("_create_ceiling()")
        width  = self.bounds.dimensions[0]  # Oś X
        length = self.bounds.dimensions[1]  # Oś Y
        height = self.bounds.dimensions[2]  # Oś Z

        # Najpierw dodajemy margines przestrzenny (np. 100), a na koniec dzielimy na pół do halfExtents
        half_wid = (width + 100.0) / 2.0
        half_len = (length + 100.0) / 2.0

        col_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half_wid, half_len, 0.5]
        )
        vis_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[half_wid, half_len, 0.5],
            rgbaColor=[0, 0, 1, 0.1]
        )
        
        # Środek sufitu musi być w połowie realnej szerokości i długości (bez nałożonego marginesu wizualnego)
        p.createMultiBody(0, col_id, vis_id, [width / 2.0, length / 2.0, height])
        print(f"[DEBUG] Ceiling created at height {height}m")

    def _setup_environment(self, ground_color=[0.5, 0.5, 0.55, 1.0]):
        """Używa self.bounds — nie wymaga już ręcznego przekazywania wymiarów."""
        print("[DEBUG] Setting up environment...")
        self._create_ground(color=ground_color)
        if self.bounds.dimensions[2] > 0:
            self._create_ceiling()
        print("[DEBUG] Environment setup complete.")

    # ------------------------------------------------------------------ #
    # Needed by BaseAviary / Gymnasium                                     #
    # ------------------------------------------------------------------ #

    def _addObstacles(self):
        self._clear_default_plane()        # usuń plane.urdf BaseAviary
        self._setup_environment()          # podłoże + sufit (wspólne)
        self._init_render_silently()
        self.draw_obstacles()              # rysowanie przeszkód 
        self._finalize_render()
        
    def _actionSpace(self):
        print("_actionSpace()")
        act_lower = np.array([[0., 0., 0., 0.]] * self.NUM_DRONES)
        act_upper = np.array([[self.MAX_RPM] * 4]  * self.NUM_DRONES)
        return spaces.Box(low=act_lower, high=act_upper, dtype=np.float32)

    def _observationSpace(self):
        print("_observationSpace()")
        return spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.NUM_DRONES, 20), dtype=np.float32
        )

    def _computeObs(self):                  return self._getDroneStateVector(0)
    def _preprocessAction(self, action):    return action
    def _computeReward(self):               return -1.0
    def _computeTerminated(self):           return False
    def _computeTruncated(self):            return False
    def _computeInfo(self):                 return {"answer": 42}

    # ------------------------------------------------------------------ #
    # ABSTRAKCJA — każda klasa potomna MUSI to zaimplementować           #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def draw_obstacles(self) -> None:
        """Rysuje przeszkody specyficzne dla danego środowiska."""
        raise NotImplementedError
    
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

