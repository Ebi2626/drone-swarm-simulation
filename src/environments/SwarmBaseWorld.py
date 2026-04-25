import inspect
from abc import abstractmethod
import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.abstraction.generate_world_boundaries import WorldData

class SwarmBaseWorld(BaseAviary):
    def __init__(
        self,
        world_data: WorldData,
        obstacles_data: ObstaclesData,
        num_drones: int | None = None,
        primary_num_drones: int | None = None,
        dynamic_obstacles_enabled: bool = False,
        num_dynamic_obstacles: int = 0,
        initial_xyzs=None,
        initial_rpys=None,
        **kwargs
    ):
        self.bounds: WorldData = world_data
        self.obstacles: ObstaclesData = obstacles_data
        self.ground_position = world_data.min_bounds[2]

        inferred_total_agents = num_drones
        if inferred_total_agents is None and primary_num_drones is None:
            raise ValueError("Brak informacji o liczbie agentów.")

        self.primary_num_drones = int(primary_num_drones if primary_num_drones is not None else inferred_total_agents)
        self.dynamic_obstacles_enabled = bool(dynamic_obstacles_enabled)
        self.num_dynamic_obstacles = int(num_dynamic_obstacles if self.dynamic_obstacles_enabled else 0)
        self.total_agents = int(inferred_total_agents if inferred_total_agents is not None else self.primary_num_drones + self.num_dynamic_obstacles)

        self.ground_body_id = None
        self.ceiling_body_id = None

        # Tylko bezpieczne klucze z kwargs trafiają do BaseAviary
        valid_keys = inspect.signature(BaseAviary.__init__).parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

        # Kluczowe argumenty tablicowe przekazujemy z ominięciem kwargs!
        super().__init__(
            num_drones=self.total_agents,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            **filtered_kwargs
        )
    # ------------------------------------------------------------------ #
    # Reset & render                                                     #
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
        print("[DEBUG]: SwarmBaseWorld finalize_render()")
        if self.GUI:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # ------------------------------------------------------------------ #
    # Helpers: role / indexing                                           #
    # ------------------------------------------------------------------ #

    def get_primary_agent_indices(self):
        return list(range(self.primary_num_drones))

    def get_dynamic_obstacle_indices(self):
        if not self.dynamic_obstacles_enabled:
            return []
        return list(range(self.primary_num_drones, self.total_agents))

    def is_dynamic_obstacle(self, agent_idx: int) -> bool:
        return self.dynamic_obstacles_enabled and self.primary_num_drones <= agent_idx < self.total_agents

    def get_agent_index_from_body_id(self, body_id: int):
        for idx, drone_body_id in enumerate(self.DRONE_IDS):
            if drone_body_id == body_id:
                return idx
        return None

    def get_body_role(self, body_id: int) -> str:
        agent_idx = self.get_agent_index_from_body_id(body_id)
        if agent_idx is not None:
            return "dynamic_obstacle" if self.is_dynamic_obstacle(agent_idx) else "drone"
        if body_id == self.ground_body_id:
            return "ground"
        if body_id == self.ceiling_body_id:
            return "ceiling"
        return "static_obstacle"

    # ------------------------------------------------------------------ #
    # Building world geometry                                            #
    # ------------------------------------------------------------------ #

    def _clear_default_plane(self):
        print("[DEBUG] Deleting default plane...")
        body_ids = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
        for body_id in reversed(body_ids):
            body_info = p.getBodyInfo(body_id)
            body_name = body_info[1].decode("utf-8").lower()
            if "plane" in body_name:
                p.removeBody(body_id)

    def _create_ground(self, color=[0.8, 0.5, 0.55, 1.0], thickness=0.1):
        print("[DEBUG] _create_ground()")
        width = self.bounds.dimensions[0]
        length = self.bounds.dimensions[1]

        half_len = (length + 100.0) / 2.0
        half_wid = (width + 100.0) / 2.0
        z_pos = self.ground_position - (thickness / 2.0)

        col_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half_wid, half_len, thickness / 2.0]
        )
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half_wid, half_len, thickness / 2.0],
            rgbaColor=color
        )

        self.ground_body_id = p.createMultiBody(
            0, col_id, vis_id, [width / 2.0, length / 2.0, z_pos]
        )
        print(f"[DEBUG] Ground created ({length}x{width})")

    def _create_ceiling(self):
        print("[DEBUG] _create_ceiling()")
        width = self.bounds.dimensions[0]
        length = self.bounds.dimensions[1]
        height = self.bounds.dimensions[2]

        half_wid = (width + 100.0) / 2.0
        half_len = (length + 100.0) / 2.0

        col_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half_wid, half_len, 0.5]
        )
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half_wid, half_len, 0.5],
            rgbaColor=[0, 0, 1, 0.1]
        )

        self.ceiling_body_id = p.createMultiBody(
            0, col_id, vis_id, [width / 2.0, length / 2.0, height]
        )
        print(f"[DEBUG] Ceiling created at height {height}m")

    def _setup_environment(self, ground_color=[0.5, 0.5, 0.55, 1.0]):
        print("[DEBUG] Setting up environment...")
        self._create_ground(color=ground_color)
        if self.bounds.dimensions[2] > 0:
            self._create_ceiling()
        print("[DEBUG] Environment setup complete.")

    # ------------------------------------------------------------------ #
    # Needed by BaseAviary / Gymnasium                                   #
    # ------------------------------------------------------------------ #

    def _addObstacles(self):
        self._clear_default_plane()
        self._setup_environment()
        self._init_render_silently()
        self.draw_obstacles()
        self._finalize_render()

    def _actionSpace(self):
        print("[DEBUG] _actionSpace()")
        # self.NUM_DRONES nie jest jeszcze zainicjalizowane przez BaseAviary!
        # Używamy naszego self.total_agents zadeklarowanego przed super()
        act_lower = np.array([[0., 0., 0., 0.]] * self.total_agents)
        
        # self.MAX_RPM też nie istnieje w momencie wołania tego przez __init__ BaseAviary
        # Dlatego dla zdefiniowania samej struktury spacji można podać twardy limit
        # lub wyciągnąć go awaryjnie np. 21702.644 (wartość dla CF2X)
        fallback_rpm = getattr(self, "MAX_RPM", 21703)
        act_upper = np.array([[fallback_rpm] * 4] * self.total_agents)
        
        return spaces.Box(low=act_lower, high=act_upper, dtype=np.float64)

    def _observationSpace(self):
        print("[DEBUG] _observationSpace()")
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.total_agents, 20),
            dtype=np.float64
        )

    def _computeObs(self):
        return np.vstack(
            [self._getDroneStateVector(i) for i in range(self.total_agents)]
        ).astype(np.float64)

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

    # ------------------------------------------------------------------ #
    # ABSTRAKCJA                                                         #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def draw_obstacles(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # COLLISIONS                                                         #
    # ------------------------------------------------------------------ #

    def get_detailed_collisions(self, include_dynamic_obstacles: bool = False):
        """
        Zwraca listę kolizji jako:
            (agent_idx, other_body_id)

        Domyślnie raportuje kolizje tylko dla głównych dronów.
        Jeśli include_dynamic_obstacles=True, sprawdza wszystkich agentów.
        """
        collisions = []
        max_agent_idx = self.NUM_DRONES if include_dynamic_obstacles else self.primary_num_drones

        for agent_idx in range(max_agent_idx):
            agent_body_id = self.DRONE_IDS[agent_idx]
            contact_points = p.getContactPoints(bodyA=agent_body_id)

            if not contact_points:
                continue

            for contact in contact_points:
                other_body_id = contact[2]
                if other_body_id != agent_body_id:
                    collisions.append((agent_idx, other_body_id))

        return list(set(collisions))

    def get_agent_collisions(self, include_dynamic_obstacles: bool = False):
        """
        Zwraca kolizje agent-agent jako:
            (agent_idx, other_agent_idx)

        Przydatne, jeśli później będziesz chciał odróżniać:
        dron-vs-dron, dron-vs-dynamic_obstacle, dynamic_obstacle-vs-dynamic_obstacle.
        """
        collisions = []
        for agent_idx, other_body_id in self.get_detailed_collisions(
            include_dynamic_obstacles=include_dynamic_obstacles
        ):
            other_agent_idx = self.get_agent_index_from_body_id(other_body_id)
            if other_agent_idx is not None and other_agent_idx != agent_idx:
                collisions.append((agent_idx, other_agent_idx))
        return list(set(collisions))