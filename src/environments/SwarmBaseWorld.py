import inspect
from abc import abstractmethod
import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.abstraction.generate_world_boundaries import WorldData

class SwarmBaseWorld(BaseAviary):
    """Bazowe środowisko PyBullet dla roju — granice świata, podłoga, sufit, agenci.

    Łączy główny rój (`primary_num_drones`) i opcjonalnych dynamicznych
    obstakli (`num_dynamic_obstacles`) jako jednorodne ciała PyBullet,
    rozróżniane semantycznie przez `is_dynamic_obstacle`/`get_body_role`.
    Konkretne klasy potomne implementują `draw_obstacles` (las, miasto, …).
    """

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

    def get_primary_agent_indices(self):
        """Zwróć listę indeksów `[0, primary_num_drones)` — głównych dronów roju."""
        return list(range(self.primary_num_drones))

    def get_dynamic_obstacle_indices(self):
        """Zwróć indeksy dynamicznych obstakli `[primary, total)`; pusta przy wyłączonych."""
        if not self.dynamic_obstacles_enabled:
            return []
        return list(range(self.primary_num_drones, self.total_agents))

    def is_dynamic_obstacle(self, agent_idx: int) -> bool:
        """`True`, gdy `agent_idx` mieści się w zakresie dynamicznych obstakli."""
        return self.dynamic_obstacles_enabled and self.primary_num_drones <= agent_idx < self.total_agents

    def get_agent_index_from_body_id(self, body_id: int):
        """Mapuj `body_id` PyBullet na indeks agenta; `None`, gdy nie należy do dronów."""
        for idx, drone_body_id in enumerate(self.DRONE_IDS):
            if drone_body_id == body_id:
                return idx
        return None

    def get_body_role(self, body_id: int) -> str:
        """Zwróć semantyczną rolę ciała: `drone / dynamic_obstacle / ground / ceiling / static_obstacle`."""
        agent_idx = self.get_agent_index_from_body_id(body_id)
        if agent_idx is not None:
            return "dynamic_obstacle" if self.is_dynamic_obstacle(agent_idx) else "drone"
        if body_id == self.ground_body_id:
            return "ground"
        if body_id == self.ceiling_body_id:
            return "ceiling"
        return "static_obstacle"

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

    # BaseAviary / Gymnasium hook.
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

    @abstractmethod
    def draw_obstacles(self) -> None:
        """Wystaw przeszkody w PyBullet — implementowane przez klasy potomne."""
        raise NotImplementedError

    def get_detailed_collisions(self, include_dynamic_obstacles: bool = False):
        """Zwróć listę `(agent_idx, other_body_id)` z fizycznych kontaktów PyBullet.

        Args:
            include_dynamic_obstacles: `True` ⇒ uwzględnij dynamiczne
                obstakle jako agentów źródłowych; domyślnie tylko `primary_num_drones`.

        Returns:
            Zbiór par bez duplikatów (kontakt z samym sobą wykluczony).
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
        """Zwróć kolizje agent-agent jako `(agent_idx, other_agent_idx)`.

        Args:
            include_dynamic_obstacles: Patrz `get_detailed_collisions`.

        Returns:
            Zbiór par agent-agent bez duplikatów; pomija ciała statyczne
            (ground/ceiling/static obstacles).
        """
        collisions = []
        for agent_idx, other_body_id in self.get_detailed_collisions(
            include_dynamic_obstacles=include_dynamic_obstacles
        ):
            other_agent_idx = self.get_agent_index_from_body_id(other_body_id)
            if other_agent_idx is not None and other_agent_idx != agent_idx:
                collisions.append((agent_idx, other_agent_idx))
        return list(set(collisions))

    # PROXIMITY-BASED inter-drone collision detection.
    # Threshold 0.5m derivowany z PyBullet LCP solver: contact impulse
    # generuje się BINARNIE przy dist≤0.12m (cylinder collision shape r=0.06m).
    # 0.15m (1.25× contact) łapał tylko fizyczny styk; tracił „near-miss"
    # gdzie drony są blisko (0.5-2m), nie dotykają, ale przy zbliżeniu PID
    # wpada w niestabilność i drone spada (test:
    # `test_near_miss_drones_must_log_inter_drone_event_not_ground`).
    #   ("Te '0' to kolizje między dronami" — user 2026-05-07).
    # - 0.5m (~4.2× contact) łapie również near-miss, dodaje ~70ms
    #   wyprzedzenia przy v=5m/s, vs 30ms dla 0.15m. Generuje minimalne
    #   false positives bo drony w roju (n=3, korytarz 600×60×11) typowo
    #   utrzymują dystans >2m (z Kamień 2 analizy real run min(0,1)=2.18m).
    INTER_DRONE_COLLISION_THRESHOLD_M = 0.5

    def get_inter_drone_proximity_collisions(
        self, threshold_m: float | None = None,
    ) -> list[tuple[int, int, float]]:
        """Wykryj pary dronów (primary) z odległością center-to-center ≤ `threshold_m`.

        Komplementarne do `get_detailed_collisions`, który wymaga fizycznego
        kontaktu (LCP) — proximity wyłapuje również near-miss.

        Args:
            threshold_m: Próg odległości [m]; `None` ⇒ `INTER_DRONE_COLLISION_THRESHOLD_M`.

        Returns:
            Lista `(agent_idx_a, agent_idx_b, distance_m)` z `a < b`; pusta,
            gdy żadne pary nie kwalifikują się.
        """
        if threshold_m is None:
            threshold_m = self.INTER_DRONE_COLLISION_THRESHOLD_M

        n = self.primary_num_drones
        if n < 2:
            return []

        # Pobierz pozycje wszystkich primary drones jednym przebiegiem.
        positions = np.array(
            [p.getBasePositionAndOrientation(self.DRONE_IDS[i])[0] for i in range(n)],
            dtype=np.float64,
        )

        pairs: list[tuple[int, int, float]] = []
        for a in range(n):
            for b in range(a + 1, n):
                dist = float(np.linalg.norm(positions[a] - positions[b]))
                if dist <= threshold_m:
                    pairs.append((a, b, dist))
        return pairs

    def get_all_inter_drone_collisions(
        self, threshold_m: float | None = None,
    ) -> list[tuple[int, int, float, str]]:
        """Połącz fizyczne (LCP) i proximity-based kolizje dron-dron do jednej listy.

        Args:
            threshold_m: Próg dla wykrywania proximity (`None` ⇒ wartość domyślna).

        Returns:
            Lista `(agent_idx_a, agent_idx_b, distance_m, source)` z
            `source ∈ {"contact", "proximity"}`. Para wykryta oboma metodami
            raportowana jest raz jako `"contact"` (silniejszy wskaźnik).
        """
        # Fizyczne kontakty: agent-agent pairs z get_agent_collisions, tylko
        # primary drones (nie dynamic obstacles).
        contact_pairs: dict[tuple[int, int], float] = {}
        for a, b in self.get_agent_collisions(include_dynamic_obstacles=False):
            if b >= self.primary_num_drones:
                continue  # b to dynamic obstacle — nie inter-drone
            key = (min(int(a), int(b)), max(int(a), int(b)))
            if key in contact_pairs:
                continue
            # Wyznacz dist dla raportu (z getClosestPoints, nie wymaga LCP).
            try:
                cp = p.getClosestPoints(
                    bodyA=self.DRONE_IDS[key[0]],
                    bodyB=self.DRONE_IDS[key[1]],
                    distance=10.0,
                )
                dist = min((c[8] for c in cp), default=0.0) + 2 * 0.06  # edge→center
            except Exception:
                dist = 0.0
            contact_pairs[key] = float(max(0.0, dist))

        out: list[tuple[int, int, float, str]] = [
            (a, b, d, "contact") for (a, b), d in contact_pairs.items()
        ]

        # Proximity (bez fizycznego kontaktu) — dodaj tylko pary których
        # nie ma w `contact_pairs`.
        for a, b, d in self.get_inter_drone_proximity_collisions(threshold_m):
            key = (a, b)
            if key in contact_pairs:
                continue
            out.append((a, b, d, "proximity"))

        return out