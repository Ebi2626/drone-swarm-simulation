import itertools
import numpy as np
from astar import AStar


class UAV3DGridSearch(AStar):
    """
    A* na dyskretnej siatce 3D z pełnym sprawdzaniem kolizji segment–sfera.
    Przeszukiwanie jest ograniczone do prostokątnego bounding-boxa,
    a koszt kroku uwzględnia directional bias (preferencja góra/dół/bok).
    """

    def __init__(
        self,
        obs_pos: np.ndarray,
        obs_radius: float,
        grid_res: float,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        preferred_dir: np.ndarray | None = None,
        bias_preferred: float = 1.0,
        bias_perpendicular: float = 1.4,
        bias_oppose: float = 2.5,
    ):
        self.obs_pos = np.asarray(obs_pos, dtype=np.float64)
        self.obs_radius = float(obs_radius)
        self.obs_radius_sq = self.obs_radius ** 2
        self.grid_res = float(grid_res)

        self.bbox_min = np.asarray(bbox_min, dtype=np.float64)
        self.bbox_max = np.asarray(bbox_max, dtype=np.float64)

        if preferred_dir is not None:
            pdir = np.asarray(preferred_dir, dtype=np.float64)
            pnorm = float(np.linalg.norm(pdir))
            self.preferred_dir = pdir / pnorm if pnorm > 1e-6 else None
        else:
            self.preferred_dir = None

        self.bias_preferred = float(bias_preferred)
        self.bias_perpendicular = float(bias_perpendicular)
        self.bias_oppose = float(bias_oppose)

        # 26 kierunków sąsiedztwa (bez wektora zerowego)
        offsets = np.array(
            list(itertools.product([-grid_res, 0.0, grid_res], repeat=3)),
            dtype=np.float64,
        )
        self.directions = offsets[np.any(offsets != 0, axis=1)]

    # ------------------------------------------------------------------ #
    # Interfejs AStar                                                      #
    # ------------------------------------------------------------------ #

    def heuristic_cost_estimate(self, current, goal) -> float:
        return float(np.linalg.norm(np.array(current) - np.array(goal)))

    def distance_between(self, n1, n2) -> float:
        a = np.array(n1, dtype=np.float64)
        b = np.array(n2, dtype=np.float64)
        step = b - a
        step_len = float(np.linalg.norm(step))
        if step_len < 1e-9:
            return 0.0

        if self.preferred_dir is None:
            return step_len

        alignment = float(np.dot(step / step_len, self.preferred_dir))
        if alignment > 0.5:
            mult = self.bias_preferred
        elif alignment < -0.5:
            mult = self.bias_oppose
        else:
            mult = self.bias_perpendicular
        return step_len * mult

    def is_goal_reached(self, current, goal) -> bool:
        return (
            float(np.linalg.norm(np.array(current) - np.array(goal)))
            <= self.grid_res * 1.5
        )

    def neighbors(self, node):
        node_arr = np.array(node, dtype=np.float64)
        candidates = node_arr + self.directions

        in_bbox = np.all(
            (candidates >= self.bbox_min) & (candidates <= self.bbox_max),
            axis=1,
        )
        candidates = candidates[in_bbox]
        if len(candidates) == 0:
            return

        valid = []
        obs = self.obs_pos

        for c in candidates:
            diff_c = c - obs
            if float(np.dot(diff_c, diff_c)) < self.obs_radius_sq:
                continue  # węzeł wewnątrz sfery

            ab = c - node_arr
            ap = obs - node_arr
            ab_sq = float(np.dot(ab, ab))
            if ab_sq < 1e-12:
                valid.append(c)
                continue

            t = float(np.clip(np.dot(ap, ab) / ab_sq, 0.0, 1.0))
            closest = node_arr + t * ab
            diff_seg = closest - obs
            if float(np.dot(diff_seg, diff_seg)) < self.obs_radius_sq:
                continue  # segment przecina sferę

            valid.append(c)

        yield from map(tuple, valid)
