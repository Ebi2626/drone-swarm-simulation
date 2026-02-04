import numpy as np
from typing import Any, Dict, Optional, List
from numpy.typing import NDArray
import scipy.interpolate as si

# Pymoo imports
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

# Twoje importy
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.abstraction.generate_obstacles import ObstaclesData

def get_xp(arr):
    return np

def precompute_bspline_basis(n_waypoints: int, n_control_points_inner: int):
    """
    Prekomputacja macierzy bazowej B-Spline.
    Wynikowa trajektoria T = Basis @ ControlPoints_Full
    ControlPoints_Full ma rozmiar (n_control_inner + 2, 3) (Start + Inner + End)
    """
    # Liczba punktów kontrolnych łącznie ze startem i metą
    n_total = n_control_points_inner + 2
    degree = min(3, n_total - 1)
    
    # POPRAWKA: Węzły (Knots) - clamped uniform
    # Aby B-spline przechodził przez skrajne punkty, węzły muszą mieć krotność degree+1 na końcach.
    # Liczba węzłów = n_total + degree + 1
    # Struktura: [0, 0, 0, 0, t1, t2, ..., 1, 1, 1, 1] dla degree=3
    
    n_internal_knots = n_total - (degree + 1)
    
    if n_internal_knots < 0:
        # Fallback dla bardzo małej liczby punktów (np. 2 punkty, stopień 1)
        knots = np.concatenate([np.zeros(degree+1), np.ones(degree+1)])
    else:
        # Wewnętrzne węzły równomiernie rozmieszczone
        internal_knots = np.linspace(0, 1, n_internal_knots + 2)[1:-1]
        knots = np.concatenate([
            np.zeros(degree + 1), 
            internal_knots, 
            np.ones(degree + 1)
        ])
    
    # Ewaluacja w n_waypoints punktach czasu
    t_eval = np.linspace(0, 1, n_waypoints)
    
    # Macierz bazowa
    basis_matrix = np.zeros((n_waypoints, n_total))
    
    for i in range(n_total):
        c = np.zeros(n_total)
        c[i] = 1.0
        # Tworzymy obiekt B-Spline
        bspl = si.BSpline(knots, c, degree)
        basis_matrix[:, i] = bspl(t_eval)
        
    return basis_matrix

def constr_inter_agent_separation_segments(
    trajectories: NDArray[np.float64], 
    min_dist: float = 1.5,
    ignore_ratio: float = 0.1
) -> NDArray[np.float64]:
    xp = get_xp(trajectories)
    pop_size, n_drones, n_steps, _ = trajectories.shape
    
    if ignore_ratio >= 0.5:
        ignore_ratio = 0.45 
        
    start_idx = int(n_steps * ignore_ratio)
    end_idx = int(n_steps * (1.0 - ignore_ratio))
    
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = n_steps
    
    total_cv = xp.zeros(pop_size)
    
    for t in range(start_idx, end_idx):
        pos_t = trajectories[:, :, t, :]
        diff = pos_t[:, :, None, :] - pos_t[:, None, :, :]
        dist_sq = xp.sum(diff**2, axis=-1)
        dist = xp.sqrt(dist_sq)
        
        violation_mask = (dist < min_dist) & (dist > 1e-5)
        val = xp.maximum(0.0, min_dist - dist)
        val = val * violation_mask
        
        step_cv = xp.sum(xp.sum(val, axis=-1), axis=-1) / 2.0
        total_cv += step_cv
        
    return total_cv

def constr_static_obstacles_vectorized(
    trajectories: NDArray[np.float64],
    obstacles_list: List[ObstaclesData],
    safety_margin: float = 1.0
) -> NDArray[np.float64]:
    xp = get_xp(trajectories)
    pop_size, n_drones, n_steps, _ = trajectories.shape
    
    flat_traj = trajectories.reshape(-1, 3)
    tx, ty, tz = flat_traj[:, 0], flat_traj[:, 1], flat_traj[:, 2]
    
    total_pen_flat = xp.zeros(flat_traj.shape[0])
    
    for batch in obstacles_list:
        data = batch.data
        obs_x = data[:, 0]
        obs_y = data[:, 1]
        obs_z = data[:, 2]
        d1 = data[:, 3]
        d2 = data[:, 4]
        d3 = data[:, 5]
        
        if batch.shape_type == 'CYLINDER':
            dx = tx[:, None] - obs_x[None, :]
            dy = ty[:, None] - obs_y[None, :]
            dist_xy = xp.sqrt(dx**2 + dy**2)
            
            radii = d1[None, :] + safety_margin
            heights = d2[None, :]
            z_bottom = obs_z[None, :]
            z_top = obs_z[None, :] + heights
            
            h_mask = (tz[:, None] >= z_bottom) & (tz[:, None] <= z_top)
            rad_pen = xp.maximum(0.0, radii - dist_xy)
            total_pen_flat += xp.sum(rad_pen * h_mask, axis=1)
            
        elif batch.shape_type == 'BOX':
            min_x = obs_x - d1/2 - safety_margin
            max_x = obs_x + d1/2 + safety_margin
            min_y = obs_y - d2/2 - safety_margin
            max_y = obs_y + d2/2 + safety_margin
            min_z = obs_z - d3/2 - safety_margin
            max_z = obs_z + d3/2 + safety_margin
            
            in_x = (tx[:, None] >= min_x[None, :]) & (tx[:, None] <= max_x[None, :])
            in_y = (ty[:, None] >= min_y[None, :]) & (ty[:, None] <= max_y[None, :])
            in_z = (tz[:, None] >= min_z[None, :]) & (tz[:, None] <= max_z[None, :])
            
            is_inside = in_x & in_y & in_z
            total_pen_flat += xp.sum(is_inside.astype(float), axis=1)

    total_pen = total_pen_flat.reshape(pop_size, n_drones, n_steps)
    return xp.sum(xp.sum(total_pen, axis=-1), axis=-1)

class VectorizedGlobalSwarmProblem(Problem):
    def __init__(
        self, 
        start_positions: NDArray, 
        target_positions: NDArray, 
        obstacles_list: list,
        world_bounds: NDArray,
        n_waypoints: int,
        n_drones: int,
        n_control_points: int = 3
    ):
        self.starts = start_positions
        self.targets = target_positions
        self.obstacles = obstacles_list
        self.n_waypoints = n_waypoints
        self.n_drones = n_drones
        self.n_control = n_control_points
        
        self.basis_matrix = precompute_bspline_basis(n_waypoints, n_control_points)
        
        dim = n_drones * n_control_points * 3
        
        xl_one = world_bounds[:, 0]
        xu_one = world_bounds[:, 1]
        xl = np.tile(xl_one, n_drones * n_control_points)
        xu = np.tile(xu_one, n_drones * n_control_points)
        
        super().__init__(n_var=dim, n_obj=1, n_ieq_constr=2, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        pop_size = x.shape[0]
        inner_cps = x.reshape((pop_size, self.n_drones, self.n_control, 3))
        
        starts_bc = np.tile(self.starts[None, :, None, :], (pop_size, 1, 1, 1))
        targets_bc = np.tile(self.targets[None, :, None, :], (pop_size, 1, 1, 1))
        
        all_cps = np.concatenate([starts_bc, inner_cps, targets_bc], axis=2)
        
        trajectories = np.einsum('sk,pdkc->pdsc', self.basis_matrix, all_cps)
        
        diffs = np.diff(trajectories, axis=2)
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        total_lengths = np.sum(np.sum(dists, axis=-1), axis=-1)
        
        static_pen = constr_static_obstacles_vectorized(trajectories, self.obstacles)
        agent_pen = constr_inter_agent_separation_segments(trajectories, min_dist=1.5)
        
        out["F"] = total_lengths
        out["G"] = np.column_stack([static_pen, agent_pen])


def nsga3_swarm_strategy(
    *, 
    start_positions: NDArray[np.float64],
    target_positions: NDArray[np.float64],
    obstacles_data: ObstaclesData,
    world_data: WorldData,
    number_of_waypoints: int,
    drone_swarm_size: int, 
    algorithm_params: Optional[Dict[str, Any]] = None
) -> NDArray[np.float64]:
    
    if isinstance(obstacles_data, list):
        obs_list = obstacles_data
    else:
        obs_list = [obstacles_data]

    params = algorithm_params or {}
    pop_size = params.get("pop_size", 100)
    n_gen = params.get("n_gen", 50)
    n_control = params.get("n_control_points", 3)
    
    if "swarm_start_pos" in params:
        start_positions = params["swarm_start_pos"]
        
    try:
        bounds = world_data.bounds
    except AttributeError:
        mx, my, mz = world_data.max_bounds
        bounds = np.array([[0, mx], [0, my], [0, mz]])

    print(f"NSGA-III (Clamped B-Spline). Pop: {pop_size}, Gen: {n_gen}, Bounds: {bounds[:,1]}")

    problem = VectorizedGlobalSwarmProblem(
        start_positions=start_positions,
        target_positions=target_positions,
        obstacles_list=obs_list,
        world_bounds=bounds,
        n_waypoints=number_of_waypoints,
        n_drones=drone_swarm_size,
        n_control_points=n_control
    )
    
    ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=12)
    
    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_gen),
        seed=42,
        verbose=True
    )
    
    if res.X is not None:
        best_X = res.X[0] if len(res.X.shape) > 1 else res.X
        
        inner_cps = best_X.reshape((1, drone_swarm_size, n_control, 3))
        starts_bc = start_positions[None, :, None, :]
        targets_bc = target_positions[None, :, None, :]
        all_cps = np.concatenate([starts_bc, inner_cps, targets_bc], axis=2)
        
        traj = np.einsum('sk,pdkc->pdsc', problem.basis_matrix, all_cps)
        return traj[0]
    else:
        print("Nie znaleziono rozwiązania.")
        return np.zeros((drone_swarm_size, number_of_waypoints, 3))
