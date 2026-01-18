import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from ElementWiseProblem import ElementwiseProblem

class SwarmDroneOptimizer:
    def __init__(self, 
                 space_limits: list, # limitations of the space [max_x, max_y, max_z]
                 n_drones: int, # number of drones in the swarm
                 n_waypoints: int, # number of waypoints per drone
                 start_positions: list, # starting positions for each drone
                 end_positions: list, # ending positions for each drone
                 obstacles: np.ndarray, # array of obstacles [x, y, radius, height]
                 **kwargs):
        super().__init__(**kwargs)
        self.problem = ElementwiseProblem(
            space_limits=space_limits,
            n_drones=n_drones,
            n_waypoints=n_waypoints,
            start_positions=start_positions,
            end_positions=end_positions,
            obstacles=obstacles)

    def run_optimization(self, pop_size=100, n_gen=200):
        # Konfiguracja algorytmu NSGA-III
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
        algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)

        # Uruchomienie minimalizacji
        res = minimize(
            self.problem,
            algorithm,
            termination=('n_gen', n_gen),
            seed=1,
            verbose=True # Warto widzieć postęp
        )
        return res
    
    def get_best_trajectories(self, res, n=5):
        if res.X is None:
            raise ValueError("Optymalizacja nie zwróciła wyników (res.X is None)")

        # 1. Sortowanie wyników względem pierwszego celu (długość trasy)
        sorted_indices = np.argsort(res.F[:, 0])
        
        # 2. Wybranie n najlepszych
        # Zabezpieczenie: jeśli znaleziono mniej rozwiązań niż n, bierzemy wszystkie
        n = min(n, len(sorted_indices))
        best_indices = sorted_indices[:n]
        
        best_X = res.X[best_indices]
        
        # 3. POPRAWNY RESHAPE
        # Wymiary: (Liczba rozwiązań, Liczba dronów, Liczba punktów, Współrzędne XYZ)
        trajectories = best_X.reshape((n, self.n_drones, self.n_waypoints, 3))
        
        return trajectories
