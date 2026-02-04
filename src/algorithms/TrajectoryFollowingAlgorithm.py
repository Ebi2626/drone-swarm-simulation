import numpy as np
from src.algorithms.BaseAlgorithm import BaseAlgorithm
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
import matplotlib.pyplot as plt

class TrajectoryFollowingAlgorithm(BaseAlgorithm):
    def __init__(self, parent, num_drones, params=None):
        super().__init__(parent, num_drones, params)
        self.parent = parent
        
        # Inicjalizacja kontrolerów PID dla każdego drona
        self.controllers = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(num_drones)]
        
        # Cache na trajektorie (żeby nie przetwarzać ich w każdej klatce)
        self._cached_trajectories = None
        
        # Indeksy śledzące postęp każdego drona (waypoint index)
        self.current_waypoint_indices = np.zeros(num_drones, dtype=int)
        
        # Promień akceptacji (Sphere of Acceptance)
        self.acceptance_radius = self.params.get("acceptance_radius", 0.2)

    def _prepare_trajectories(self):
        """
        Pobiera wyniki optymalizacji, wybiera najlepsze rozwiązanie i 
        rekonstruuje pełną ścieżkę (Start -> Waypoints -> End).
        """
        # 1. Pobierz surowe wyniki z optymalizatora (Best 5)
        # Kształt: (n_solutions, n_drones, n_waypoints, 3)
        raw_trajectories = self.parent.best_trajectories
        
        if raw_trajectories is None:
            raise ValueError("Brak wyników optymalizacji w parent.best_trajectories!")

        # 2. Wybierz NAJLEPSZE rozwiązanie (pierwsze z posortowanych)
        # Kształt: (n_drones, n_waypoints, 3)
        best_solution_waypoints = raw_trajectories[0] 
        
        # 3. Rekonstrukcja pełnej trasy (dodanie Start i End)
        full_trajectories = []
        
        # Musimy mieć dostęp do pozycji startowych i końcowych.
        # Pobieramy je z instancji optymalizatora w parent
        optimizer = self.parent.nsga3_optimizer
        start_positions = optimizer.start_positions # Kształt (n_drones, 3)
        end_positions = optimizer.end_positions     # Kształt (n_drones, 3)

        for i in range(self.num_drones):
            start = start_positions[i].reshape(1, 3)
            waypoints = best_solution_waypoints[i] # Punkty pośrednie z NSGA-III
            end = end_positions[i].reshape(1, 3)
            
            # Sklejanie: [Start] + [Waypoints] + [End]
            full_path = np.vstack([start, waypoints, end])
            full_trajectories.append(full_path)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        colors = ['r', 'g', 'b', 'y', 'c']

        for i in range(self.num_drones):
            path = full_trajectories[i] # To co zaraz zwrócisz
            ax.plot(path[:,0], path[:,1], path[:,2], label=f'Dron {i}', color=colors[i%5])
            ax.scatter(path[0,0], path[0,1], path[0,2], marker='o') # Start
            ax.scatter(path[-1,0], path[-1,1], path[-1,2], marker='x') # Stop

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend()
        plt.show() # To zatrzyma symulację i pokaże wykres
            
        return np.array(full_trajectories)

    def compute_actions(self, current_states, current_time):
        # Inicjalizacja trajektorii tylko raz (Lazy loading)
        if self._cached_trajectories is None:
            self._cached_trajectories = self._prepare_trajectories()
            print(f"[TrajectoryFollower] Załadowano trasę o długości {self._cached_trajectories.shape[1]} punktów.")

        actions = []
        
        for i in range(self.num_drones):
            state = current_states[i]
            pos = state[0:3]
            
            # Pobierz pełną trajektorię dla drona i
            drone_path = self._cached_trajectories[i]
            
            # Logika przełączania punktów
            current_idx = self.current_waypoint_indices[i]
            
            # Celujemy w aktualny punkt
            target_pos = drone_path[current_idx]
            
            # Sprawdź odległość
            dist = np.linalg.norm(target_pos - pos)
            
            if dist < self.acceptance_radius:
                # Jeśli to nie jest ostatni punkt, przełącz na kolejny
                if current_idx < len(drone_path) - 1:
                    self.current_waypoint_indices[i] += 1
                    # print(f"Dron {i} osiągnął punkt {current_idx}. Następny: {self.current_waypoint_indices[i]}")
                    target_pos = drone_path[self.current_waypoint_indices[i]]
            
            # Obliczenie sterowania PID
            action, _, _ = self.controllers[i].computeControlFromState(
                control_timestep=self.params.get("simulation_freq_hz", 240) ** -1, # Ważne: dopasuj do symulacji
                state=state,
                target_pos=target_pos,
                target_rpy=np.array([0, 0, 0])
            )
            actions.append(action)
        
        return np.array(actions)
