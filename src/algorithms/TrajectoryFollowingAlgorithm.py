import numpy as np
from src.algorithms.BaseAlgorithm import BaseAlgorithm
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

class TrajectoryFollowingAlgorithm(BaseAlgorithm):
    def __init__(self, num_drones, params=None):
        super().__init__(num_drones, params)
        
        # Inicjalizacja kontrolerów PID dla każdego drona
        self.controllers = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(num_drones)]
        
        # Pobranie trajektorii z parametrów
        # Oczekiwany format: tablica numpy o wymiarach (n_drones, n_waypoints, 3)
        # Przykład: trajectories[0] to lista punktów [x, y, z] dla drona 0
        self.trajectories = self.params.get("trajectories")
        
        if self.trajectories is None:
            raise ValueError("TrajectoryFollowingAlgorithm wymaga przekazania 'trajectories' w parametrach.")

        # Indeksy śledzące, do którego punktu trajektorii leci aktualnie dany dron
        self.current_waypoint_indices = np.zeros(num_drones, dtype=int)
        
        # Promień akceptacji (Sphere of Acceptance)
        # Jeśli dron znajdzie się bliżej niż X metrów od punktu, uznajemy go za zaliczony.
        # Wartość zależy od prędkości drona i gęstości punktów (zwykle 0.1 - 0.5 m)
        self.acceptance_radius = self.params.get("acceptance_radius", 0.2)

    def compute_actions(self, current_states, current_time):
        actions = []
        
        for i in range(self.num_drones):
            state = current_states[i]
            pos = state[0:3]
            
            # Pobierz trajektorię dla tego konkretnego drona
            drone_trajectory = self.trajectories[i]
            
            # Pobierz indeks aktualnego celu
            current_idx = self.current_waypoint_indices[i]
            
            # Sprawdzenie, czy nie jesteśmy już na końcu trasy
            # Jeśli tak, celujemy ciągle w ostatni punkt (hover)
            if current_idx < len(drone_trajectory):
                target_pos = drone_trajectory[current_idx]
                
                # Oblicz odległość do aktualnego celu
                dist_to_target = np.linalg.norm(target_pos - pos)
                
                # --- LOGIKA PRZEŁĄCZANIA PUNKTÓW (Waypoint Switching) ---
                # Jeśli jesteśmy wystarczająco blisko celu, przełączamy na następny punkt
                if dist_to_target < self.acceptance_radius:
                    # Przełącz na następny punkt, jeśli istnieje
                    if current_idx < len(drone_trajectory) - 1:
                        self.current_waypoint_indices[i] += 1
                        # Aktualizujemy cel natychmiast na nowy
                        target_pos = drone_trajectory[self.current_waypoint_indices[i]]
            else:
                # Zabezpieczenie (fallback): leć do ostatniego znanego punktu
                target_pos = drone_trajectory[-1]

            # --- STEROWANIE ---
            # Kontroler PID otrzymuje pozycję aktualnego punktu z trajektorii jako Setpoint
            action, _, _ = self.controllers[i].computeControlFromState(
                control_timestep=1.0/48.0, # Zakładam standardowy krok symulacji PyBullet
                state=state,
                target_pos=target_pos,
                target_rpy=np.array([0, 0, 0]) # Dążymy do stabilizacji orientacji
            )
            actions.append(action)
        
        return np.array(actions)
