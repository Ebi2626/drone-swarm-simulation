import numpy as np
from src.algorithms.BaseAlgorithm import BaseAlgorithm
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

class LinearTestAlgorithm(BaseAlgorithm):
    def __init__(self, num_drones, params=None):
        super().__init__(num_drones, params)
        self.controllers = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(num_drones)]
        self.final_target_pos = np.array(self.params.get("target_pos", [10.0, 0.0, 1.0]))        
        self.cruise_speed = self.params.get("speed", 5.0) 

    def compute_actions(self, current_states, current_time):
        actions = []
        for i in range(self.num_drones):
            state = current_states[i]
            pos = state[0:3]
            
            # --- LOGIKA "MARCHEWKI NA KIJU" (Trajectory Generation) ---
            
            # 1. Wektor do ostatecznego celu
            direction_vector = self.final_target_pos - pos
            dist_to_target = np.linalg.norm(direction_vector)
            
            # 2. Jeśli jesteśmy daleko, wyznaczamy punkt pośredni
            # Krok czasu sterownika to ok. 1/48s (0.02s)
            # Dron ma przelecieć w tym czasie: speed * dt
            step_distance = self.cruise_speed * (1.0 / 48.0) 
            
            if dist_to_target > step_distance:
                # Normalizujemy wektor i mnożymy przez krok
                direction_norm = direction_vector / dist_to_target
                # Ustawiamy "chwilowy cel" kawałek przed dronem w stronę celu
                # Dodajemy to do AKTUALNEJ pozycji drona (lub przesuwamy poprzedni cel)
                # Bezpieczniej dla PID: Celujemy "trochę przed siebie"
                next_waypoint = pos + (direction_norm * step_distance * 2.0) 
                # Mnożnik *2.0 (lub więcej) wymusza ruch, tzw. "lookahead distance"
            else:
                # Jesteśmy blisko, celujemy w punkt końcowy
                next_waypoint = self.final_target_pos
            
            # Upewnijmy się, że wysokość celu jest stabilna (np. taka jak finalna)
            # Jeśli chcesz, by wznosił się powoli, usuń tę linię.
            # Jeśli ma lecieć płasko na docelowej wysokości:
            # next_waypoint[2] = self.final_target_pos[2] 

            # --- STEROWANIE ---
            
            action, _, _ = self.controllers[i].computeControlFromState(
                control_timestep=1.0/48.0,
                state=state,
                target_pos=next_waypoint, # PID dostaje cel blisko siebie -> mały uchyb -> stabilny lot
                target_rpy=np.array([0,0,0])
            )
            actions.append(action)
        
        return np.array(actions)
