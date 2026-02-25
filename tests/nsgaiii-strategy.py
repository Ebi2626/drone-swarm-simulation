"""
Skrypt diagnostyczny do weryfikacji nowej strategii Polyline.
Sprawdza czy trasa zaczyna się w Starcie i kończy w Celu.
"""

import numpy as np
from src.algorithms.abstraction.trajectory.strategies.nsga3_swarm_strategy import nsga3_swarm_strategy

# --- Mock Data Classes ---
class WorldData:
    def __init__(self):
        # Świat 1000x1000x100
        self.bounds = np.array([[0, 1000], [0, 1000], [0, 100]])

class ObstaclesData:
    def __init__(self):
        self.count = 0
        self.data = np.zeros((10, 6)) # Puste
        self.shape_type = 'CYLINDER'

def test_run():
    # 1. Konfiguracja Scenariusza
    n_drones = 2
    n_waypoints = 20 # Wyjściowa liczba punktów
    
    start_pos = np.array([
        [140.0, 1.0, 0.5],   # Dron 0
        [150.0, 1.0, 0.5]    # Dron 1
    ])
    
    target_pos = np.array([
        [140.0, 900.0, 50.0],
        [150.0, 900.0, 50.0]
    ])
    
    params = {
        "pop_size": 20,
        "n_gen": 5, # Tylko żeby sprawdzić czy działa
        "n_inner_waypoints": 3, # Mało punktów kontrolnych dla testu
        "uniformity_std": 50.0, # Luźny constraint
        "max_jerk": 500.0
    }
    
    print("\n--- Uruchamianie Testu NSGA-III (Polyline) ---\n")
    
    # 2. Uruchomienie Strategii
    final_traj = nsga3_swarm_strategy(
        start_positions=start_pos,
        target_positions=target_pos,
        obstacles_data=[ObstaclesData()],
        world_data=WorldData(),
        number_of_waypoints=n_waypoints,
        drone_swarm_size=n_drones,
        algorithm_params=params
    )
    
    # 3. Diagnostyka Wyników
    print("\n--- WYNIKI DIAGNOSTYCZNE ---\n")
    print(f"Kształt trajektorii: {final_traj.shape} (oczekiwane: {n_drones}, {n_waypoints}, 3)")
    
    for d in range(n_drones):
        print(f"\n[DRON {d}]")
        p_start = final_traj[d, 0]
        p_end = final_traj[d, -1]
        
        expected_start = start_pos[d]
        expected_end = target_pos[d]
        
        # Sprawdzenie błędu
        err_start = np.linalg.norm(p_start - expected_start)
        err_end = np.linalg.norm(p_end - expected_end)
        
        print(f"  START: Otrzymany {p_start} vs Oczekiwany {expected_start} | Błąd: {err_start:.4f}")
        print(f"  KONIEC: Otrzymany {p_end} vs Oczekiwany {expected_end}   | Błąd: {err_end:.4f}")
        
        if err_start < 1e-3 and err_end < 1e-3:
            print("  [OK] Zakotwiczenie poprawne.")
        else:
            print("  [BŁĄD] Trasa oderwana od punktów granicznych!")

if __name__ == "__main__":
    test_run()
