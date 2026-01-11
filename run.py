import os
import time
import hydra
import numpy as np
import pybullet as p
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from src import get_environment, get_algorithm
from src.utils.SimulationLogger import SimulationLogger

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    # 0. Konfiguracja parametrów symulacji
    PYB_FREQ = cfg.simulation.get("pyb_freq", 240)
    CTRL_FREQ = cfg.simulation.get("ctrl_freq", 48)

    # --- ETAP 1: DYNAMICZNE TWORZENIE ŚRODOWISKA ---
    env_name = cfg.environment.name
    print(f"\n--- ETAP 1: Inicjalizacja środowiska: {env_name.upper()} ---")
    
    # Pobieramy klasę z rejestru w src/__init__.py
    WorldClass = get_environment(env_name)
    
    env = WorldClass(
        num_drones=cfg.num_drones,
        initial_xyzs=np.array(cfg.initial_xyzs),
        pyb_freq=PYB_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=cfg.simulation.gui
    )

    # --- ETAP 2: DYNAMICZNE TWORZENIE ALGORYTMU ---
    alg_class_name = cfg.algorithm.class_name
    print(f"--- ETAP 2: Inicjalizacja algorytmu: {cfg.algorithm.name} ({alg_class_name}) ---")
    
    # Pobieramy klasę z rejestru w src/__init__.py
    AlgClass = get_algorithm(alg_class_name)
    
    algorithm = AlgClass(
        num_drones=cfg.num_drones, 
        params=cfg.algorithm.params
    )
    
    current_tracked_id = cfg.visualization.get("tracked_drone_id", 0)
    
    # Upewniamy się, że nie wykracza poza zakres na starcie
    if current_tracked_id >= cfg.num_drones:
        current_tracked_id = 0

    # Ustawienie kamery, jeśli włączono śledzenie drona
    if cfg.visualization.camera_follow:
        target_id = cfg.visualization.tracked_drone_id
        # Pobieramy pozycję startową [x, y, z] wybranego drona
        start_pos = cfg.initial_xyzs[target_id] 
        p.resetDebugVisualizerCamera(
            cameraDistance=cfg.visualization.camera_distance,
            cameraYaw=cfg.visualization.camera_yaw,
            cameraPitch=cfg.visualization.camera_pitch,
            cameraTargetPosition=start_pos
        )

    logger = None
    if cfg.logging.enabled:
        try:
            output_dir = HydraConfig.get().runtime.output_dir
        except Exception:
            # Fallback dla uruchomienia bez @hydra.main (np. debugowanie w IDE)
            output_dir = os.getcwd()
            
        logger = SimulationLogger(
            output_dir=output_dir,
            log_freq=cfg.logging.log_freq,
            ctrl_freq=CTRL_FREQ,
            num_drones=cfg.num_drones
        )

    # Reset i uruchomienie renderingu
    obs, info = env.reset()

    print(">>> System gotowy.")
    print(">>> Instrukcja: [0-9] Zmień drona | [ENTER] Start symulacji")

    # --- PĘTLA OCZEKIWANIA NA START (WAITING LOOP) ---
    if cfg.simulation.gui:
        while True:
            # Obsługa klawiszy w fazie oczekiwania
            keys = p.getKeyboardEvents()
            
            # 1. Sprawdzenie czy wciśnięto ENTER (Start)
            if p.B3G_RETURN in keys and (keys[p.B3G_RETURN] & p.KEY_WAS_TRIGGERED):
                print(">>> START SYMULACJI!")
                break
            
            # 2. Sprawdzenie czy wciśnięto 0-9 (Zmiana kamery)
            for k, v in keys.items():
                if (v & p.KEY_WAS_TRIGGERED):
                    if ord('0') <= k <= ord('9'):
                        selected_id = k - ord('0')
                        if selected_id < cfg.num_drones:
                            current_tracked_id = selected_id
                            print(f"-> Kamera: Dron {current_tracked_id}")

            # 3. Aktualizacja kamery (żeby działała też przed startem)
            if cfg.visualization.camera_follow:
                # W fazie oczekiwania pobieramy stan początkowy (drony stoją)
                # Musimy pobrać go "świeżego" z pybulleta
                # Uwaga: env._getDroneStateVector wymaga, by symulacja istniała
                target_state = env._getDroneStateVector(current_tracked_id)
                target_pos = target_state[0:3]
                target_yaw_rad = target_state[9]
                target_yaw_deg = np.degrees(target_yaw_rad)
                
                camera_yaw = target_yaw_deg - 90.0

                p.resetDebugVisualizerCamera(
                    cameraDistance=cfg.visualization.camera_distance,
                    cameraYaw=camera_yaw,
                    cameraPitch=cfg.visualization.camera_pitch,
                    cameraTargetPosition=target_pos
                )
            
            # Krótki sleep, żeby nie spalić procesora w pętli while True
            time.sleep(0.01)

    # --- ETAP 3: PĘTLA SYMULACJI ---
    print(f"--- ETAP 3: Start symulacji ({cfg.simulation.duration_sec}s) ---")
    
    start_time = time.time()
    total_steps = int(cfg.simulation.duration_sec * CTRL_FREQ)
    
    for i in range(total_steps):
        step_start = time.time()
        current_sim_time = i / CTRL_FREQ
        
        # 1. Algorytm
        all_states = [env._getDroneStateVector(d) for d in range(cfg.num_drones)]
        actions = algorithm.compute_actions(all_states, current_time=i/CTRL_FREQ)
        
        # 2. Krok środowiska
        obs, reward, terminated, truncated, info = env.step(actions)

        if logger:
            # A. Logowanie trajektorii (rzadko, np. co 1s)
            logger.log_step(i, current_sim_time, all_states)
            
            # B. Logowanie kolizji (zawsze, w każdym kroku!)
            # Musimy to sprawdzać ciągle, bo kolizja trwa ułamek sekundy
            collisions = env.get_detailed_collisions()
            for drone_id, other_id in collisions:
                logger.log_collision(current_sim_time, drone_id, other_id)
        
        # 3. Synchronizacja czasu
        if cfg.simulation.gui:

            # --- OBSŁUGA KLAWIATURY [NOWOŚĆ] ---
            # Pobieramy zdarzenia klawiatury z PyBullet
            keys = p.getKeyboardEvents()
            for k, v in keys.items():
                # Sprawdzamy czy klawisz został właśnie naciśnięty (KEY_WAS_TRIGGERED)
                if (v & p.KEY_WAS_TRIGGERED):
                    # Sprawdzamy kody ASCII dla cyfr 0-9
                    # ord('0') to 48, ord('9') to 57
                    if ord('0') <= k <= ord('9'):
                        selected_id = k - ord('0') - 1
                        # Sprawdzamy czy takie ID drona istnieje
                        if selected_id < cfg.num_drones:
                            current_tracked_id = selected_id
                            print(f"-> Przełączono kamerę na drona ID: {current_tracked_id}")
            # -----------------------------------

            # --- KAMERA ŚLEDZĄCA [NOWOŚĆ] ---
            if cfg.visualization.camera_follow:
                target_id = current_tracked_id
                
                # Stan drona to wektor: [x, y, z, qx, qy, qz, qw, roll, pitch, yaw, vx, vy, vz, ...]
                # Interesują nas pierwsze 3 elementy (pozycja)
                target_state = all_states[target_id]
                target_pos = target_state[0:3]
                
                p.resetDebugVisualizerCamera(
                    cameraDistance=cfg.visualization.camera_distance,
                    cameraYaw=cfg.visualization.camera_yaw,
                    cameraPitch=cfg.visualization.camera_pitch,
                    cameraTargetPosition=target_pos
                )
            # --------------------------------

            env.render()
            step_end = time.time()
            elapsed = step_end - step_start
            if elapsed < (1.0 / CTRL_FREQ):
                time.sleep((1.0 / CTRL_FREQ) - elapsed)

    print("--- Koniec symulacji ---")
    print(f"Czas rzeczywisty: {time.time() - start_time:.2f}s")
    # Zapis wyników
    if logger:
        logger.save()
    env.close()

if __name__ == "__main__":
    main()
