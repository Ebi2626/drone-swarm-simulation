import time
import sys
import numpy as np
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import pybullet as p

def test_headless():
    print("[1/3] Test symulacji HEADLESS (bez okienka)...")
    try:
        # Inicjalizacja środowiska bez GUI
        env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1, physics=Physics.PYB, gui=False)
        env.reset()
        
        start = time.time()
        for i in range(1000):
            # Losowe działanie (obroty śmigieł)
            action = np.array([[10000, 10000, 10000, 10000]]) 
            obs, reward, done, info = env.step(action)
        
        env.close()
        fps = 1000 / (time.time() - start)
        print(f"   -> SUKCES! Prędkość: {fps:.0f} kroków/s")
    except Exception as e:
        print(f"   -> BŁĄD: {e}")

def test_gui():
    print("\n[2/3] Test symulacji GUI (okienko PyBullet)...")
    print("   -> Za chwilę powinno pojawić się okno. Sprawdź czy widzisz drona.")
    try:
        # Inicjalizacja z GUI
        env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1, physics=Physics.PYB, gui=True)
        env.reset()
        
        for i in range(240):  # Krótki lot (1 sekunda symulacji)
            action = np.array([[.5, .5, .5, .5]]) 
            env.step(action)
            p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0,0,0])
            time.sleep(1/240) # Spowolnienie do czasu rzeczywistego
            
        env.close()
        print("   -> SUKCES! Okno otworzyło się i zamknęło.")
    except Exception as e:
        print(f"   -> BŁĄD GUI (OpenGL/Sterowniki): {e}")

if __name__ == "__main__":
# Poprawione odwołanie do sys.version (bezpośrednio, nie przez np)
    print(f"Środowisko: Python {sys.version.split()[0]}, NumPy {np.__version__}")
    test_headless()
    test_gui()
