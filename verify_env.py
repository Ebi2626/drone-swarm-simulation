import traceback
import time
import hydra
import pybullet as p
import numpy as np
from omegaconf import DictConfig, OmegaConf
# Importujemy naszą fabrykę środowisk
from src import get_environment


# --- MONKEY PATCH ---
# Zapisujemy oryginał
_original_connect = p.connect

def patched_connect(method, options=""):
    # Logujemy próbę połączenia
    print(f"[DEBUG MONKEY PATCH] Wywołano p.connect z metodą: {method}")
    
    if method == p.GUI:
        print("[INFO] Monkey Patch: Włączam ULTRA MSAA x16 + 4K!")
        
        # Dodajemy flagi tylko jeśli ich tam jeszcze nie ma
        new_flags = " --samples=16 --width=1920 --height=1080"
        if options:
            options += new_flags
        else:
            options = new_flags
            
        print(f"[DEBUG] Nowe opcje: {options}")
    
    # Wywołujemy oryginał i ZWRACAMY wynik (Client ID)
    client_id = _original_connect(method, options=options)
    
    if client_id < 0:
        print("[CRITICAL] Nie udało się połączyć z PyBullet!")
    else:
        print(f"[DEBUG] Połączono pomyślnie. Client ID: {client_id}")
        
    return client_id

# Podmieniamy funkcję
p.connect = patched_connect
# -------------------------------

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"Loading environment: {cfg.environment.name}")
    print("-" * 40)
    
    # Wyświetlenie aktualnej konfiguracji dla pewności
    print(OmegaConf.to_yaml(cfg.environment))
    
    # Pobranie klasy i inicjalizacja środowiska
    # Hydra automatycznie wstrzykuje parametry z sekcji 'params'
    try:
        EnvClass = get_environment(cfg.environment.name)
        print(f"DEBUG TYPE: {type(EnvClass)}")     # <--- Dodaj to
        print(f"DEBUG VAL: {EnvClass}")            
        env = EnvClass(**cfg.environment.params)

        # Pobierz informacje o połączeniu
        connection_info = p.getConnectionInfo()
        print("\n" + "="*40)
        print(f"PYBULLET BACKEND INFO")
        print("="*40)
        print(f"Is Connected: {p.isConnected()}")
        print(f"Connection Method: {connection_info['connectionMethod']}") 
        # 1 = GUI (OpenGL), 2 = DIRECT (CPU only), 3 = SHARED_MEMORY

        # Pobierz informacje o OpenGL (działa tylko w trybie GUI)
        if p.isConnected() and connection_info['connectionMethod'] == p.GUI:
            data = p.getVREvents() # Hack: Czasami odświeża kontekst
            # Niestety PyBullet nie ma funkcji "getOpenGLVersion", 
            # ale wypisuje to w konsoli na samym początku.
            # Szukaj w logach linii: "Created GL 3.3 context" lub "GL_VENDOR"
            # Sprawdź, czy używasz sprzętowego OpenGL czy TinyRenderer (programowy)
            # TinyRenderer to powolny, brzydki renderer CPU używany gdy nie ma GPU.
            # Jeśli masz Nvidia, powinien być wyłączony.
            vis_info = p.getDebugVisualizerCamera()
            print(f"Renderer Detected (Indirect check):")
            # Niestety nie ma flagi "isTinyRenderer", ale logi terminala są kluczem.

            print("="*40 + "\n")
        
        print(f"\nEnvironment {cfg.environment.name} created successfully!")
        print("Controls:")
        print("  - CTRL + Mouse Left: Rotate Camera")
        print("  - CTRL + Mouse Middle: Pan Camera")
        print("  - CTRL + Mouse Right: Zoom Camera")
        print("  - Close window to exit")

    except Exception as e:
        print(f"Error creating environment: {e}")
        print("\n" + "="*40)
        print("CRITICAL ERROR IN ENVIRONMENT CREATION")
        print("="*40)
        traceback.print_exc()  # <--- TO JEST KLUCZOWE: DRUKUJE PEŁNY BŁĄD
        print("="*40)
        return

    # Ustawienie kamery, aby widzieć drona i otoczenie
    # Dla Urban/Forest chcemy widzieć perspektywę
    p.resetDebugVisualizerCamera(
        cameraDistance=5.0,
        cameraYaw=-90,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0]
    )


    print("\nSimulation running... Press CTRL+C in terminal to stop.")
    
    try:
        # ZMIANA: Nieskończona pętla zamiast for i in range(...)
        while True: 
            
            # Sprawdzenie czy okno jest nadal otwarte (jeśli zamkniesz 'X' myszką)
            if not p.isConnected():
                print("GUI window closed by user.")
                break

            # Pusty krok symulacji (aby fizyka działała, np. woda, wiatr, grawitacja)
            action = np.zeros((1, 4)) 
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Opcjonalnie: Resetuj drona jeśli spadnie, żebyś miał co oglądać
            # (zakładając że env.DRONE_IDS[0] to ID drona)
            pos, _ = p.getBasePositionAndOrientation(env.DRONE_IDS[0])
            if pos[2] < -1.0: # Jeśli spadł pod ziemię
                p.resetBasePositionAndOrientation(env.DRONE_IDS[0], [0,0,0.5], [0,0,0,1])

            # Synchronizacja czasu (ważne, żeby nie przewijało za szybko)
            time.sleep(1 / 240)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user (CTRL+C).")
    except Exception as e:
        print("\n" + "="*40)
        print("CRITICAL ERROR IN Simulation")
        print("="*40)
        traceback.print_exc()
        print("="*40)
    finally:
        # Sprzątanie po wyjściu z pętli
        env.close()

if __name__ == "__main__":
    main()
