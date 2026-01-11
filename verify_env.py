import traceback
import time
import hydra
import pybullet as p
import numpy as np
from omegaconf import DictConfig, OmegaConf
from src import get_environment

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("=== Drone Swarm Simulation Environment Verification ===\n")
    print(OmegaConf.to_yaml(cfg.environment))
    
    try:
        EnvClass = get_environment(cfg.environment.name)      
        env = EnvClass(**cfg.environment.params)
        print(f"\nEnvironment {cfg.environment.name} created successfully!")

    except Exception as e:
        print("\n" + "="*40)
        print("CRITICAL ERROR IN ENVIRONMENT CREATION")
        print("="*40)
        print(f"Exception during environment creation: {e}")
        traceback.print_exc()
        print("="*40)
        return

    p.resetDebugVisualizerCamera(
        cameraDistance=5.0,
        cameraYaw=-90,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0]
    )

    print("\nSimulation running... Press CTRL+C in terminal to stop.")

    try:
        while True: 
            if not p.isConnected():
                print("GUI window closed by user.")
                break

            action = np.zeros((1, 4)) 
            obs, reward, terminated, truncated, info = env.step(action)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

            pos, _ = p.getBasePositionAndOrientation(env.DRONE_IDS[0])
            if pos[2] < -1.0:
                p.resetBasePositionAndOrientation(env.DRONE_IDS[0], [0,0,0.5], [0,0,0,1])

            time.sleep(1 / 240)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user (CTRL+C).")
    except Exception as e:
        print("\n" + "="*40)
        print("CRITICAL ERROR IN Simulation")
        print("="*40)
        print(f"Exception during simulation: {e}")
        traceback.print_exc()
        print("="*40)
    finally:
        env.close()

if __name__ == "__main__":
    main()
