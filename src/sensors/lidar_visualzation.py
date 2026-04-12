import time
import numpy as np
import pybullet as p
import pybullet_data

from LidarSensor import LidarSensor

def main():
    # 1. Inicjalizacja środowiska fizycznego z GUI
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 2. Załadowanie obiektów statycznych
    # Ładujemy podłoże (będzie wykryte przez promienie skierowane w dół)
       # 2. Załadowanie obiektów statycznych
    plane_id = p.loadURDF("plane.urdf")
    
    cube_start_pos = [5.0, 0.0, 1.0]
    cube_start_ori = p.getQuaternionFromEuler([0, 0, 0])
    
    # DODANO: useFixedBase=True zamraża obiekt w przestrzeni (wyłącza grawitację dla niego)
    cube_id = p.loadURDF("cube.urdf", cube_start_pos, cube_start_ori, useFixedBase=True)
    
    print(f"ID podłoża: {plane_id}, ID sześcianu: {cube_id}")

    # DODANO: Wymuszenie aktualizacji struktur kolizyjnych silnika PyBullet
    p.stepSimulation()

    # 3. Inicjalizacja Lidaru
    lidar = LidarSensor(physics_client_id=client_id)
    
    # Symulowana pozycja drona (na wysokości 1 metra, patrzącego na sześcian)
    drone_position = np.array([0.0, 0.0, 1.0])
    
    # 4. Wykonanie skanowania
    print("\n--- Rozpoczynam skanowanie Lidar ---")
    start_time = time.time()
    hits = lidar.scan(drone_position)
    compute_time = time.time() - start_time
    
    print(f"Skanowanie zakończone w {compute_time:.5f} s.")
    print(f"Wykryto {len(hits)} punktów przecięcia z obiektami.\n")
    
    # 5. Analiza wyników
    for hit in hits:
        # Płyta podłogowa ma zazwyczaj ID 0, a pierwszy dodany obiekt ID 1
        if hit.object_id == cube_id:
            print(f"[PRZESZKODA] Trafiono sześcian! Dystans: {hit.distance:.2f} m, "
                  f"Pozycja: [{hit.hit_position[0]:.2f}, {hit.hit_position[1]:.2f}, {hit.hit_position[2]:.2f}]")
        elif hit.object_id == plane_id:
            # Ignorujemy logowanie każdego promienia bijącego w ziemię, żeby nie zaśmiecać konsoli
            pass
        else:
            print(f"[INNE] Trafiono obiekt ID {hit.object_id} w odległości {hit.distance:.2f} m")

    # 6. Pętla symulacji pozwalająca na inspekcję wizualną w GUI
    print("\nSymulacja działa. Możesz obejrzeć scenę w oknie PyBullet.")
    print("Naciśnij Ctrl+C w konsoli, aby zakończyć.")
    
    # Rysujemy promienie dla celów debugowania wizualnego w tym teście
    # (Zgodnie z koncepcją promieni debugujących dla systemów UAV)
    ray_from = np.tile(drone_position, (lidar._num_rays, 1))
    ray_to = drone_position + lidar._ray_directions * lidar.MAX_RANGE
    
    # Zaznaczmy na czerwono promienie, które coś trafiły
    hit_directions = {tuple(hit.ray_direction) for hit in hits}
    
    for i in range(lidar._num_rays):
        dir_tuple = tuple(lidar._ray_directions[i])
        if dir_tuple in hit_directions:
            # Promień trafił w cel (kolor czerwony)
            p.addUserDebugLine(ray_from[i], ray_to[i], [1, 0, 0], physicsClientId=client_id)
        else:
            # Promień nic nie trafił (kolor zielony)
            p.addUserDebugLine(ray_from[i], ray_to[i], [0, 1, 0], physicsClientId=client_id)

    try:
        while p.isConnected(client_id):
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        p.disconnect(client_id)

if __name__ == "__main__":
    main()