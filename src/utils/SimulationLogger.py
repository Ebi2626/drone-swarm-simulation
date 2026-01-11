import os
import csv
import numpy as np

class SimulationLogger:
    def __init__(self, output_dir, log_freq, ctrl_freq, num_drones):
        self.output_dir = output_dir
        self.log_step_interval = max(1, int(ctrl_freq / log_freq))
        self.num_drones = num_drones
        
        # Bufory danych
        self.trajectory_buffer = []
        self.collision_buffer = []
        
        # Zbiór do śledzenia rozbitych dronów (zapobiega duplikatom)
        self.crashed_drones = set()
        
        print(f"[LOGGER] Buforowanie w RAM. Zapis na dysk po zakończeniu.")

    def log_step(self, step_idx, current_time, all_states):
        """Loguje trajektorie."""
        if step_idx % self.log_step_interval == 0:
            for drone_id, state in enumerate(all_states):
                # Opcjonalnie: Możesz tu dodać: if drone_id in self.crashed_drones: continue
                # jeśli nie chcesz logować trasy wraku po zderzeniu.
                
                record = (
                    round(current_time, 3),
                    drone_id,
                    round(state[0], 3),
                    round(state[1], 3),
                    round(state[2], 3),
                    round(state[7], 3),
                    round(state[8], 3),
                    round(state[9], 3),
                    round(state[10], 3),
                    round(state[11], 3),
                    round(state[12], 3)
                )
                self.trajectory_buffer.append(record)

    def log_collision(self, current_time, drone_id, other_body_id):
        """
        Loguje kolizję TYLKO JEŚLI to pierwsza kolizja tego drona.
        """
        if current_time < 1:
            # Ignorujemy kolizję związaną ze startem z ziemi
            return
        
        if drone_id not in self.crashed_drones:
            # 1. Dodajemy drona do listy "rozbitych"
            self.crashed_drones.add(drone_id)
            
            # 2. Rejestrujemy zdarzenie
            # Zgodnie z życzeniem: drone_id, time, other_body_id
            self.collision_buffer.append((
                drone_id,
                round(current_time, 3), 
                other_body_id
            ))
            
            print(f"[LOGGER] Kolizja! Dron {drone_id} uderzył w obiekt {other_body_id} (t={current_time:.2f}s)")

    def save(self):
        """Zrzut danych na dysk."""
        print(f"[LOGGER] Rozpoczynam zapis danych na dysk...")
        
        # 1. Trajektorie (bez zmian)
        if self.trajectory_buffer:
            path = os.path.join(self.output_dir, "trajectories.csv")
            headers = ["time", "drone_id", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz"]
            
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(self.trajectory_buffer)
            print(f"[LOGGER] Zapisano trajektorie: trajectories.csv")
            self.trajectory_buffer.clear()
        
        # 2. Kolizje (ZMIA NA)
        if self.collision_buffer:
            path = os.path.join(self.output_dir, "collisions.csv")
            # Struktura o którą prosiłeś:
            headers = ["drone_id", "time", "other_body_id"]
            
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(self.collision_buffer)
                
            print(f"[LOGGER] Zapisano kolizje: collisions.csv ({len(self.collision_buffer)} zdarzeń)")
            self.collision_buffer.clear()
        else:
            # Jeśli bufor jest pusty, plik NIE powstanie
            print(f"[LOGGER] Brak kolizji - plik collisions.csv nie został utworzony.")
