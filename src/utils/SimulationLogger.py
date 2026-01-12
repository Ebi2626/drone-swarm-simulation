import os
import csv

class SimulationLogger:
    def __init__(self, output_dir, log_freq, ctrl_freq, num_drones):
        self.output_dir = output_dir
        self.log_step_interval = max(1, int(ctrl_freq / log_freq))
        self.num_drones = num_drones        
        self.trajectory_buffer = []
        self.collision_buffer = []        
        self.crashed_drones = set()
        print("[LOGGER] Buffering in RAM. Writing to disk after completion.")

    def log_step(self, step_idx, current_time, all_states):
        if step_idx % self.log_step_interval == 0:
            for drone_id, state in enumerate(all_states):
                if drone_id in self.crashed_drones:
                    continue
                
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
        if current_time < 1:
            return
        
        if drone_id not in self.crashed_drones:
            self.crashed_drones.add(drone_id)
            
            self.collision_buffer.append((
                round(current_time, 3),
                drone_id,
                other_body_id
            ))
            
            print(f"[LOGGER] Collision! Drone {drone_id} hit object {other_body_id} (t={current_time:.2f}s)")

    def save(self):
        print("[LOGGER] Saving data to disk...")
        
        if self.trajectory_buffer:
            path = os.path.join(self.output_dir, "trajectories.csv")
            headers = ["time", "drone_id", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz"]
            
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(self.trajectory_buffer)
            print("[LOGGER] Trajectories saved: trajectories.csv")
            self.trajectory_buffer.clear()
        
        if self.collision_buffer:
            path = os.path.join(self.output_dir, "collisions.csv")
            headers = ["time", "drone_id", "other_body_id"]
            
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(self.collision_buffer)
                
            print(f"[LOGGER] Collisions saved: collisions.csv ({len(self.collision_buffer)} events)")
            self.collision_buffer.clear()
        else:
            print("[LOGGER] No collisions - file collisions.csv not created.")
