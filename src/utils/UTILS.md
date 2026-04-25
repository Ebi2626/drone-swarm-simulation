# src/utils/ — Narzędzia pomocnicze symulacji

Katalog zawiera **infrastrukturę reproducible research** + **utils** dla całego pipeline'u. Kluczowy dla walidacji algorytmów ewolucyjnych w pracy magisterskiej.

## 🏗️ Struktura (8 plików)

```
src/utils/
├── config_parser.py                # Parsowanie argumentów z Hydry 
├── input_utils.py                  # Keyboard CLI 
├── optimization_history_writer.py  # Optymalizacja HDF5/npz 
├── positions_to_tensor.py          # List→Tensor 
├── pybullet_utils.py               # Funkcje pomocnicze PyBullet 
├── save_obstacles_to_csv.py        # Obstacles→CSV 
├── SimulationLogger.py             # Silnik logowania 
└── ValidationMessage.py            # Enum z komunikatami błędów
```

## 🎯 SimulationLogger — Centralny logger

**Bufor RAM → CSV** (flush na końcu):
```
CSV outputs:
├── counted_trajectories.csv     # Optymalne trajektorie
├── world_boundaries.csv         # Tunele misji (X/Y/Z bounds)
├── generated_obstacles.csv      # Przeszkody (BOX/CYL)
├── trajectories.csv             # Real-time pozycje RPY/V
├── collisions.csv               # Kolizje
├── lidar_hits.csv               # LiDAR trafienia (niefiltrowane)
└── optimization_timings.csv     # Pomiary czasu
```

**Logi**:
- `log_step()`: pozycje/prędkości (co `log_freq` kroków)
- `log_collision(time, drone_id, object_id)`
- `log_lidar_hit(time, drone_id, LidarHit)`
- `log_optimization_timing()`: wall/cpu time per generation

## 📊 optimization_history_writer.py — Historia ewolucji

**Asynchroniczny producer-consumer** (thread-safe):
- **Input**: `put_generation_data({"objectives": matrix, "decisions": matrix})`
- **Output**: `optimization_history.h5` (HDF5 gzip) lub `.npz chunks`
- **Flush**: Co 100 generacji (`BUFFER_FLUSH_SIZE=100`)

**Format**: `Generations × Individuals × Features` (stackowane NumPy).

## 🔧 Walidacja + Parsing

| Plik | Rola |
|------|------|
| **`ValidationMessage`** | Enum błędów: `INVALID_INITIAL_POINTS`, `WRONG_GROUND_POSITION` |
| **`config_parser.py`** | Hydra config → struktury (`environment.params`) |
| **`positions_to_tensor()`** | `List[List[x,y,z]] → NDArray[N,3]` |
| **`save_obstacles_to_csv()`** | `ObstaclesData → CSV` (BOX: x,y,z,l,w,h) |

## 🎮 UI + PyBullet

| Plik | Sterowanie |
|------|------------|
| **`input_utils.py`** | **Keyboard CLI**: `SPACE`=pause, `L`=LiDAR, `0-9`=switch drone cam |
| **`pybullet_utils.py`** | PyBullet wrappers (load, step, reset) |

## 🔄 Diagram dataflow

```mermaid
graph TD
    A[Optimizer NSGA/MSFOA] --> B[GenerationDataStrategy]
    B --> C[SimulationLogger<br/>log_chosen_trajectories()]
    C --> D[CSV: trajectories/obstacles/world]
    
    E[Opt Loop] --> F[HistoryWriter<br/>put_generation_data()]
    F --> G[HDF5: objectives/decisions]
    
    H[Sim Loop] --> I[log_step/lidar/collision]
    I --> J[realtime CSV]
    
    K[CLI --replay] --> L[ReplayDataStrategy<br/>CSV→Tensors]
```

## 🚀 Przykładowe użycie

```python
# Logger init
logger = SimulationLogger("results/", log_freq=10, ctrl_freq=240, num_drones=5)

# Optymalizacja
logger.log_chosen_trajectories(trajectories)
logger.log_optimization_timing(run_id="msfoa_001")

# Sim loop
for t, states in enumerate(sim):
    logger.log_step(t, states)           # Pozycje
    logger.log_lidar_hit(t, drone_id, hit)  # Sensory

logger.save()  # Flush CSV
history_writer.close()  # Flush HDF5
```

## 📈 Funkcje badawcze związane z pomiarami i reprodukowalnością

✅ **Full provenance**: Timings + params JSON  
✅ **HDF5 history**: Pareto fronts per generation  
✅ **Replay 100%**: CSV deterministic  
✅ **Validation**: Structured error messages  
✅ **Batch logging**: RAM buffer → disk  

**Wymagania**: `h5py` (opcjonalne), `pandas`, `numpy`.
