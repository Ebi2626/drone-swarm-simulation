# src/sensors/ — Sensory symulacji UAV (LiDAR 3D)

Katalog implementuje **LiDAR 3D** dla fazy **online avoidance** w symulacji roju dronów. Używa `pybullet.rayTestBatch()` do **ray casting** z optymalizacją batchową dla wielu UAV. Zintegrowany z PyBullet physics.

## 🏗️ Struktura

```
src/sensors/
├── __init__.py
├── LidarSensor.py          # Core sensor
└── lidar_visualzation.py   # Demo + debug
```

## 🎯 LidarSensor — Specyfikacja techniczna

**Parametry skanowania** (stożek **spotlight FOV**):
- **FOV**: 30° rozwarcia (15° od osi, przód drona CF2X)
- **Promienie**: **123** (7 koncentrycznych pierścieni, gęstość ~1-1.5°)
- **Zasięg**: **100m**
- **Batch**: `rayTestBatch()` — <1ms dla 5×123=615 promieni

**Pierścienie** (`_compute_ray_directions()`):
```
R0: 0.0° → 1 ray   (centralny)
R1: 0.5° → 4 rays  (rdzeń ochronny)
R2: 1.5° → 10 rays
...
R6: 15.0° → 84 rays (krawędź FOV)
```

**Dataclass `LidarHit`**:
```python
@dataclass
class LidarHit:
    object_id: int          # PyBullet ID (np. obstacle=1, ground=0)
    distance: float         #  m
    hit_position: NDArray   # [x,y,z] punkt trafienia
    ray_direction: NDArray  # Jednostkowy wektor promienia
```

## 🔄 Kluczowe metody

| Metoda | Opis | Użycie |
|--------|------|--------|
| `scan(position, quat=None)` | Pojedynczy skan z rotacją | `hits = lidar.scan(drone_pos)` |
| `batch_ray_test(positions, quats)` | Rój UAV (N×123 rays) | `raw = LidarSensor.batch_ray_test(all_drones)` |
| `process_batch_results(raw, logger)` | Parse + log | `hits = lidar.process_batch_results(raw)` |
| `draw_debug_lines(pos)` | Wizualizacja (czerwony=hit, zielony=miss) | Debug PyBullet GUI |

**Rotacja**: `scipy.spatial.transform.Rotation.from_quat()` — pełna 6DoF.

## 🧪 lidar_visualzation.py — Demo

**Scenariusz testowy**:
```
1. PyBullet GUI + plane.urdf + cube.urdf (useFixedBase=True)
2. Drone @ [0,0,1m] → skanuje cube @ [5,0,1m]
3. Debug lines: czerwony=hit, zielony=miss
4. Logi: "Trafiono sześcian! Dystans: X.XX m"
```

**Wynik**:
```
Skanowanie zakończone w 0.000XX s.
Wykryto N punktów przecięcia z obiektami.
[PRZESZKODA] Trafiono sześcian! Dystans: 5.00 m
```

## 🔄 Diagram integracji

```mermaid
graph TD
    A[Drone Position<br/>+ Quaternion] --> B[LidarSensor.scan()]
    B --> C[pybullet.rayTestBatch<br/>123 rays]
    C --> D[LidarHit list]
    D --> E[Avoidance<br/>algorithms/avoidance/]
    D --> F[Logger<br/>log_lidar_hit()]
    
    G[Rój 5 UAV] --> H[LidarSensor.batch_ray_test()]
    H --> I[Parse + Process]
    
    J[PyBullet GUI] --> K[draw_debug_lines<br/>Czerwony/Zielony]
```

## 🚀 Użycie w symulacji

```python
# Pojedynczy dron
lidar = LidarSensor(client_id)
hits = lidar.scan(drone_pos, drone_quat)

# Rój (batch)
raw_hits = LidarSensor.batch_ray_test(all_drone_positions)
for i, raw in enumerate(raw_hits):
    drone_hits = lidar.process_batch_results(raw, logger, t, drone_id=i)

# Debug wizualny
lidar.draw_debug_lines(drone_pos)
```

**Integracja z `runner/`**: Logi do CSV replay.

## 📊 Wydajność

| Konfiguracja | Promienie/krok | Czas CPU |
|--------------|----------------|----------|
| 1 UAV | 123 | <0.1ms |
| 5 UAV | 615 | <0.5ms |
| 20 UAV | 2460 | ~2ms |

**Optymalizacja**: C++ backend PyBullet, vektoryzacja NumPy.

## 🛠️ Zastosowanie w pracy magisterskiej

- **Online avoidance**: Input do A* lub innych algorytmów (`algorithms/avoidance/`)
- **APF forces**: Skalowanie sił odpychających ∝ 1/distance
- **Walidacja offline planów**: Porównanie z `counted_trajectories.csv`
- **Metryki bezpieczeństwa**: Min distance do przeszkód


**Autor**: Edwin — Symulacja sensoryczna roju UAV  
**Wersja**: 1.0 (Kwiecień 2026) [file:24][file:25]
