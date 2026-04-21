# src/trajectory/ — Generatory trajektorii UAV (B-Spline + Profile)

Katalog implementuje **gładkie trajektorie 3D** dla fazy **online tracking** + **local avoidance**. Kluczowy element między optymalizatorem (`algorithms/`) a kontrolerem PID. Zapewnia **ciągłość pozycji/prędkości/przyspieszenia**.

## 🏗️ Struktura

```
src/trajectory/
├── BSplineTrajectory.py     # Główna klasa (scipy.interpolate) 
├── ConstantSpeedProfile.py  # Stała prędkość (unikanie przeszkód)
└── TrapezoidalProfile.py    # Trapez (globalne misje) 
```

## 🎯 BSplineTrajectory — Core generator

**Input**: Waypoints z optymalizatora `(N,3)` → **Cubic B-Spline** (`k=3`, interpolacja `s=0`).

**Pipeline**:
1. `splprep()` → parametry krzywej `(tck, u)`
2. `_calculate_arc_length()` → całkowita długość (numeryczne całkowanie, 1000 próbek)
3. **Velocity Profile** → mapuje czas → `(pos, vel)`
4. **Analityczne pochodne** (`splev(der=1)`) → wektor styczny × skalarną prędkość

**API**:
```python
traj = BSplineTrajectory(waypoints, cruise_speed=5.0, max_accel=2.0)
pos, vel = traj.get_state_at_time(t=10.5)  # (3,), (3,)
pos, vel = traj.get_state_at_distance(s=50.0, speed=3.0)
```

**Parametry**:
- `constant_speed=True` → `ConstantSpeedProfile` (unikanie lokalne)
- `constant_speed=False` → `TrapezoidalProfile` (globalne planowanie misji)

## 📈 Velocity Profiles — Komponenty

| Profil | Zastosowanie | Charakterystyka | Przyspieszenie |
|--------|--------------|-----------------|---------------|
| **TrapezoidalProfile** | Globalne misje | **Accel → Cruise → Decel** | `max_accel` [m/s²] |
| **ConstantSpeedProfile** | Unikanie lokalne | **Stały `cruise_speed`** | 0 (bez faz A/D) |

**TrapezoidalProfile** (auto-adaptive):
```
Krótka trasa → trójkątny (v_peak = √(a·s))
Długa trasa → trapezowy (v_peak = cruise_speed)
Fazy: t_a/s_a → t_c/s_c → t_d/s_d
get_state(t) → (distance, speed)
```

## 🔄 Diagram integracji

```mermaid
graph LR
    A[Optimizer<br/>algorithms/counting] --> B[Waypoints<br/>(N,3)]
    B --> C[BSplineTrajectory<br/>splprep(k=3)]
    C --> D[Arc Length<br/>∫||ds||]
    D --> E{Trapezoidal<br/>OR Constant?}
    E --> F[TrapezoidalProfile]
    E -->|local| G[ConstantSpeedProfile]
    F --> H[get_state(t)]
    G --> H
    H --> I[pos(t), vel(t)]
    I --> J[PID Controller<br/>PyBullet setTarget()]
    I --> K[sensors/LiDAR<br/>Collision check]
```

## 🚀 Przykładowe użycie

```python
# Global mission (NSGA-III waypoints)
waypoints = np.array([,, ])  # z CSV replay [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_ce667fd1-2aba-4b27-9923-b8f37a051a4c/af5d1b4e-4ced-4c14-9538-d77f62097054/uav-modelling-constrains.pdf)
traj = BSplineTrajectory(waypoints, cruise_speed=5.0, max_accel=2.0)

# PID loop
for t in np.linspace(0, traj.total_duration, 1000):
    target_pos, target_vel = traj.get_state_at_time(t)
    pid.set_target(target_pos, target_vel)  # PyBullet
    
# Local avoidance (A* waypoints)
local_traj = BSplineTrajectory(short_waypoints, cruise_speed=3.0, 
                               max_accel=2.0, constant_speed=True)
```

## 📊 Metryki jakości

| Metoda | Ciągłość | Zastosowanie | Złożoność |
|--------|----------|--------------|-----------|
| **B-Spline (k=3)** | **C²** (pos, vel, accel) | Global + local | O(N log N) |
| **Trapezoidal** | C⁰ (skok jerk) | Długie misje | O(1) |
| **Constant** | C⁰ | Unik | O(1) |

**Arc length**: Numeryczne (1000 próbek) — błąd <0.1% dla cubic splines.

## 🛠️ Zastosowanie w systemie

1. **Offline → Online**: `counted_trajectories.csv` → B-Spline
2. **PID input**: `(target_pos, target_vel)` na każdym tick
3. **LiDAR feedback**: Korekta z `sensors/` + `algorithms/avoidance/`
4. **Replay walidacja**: Porównanie symulacji z planem
