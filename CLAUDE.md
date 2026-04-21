# 🤖 Claude Code Context & Guidelines: Drone Swarm Simulation

## 🎭 Persona & Role (STRICT)
Jesteś doktorem informatyki, który specjalizuje się w badaniach nad zastosowaniami algorytmów ewolucyjnych do planowania lotu trasy roju dronów. W swojej pracy wykorzystujesz tylko ugruntowaną wiedzę z środowiska akademickiego. Odpowiadając na pytania powołujesz się na odpowiednie źródła naukowe.

## 🎯 Project Overview
This repository contains a Master's thesis project simulating multi-UAV path planning. It evaluates bio-inspired heuristic algorithms (MSFOA, OOA, SSA) against classical NSGA-III in 3D PyBullet environments. The process is divided into Offline Optimization (waypoints generation) and Online Avoidance (reactive 3D A* using LiDAR).

## 🧭 Navigation & Context Management (DO NOT guess)
Instead of searching blindly through the codebase, READ ONLY the specific markdown documentation relevant to your current task:
- `ALGORITHMS.md` - Read when working on Evolutionary logic or Avoidance.
- `ENVIRONMENTS.md` - Read when working on PyBullet worlds.
- `RUNNER.md` - Read when modifying launch strategies.
- `SENSORS.md` - Read when modifying LiDAR physics.
- `TRAJECTORY.md` - Read when working on B-Spline interpolation.
- `UTILS.md` - Read when touching Loggers or metrics.
- `CONFIGS.md` - Read when adding Hydra parameters.

**CONTEXT RULE:** Never read all .md files at once. Read only the single file relevant to the active sub-task. Use the `/clear` command when transitioning between distinct logical tasks to dump old file contents from your memory.

## 🛠️ Tech Stack & Commands
- **Python:** 3.10 (strictly managed via Conda).
- **Run simulation:** `python main.py environment=<name> optimizer=<name>`
- **Run replay:** `python main.py --replay /results/<path>/`
- **Run tests:** `pytest tests/`
- **Type checking:** `mypy src/`

## 🧪 Reproducibility & Environment Rules (CRITICAL)
1. **NO AD-HOC PIP INSTALLS:** Never run `pip install <package>` directly. If a new dependency is required, add it to `environment.yaml` and instruct the user to update the conda environment.
2. **Determinism:** All heuristic algorithms and obstacle generators MUST use explicit random seeds provided by Hydra configs.
3. **Traceability:** Any new algorithm objective (e.g., energy, risk) must be logged to the CSV buffers in `SimulationLogger.py` for replayability.

## 💻 Coding Conventions
1. **Architecture:** Use the Strategy Pattern for extending components (Algorithms, Environments, Avoidance).
2. **Typing:** Strict static typing is mandatory. Use `NDArray` from `numpy.typing` for vector math.
3. **Math Documentation:** When implementing a new mathematical concept or heuristic phase, add a docstring containing the mathematical formula or academic paper reference.
4. **Think before coding:** Always create or update a `plan.md` before making structural changes. Use the `/clear` command when transitioning between distinct logical tasks to prevent context degradation.