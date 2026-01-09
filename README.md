# drone-swarm-simulation
Simulation of drone swarms path planning with modern biologically inspired heuristic algorithms,

## Project structure
- `/configs` - files for hydra to make experiments and their results easier to parametrize and analyze
    - `/algorithm` - parameters for different algorithms
    - `/environment` - parameters for different worlds
- `/experiments` - launch scripts for experiments
- `/notebooks` - notes about the project
- `/src` - main logic
- `/tests` - unit tests

## Main concepts
- experiments should be reproducible
- experiments should compare different algorithms in the same environment
- experiments should be prepared in matrix of:
    - 3 different worlds with increasing amount of obstacles
    - 3 different algorithms to verify (OOA, SSA, MSFFOA) and classical NSGA-III as a reference point
    - 2 variants of each world - static and dynamic
- experiments should verify drone swarm path planning as a whole
- experiments should contain real-time recalculation for obstacle avoidance verification
- evaluation measures for algorithms are:
    - time to reach target
    - amount of collisions (including collisions between drones in the swarm)
    - smoothness of the trajectory
    - behavior in the moment of dynamic obstacle occurance

## Technologies
- python 3.10 - main language of the project - version determined by main library of the project (gym-pybullet-drones)
- numpy - for fast calculations
- matplotlib - for plots
- pandas - for data analytics
- scipy - for scientific calculations
- hydra-core - for configuration management
- jupyter - for notebooks
- pytest - for testing
- pip - package manager
    - gym-pybullet-drones - framework for simulation

## Installation
1. Download this repository
2. Ensure you have Conda on your OS
3. Enter to the project eg. `cd ~/drone-swarm-simulation`
4. Dowload dependencies with conda: `conda env create -f environment.yml`

## Run simulation
1. Enter environment with `conda activate drone-swarm-env`
2. Call experiment eg. from gym-pybullet-drones `python ~miniconda3/envs/drone-swarm-env/lib/python3.10/site-packages/gym_pybullet_drones/examples/learn.py`
3. Check results in the `/results`

## My setup
- Procesor: AMD Ryzen 7 7700
- Ram: DDR5 64GB/6000 (2x32GB) CL30
- Graphic: Nvidia GIGABYTE RTX 4060 TI
- Disk: SSD UD85 2TB PCIe M.2 2280 NVMe Gen 4x4 3600/2800
- OS: Fedora 43