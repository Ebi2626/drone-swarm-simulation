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
- numpy
- matplotlib
- pandas
- scipy
- hydra-core
- jupyter
- pytest
- pip
