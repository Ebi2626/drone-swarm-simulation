from src.algorithms.TrajectoryFollowingAlgorithm import TrajectoryFollowingAlgorithm
from .environments.EmptyWorld import EmptyWorld
from .environments.ForestWorld import ForestWorld
from .environments.UrbanWorld import UrbanWorld
from .algorithms.LinearTestAlgorithm import LinearTestAlgorithm

ENVIRONMENT_REGISTRY = {
    "empty": EmptyWorld,
    "forest": ForestWorld,
    "urban": UrbanWorld,
}

ALGORITHM_REGISTRY = {
    "LinearTestAlgorithm": LinearTestAlgorithm, ## Simple algorithm for testing
    "TrajectoryFollowingAlgorithm": TrajectoryFollowingAlgorithm
    # "OOA": OOAPlanner,
    # "MSFFOA": MSFFOAPlanner,
    # "SSA": SSAPlanner,
    # "NSGA": NSGAPlanner,
}

def get_environment(name):
    if name not in ENVIRONMENT_REGISTRY:
        available = list(ENVIRONMENT_REGISTRY.keys())
        raise ValueError(f"Envrionment '{name}' not found in registry. Available: {available}")
    
    return ENVIRONMENT_REGISTRY[name]

def get_algorithm(name):
    if name not in ALGORITHM_REGISTRY:
        available = list(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Algorithm '{name}' not found in registry. Available: {available}")
    
    return ALGORITHM_REGISTRY[name]