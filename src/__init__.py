from .simulation.worlds.EmptyWorld import EmptyWorld
from .simulation.worlds.ForestWorld import ForestWorld
from .simulation.worlds.UrbanWorld import UrbanWorld

ENVIRONMENT_REGISTRY = {
    "empty": EmptyWorld,
    "forest": ForestWorld,
    "urban": UrbanWorld, # Rejestracja
}

ALGORITHM_REGISTRY = {
    # "pso": PSOPlanner,
    # "gwo": GWOPlanner,
    # "ssa": SSAPlanner,
    # "de": DEPlanner,
}

def get_environment(name):
    """
    Fabryka zwracająca klasę środowiska na podstawie nazwy z konfiguracji.
    """
    if name not in ENVIRONMENT_REGISTRY:
        # Wyświetl dostępne opcje, aby ułatwić debugowanie
        available = list(ENVIRONMENT_REGISTRY.keys())
        raise ValueError(f"Środowisko '{name}' nie zostało znalezione w rejestrze. Dostępne: {available}")
    
    return ENVIRONMENT_REGISTRY[name]

def get_algorithm(name):
    """
    Fabryka zwracająca klasę algorytmu na podstawie nazwy z konfiguracji.
    """
    if name not in ALGORITHM_REGISTRY:
        available = list(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Algorytm '{name}' nie został znaleziony w rejestrze. Dostępne: {available}")
    
    return ALGORITHM_REGISTRY[name]