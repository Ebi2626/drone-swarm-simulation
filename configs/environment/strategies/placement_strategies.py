# src/environments/strategies/placement_strategies.py


from src.environments.abstraction.generate_obstacles import strategy_grid_jitter, strategy_random_uniform

# Rejestr: klucz = nazwa z YAML, wartość = funkcja
PLACEMENT_STRATEGY_REGISTRY: dict[str, callable] = {
    "strategy_grid_jitter": strategy_grid_jitter,
    "strategy_random_uniform": strategy_random_uniform,
}

def get_placement_strategy(name: str) -> callable:
    if name not in PLACEMENT_STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: '{name}'. "
            f"Available: {list(PLACEMENT_STRATEGY_REGISTRY.keys())}"
        )
    return PLACEMENT_STRATEGY_REGISTRY[name]
