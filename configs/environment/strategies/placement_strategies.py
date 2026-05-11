"""Rejestr strategii rozmieszczania przeszkód używanych w `generate_obstacles`.

Klucze odpowiadają nazwom strategii w plikach konfiguracyjnych YAML.
"""

from src.environments.abstraction.generate_obstacles import strategy_grid_jitter, strategy_random_uniform


PLACEMENT_STRATEGY_REGISTRY: dict[str, callable] = {
    "strategy_grid_jitter": strategy_grid_jitter,
    "strategy_random_uniform": strategy_random_uniform,
}
"""Klucz — nazwa strategii w pliku YAML; wartość — funkcja rozmieszczania."""


def get_placement_strategy(name: str) -> callable:
    """Zwróć funkcję rozmieszczania przeszkód zarejestrowaną pod `name`.

    Args:
        name: Klucz strategii zgodny z `PLACEMENT_STRATEGY_REGISTRY`.

    Returns:
        Funkcja o sygnaturze strategii rozmieszczania
        (zob. `generate_obstacles`).

    Raises:
        ValueError: Gdy `name` nie występuje w rejestrze.
    """
    if name not in PLACEMENT_STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: '{name}'. "
            f"Available: {list(PLACEMENT_STRATEGY_REGISTRY.keys())}"
        )
    return PLACEMENT_STRATEGY_REGISTRY[name]
