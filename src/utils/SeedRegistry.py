from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable
import random
import numpy as np
from numba import njit

DEFAULT_SEED_NAMESPACES = (
    "global", # global entry seed
    "numba", # global seed for numba jit functions
    "environment", # seed for random obstacles placement
    "optimizer", # seed for optimizers
    "sampling", # seed for initial population
    "avoidance" # seed for collision avoidance optimizers
)

@dataclass
class SeedRegistry:
    """Hierarchiczny rejestr ziaren z `np.random.SeedSequence` per namespace.

    Master seed → `SeedSequence.spawn(len(namespaces))` ⇒ niezależne, ale
    deterministyczne sub-seeds, jeden per komponent (`environment`,
    `optimizer`, `sampling`, `avoidance`, …).
    """

    master_seed: int
    namespaces: Iterable[str] = DEFAULT_SEED_NAMESPACES
    _children: Dict[str, np.random.SeedSequence] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Zbuduj `_children` przez `SeedSequence.spawn` z deduplikowanej listy nazw."""
        names = list(dict.fromkeys(self.namespaces))
        if not names:
            raise ValueError("At least one namespace is required.")
        root = np.random.SeedSequence(int(self.master_seed))
        spawned = root.spawn(len(names))
        self._children = dict(zip(names, spawned))

    def seed(self, namespace: str) -> int:
        """Zwróć skalarne ziarno `uint32` dla `namespace` (deterministyczne per master_seed)."""
        ss = self._children[namespace]
        return int(ss.generate_state(1, dtype=np.uint32)[0])

    def rng(self, namespace: str) -> np.random.Generator:
        """Zwróć `np.random.Generator` zasilony `SeedSequence` namespace'u."""
        return np.random.default_rng(self._children[namespace])

    def export(self) -> dict:
        """Zwróć słownik `{master_seed, children: {namespace: seed}}` do logowania."""
        return {
            "master_seed": int(self.master_seed),
            "children": {name: self.seed(name) for name in self._children}
        }


@njit(cache=True)
def seed_numba(seed: int) -> None:
    """Ustaw globalne ziarno wewnątrz JIT-skompilowanego kodu Numba."""
    np.random.seed(seed)


def bootstrap_global_seed(seed: int) -> None:
    """Ustaw globalne ziarna `random` i `numpy.random` (poza Numba)."""
    random.seed(seed)
    np.random.seed(seed)