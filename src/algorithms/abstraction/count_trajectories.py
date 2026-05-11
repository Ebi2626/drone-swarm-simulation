"""Strategy Pattern dla generacji trajektorii w fazie offline.

`TrajectoryStrategyProtocol` definiuje wspólny kontrakt wszystkich
algorytmów planowania (NSGA-III, MSFFOA, OOA, SSA); `count_trajectories`
jest cienkim wrapperem delegującym do wybranej strategii.
"""
from typing import Any, Dict, Optional, Protocol
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.abstraction.generate_obstacles import ObstaclesData
from numpy.typing import NDArray
import numpy as np


class TrajectoryStrategyProtocol(Protocol):
    """Kontrakt strategii planowania trajektorii dla roju dronów.

    Wszystkie implementacje (np. `nsga3_swarm_strategy`,
    `msffoa_strategy`) muszą akceptować pełen zestaw argumentów
    wymagany przez planowanie wielokryterialne i zwracać wspólny
    format tensora trajektorii.
    """
    def __call__(
        self,
        *,
        start_positions: NDArray[np.float64],
        target_positions: NDArray[np.float64],
        obstacles_data: ObstaclesData,
        world_data: WorldData,
        number_of_waypoints: int,
        drone_swarm_size: int,
        algorithm_params: Optional[Dict[str, Any]] = None
    ) -> NDArray[np.float64]:
        """Zaplanuj trajektorie dla całego roju.

        Args:
            start_positions: `(N, 3)` pozycje startowe dronów [m].
            target_positions: `(N, 3)` pozycje docelowe dronów [m].
            obstacles_data: Geometria przeszkód statycznych w środowisku.
            world_data: Wymiary i granice świata symulacji.
            number_of_waypoints: Liczba punktów `W` w wynikowej trajektorii.
            drone_swarm_size: Rozmiar roju `N` (zgodny z liczbą wierszy
                w `start_positions` i `target_positions`).
            algorithm_params: Hiperparametry algorytmu (rozmiar populacji,
                wagi obiektywów itp.); `None` ⇒ wartości domyślne.

        Returns:
            Tensor `(N, W, 3)` waypointów `[x, y, z]` — po jednym wierszu
            na drona, `W` punktów na trajektorię.
        """
        pass


def count_trajectories(
        world_data: WorldData,
        obstacles_data: ObstaclesData,
        counting_protocol: TrajectoryStrategyProtocol,
        drone_swarm_size: int,
        number_of_waypoints: int,
        start_positions: NDArray[np.float64],
        target_positions: NDArray[np.float64],
        algorithm_params: Optional[Dict[str, Any]] = None
        ) -> NDArray[np.float64]:
    """Wywołaj wybraną strategię planowania i zwróć tensor trajektorii.

    Cienki wrapper porządkujący argumenty przed delegacją — wszystkie
    parametry są przekazywane do `counting_protocol` jako keyword-only.

    Args:
        world_data: Wymiary i granice świata symulacji.
        obstacles_data: Geometria przeszkód statycznych.
        counting_protocol: Implementacja `TrajectoryStrategyProtocol`
            (konkretny algorytm planujący).
        drone_swarm_size: Rozmiar roju `N`.
        number_of_waypoints: Liczba punktów `W` w trajektorii.
        start_positions: `(N, 3)` pozycje startowe [m].
        target_positions: `(N, 3)` pozycje docelowe [m].
        algorithm_params: Hiperparametry przekazywane do strategii.

    Returns:
        Tensor `(N, W, 3)` waypointów `[x, y, z]`.
    """
    trajectories = counting_protocol(
        start_positions=start_positions,
        target_positions=target_positions,
        obstacles_data=obstacles_data,
        world_data=world_data,
        number_of_waypoints=number_of_waypoints,
        drone_swarm_size=drone_swarm_size,
        algorithm_params=algorithm_params
    )

    return trajectories
