import functools
from typing import TYPE_CHECKING
from hydra.utils import instantiate

from configs.environment.strategies.placement_strategies import get_placement_strategy
from src.environments.abstraction.generate_obstacles import generate_obstacles
from src.environments.abstraction.generate_world_boundaries import generate_world_boundaries
from src.runner.ExperimentDataStrategy import ExperimentDataStrategy
from src.utils.trajectory_validator import validate_trajectories
from src.utils.SeedRegistry import SeedRegistry

if TYPE_CHECKING:
    from main import ExperimentRunner


class GenerationDataStrategy(ExperimentDataStrategy):
    def prepare_data(self, runner: "ExperimentRunner", seeds: SeedRegistry):
        print("[INFO] Generowanie nowego środowiska i optymalizacja trajektorii (Offline Path-Planning)...")
        # 1. Generowanie granic świata
        runner.world_data = generate_world_boundaries(
            width=runner.track_width, 
            length=runner.track_length,
            height=runner.track_height, 
            ground_height=runner.ground_position
        )
        
        # 2. Generowanie przeszkód
        if runner.placement_strategy_name is not None:
            runner.obstacles_data = generate_obstacles(
                runner.world_data,
                n_obstacles=runner.obstacles_number,
                shape_type=runner.shape_type,
                placement_strategy=get_placement_strategy(runner.placement_strategy_name),
                size_params={
                    'length': runner.obstacle_length,
                    'width': runner.obstacle_width,
                    'height': runner.obstacle_height,
                },
                start_positions=runner.start_positions,
                target_positions=runner.end_positions,
                safe_radius=runner.safe_radius,
                rng=seeds.rng("environment")
            )
        
        # 3. Zgodnie z oryginałem - wstrzykujemy DOKŁADNIE 6 argumentów do strategii.
        # Wcześniejsze dodanie "algorithm_params" uaktywniło inny tryb obliczeniowy kary 
        # i spowodowało przekazanie 5 argumentów zamiast 4 do funkcji `_dist_segment_to_box`.
        counting_strategy = instantiate(runner.cfg.optimizer)
        counting_protocol = functools.partial(
            counting_strategy,
            start_positions=runner.start_positions,
            target_positions=runner.end_positions,
            obstacles_data=runner.obstacles_data,
            world_data=runner.world_data,
            number_of_waypoints=runner.number_of_waypoints,
            drone_swarm_size=runner.num_drones,
            seeds=seeds
        )
        
        print(f"\n🚀 Uruchamianie obliczeń algorytmu metaheurystycznego...")
        runner.drones_trajectories = counting_protocol()

        # Sanity-check zaraz po wyjściu ze strategii — patologie typu
        # „drony stojące w starcie" (plan.md, Krok 2) muszą być widoczne
        # w stdout PRZED startem PyBullet, a nie dopiero w post-mortem ETL.
        opt_label = str(runner.cfg.optimizer.get("_target_", "strategy")).split(".")[-1]
        validate_trajectories(
            runner.drones_trajectories,
            runner.start_positions,
            label=opt_label,
        )

        # 4. Archiwizacja stanu początkowego
        if runner.logger is not None:
            runner.logger.log_chosen_trajectories(runner.drones_trajectories)
            runner.logger.log_world_dimensions(runner.world_data)
            runner.logger.log_obstacles(runner.obstacles_data)