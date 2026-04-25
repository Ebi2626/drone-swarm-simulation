import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from mealpy import FloatVar

from src.algorithms.abstraction.trajectory.strategies.ssa_strategy import (
    SSAProblemAdapter,
    LoggedOriginalSSA,
    ssa_swarm_strategy,
)

TARGET_MODULE = "src.algorithms.abstraction.trajectory.strategies.ssa_strategy"

# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def mock_world_data():
    world = MagicMock()
    world.min_bounds = np.array([0.0, 0.0, 0.0])
    world.max_bounds = np.array([100.0, 100.0, 20.0])
    world.bounds = np.array([[0.0, 100.0], [0.0, 100.0], [0.0, 20.0]])
    return world

@pytest.fixture
def basic_positions():
    """Pozycje startowe i docelowe dla 2 dronow."""
    starts = np.array([[10.0, 10.0, 2.0], [20.0, 20.0, 2.0]])
    targets = np.array([[90.0, 90.0, 5.0], [80.0, 80.0, 5.0]])
    return starts, targets

@pytest.fixture
def single_drone_positions():
    """Pozycje startowe i docelowe dla 1 drona."""
    starts = np.array([[10.0, 10.0, 2.0]])
    targets = np.array([[90.0, 90.0, 5.0]])
    return starts, targets

@pytest.fixture
def mock_evaluator():
    """Atrapa ewaluatora wektoryzowanego (zwraca zera jako F i brak naruszeń dla G)."""
    evaluator = MagicMock()
    def side_effect(trajectories, out):
        pop_size = trajectories.shape[0]
        out["F"] = np.ones((pop_size, 3)) * 10.0
        out["G"] = np.full((pop_size, 5), -1.0)
    evaluator.evaluate.side_effect = side_effect
    return evaluator


# ===========================================================================
# TESTY: SSAProblemAdapter
# ===========================================================================

class TestSSAProblemAdapter:

    def _create_adapter(self, starts, targets, mock_scalar_adapter, mock_evaluator):
        n_drones = starts.shape[0]
        n_inner = 3
        d = n_drones * n_inner * 3
        bounds = FloatVar(lb=np.full(d, -50.0), ub=np.full(d, 150.0))
        
        return SSAProblemAdapter(
            bounds=bounds,
            evaluator=mock_evaluator,
            scalar_adapter=mock_scalar_adapter,
            start_pos=starts,
            target_pos=targets,
            n_drones=n_drones,
            n_inner=n_inner,
            n_output_samples=20,
        )

    def test_amend_position_safeguards_against_nan_and_bounds(self, basic_positions):
        """SSA bywa niestabilne numerycznie (np. rzuca nan przez dzielenie), weryfikacja czy amend_position go chroni."""
        starts, targets = basic_positions
        adapter = self._create_adapter(starts, targets, MagicMock(), MagicMock())
        
        # Obliczenie poprawnego wymiaru (2 drony * 3 węzły wewnętrzne * 3 koordynaty)
        n_drones = starts.shape[0]
        n_inner = 3
        d = n_drones * n_inner * 3  # = 18
        
        # Tworzymy wektor docelowy i ustawiamy pierwsze trzy wartości na skrajne/błędne
        pos = np.zeros(d)
        pos[0] = 160.0   # Powyżej limitu ub (150)
        pos[1] = -60.0   # Poniżej limitu lb (-50)
        pos[2] = np.nan  # Do zastąpienia (według logiki nan -> ub)
        
        amended = adapter.amend_position(pos)
        
        # Oczekiwany wektor po korekcie
        expected = np.zeros(d)
        expected[0] = 150.0
        expected[1] = -50.0
        expected[2] = 150.0
        
        np.testing.assert_array_equal(amended, expected)

    def test_obj_func_delegates_to_scalar_adapter(self, basic_positions):
        """Sprawdza czy funkcja celu rzuca zadanie do adaptera po wstępnym klipowaniu wektora decyzyjnego."""
        starts, targets = basic_positions
        n_drones = starts.shape[0]
        n_inner = 3
        d = n_drones * n_inner * 3

        mock_scalar_adapter = MagicMock()
        mock_scalar_adapter.return_value = np.array([42.5])
        
        adapter = self._create_adapter(starts, targets, mock_scalar_adapter, MagicMock())
        x = np.random.rand(d)
        fitness = adapter.obj_func(x)
        
        assert fitness == 42.5
        mock_scalar_adapter.assert_called_once()

    def test_evaluate_population_returns_correct_shape(self, basic_positions):
        """Wektorowe przetwarzanie populacji musi zachować rozmiar z mealpy (pop_size,)."""
        starts, targets = basic_positions
        n_drones = starts.shape[0]
        n_inner = 3
        d = n_drones * n_inner * 3
        pop_size = 5

        mock_scalar_adapter = MagicMock()
        mock_scalar_adapter.return_value = np.full(pop_size, 10.0)
        
        adapter = self._create_adapter(starts, targets, mock_scalar_adapter, MagicMock())
        population = np.random.rand(pop_size, d)
        fitness = adapter.evaluate_population(population)
        
        assert fitness.shape == (pop_size,)
        np.testing.assert_allclose(fitness, np.full(pop_size, 10.0))


# ===========================================================================
# TESTY: ssa_swarm_strategy
# ===========================================================================

class TestSSASwarmStrategy:

    @pytest.fixture
    def patch_deps(self, mock_evaluator):
        """Mockuje wszystkie wywołania logiki wspólnej, aby testować jedynie logikę samej orkiestracji ssa_swarm_strategy bez śmiecenia dysku."""
        # W module SSA używasz np. TrajectorySOOAdapter i VectorizedEvaluator.
        # Musimy zablokować tworzenie plików historycznych wewnątrz SOO.
        with patch(f"{TARGET_MODULE}.VectorizedEvaluator", return_value=mock_evaluator), \
             patch(f"{TARGET_MODULE}.TrajectorySOOAdapter") as MockAdapter, \
             patch(f"{TARGET_MODULE}.SwarmOptimizationProblem") as MockProblem, \
             patch(f"{TARGET_MODULE}.StraightLineNoiseSampling") as MockSampling, \
             patch(f"{TARGET_MODULE}.generate_bspline_batch") as MockBSpline, \
             patch("hydra.core.hydra_config.HydraConfig") as mock_hydra, \
             patch(f"{TARGET_MODULE}.OptimizationHistoryWriter") as mock_writer, \
             patch(f"{TARGET_MODULE}.TimingCollector") as mock_timing:
             
            # Izolacja wyjścia hydry żeby zapobiec ewentualnym ucieczkom katalogu
            mock_hydra.get.return_value.runtime.output_dir = "dummy_dir_in_memory"

            # Wymuszone zignorowanie zapisu czasów
            mock_timing.return_value.save_csv = MagicMock()
            
            # Wymuszone zignorowanie zapisu historii
            mock_writer.return_value.put_generation_data = MagicMock()
            mock_writer.return_value.close = MagicMock()

            # Konfiguracja zwracanych obiektów
            mock_scalar_instance = MagicMock()
            mock_scalar_instance._f_ref = np.array([1.0, 1.0, 1.0])
            MockAdapter.return_value = mock_scalar_instance
            
            mock_prob_instance = MagicMock()
            mock_prob_instance.xl = np.full(10, -50.0)
            mock_prob_instance.xu = np.full(10, 150.0)
            MockProblem.return_value = mock_prob_instance
            
            mock_samp_instance = MagicMock()
            mock_samp_instance._do.return_value = np.zeros((5, 10))
            MockSampling.return_value = mock_samp_instance
            
            def bspline_side_effect(sparse, num_samples):
                drones = sparse.shape[1]
                return np.zeros((1, drones, num_samples, 3))
            MockBSpline.side_effect = bspline_side_effect
            
            yield
    
    def test_output_shape(self, mock_world_data, basic_positions, patch_deps):
        """Wynik zoptymalizowanego lotu powinien poprawnie się formować poprzez b-spline batch generator."""
        starts, targets = basic_positions
        n_waypoints = 20
        n_drones = 2

        with patch(f"{TARGET_MODULE}.LoggedOriginalSSA") as MockSSA:
            mock_model = MagicMock()
            mock_agent = MagicMock()
            mock_agent.solution = np.zeros(2 * 3 * 3)
            mock_agent.target.fitness = 10.0
            mock_model.solve.return_value = mock_agent
            MockSSA.return_value = mock_model

            result = ssa_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=[],
                world_data=mock_world_data,
                number_of_waypoints=n_waypoints,
                drone_swarm_size=n_drones,
                algorithm_params={"pop_size": 5, "n_gen": 1, "n_inner_waypoints": 3},
            )

        assert result.shape == (n_drones, n_waypoints, 3)

    def test_single_drone(self, mock_world_data, single_drone_positions, patch_deps):
        """UAV Swarm Optimization dla pojedynczego drona z użyciem Mealpy."""
        starts, targets = single_drone_positions
        n_waypoints = 15

        with patch(f"{TARGET_MODULE}.LoggedOriginalSSA") as MockSSA:
            mock_model = MagicMock()
            mock_agent = MagicMock()
            mock_agent.solution = np.zeros(1 * 3 * 3)
            mock_agent.target.fitness = 10.0
            mock_model.solve.return_value = mock_agent
            MockSSA.return_value = mock_model

            result = ssa_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=[],
                world_data=mock_world_data,
                number_of_waypoints=n_waypoints,
                drone_swarm_size=1,
                algorithm_params={"pop_size": 5, "n_gen": 1, "n_inner_waypoints": 3},
            )

        assert result.shape == (1, n_waypoints, 3)

    def test_fallback_on_exception(self, mock_world_data, basic_positions, patch_deps):
        """Wykrycie błędu krytycznego podczas ewaluacji populacji uruchamia bezpieczny lot po linii prostej."""
        starts, targets = basic_positions
        n_waypoints = 10
        n_drones = 2

        with patch(f"{TARGET_MODULE}.LoggedOriginalSSA") as MockSSA:
            mock_model_instance = MagicMock()
            mock_model_instance.solve.side_effect = RuntimeError("Symulowany blad SSA.")
            MockSSA.return_value = mock_model_instance

            result = ssa_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=[],
                world_data=mock_world_data,
                number_of_waypoints=n_waypoints,
                drone_swarm_size=n_drones,
                algorithm_params={"pop_size": 5, "n_gen": 1, "n_inner_waypoints": 3},
            )

        assert result.shape == (n_drones, n_waypoints, 3)
        np.testing.assert_array_almost_equal(result[0, 0, :2], starts[0, :2])
        np.testing.assert_array_almost_equal(result[0, -1, :2], targets[0, :2])

    def test_fallback_z_altitude_clamped(self, mock_world_data, patch_deps):
        """Fallback powinien wymuszać minimalną wysokość Z >= 0.5 lub params['min_safe_altitude']."""
        starts = np.array([[10.0, 10.0, 0.0]])
        targets = np.array([[90.0, 90.0, 0.0]])
        n_waypoints = 10

        with patch(f"{TARGET_MODULE}.LoggedOriginalSSA") as MockSSA:
            mock_model_instance = MagicMock()
            mock_model_instance.solve.side_effect = RuntimeError("Blad")
            MockSSA.return_value = mock_model_instance

            result = ssa_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=[],
                world_data=mock_world_data,
                number_of_waypoints=n_waypoints,
                drone_swarm_size=1,
                algorithm_params={"pop_size": 5, "n_gen": 2, "n_inner_waypoints": 3, "min_safe_altitude": 0.5},
            )

        assert np.all(result[0, :, 2] >= 0.5)

    def test_mealpy_called_correctly_with_biological_params(self, mock_world_data, basic_positions, patch_deps):
        """Weryfikuje czy biologiczne parametry ST, PD, SD algorytmu Sparrow Search wchodzą do Mealpy."""
        starts, targets = basic_positions
        custom_params = {
            "pop_size": 12, "n_gen": 5, "n_workers": 2, "seed": 42,
            "st": 0.9, "pd_ratio": 0.3, "sd_ratio": 0.15
        }

        with patch(f"{TARGET_MODULE}.LoggedOriginalSSA") as MockSSA:
            mock_model_instance = MagicMock()
            mock_best_agent = MagicMock()
            mock_best_agent.solution = np.zeros(2 * 3 * 3)
            mock_best_agent.target.fitness = 10.0
            mock_model_instance.solve.return_value = mock_best_agent
            MockSSA.return_value = mock_model_instance
            
            ssa_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=[],
                world_data=mock_world_data,
                number_of_waypoints=20,
                drone_swarm_size=2,
                algorithm_params=custom_params,
            )

            call_args, call_kwargs = MockSSA.call_args
            assert call_kwargs["epoch"] == 5
            assert call_kwargs["pop_size"] == 12
            assert call_kwargs["ST"] == 0.9
            assert call_kwargs["PD"] == 0.3
            assert call_kwargs["SD"] == 0.15
            
            solve_kwargs = mock_model_instance.solve.call_args.kwargs
            assert solve_kwargs["mode"] == "thread"
            assert solve_kwargs["n_workers"] == 2