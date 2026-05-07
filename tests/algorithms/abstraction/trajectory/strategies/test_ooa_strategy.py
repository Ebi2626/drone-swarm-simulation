import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from mealpy import FloatVar

from src.algorithms.abstraction.trajectory.strategies.ooa_strategy import (
    OOAProblemAdapter,
    osprey_swarm_strategy,
    LoggedOriginalOOA
)
from src.utils.SeedRegistry import SeedRegistry

TARGET_MODULE = "src.algorithms.abstraction.trajectory.strategies.ooa_strategy"


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
    """Pozycje startowe i docelowe dla 2 dronów."""
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
    """Podstawowy mock ewaluatora zwracający zera dla celów wektoryzacji."""
    evaluator = MagicMock()
    def side_effect(trajectories, out):
        pop_size = trajectories.shape[0]
        out["F"] = np.ones((pop_size, 3)) * 10.0
        out["G"] = np.full((pop_size, 5), -1.0)
    evaluator.evaluate.side_effect = side_effect
    return evaluator

@pytest.fixture
def mock_master_seed():
    seeds = SeedRegistry(master_seed=int(42))
    return seeds

# ===========================================================================
# TESTY: OOAProblemAdapter
# ===========================================================================

class TestOOAProblemAdapter:

    def _create_adapter(self, starts, targets, mock_scalar_adapter, mock_evaluator):
        n_drones = starts.shape[0]
        n_inner = 3
        d = n_drones * n_inner * 3
        bounds = FloatVar(lb=np.full(d, -50.0), ub=np.full(d, 150.0))
        
        return OOAProblemAdapter(
            bounds=bounds,
            evaluator=mock_evaluator,
            scalar_adapter=mock_scalar_adapter,
            start_pos=starts,
            target_pos=targets,
            n_drones=n_drones,
            n_inner=n_inner,
            n_output_samples=20,
        )

    def test_amend_position_clamps_and_handles_nan(self, basic_positions):
        """Metoda amend_position powinna chronić przed przekroczeniem barier przestrzeni oraz wartościami NaN."""
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
        """Sprawdza czy adapter rygorystycznie deleguje logikę ewaluacyjną fitness do scalera."""
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
        """Funkcja ewaluacji wsadowej wektora powinna zachować pierwotny rozmiar populacji."""
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
# TESTY: osprey_swarm_strategy
# ===========================================================================

class TestOspreySwarmStrategy:

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
        """Wynik zoptymalizowanego lotu powinien zachować kształt narzucony przez architekturę B-Spline."""
        starts, targets = basic_positions
        n_waypoints = 20
        n_drones = 2

        with patch(f"{TARGET_MODULE}.LoggedOriginalOOA") as MockOOA:
            mock_model = MagicMock()
            mock_agent = MagicMock()
            mock_agent.solution = np.zeros(2 * 3 * 3)
            mock_agent.target.fitness = 10.0
            mock_model.solve.return_value = mock_agent
            MockOOA.return_value = mock_model

            result = osprey_swarm_strategy(
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
        """Strategia optymalizacji wieloobiektowej powinna poprawnie rzutować się na jedno-agentowe loty UAV."""
        starts, targets = single_drone_positions
        n_waypoints = 15

        with patch(f"{TARGET_MODULE}.LoggedOriginalOOA") as MockOOA:
            mock_model = MagicMock()
            mock_agent = MagicMock()
            mock_agent.solution = np.zeros(1 * 3 * 3)
            mock_agent.target.fitness = 10.0
            mock_model.solve.return_value = mock_agent
            MockOOA.return_value = mock_model

            result = osprey_swarm_strategy(
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
        """Sprawdza mechanizm uodpornienia (fallback) generujący bezpieczny lot liniowy w przypadku błędu Mealpy."""
        starts, targets = basic_positions
        n_waypoints = 10
        n_drones = 2

        with patch(f"{TARGET_MODULE}.LoggedOriginalOOA") as MockOOA:
            mock_model_instance = MagicMock()
            mock_model_instance.solve.side_effect = RuntimeError("Symulowany blad Mealpy/Evaluatora")
            MockOOA.return_value = mock_model_instance

            result = osprey_swarm_strategy(
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

    def test_logged_ooa_caches_FG_on_agents_and_skips_logger_reeval(self, basic_positions):
        """Regresja perf-fix 2026-05-07.

        `LoggedOriginalOOA.generate_agent` ma przykleić F/G (z cache
        scalar_adapter) do `agent._F_row/_G_row` (atrybut na AGENCIE
        przetrwa `agent.copy()`, na targecie nie). `evolve` przy logowaniu
        używa tych F/G zamiast wywoływać `evaluate_full(decisions)`
        (eliminacja nadmiarowej re-ewaluacji 1× pop_size NFE/gen).
        """
        starts, targets = basic_positions
        n_drones = starts.shape[0]
        n_inner = 3
        d = n_drones * n_inner * 3
        pop_size = 5
        n_obj = 5
        n_g = 3

        scalar_adapter = MagicMock()
        # Symulacja TrajectorySOOAdapter.__call__: po każdym wywołaniu
        # ustawia last_objectives/last_constraints na ndarray (1, M).
        def scalar_call_side_effect(inner):
            batch = inner.shape[0]
            scalar_adapter.last_objectives = np.full((batch, n_obj), 0.7)
            scalar_adapter.last_constraints = np.full((batch, n_g), -0.2)
            return np.full(batch, 11.0)
        scalar_adapter.side_effect = scalar_call_side_effect

        evaluator = MagicMock()
        evaluator.individuals_evaluated = 0

        adapter = OOAProblemAdapter(
            bounds=FloatVar(lb=np.full(d, -50.0), ub=np.full(d, 150.0)),
            evaluator=evaluator,
            scalar_adapter=scalar_adapter,
            start_pos=starts,
            target_pos=targets,
            n_drones=n_drones,
            n_inner=n_inner,
            n_output_samples=20,
        )

        history_writer = MagicMock()
        ooa = LoggedOriginalOOA(
            epoch=1,
            pop_size=pop_size,
            history_writer=history_writer,
            history_problem=adapter,
        )
        # mealpy.Optimizer.get_target deleguje do self.problem.get_target —
        # w pełnym flow ustawiane przez solve(); w teście jednostkowym
        # podpinamy wprost ten sam adapter.
        ooa.problem = adapter

        # 1) generate_agent przykleja F_row/G_row na AGENCIE.
        agent = ooa.generate_agent(np.zeros(d))
        assert isinstance(getattr(agent, "_F_row", None), np.ndarray)
        assert agent._F_row.shape == (n_obj,)
        assert isinstance(getattr(agent, "_G_row", None), np.ndarray)
        assert agent._G_row.shape == (n_g,)

        # 2) evolve nie woła evaluate_full gdy _F_row/_G_row są dostępne.
        ooa.pop = [ooa.generate_agent(np.zeros(d)) for _ in range(pop_size)]

        adapter.evaluate_full = MagicMock(side_effect=AssertionError(
            "evaluate_full nie powinno być wołane gdy _F_row/_G_row są zacache'owane"
        ))

        with patch("mealpy.swarm_based.OOA.OriginalOOA.evolve") as super_evolve:
            super_evolve.return_value = None
            ooa.evolve(epoch=0)

        assert history_writer.put_generation_data.call_count == 1
        logged = history_writer.put_generation_data.call_args.args[0]
        assert logged["objectives_matrix"].shape == (pop_size, n_obj)

    def test_mealpy_called_correctly(self, mock_world_data, basic_positions, patch_deps, mock_master_seed):
        """Weryfikuje, czy Mealpy jest parametryzowany zgodnie z konfiguracją wejściową algorytmu z logowaniem danych."""
        starts, targets = basic_positions
        custom_params = {"pop_size": 12, "n_gen": 5, "n_workers": 2, "seed": 42, "n_inner_waypoints": 3}

        with patch(f"{TARGET_MODULE}.LoggedOriginalOOA") as MockOOA:
            mock_model_instance = MagicMock()
            mock_best_agent = MagicMock()
            mock_best_agent.solution = np.zeros(2 * 3 * 3)
            mock_best_agent.target.fitness = 10.0
            mock_model_instance.solve.return_value = mock_best_agent
            MockOOA.return_value = mock_model_instance
            
            osprey_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=[],
                world_data=mock_world_data,
                number_of_waypoints=20,
                drone_swarm_size=2,
                algorithm_params=custom_params,
                seeds=mock_master_seed
            )

            call_args, call_kwargs = MockOOA.call_args
            assert call_kwargs["epoch"] == 5
            assert call_kwargs["pop_size"] == 12
            
            solve_kwargs = mock_model_instance.solve.call_args.kwargs
            assert solve_kwargs["mode"] == "thread"
            assert solve_kwargs["n_workers"] == 2
            assert solve_kwargs["seed"] == 3276785861 # master-seed 42
            assert "problem" in solve_kwargs
            assert "starting_solutions" in solve_kwargs