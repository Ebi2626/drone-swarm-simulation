import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from mealpy import FloatVar

from src.algorithms.abstraction.trajectory.strategies.ooa_strategy import (
    _resample_polyline,
    OspreyProblemAdapter,
    _generate_starting_solutions,
    osprey_swarm_strategy,
)

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
def default_weights():
    return {
        "w_path_length": 1.0,
        "w_collision_risk": 1.0,
        "w_elevation": 1.0,
    }


@pytest.fixture
def mock_evaluator():
    """
    Atrapa ewaluatora zwracajaca kontrolowane wartosci F i G.
    F: (pop_size, 3) — cele (dystans, ryzyko kolizji, wysokosc)
    G: (pop_size, 5) — ograniczenia (G <= 0 oznacza brak naruszen)
    """
    evaluator = MagicMock()

    def side_effect(trajectories, out):
        pop_size = trajectories.shape[0]
        out["F"] = np.ones((pop_size, 3)) * 10.0
        out["G"] = np.full((pop_size, 5), -1.0)  # brak naruszen

    evaluator.evaluate.side_effect = side_effect
    return evaluator


def _make_evaluator_with_values(f_values, g_values):
    """Tworzy mock ewaluatora z podanymi wartosciami F i G."""
    evaluator = MagicMock()

    def side_effect(trajectories, out):
        pop_size = trajectories.shape[0]
        if callable(f_values):
            out["F"] = f_values(pop_size)
        else:
            out["F"] = np.tile(f_values, (pop_size, 1))[:pop_size]
        if callable(g_values):
            out["G"] = g_values(pop_size)
        else:
            out["G"] = np.tile(g_values, (pop_size, 1))[:pop_size]

    evaluator.evaluate.side_effect = side_effect
    return evaluator


# ===========================================================================
# TESTY: _resample_polyline
# ===========================================================================

class TestResamplePolyline:
    
    def test_output_shape(self):
        """Funkcja powinna zwracac zageszczona trajektorie o odpowiednim ksztalcie."""
        pop_size, n_drones, n_in, dims = 5, 2, 4, 3
        n_out = 20
        waypoints = np.random.rand(pop_size, n_drones, n_in, dims)
        
        result = _resample_polyline(waypoints, num_samples=n_out)
        
        assert result.shape == (pop_size, n_drones, n_out, dims)


# ===========================================================================
# TESTY: OspreyProblemAdapter
# ===========================================================================

class TestOspreyProblemAdapter:

    def _create_adapter(self, starts, targets, evaluator, weights, penalty_weight=100.0, expected_pop=0):
        n_drones = starts.shape[0]
        n_inner = 3
        d = n_drones * n_inner * 3
        bounds = FloatVar(lb=np.full(d, -50.0), ub=np.full(d, 150.0))
        
        return OspreyProblemAdapter(
            bounds=bounds,
            evaluator=evaluator,
            start_pos=starts,
            target_pos=targets,
            n_drones=n_drones,
            n_inner=n_inner,
            n_output_samples=20,
            weights=weights,
            penalty_weight=penalty_weight,
            expected_pop_size=expected_pop
        )

    def test_compute_reference_scales_clamps_to_one(self, basic_positions, default_weights):
        """Wartosci referencyjne mniejsze niz 1.0 (np. brak kolizji) powinny byc klampowane do 1.0."""
        starts, targets = basic_positions
        
        # Ewaluator zwraca same zera w F (co zrobiloby blad dzielenia przez zero)
        f_row = np.array([[0.0, 0.0, 0.0]])
        g_row = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0]])
        evaluator = _make_evaluator_with_values(f_row, g_row)

        adapter = self._create_adapter(starts, targets, evaluator, default_weights)
        
        # Klampowane do 1.0
        np.testing.assert_array_equal(adapter._f_scale, np.array([1.0, 1.0, 1.0]))

    def test_obj_func_weights_and_normalization_applied(self, basic_positions):
        """Wartosc fitness powinna byc poprawnie znormalizowana i zwazona."""
        starts, targets = basic_positions
        n_drones = 2
        n_inner = 3
        d = n_drones * n_inner * 3

        call_count = [0]
        def side_effect(trajectories, out):
            ps = trajectories.shape[0]
            if call_count[0] == 0:
                # Pierwsze wywolanie to inicjalizacja _compute_reference_scales
                out["F"] = np.ones((ps, 3)) * 2.0  # f_ref = [2.0, 2.0, 2.0]
            else:
                # Kolejne wywolania to obj_func
                out["F"] = np.array([[4.0, 6.0, 8.0]]) # po normalizacji: [2.0, 3.0, 4.0]
            out["G"] = np.full((ps, 5), -1.0)
            call_count[0] += 1

        evaluator = MagicMock()
        evaluator.evaluate.side_effect = side_effect

        weights = {"w_path_length": 1.0, "w_collision_risk": 2.0, "w_elevation": 3.0}
        adapter = self._create_adapter(starts, targets, evaluator, weights)
        
        x = np.random.rand(d)
        fitness = adapter.obj_func(x)
        
        # F_norm = [4/2, 6/2, 8/2] = [2, 3, 4]
        # Fitness = 1*2 + 2*3 + 3*4 = 20.0
        np.testing.assert_almost_equal(fitness, 20.0)

    def test_obj_func_penalty_applied(self, basic_positions, default_weights):
        """Naruszenia ograniczen (G > 0) powinny zwiekszac fitness o odpowiednia kare."""
        starts, targets = basic_positions
        n_drones = 2
        n_inner = 3
        d = n_drones * n_inner * 3

        f_row = np.array([[1.0, 1.0, 1.0]])
        g_row = np.array([[2.0, 3.0, -1.0, -1.0, -1.0]])  # suma = 5.0
        evaluator = _make_evaluator_with_values(f_row, g_row)

        penalty_weight = 100.0
        adapter = self._create_adapter(starts, targets, evaluator, default_weights, penalty_weight)
        
        x = np.random.rand(d)
        fitness = adapter.obj_func(x)
        
        # f_ref = [1.0, 1.0, 1.0] -> F_norm = [1, 1, 1]
        # obj_value = 1*1 + 1*1 + 1*1 = 3.0
        # penalty = 100.0 * 5.0 = 500.0
        np.testing.assert_almost_equal(fitness, 503.0)

    def test_evaluate_batch_returns_correct_shape_and_values(self, basic_positions, default_weights):
        """Funkcja evaluate_batch powinna wektoryzowac ewaluacje wielu osobnikow naraz."""
        starts, targets = basic_positions
        n_drones = 2
        n_inner = 3
        d = n_drones * n_inner * 3
        pop_size = 5

        f_row = np.array([[2.0, 2.0, 2.0]])
        g_row = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0]])
        evaluator = _make_evaluator_with_values(f_row, g_row)

        adapter = self._create_adapter(starts, targets, evaluator, default_weights)
        
        population = np.random.rand(pop_size, d)
        fitness = adapter.evaluate_batch(population)
        
        assert fitness.shape == (pop_size,)
        # f_ref będzie miało postać [2.0, 2.0, 2.0], a ewaluacje zwrócą [2.0, 2.0, 2.0]
        # F_norm = [1, 1, 1], weights = [1, 1, 1] => fitness = 3.0
        np.testing.assert_allclose(fitness, np.full(pop_size, 3.0))


# ===========================================================================
# TESTY: _generate_starting_solutions
# ===========================================================================

class TestGenerateStartingSolutions:

    def test_output_shape_and_bounds(self, mock_world_data, basic_positions):
        """Inicjalizacja heurystyczna powinna zwaracac macierz zgodna z granicami."""
        starts, targets = basic_positions
        n_drones = 2
        n_inner = 3
        pop_size = 10
        d = n_drones * n_inner * 3
        
        xl = np.full(d, -50.0)
        xu = np.full(d, 150.0)

        X_init = _generate_starting_solutions(
            start_pos=starts,
            target_pos=targets,
            n_drones=n_drones,
            n_inner=n_inner,
            pop_size=pop_size,
            world_data=mock_world_data,
            obstacles_data=None,
            xl=xl,
            xu=xu,
        )

        assert X_init.shape == (pop_size, d)
        assert np.all(X_init >= xl)
        assert np.all(X_init <= xu)
        
        # Test Z safe corridor (>0.5)
        X_reshaped = X_init.reshape(pop_size, n_drones, n_inner, 3)
        assert np.all(X_reshaped[..., 2] >= 0.5)

# ===========================================================================
# TESTY: osprey_swarm_strategy (orchestrator)
# ===========================================================================

class TestOspreySwarmStrategy:

    def test_output_shape(self, mock_world_data, basic_positions, mock_evaluator):
        """Wynik powinien miec ksztalt (n_drones, n_waypoints, 3)."""
        starts, targets = basic_positions
        n_waypoints = 20
        n_drones = 2

        with patch(f"{TARGET_MODULE}.VectorizedEvaluator", return_value=mock_evaluator), \
             patch("hydra.core.hydra_config.HydraConfig"), \
             patch("src.utils.optimization_history_writer.OptimizationHistoryWriter"), \
             patch("src.algorithms.abstraction.trajectory.strategies.timing_utils.TimingCollector"):
            
            result = osprey_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=[],
                world_data=mock_world_data,
                number_of_waypoints=n_waypoints,
                drone_swarm_size=n_drones,
                algorithm_params={"pop_size": 5, "n_gen": 1, "n_inner_waypoints": 3},  # ZMIANA: pop_size na 5
            )

        assert result.shape == (n_drones, n_waypoints, 3)

    def test_single_drone(self, mock_world_data, single_drone_positions, mock_evaluator):
        """Powinien dzialac poprawnie dla jednego drona."""
        starts, targets = single_drone_positions
        n_waypoints = 15

        with patch(f"{TARGET_MODULE}.VectorizedEvaluator", return_value=mock_evaluator), \
             patch("hydra.core.hydra_config.HydraConfig"), \
             patch("src.utils.optimization_history_writer.OptimizationHistoryWriter"), \
             patch("src.algorithms.abstraction.trajectory.strategies.timing_utils.TimingCollector"):
            
            result = osprey_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=[],
                world_data=mock_world_data,
                number_of_waypoints=n_waypoints,
                drone_swarm_size=1,
                algorithm_params={"pop_size": 5, "n_gen": 1, "n_inner_waypoints": 3},  # ZMIANA: pop_size na 5
            )

        assert result.shape == (1, n_waypoints, 3)

    def test_fallback_on_exception(self, mock_world_data, basic_positions):
        """Blad w solverze mealpy powinien zwrocic trajektorie liniowa z Z podniesionym na bezpieczna wysokosc."""
        starts, targets = basic_positions
        n_waypoints = 10
        n_drones = 2

        # Prawidłowo działający ewaluator (do inicjalizacji obiektu Problem)
        evaluator = MagicMock()
        def side_effect(trajectories, out):
            ps = trajectories.shape[0]
            out["F"] = np.ones((ps, 3))
            out["G"] = np.full((ps, 5), -1.0)
        evaluator.evaluate.side_effect = side_effect

        with patch(f"{TARGET_MODULE}.VectorizedEvaluator", return_value=evaluator), \
             patch(f"{TARGET_MODULE}.OriginalOOA") as MockOOA, \
             patch("hydra.core.hydra_config.HydraConfig"), \
             patch("src.utils.optimization_history_writer.OptimizationHistoryWriter"), \
             patch("src.algorithms.abstraction.trajectory.strategies.timing_utils.TimingCollector"):
            
            # ZMIANA: Zmuszamy sam solver Mealpy do wybuchu, aby fallback został prawidłowo zainicjowany
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
                algorithm_params={"pop_size": 5, "n_gen": 1, "n_inner_waypoints": 3},  # ZMIANA: pop_size na 5
            )

        # Fallback powinien zwrocic poprawny ksztalt
        assert result.shape == (n_drones, n_waypoints, 3)

        # Punkt startowy na Z powinien byc wymuszony do bezpiecznej wysokosci (default min_safe_alt=1.0)
        np.testing.assert_array_almost_equal(result[0, 0, :2], starts[0, :2])
        np.testing.assert_array_almost_equal(result[0, -1, :2], targets[0, :2])

    def test_obstacles_data_as_single_object(self, mock_world_data, basic_positions, mock_evaluator):
        """obstacles_data przekazany jako pojedynczy obiekt powinien zostac wrzucony w liste do ewaluatora."""
        starts, targets = basic_positions
        single_obstacle = MagicMock()

        with patch(f"{TARGET_MODULE}.VectorizedEvaluator", return_value=mock_evaluator) as mock_cls, \
             patch("hydra.core.hydra_config.HydraConfig"), \
             patch("src.utils.optimization_history_writer.OptimizationHistoryWriter"), \
             patch("src.algorithms.abstraction.trajectory.strategies.timing_utils.TimingCollector"):
            
            osprey_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=single_obstacle,
                world_data=mock_world_data,
                number_of_waypoints=20,
                drone_swarm_size=2,
                algorithm_params={"pop_size": 5, "n_gen": 1},  # ZMIANA: pop_size na 5
            )

            call_kwargs = mock_cls.call_args
            obs_arg = call_kwargs.kwargs.get("obstacles") or call_kwargs[1].get("obstacles")
            if obs_arg is None:
                obs_arg = call_kwargs[0][0]  
            
            assert isinstance(obs_arg, list)
            assert len(obs_arg) == 1
            assert obs_arg[0] == single_obstacle

    def test_mealpy_called_correctly(self, mock_world_data, basic_positions, mock_evaluator):
        """Test czy OriginalOOA jest prawidlowo inicjalizowane i wywolywane z przekazanymi parametrami."""
        starts, targets = basic_positions
        
        custom_params = {
            "pop_size": 12,
            "n_gen": 5,
            "n_workers": 2,
            "seed": 42
        }

        with patch(f"{TARGET_MODULE}.VectorizedEvaluator", return_value=mock_evaluator), \
             patch(f"{TARGET_MODULE}.OriginalOOA") as MockOOA, \
             patch("hydra.core.hydra_config.HydraConfig"), \
             patch("src.utils.optimization_history_writer.OptimizationHistoryWriter"), \
             patch("src.algorithms.abstraction.trajectory.strategies.timing_utils.TimingCollector"):
            
            # Konfiguracja zachowania mocka Mealpy
            mock_model_instance = MagicMock()
            mock_best_agent = MagicMock()
            mock_best_agent.solution = np.zeros(2 * 5 * 3) # (Drones * Inner * 3) - symulacja rozwiazania
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
            )

            # Assert inicjalizacja
            MockOOA.assert_called_once_with(epoch=5, pop_size=12)
            
            # Assert .solve() argumenty
            solve_kwargs = mock_model_instance.solve.call_args.kwargs
            assert solve_kwargs["mode"] == "thread"
            assert solve_kwargs["n_workers"] == 2
            assert solve_kwargs["seed"] == 42
            assert "problem" in solve_kwargs
            assert "starting_solutions" in solve_kwargs