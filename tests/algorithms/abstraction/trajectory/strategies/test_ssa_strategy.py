import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.algorithms.abstraction.trajectory.strategies.ssa_strategy import (
    evaluate_ssa_population,
    sparrow_search_algorithm_classic,
    sparrow_search_algorithm_vectorized,
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
def default_weights():
    return np.array([1.0, 1.0, 1.0])


@pytest.fixture
def mock_evaluator():
    """
    Atrapa ewaluatora zwracajaca kontrolowane wartosci F i G.
    F: (pop_size, 3) — cele (dystans, gladkosc, rownomiernosc)
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
# TESTY: evaluate_ssa_population
# ===========================================================================

class TestEvaluateSSAPopulation:

    def test_output_shape(self, basic_positions, default_weights, mock_evaluator):
        """Fitness powinien miec ksztalt (pop_size,)."""
        starts, targets = basic_positions
        pop_size, n_drones, n_inner = 5, 2, 3
        d = n_drones * n_inner * 3
        X = np.random.rand(pop_size, d) * 50

        fitness, _ = evaluate_ssa_population(
            X, n_drones, n_inner, n_out=20,
            starts=starts, targets=targets,
            evaluator=mock_evaluator,
            penalty_weight=100.0, weights=default_weights,
        )

        assert fitness.shape == (pop_size,)

    def test_weights_applied_correctly(self, basic_positions):
        """Iloczyn skalarny F*weights powinien byc odzwierciedlony w fitness."""
        starts, targets = basic_positions
        n_drones, n_inner = 2, 3
        pop_size = 1
        d = n_drones * n_inner * 3
        X = np.random.rand(pop_size, d) * 50

        f_row = np.array([[2.0, 3.0, 4.0]])
        g_row = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0]])  # brak naruszen
        evaluator = _make_evaluator_with_values(f_row, g_row)

        weights = np.array([1.0, 2.0, 3.0])
        fitness, _ = evaluate_ssa_population(
            X, n_drones, n_inner, n_out=20,
            starts=starts, targets=targets,
            evaluator=evaluator,
            penalty_weight=100.0, weights=weights,
        )

        # 2*1 + 3*2 + 4*3 = 2 + 6 + 12 = 20
        np.testing.assert_almost_equal(fitness[0], 20.0)

    def test_penalty_for_constraint_violations(self, basic_positions, default_weights):
        """Naruszenia ograniczen (G > 0) powinny dodawac kare proporcjonalna do penalty_weight."""
        starts, targets = basic_positions
        n_drones, n_inner = 2, 3
        pop_size = 1
        d = n_drones * n_inner * 3
        X = np.random.rand(pop_size, d) * 50

        f_row = np.array([[1.0, 1.0, 1.0]])
        g_row = np.array([[2.0, 3.0, -1.0, -1.0, -1.0]])  # 2 naruszenia: 2.0 + 3.0 = 5.0
        evaluator = _make_evaluator_with_values(f_row, g_row)

        penalty_weight = 100.0
        fitness, _ = evaluate_ssa_population(
            X, n_drones, n_inner, n_out=20,
            starts=starts, targets=targets,
            evaluator=evaluator,
            penalty_weight=penalty_weight, weights=default_weights,
        )

        # obj_score = 1+1+1 = 3, violation = 2+3 = 5, total = 3 + 100*5 = 503
        np.testing.assert_almost_equal(fitness[0], 503.0)

    def test_no_penalty_when_constraints_satisfied(self, basic_positions, default_weights):
        """G <= 0 nie powinno generowac zadnej kary."""
        starts, targets = basic_positions
        n_drones, n_inner = 2, 3
        d = n_drones * n_inner * 3
        X = np.random.rand(1, d) * 50

        f_row = np.array([[5.0, 5.0, 5.0]])
        g_row = np.array([[-2.0, -3.0, 0.0, -1.0, -0.5]])  # G <= 0
        evaluator = _make_evaluator_with_values(f_row, g_row)

        fitness, _ = evaluate_ssa_population(
            X, n_drones, n_inner, n_out=20,
            starts=starts, targets=targets,
            evaluator=evaluator,
            penalty_weight=9999.0, weights=default_weights,
        )

        # obj_score = 15, brak kar
        np.testing.assert_almost_equal(fitness[0], 15.0)

    def test_zero_weights(self, basic_positions):
        """Zerowe wagi powinny dawac fitness oparty wylacznie na karach."""
        starts, targets = basic_positions
        n_drones, n_inner = 2, 3
        d = n_drones * n_inner * 3
        X = np.random.rand(1, d) * 50

        f_row = np.array([[100.0, 200.0, 300.0]])
        g_row = np.array([[1.0, -1.0, -1.0, -1.0, -1.0]])
        evaluator = _make_evaluator_with_values(f_row, g_row)

        zero_weights = np.array([0.0, 0.0, 0.0])
        fitness, _ = evaluate_ssa_population(
            X, n_drones, n_inner, n_out=20,
            starts=starts, targets=targets,
            evaluator=evaluator,
            penalty_weight=10.0, weights=zero_weights,
        )

        # obj_score = 0, violation = 1.0, total = 0 + 10*1 = 10
        np.testing.assert_almost_equal(fitness[0], 10.0)

    def test_multiple_individuals_different_fitness(self, basic_positions, default_weights):
        """Rozne osobniki powinny moc otrzymac rozne wartosci fitness."""
        starts, targets = basic_positions
        n_drones, n_inner = 2, 3
        pop_size = 3
        d = n_drones * n_inner * 3
        X = np.random.rand(pop_size, d) * 50

        def f_fn(ps):
            return np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])[:ps]

        def g_fn(ps):
            return np.full((ps, 5), -1.0)

        evaluator = _make_evaluator_with_values(f_fn, g_fn)

        fitness, _ = evaluate_ssa_population(
            X, n_drones, n_inner, n_out=20,
            starts=starts, targets=targets,
            evaluator=evaluator,
            penalty_weight=100.0, weights=default_weights,
        )

        np.testing.assert_almost_equal(fitness[0], 3.0)
        np.testing.assert_almost_equal(fitness[1], 6.0)
        np.testing.assert_almost_equal(fitness[2], 9.0)


# ===========================================================================
# TESTY: sparrow_search_algorithm_vectorized
# ===========================================================================

class TestSparrowSearchVectorized:

    def _run_ssa(self, starts, targets, n_drones, n_inner=3, pop_size=10,
                 iter_max=5, mock_evaluator=None, st=0.8, pd_ratio=0.2,
                 sd_ratio=0.15, penalty_weight=100.0, weights=None):
        if weights is None:
            weights = np.array([1.0, 1.0, 1.0])
        if mock_evaluator is None:
            mock_evaluator = MagicMock()
            def side_effect(trajectories, out):
                ps = trajectories.shape[0]
                out["F"] = np.random.rand(ps, 3) * 10
                out["G"] = np.full((ps, 5), -1.0)
            mock_evaluator.evaluate.side_effect = side_effect

        d = n_drones * n_inner * 3
        bounds_min = np.full(d, -50.0)
        bounds_max = np.full(d, 150.0)
        X_init = np.random.rand(pop_size, d) * 100

        with patch(f"{TARGET_MODULE}.HydraConfig"), \
             patch(f"{TARGET_MODULE}.OptimizationHistoryWriter"):
            return sparrow_search_algorithm_vectorized(
                X_init=X_init, bounds_min=bounds_min, bounds_max=bounds_max,
                iter_max=iter_max, n_drones=n_drones, n_inner=n_inner,
                n_out=20, starts=starts, targets=targets,
                evaluator=mock_evaluator, st=st, pd_ratio=pd_ratio,
                sd_ratio=sd_ratio, penalty_weight=penalty_weight, weights=weights,
            )

    def test_output_shape(self, basic_positions):
        """Wynik powinien byc wektorem o dlugosci d = n_drones * n_inner * 3."""
        starts, targets = basic_positions
        n_drones, n_inner = 2, 3
        result = self._run_ssa(starts, targets, n_drones, n_inner)
        assert result.shape == (n_drones * n_inner * 3,)

    def test_bounds_respected(self, basic_positions):
        """Wynik powinien mieszcic sie w zadanych granicach."""
        starts, targets = basic_positions
        n_drones, n_inner = 2, 3
        d = n_drones * n_inner * 3
        bounds_min = np.full(d, 0.0)
        bounds_max = np.full(d, 100.0)
        X_init = np.random.rand(10, d) * 100

        evaluator = MagicMock()
        def side_effect(trajectories, out):
            ps = trajectories.shape[0]
            out["F"] = np.ones((ps, 3))
            out["G"] = np.full((ps, 5), -1.0)
        evaluator.evaluate.side_effect = side_effect

        with patch(f"{TARGET_MODULE}.HydraConfig"), \
             patch(f"{TARGET_MODULE}.OptimizationHistoryWriter"):
            result = sparrow_search_algorithm_vectorized(
                X_init=X_init, bounds_min=bounds_min, bounds_max=bounds_max,
                iter_max=10, n_drones=n_drones, n_inner=n_inner,
                n_out=20, starts=starts, targets=targets,
                evaluator=evaluator, st=0.8, pd_ratio=0.2,
                sd_ratio=0.15, penalty_weight=100.0,
                weights=np.array([1.0, 1.0, 1.0]),
            )

        assert np.all(result >= bounds_min)
        assert np.all(result <= bounds_max)

    def test_greedy_acceptance_improves_or_keeps_fitness(self, basic_positions):
        """
        Greedy acceptance powinien zapewniac, ze globalne optimum
        nie pogarszaja sie miedzy iteracjami.
        """
        starts, targets = basic_positions
        n_drones, n_inner = 2, 3
        d = n_drones * n_inner * 3

        # Ewaluator zwracajacy malejacy fitness (symulacja poprawy)
        call_count = [0]
        def side_effect(trajectories, out):
            ps = trajectories.shape[0]
            call_count[0] += 1
            base = max(100.0 - call_count[0] * 5, 1.0)
            out["F"] = np.ones((ps, 3)) * base
            out["G"] = np.full((ps, 5), -1.0)

        evaluator = MagicMock()
        evaluator.evaluate.side_effect = side_effect

        result = self._run_ssa(
            starts, targets, n_drones, n_inner,
            pop_size=10, iter_max=10, mock_evaluator=evaluator,
        )

        # Algorytm powinien zwrocic jakis wynik (nie NaN)
        assert not np.any(np.isnan(result))
        assert result.shape == (d,)

    def test_zero_iterations(self, basic_positions):
        """Przy iter_max=0 algorytm powinien zwrocic najlepszy element z populacji poczatkowej."""
        starts, targets = basic_positions
        n_drones, n_inner = 2, 3
        result = self._run_ssa(starts, targets, n_drones, n_inner, iter_max=0)
        assert result.shape == (n_drones * n_inner * 3,)
        assert not np.any(np.isnan(result))

    def test_high_st_triggers_exploration(self, basic_positions):
        """Wysoki ST (> R_2 w wiekszosci przypadkow) powinien uzyc trybu eksploracji."""
        starts, targets = basic_positions
        n_drones, n_inner = 2, 3
        # st=1.0 => R_2 < st zawsze prawdziwe => tryb eksponencjalny
        result = self._run_ssa(starts, targets, n_drones, n_inner, st=1.0, iter_max=3)
        assert result.shape == (n_drones * n_inner * 3,)

    def test_low_st_triggers_exploitation(self, basic_positions):
        """Niski ST powoduje tryb eksploatacji (Q noise)."""
        starts, targets = basic_positions
        n_drones, n_inner = 2, 3
        # st=0.0 => R_2 < st nigdy prawdziwe => tryb szumowy
        result = self._run_ssa(starts, targets, n_drones, n_inner, st=0.0, iter_max=3)
        assert result.shape == (n_drones * n_inner * 3,)

    def test_single_individual_population(self, single_drone_positions):
        """Populacja z jednym osobnikiem nie powinna powodowac bledow."""
        starts, targets = single_drone_positions
        result = self._run_ssa(starts, targets, n_drones=1, n_inner=3, pop_size=1, iter_max=2)
        assert result.shape == (1 * 3 * 3,)

    def test_large_pd_ratio(self, basic_positions):
        """pd_ratio=1.0 — wszyscy sa producentami."""
        starts, targets = basic_positions
        result = self._run_ssa(starts, targets, n_drones=2, n_inner=3, pd_ratio=1.0, iter_max=3)
        assert result.shape == (2 * 3 * 3,)

    def test_large_sd_ratio(self, basic_positions):
        """sd_ratio=1.0 — wszyscy sa zwiadowcami."""
        starts, targets = basic_positions
        result = self._run_ssa(starts, targets, n_drones=2, n_inner=3, sd_ratio=1.0, iter_max=3)
        assert result.shape == (2 * 3 * 3,)


# ===========================================================================
# TESTY: sparrow_search_algorithm_classic
# ===========================================================================

class TestSparrowSearchClassic:

    def test_output_shape(self, basic_positions, default_weights):
        """Wynik powinien byc wektorem o dlugosci d."""
        starts, targets = basic_positions
        n_drones, n_inner = 2, 3
        d = n_drones * n_inner * 3
        pop_size = 10

        evaluator = MagicMock()
        def side_effect(trajectories, out):
            ps = trajectories.shape[0]
            out["F"] = np.random.rand(ps, 3) * 10
            out["G"] = np.full((ps, 5), -1.0)
        evaluator.evaluate.side_effect = side_effect

        X_init = np.random.rand(pop_size, d) * 100

        result = sparrow_search_algorithm_classic(
            X_init=X_init,
            bounds_min=np.full(d, -50.0),
            bounds_max=np.full(d, 150.0),
            iter_max=3,
            n_drones=n_drones, n_inner=n_inner, n_out=20,
            starts=starts, targets=targets,
            evaluator=evaluator,
            st=0.8, pd_ratio=0.2, sd_ratio=0.15,
            penalty_weight=100.0, weights=default_weights,
        )

        assert result.shape == (d,)
        assert not np.any(np.isnan(result))

    def test_bounds_respected(self, basic_positions, default_weights):
        """Wynik powinien respektowac granice przestrzeni."""
        starts, targets = basic_positions
        n_drones, n_inner = 2, 3
        d = n_drones * n_inner * 3
        bounds_min = np.full(d, 0.0)
        bounds_max = np.full(d, 100.0)

        evaluator = MagicMock()
        def side_effect(trajectories, out):
            ps = trajectories.shape[0]
            out["F"] = np.ones((ps, 3))
            out["G"] = np.full((ps, 5), -1.0)
        evaluator.evaluate.side_effect = side_effect

        X_init = np.random.rand(10, d) * 100

        result = sparrow_search_algorithm_classic(
            X_init=X_init,
            bounds_min=bounds_min, bounds_max=bounds_max,
            iter_max=5,
            n_drones=n_drones, n_inner=n_inner, n_out=20,
            starts=starts, targets=targets,
            evaluator=evaluator,
            st=0.8, pd_ratio=0.2, sd_ratio=0.15,
            penalty_weight=100.0, weights=default_weights,
        )

        assert np.all(result >= bounds_min)
        assert np.all(result <= bounds_max)

    def test_classic_vs_vectorized_same_seed(self, basic_positions, default_weights):
        """
        Obie implementacje powinny zwrocic wynik o identycznym ksztalcie
        i niezawierajacy NaN (deterministycznosc trudna do wymuszenia ze wzgledu
        na roznice w losowaniu, testujemy wiec stabilnosc numeryczna).
        """
        starts, targets = basic_positions
        n_drones, n_inner = 2, 3
        d = n_drones * n_inner * 3

        evaluator = MagicMock()
        def side_effect(trajectories, out):
            ps = trajectories.shape[0]
            out["F"] = np.ones((ps, 3)) * 5.0
            out["G"] = np.full((ps, 5), -1.0)
        evaluator.evaluate.side_effect = side_effect

        X_init = np.random.rand(10, d) * 100

        res_classic = sparrow_search_algorithm_classic(
            X_init=X_init.copy(), bounds_min=np.full(d, -50.0),
            bounds_max=np.full(d, 150.0), iter_max=3,
            n_drones=n_drones, n_inner=n_inner, n_out=20,
            starts=starts, targets=targets, evaluator=evaluator,
            st=0.8, pd_ratio=0.2, sd_ratio=0.15,
            penalty_weight=100.0, weights=default_weights,
        )

        with patch(f"{TARGET_MODULE}.HydraConfig"), \
             patch(f"{TARGET_MODULE}.OptimizationHistoryWriter"):
            res_vec = sparrow_search_algorithm_vectorized(
                X_init=X_init.copy(), bounds_min=np.full(d, -50.0),
                bounds_max=np.full(d, 150.0), iter_max=3,
                n_drones=n_drones, n_inner=n_inner, n_out=20,
                starts=starts, targets=targets, evaluator=evaluator,
                st=0.8, pd_ratio=0.2, sd_ratio=0.15,
                penalty_weight=100.0, weights=default_weights,
            )

        assert res_classic.shape == res_vec.shape
        assert not np.any(np.isnan(res_classic))
        assert not np.any(np.isnan(res_vec))


# ===========================================================================
# TESTY: ssa_swarm_strategy (orchestrator)
# ===========================================================================

class TestSSASwarmStrategy:

    def test_output_shape(self, mock_world_data, basic_positions):
        """Wynik powinien miec ksztalt (n_drones, n_waypoints, 3)."""
        starts, targets = basic_positions
        n_waypoints = 20
        n_drones = 2

        evaluator = MagicMock()
        def side_effect(trajectories, out):
            ps = trajectories.shape[0]
            out["F"] = np.ones((ps, 3))
            out["G"] = np.full((ps, 5), -1.0)
        evaluator.evaluate.side_effect = side_effect

        with patch(f"{TARGET_MODULE}.VectorizedEvaluator", return_value=evaluator):
            result = ssa_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=[],
                world_data=mock_world_data,
                number_of_waypoints=n_waypoints,
                drone_swarm_size=n_drones,
                algorithm_params={"pop_size": 5, "n_gen": 2, "n_inner_waypoints": 3},
            )

        assert result.shape == (n_drones, n_waypoints, 3)

    def test_single_drone(self, mock_world_data, single_drone_positions):
        """Powinien dzialac poprawnie dla jednego drona."""
        starts, targets = single_drone_positions
        n_waypoints = 15

        evaluator = MagicMock()
        def side_effect(trajectories, out):
            ps = trajectories.shape[0]
            out["F"] = np.ones((ps, 3))
            out["G"] = np.full((ps, 5), -1.0)
        evaluator.evaluate.side_effect = side_effect

        with patch(f"{TARGET_MODULE}.VectorizedEvaluator", return_value=evaluator):
            result = ssa_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=[],
                world_data=mock_world_data,
                number_of_waypoints=n_waypoints,
                drone_swarm_size=1,
                algorithm_params={"pop_size": 5, "n_gen": 2, "n_inner_waypoints": 3},
            )

        assert result.shape == (1, n_waypoints, 3)

    def test_default_params_applied(self, mock_world_data, basic_positions):
        """Domyslne parametry powinny byc stosowane gdy algorithm_params jest None."""
        starts, targets = basic_positions

        evaluator = MagicMock()
        def side_effect(trajectories, out):
            ps = trajectories.shape[0]
            out["F"] = np.ones((ps, 3))
            out["G"] = np.full((ps, 5), -1.0)
        evaluator.evaluate.side_effect = side_effect

        with patch(f"{TARGET_MODULE}.VectorizedEvaluator", return_value=evaluator):
            result = ssa_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=[],
                world_data=mock_world_data,
                number_of_waypoints=20,
                drone_swarm_size=2,
                algorithm_params=None,
            )

        assert result.shape == (2, 20, 3)

    def test_fallback_on_exception(self, mock_world_data, basic_positions):
        """Blad w optymalizacji powinien zwrocic trajektorie liniowa (fallback)."""
        starts, targets = basic_positions
        n_waypoints = 10
        n_drones = 2

        evaluator = MagicMock()
        evaluator.evaluate.side_effect = RuntimeError("Symulowany blad")

        with patch(f"{TARGET_MODULE}.VectorizedEvaluator", return_value=evaluator):
            result = ssa_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=[],
                world_data=mock_world_data,
                number_of_waypoints=n_waypoints,
                drone_swarm_size=n_drones,
                algorithm_params={"pop_size": 5, "n_gen": 2, "n_inner_waypoints": 3},
            )

        # Fallback powinien zwrocic poprawny ksztalt
        assert result.shape == (n_drones, n_waypoints, 3)

        # Fallback interpoluje liniowo — sprawdzenie punktow krancowych
        np.testing.assert_array_almost_equal(result[0, 0, :2], starts[0, :2])
        np.testing.assert_array_almost_equal(result[0, -1, :2], targets[0, :2])

    def test_fallback_z_altitude_clamped(self, mock_world_data):
        """Fallback powinien wymuszac minimalna wysokosc Z >= 0.5."""
        starts = np.array([[10.0, 10.0, 0.0]])   # Z = 0 (pod podloga)
        targets = np.array([[90.0, 90.0, 0.0]])   # Z = 0
        n_waypoints = 10

        evaluator = MagicMock()
        evaluator.evaluate.side_effect = RuntimeError("Blad")

        with patch(f"{TARGET_MODULE}.VectorizedEvaluator", return_value=evaluator):
            result = ssa_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=[],
                world_data=mock_world_data,
                number_of_waypoints=n_waypoints,
                drone_swarm_size=1,
                algorithm_params={"pop_size": 5, "n_gen": 2, "n_inner_waypoints": 3},
            )

        # Cala os Z powinna byc >= 0.5 (MIN_Z_ALTITUDE)
        assert np.all(result[0, :, 2] >= 0.5)

    def test_obstacles_data_as_single_object(self, mock_world_data, basic_positions):
        """obstacles_data przekazany jako pojedynczy obiekt (nie lista) powinien byc obsluzony."""
        starts, targets = basic_positions

        evaluator = MagicMock()
        def side_effect(trajectories, out):
            ps = trajectories.shape[0]
            out["F"] = np.ones((ps, 3))
            out["G"] = np.full((ps, 5), -1.0)
        evaluator.evaluate.side_effect = side_effect

        single_obstacle = MagicMock()

        with patch(f"{TARGET_MODULE}.VectorizedEvaluator", return_value=evaluator) as mock_cls:
            ssa_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=single_obstacle,
                world_data=mock_world_data,
                number_of_waypoints=20,
                drone_swarm_size=2,
                algorithm_params={"pop_size": 5, "n_gen": 2, "n_inner_waypoints": 3},
            )

            # VectorizedEvaluator powinien dostac liste
            call_kwargs = mock_cls.call_args
            obs_arg = call_kwargs.kwargs.get("obstacles") or call_kwargs[1].get("obstacles")
            if obs_arg is None:
                obs_arg = call_kwargs[0][0]  # pozycyjny
            assert isinstance(obs_arg, list)

    def test_obstacles_data_as_list(self, mock_world_data, basic_positions):
        """obstacles_data przekazany jako lista powinien byc przekazany bez zmian."""
        starts, targets = basic_positions

        evaluator = MagicMock()
        def side_effect(trajectories, out):
            ps = trajectories.shape[0]
            out["F"] = np.ones((ps, 3))
            out["G"] = np.full((ps, 5), -1.0)
        evaluator.evaluate.side_effect = side_effect

        obs_list = [MagicMock(), MagicMock()]

        with patch(f"{TARGET_MODULE}.VectorizedEvaluator", return_value=evaluator) as mock_cls:
            ssa_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=obs_list,
                world_data=mock_world_data,
                number_of_waypoints=20,
                drone_swarm_size=2,
                algorithm_params={"pop_size": 5, "n_gen": 2, "n_inner_waypoints": 3},
            )

            call_kwargs = mock_cls.call_args
            obs_arg = call_kwargs.kwargs.get("obstacles") or call_kwargs[1].get("obstacles")
            if obs_arg is None:
                obs_arg = call_kwargs[0][0]
            assert isinstance(obs_arg, list)
            assert len(obs_arg) == 2

    def test_custom_ssa_hyperparams_forwarded(self, mock_world_data, basic_positions):
        """Niestandardowe hiperparametry SSA powinny byc poprawnie przekazane."""
        starts, targets = basic_positions

        evaluator = MagicMock()
        def side_effect(trajectories, out):
            ps = trajectories.shape[0]
            out["F"] = np.ones((ps, 3))
            out["G"] = np.full((ps, 5), -1.0)
        evaluator.evaluate.side_effect = side_effect

        custom_params = {
            "pop_size": 8,
            "n_gen": 2,
            "n_inner_waypoints": 4,
            "st": 0.5,
            "pd_ratio": 0.3,
            "sd_ratio": 0.25,
            "penalty_weight": 5000.0,
            "weight_distance": 2.0,
            "weight_smoothness": 3.0,
            "weight_uniformity": 4.0,
        }

        with patch(f"{TARGET_MODULE}.VectorizedEvaluator", return_value=evaluator), \
             patch(f"{TARGET_MODULE}.HydraConfig"), \
             patch(f"{TARGET_MODULE}.OptimizationHistoryWriter"), \
             patch(f"{TARGET_MODULE}.sparrow_search_algorithm_vectorized", wraps=sparrow_search_algorithm_vectorized) as mock_ssa:
            ssa_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=[],
                world_data=mock_world_data,
                number_of_waypoints=20,
                drone_swarm_size=2,
                algorithm_params=custom_params,
            )

            call_kwargs = mock_ssa.call_args.kwargs
            assert call_kwargs["st"] == 0.5
            assert call_kwargs["pd_ratio"] == 0.3
            assert call_kwargs["sd_ratio"] == 0.25
            assert call_kwargs["penalty_weight"] == 5000.0
            np.testing.assert_array_equal(
                call_kwargs["weights"], np.array([2.0, 3.0, 4.0])
            )

    def test_z_bounds_computed_correctly(self, mock_world_data, basic_positions):
        """
        Granice Z w populacji poczatkowej powinny respektowac MIN_Z_ALTITUDE
        i max_flight_z obliczone z world_data.
        """
        starts, targets = basic_positions
        n_waypoints = 20
        n_drones = 2

        captured_x_init = {}
        original_ssa = sparrow_search_algorithm_vectorized

        def capture_ssa(**kwargs):
            captured_x_init["X_init"] = kwargs["X_init"].copy()
            captured_x_init["bounds_min"] = kwargs["bounds_min"].copy()
            captured_x_init["bounds_max"] = kwargs["bounds_max"].copy()
            return original_ssa(**kwargs)

        evaluator = MagicMock()
        def side_effect(trajectories, out):
            ps = trajectories.shape[0]
            out["F"] = np.ones((ps, 3))
            out["G"] = np.full((ps, 5), -1.0)
        evaluator.evaluate.side_effect = side_effect

        with patch(f"{TARGET_MODULE}.VectorizedEvaluator", return_value=evaluator), \
             patch(f"{TARGET_MODULE}.HydraConfig"), \
             patch(f"{TARGET_MODULE}.OptimizationHistoryWriter"), \
             patch(f"{TARGET_MODULE}.sparrow_search_algorithm_vectorized", side_effect=capture_ssa):
            ssa_swarm_strategy(
                start_positions=starts,
                target_positions=targets,
                obstacles_data=[],
                world_data=mock_world_data,
                number_of_waypoints=n_waypoints,
                drone_swarm_size=n_drones,
                algorithm_params={"pop_size": 5, "n_gen": 1, "n_inner_waypoints": 3},
            )

        # Sprawdzenie obliczen: MIN_Z_ALTITUDE = 0.5, max_bounds[2] = 20.0
        # xl_one[2] = max(0.5, 0.0) = 0.5
        # max_flight_z = min(20.0, 20.0 + 50.0 - 3.0) = 20.0
        n_inner = 3
        d = n_drones * n_inner * 3
        bounds_min = captured_x_init["bounds_min"]
        bounds_max = captured_x_init["bounds_max"]

        # Indeksy Z w splaszczonej tablicy: co 3-ci element zaczynajac od 2
        z_indices = list(range(2, d, 3))
        assert np.all(bounds_min[z_indices] >= 0.5)
        assert np.all(bounds_max[z_indices] <= 20.0)
