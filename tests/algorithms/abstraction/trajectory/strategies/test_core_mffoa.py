"""
Testy MSFFOAOptimizer (paper-strict implementacja Shi et al. 2020).

Pokrywają:
- Inicjalizację podrojów (fallback wewnętrzny + ścieżka external initial_population)
- Konwergencję optymalizacji na łatwym landscapie (dummy_fitness)
- Format zwracanej trajektorii rzadkiej (start + inner + target)
- Walidacje wynikające z literatury:
  - coe1 + coe2 = 1 (Sec. 1)
  - n_swarms ≥ 1, pop_size ≥ n_swarms, pop_size % n_swarms == 0
  - kształt step_global_frac / step_local_frac == (3,)
  - step_*_frac wartości strictly positive
  - kształt initial_population musi pasować do (pop_size, n_drones, n_inner, 3)
- Reprodukcyjność dla tego samego seeda.
"""

import numpy as np
import pytest

from src.algorithms.abstraction.trajectory.strategies.core_msffoa import MSFFOAOptimizer


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def base_params():
    """
    Minimalna, ale wystarczająca konfiguracja dla testów MSFFOAOptimizer.
    pop_size=20, n_swarms (default)=5 → P=4 mucha/podrój.
    """
    return {
        "pop_size": 20,
        "n_drones": 3,
        "n_inner": 5,
        "world_min_bounds": np.array([0.0, 0.0, 0.0]),
        "world_max_bounds": np.array([100.0, 100.0, 50.0]),
        "start_positions": np.array([
            [10.0, 10.0, 1.0],
            [10.0, 15.0, 1.0],
            [10.0, 20.0, 1.0],
        ]),
        "target_positions": np.array([
            [90.0, 90.0, 5.0],
            [90.0, 95.0, 5.0],
            [90.0, 100.0, 5.0],
        ]),
        "max_generations": 15,
        "seed": 42,
    }


def dummy_fitness(trajectories: np.ndarray) -> np.ndarray:
    """
    Wypukła funkcja celu — odległość Euklidesowa od punktu [50, 50, 25].
    Akceptuje dowolny kształt (Pop, Drones, N_points, 3) i zwraca (Pop,).
    """
    ideal = np.array([50.0, 50.0, 25.0])
    distances = np.linalg.norm(trajectories - ideal, axis=-1)
    # Suma po wszystkich osiach poza pierwszą (Pop)
    return np.sum(distances, axis=tuple(range(1, distances.ndim)))


# ---------------------------------------------------------------------------
# Inicjalizacja podrojów
# ---------------------------------------------------------------------------

def test_internal_initialization_shapes_and_bounds(base_params):
    """
    Fallback wewnętrzny `_initialize_swarms`: liderzy generowani wokół
    zaszumionej linii prostej, klipowani do [_lb, _ub].
    """
    opt = MSFFOAOptimizer(**base_params, fitness_function=dummy_fitness)
    opt._initialize_swarms()

    expected = (opt.G, base_params["n_drones"], base_params["n_inner"], 3)
    assert opt.swarm_best_pos.shape == expected
    assert opt.swarm_best_fit.shape == (opt.G,)

    # Granice: wartości wewnątrz [_lb, _ub]
    assert np.all(opt.swarm_best_pos >= opt._lb), "Lider poniżej dolnej granicy"
    assert np.all(opt.swarm_best_pos <= opt._ub), "Lider powyżej górnej granicy"

    # Globalny best zainicjalizowany z istniejących liderów
    assert opt.global_best_pos.shape == (base_params["n_drones"], base_params["n_inner"], 3)
    assert np.isfinite(opt.global_best_fitness)
    assert opt.global_best_fitness == pytest.approx(np.min(opt.swarm_best_fit))


def test_external_initialization_picks_best_per_slice(base_params):
    """
    Gdy podana jest `initial_population`, `_initialize_swarms` dzieli ją
    na G plasterków po P osobników i wybiera lidera = najlepszy w plasterku
    (paper Sec. 1: G podrojów, każdy z lokalnym liderem).
    """
    rng = np.random.default_rng(0)
    init_pop = rng.uniform(
        low=5.0, high=45.0,
        size=(base_params["pop_size"], base_params["n_drones"], base_params["n_inner"], 3),
    )

    opt = MSFFOAOptimizer(
        **base_params,
        fitness_function=dummy_fitness,
        initial_population=init_pop,
    )
    opt._initialize_swarms()

    # Każdy z G liderów powinien mieć fitness = min ze swojego plasterka P
    P = opt.P
    for g in range(opt.G):
        slice_pop = init_pop[g * P : (g + 1) * P]
        slice_fit = dummy_fitness(slice_pop)
        expected_leader_fit = float(np.min(slice_fit))
        assert opt.swarm_best_fit[g] == pytest.approx(expected_leader_fit, rel=1e-9), (
            f"Lider podroju {g} powinien być najlepszym osobnikiem swojego plasterka"
        )


def test_initial_population_wrong_shape_raises(base_params):
    """`initial_population` o niezgodnym kształcie powinno rzucić ValueError."""
    bad_init = np.zeros((10, base_params["n_drones"], base_params["n_inner"], 3))
    with pytest.raises(ValueError, match="initial_population shape"):
        MSFFOAOptimizer(
            **base_params,
            fitness_function=dummy_fitness,
            initial_population=bad_init,
        )


# ---------------------------------------------------------------------------
# Trajektoria rzadka (start + inner + target)
# ---------------------------------------------------------------------------

def test_get_best_dense_trajectory_shape_and_endpoints(base_params):
    """
    `get_best_dense_trajectory` zwraca polilinię rzadką:
    (n_drones, n_inner + 2, 3) — z dokładnie zacumowanym startem i celem.
    Faktyczna densyfikacja B-Spline odbywa się w msffoa_strategy.py
    przez generate_bspline_batch.
    """
    opt = MSFFOAOptimizer(**base_params, fitness_function=dummy_fitness)
    opt._initialize_swarms()

    sparse = opt.get_best_dense_trajectory()
    n_drones, n_inner = base_params["n_drones"], base_params["n_inner"]
    assert sparse.shape == (n_drones, n_inner + 2, 3)

    np.testing.assert_allclose(
        sparse[:, 0, :], base_params["start_positions"],
        err_msg="Pierwszy waypoint musi być start_position",
    )
    np.testing.assert_allclose(
        sparse[:, -1, :], base_params["target_positions"],
        err_msg="Ostatni waypoint musi być target_position",
    )

    # Środkowe punkty powinny pokrywać się z global_best_pos
    np.testing.assert_allclose(sparse[:, 1:-1, :], opt.global_best_pos)


# ---------------------------------------------------------------------------
# Konwergencja
# ---------------------------------------------------------------------------

def test_optimize_returns_proper_shape_and_finite_fitness(base_params):
    """`optimize()` zwraca (best_pos, best_fitness) o oczekiwanym kształcie."""
    opt = MSFFOAOptimizer(**base_params, fitness_function=dummy_fitness)
    best_pos, final_fit = opt.optimize()

    assert best_pos.shape == (base_params["n_drones"], base_params["n_inner"], 3)
    assert np.isfinite(final_fit)
    assert final_fit < np.inf


def test_optimize_improves_over_initial_global_best(base_params):
    """
    Algorytm musi poprawiać global_best względem stanu po inicjalizacji.
    Używamy dwóch optymalizatorów z tym samym seedem:
    - opt_init: tylko `_initialize_swarms()` → punkt odniesienia
    - opt_full: pełne `optimize()` → wynik finalny
    """
    base_params["max_generations"] = 30

    opt_init = MSFFOAOptimizer(**base_params, fitness_function=dummy_fitness)
    opt_init._initialize_swarms()
    initial_global_best = opt_init.global_best_fitness

    opt_full = MSFFOAOptimizer(**base_params, fitness_function=dummy_fitness)
    _, final_fit = opt_full.optimize()

    # Z elityzmem (Eq. 18-19) global_best jest monotonicznie nierosnący
    assert final_fit <= initial_global_best, (
        f"Brak poprawy: init={initial_global_best:.2f}, final={final_fit:.2f}"
    )


def test_swarm_best_fit_is_monotonic_non_increasing(base_params):
    """
    Konsekwencja elityzmu (Sec. 4 Eq. 18-19): per-swarm `swarm_best_fit`
    nie powinien rosnąć między inicjalizacją a końcem optymalizacji.
    Tryb deterministyczny (ten sam seed) gwarantuje powtarzalne porównanie.
    """
    base_params["max_generations"] = 20

    opt_init = MSFFOAOptimizer(**base_params, fitness_function=dummy_fitness)
    opt_init._initialize_swarms()
    init_swarm_fit = opt_init.swarm_best_fit.copy()

    opt_full = MSFFOAOptimizer(**base_params, fitness_function=dummy_fitness)
    opt_full.optimize()
    final_swarm_fit = opt_full.swarm_best_fit

    assert np.all(final_swarm_fit <= init_swarm_fit + 1e-9), (
        f"Naruszenie elityzmu: init={init_swarm_fit}, final={final_swarm_fit}"
    )


def test_seed_reproducibility(base_params):
    """Ten sam seed → ten sam wynik finalny (deterministyczność)."""
    base_params["max_generations"] = 10

    _, fit_a = MSFFOAOptimizer(**base_params, fitness_function=dummy_fitness).optimize()
    _, fit_b = MSFFOAOptimizer(**base_params, fitness_function=dummy_fitness).optimize()

    assert fit_a == pytest.approx(fit_b, rel=1e-12), (
        f"Reprodukcyjność seeda zerwana: {fit_a} vs {fit_b}"
    )


# ---------------------------------------------------------------------------
# Walidacje paper-strict (constraints z literature/MSFFOA.md)
# ---------------------------------------------------------------------------

def test_validation_n_swarms_must_be_at_least_one(base_params):
    """Sec. 1 paperu: G ≥ 1."""
    with pytest.raises(ValueError, match="n_swarms"):
        MSFFOAOptimizer(**base_params, fitness_function=dummy_fitness, n_swarms=0)


def test_validation_pop_size_must_be_at_least_n_swarms(base_params):
    """Każdy podrój musi mieć co najmniej jednego osobnika (P = pop_size // G ≥ 1)."""
    base_params["pop_size"] = 3  # 3 < 5 (default n_swarms)
    with pytest.raises(ValueError, match="must be"):
        MSFFOAOptimizer(**base_params, fitness_function=dummy_fitness, n_swarms=5)


def test_validation_pop_size_must_be_divisible_by_n_swarms(base_params):
    """Sec. 1 paperu: M_pop dzielone na G równolicznych podrojów."""
    base_params["pop_size"] = 21  # 21 % 5 != 0
    with pytest.raises(ValueError, match="divisible"):
        MSFFOAOptimizer(**base_params, fitness_function=dummy_fitness, n_swarms=5)


def test_validation_coe_sum_must_equal_one(base_params):
    """Sec. 1 paperu wymaga coe1 + coe2 = 1 (Eq. 14 nie jest konwex inaczej)."""
    with pytest.raises(ValueError, match=r"coe1 \+ coe2 = 1"):
        MSFFOAOptimizer(
            **base_params,
            fitness_function=dummy_fitness,
            coe1=0.7,
            coe2=0.5,
        )


def test_validation_coe_sum_close_to_one_passes(base_params):
    """Tolerancja numeryczna 1e-6 wokół coe1 + coe2 = 1."""
    # coe1 + coe2 = 1.0 + 0.5e-7 → mieści się w tolerancji
    opt = MSFFOAOptimizer(
        **base_params,
        fitness_function=dummy_fitness,
        coe1=0.8 + 0.5e-7,
        coe2=0.2,
    )
    assert opt.coe1 == pytest.approx(0.8, abs=1e-6)


def test_validation_step_global_frac_wrong_shape(base_params):
    """`step_global_frac` musi mieć kształt (3,) — po jednej wartości na oś."""
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        MSFFOAOptimizer(
            **base_params,
            fitness_function=dummy_fitness,
            step_global_frac=np.array([0.01, 0.01]),  # tylko 2 wartości
        )


def test_validation_step_local_frac_wrong_shape(base_params):
    """`step_local_frac` musi mieć kształt (3,)."""
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        MSFFOAOptimizer(
            **base_params,
            fitness_function=dummy_fitness,
            step_local_frac=np.array([[0.003, 0.003, 0.001]]),  # zły wymiar
        )


def test_validation_step_global_frac_must_be_positive(base_params):
    """Niedodatnie wartości w step_global_frac muszą rzucać ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        MSFFOAOptimizer(
            **base_params,
            fitness_function=dummy_fitness,
            step_global_frac=np.array([0.01, -0.01, 0.005]),
        )


def test_validation_step_local_frac_zero_raises(base_params):
    """Zero w step_local_frac jest niedozwolone (równa zero amplituda kroku → brak ruchu)."""
    with pytest.raises(ValueError, match="strictly positive"):
        MSFFOAOptimizer(
            **base_params,
            fitness_function=dummy_fitness,
            step_local_frac=np.array([0.003, 0.0, 0.001]),
        )


# ---------------------------------------------------------------------------
# Konfiguracja step_*_frac propaguje się do amplitudy kroku
# ---------------------------------------------------------------------------

def test_step_fractions_propagate_to_step_arrays(base_params):
    """
    `step_global` i `step_local` to iloczyn world_size i frakcji.
    Przy world_size = [100, 100, 50] i frac=[0.02, 0.01, 0.005] otrzymujemy
    step = [2.0, 1.0, 0.25].
    """
    opt = MSFFOAOptimizer(
        **base_params,
        fitness_function=dummy_fitness,
        step_global_frac=np.array([0.02, 0.01, 0.005]),
        step_local_frac=np.array([0.005, 0.002, 0.001]),
    )
    np.testing.assert_allclose(opt.step_global, np.array([2.0, 1.0, 0.25]))
    np.testing.assert_allclose(opt.step_local, np.array([0.5, 0.2, 0.05]))


def test_default_step_fractions_when_not_provided(base_params):
    """Domyślne wartości step_*_frac (z docstringu): [0.010, 0.010, 0.005] / [0.003, 0.003, 0.001]."""
    opt = MSFFOAOptimizer(**base_params, fitness_function=dummy_fitness)
    # world_size = [100, 100, 50]
    np.testing.assert_allclose(opt.step_global, np.array([1.0, 1.0, 0.25]))
    np.testing.assert_allclose(opt.step_local, np.array([0.3, 0.3, 0.05]))
