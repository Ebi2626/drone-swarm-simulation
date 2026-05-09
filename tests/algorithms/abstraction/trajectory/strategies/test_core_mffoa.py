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
        "rng": np.random.default_rng(12345),
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
    """Ten sam seed (przekazany jako instancja generatora rng) → ten sam wynik finalny (deterministyczność)."""
    base_params["max_generations"] = 10
    
    # Skopiowanie parametrów, by nie modyfikować globalnego stanu dla innych testów
    params_a = base_params.copy()
    params_b = base_params.copy()
    
    # Nadpisanie domyślnego rng zdefiniowanego w base_params
    params_a["rng"] = np.random.default_rng(42)
    params_b["rng"] = np.random.default_rng(42)

    _, fit_a = MSFFOAOptimizer(**params_a, fitness_function=dummy_fitness).optimize()
    _, fit_b = MSFFOAOptimizer(**params_b, fitness_function=dummy_fitness).optimize()

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


# ---------------------------------------------------------------------------
# Logowanie per-gen — stan algorytmu (swarm_best_pos), nie offspring (new_pop)
# ---------------------------------------------------------------------------

class _CapturingHistoryWriter:
    """Zbiera wywołania put_generation_data dla testów logowania."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def put_generation_data(self, payload: dict) -> None:
        self.calls.append({k: (v.copy() if hasattr(v, "copy") else v)
                           for k, v in payload.items()})

    def close(self) -> None:
        pass


class _AdapterMock:
    """
    Imituje TrajectorySOOAdapter — wystawia last_objectives / last_constraints
    po każdym wywołaniu (jak Big-M adapter w soo_adapter.py).
    Wszystkie osobniki feasible (G=0) → feasible_ratio = 1.0 dla każdego liderów.
    """

    def __init__(self, n_obj: int = 3, n_g: int = 2) -> None:
        self.n_obj = n_obj
        self.n_g = n_g
        self.last_objectives: np.ndarray | None = None
        self.last_constraints: np.ndarray | None = None
        self.evaluator = None  # symuluje brak `individuals_evaluated`

    def __call__(self, pop: np.ndarray) -> np.ndarray:
        # pop shape: (N, n_drones, n_inner, 3) → fitness ~ Σ |x|
        flat = pop.reshape(pop.shape[0], -1)
        fit = np.linalg.norm(flat, axis=1)
        # Wielokolumnowe F/G żeby symulować realny przypadek (M, K > 1)
        self.last_objectives = np.column_stack(
            [fit] + [fit * (0.5 + 0.1 * k) for k in range(self.n_obj - 1)]
        )
        # Wszystkie feasible (G ≤ 0)
        self.last_constraints = np.full((pop.shape[0], self.n_g), -1.0)
        return fit


def test_history_writer_logs_swarm_leaders_not_offspring(base_params):
    """
    Po Phase 3 do history_writer wpisywany jest STAN algorytmu — G liderów
    (swarm_best_pos po elityzmie), NIE offspring (new_pop). Spójne z NSGA-III/
    OOA/SSA, które logują populację po selekcji.

    Eliminuje fałszywą degradację `feasible_ratio` raportowaną w
    exp_20260508 (offspring tuż za wąską niszą feasibility, mimo że liderzy
    sami są feasible).
    """
    base_params["max_generations"] = 4
    writer = _CapturingHistoryWriter()
    adapter = _AdapterMock(n_obj=3, n_g=2)

    opt = MSFFOAOptimizer(
        **base_params,
        fitness_function=adapter,
        history_writer=writer,
    )
    opt.optimize()

    G = opt.G
    M = adapter.n_obj
    n_var = base_params["n_drones"] * base_params["n_inner"] * 3

    # Po max_generations wywołań — jeden put_generation_data per generację
    assert len(writer.calls) == base_params["max_generations"]

    for call in writer.calls:
        # Kluczowy kontrakt: pop_size logowane = G (liderzy), nie pop_size optimizera
        assert call["objectives_matrix"].shape == (G, M)
        assert call["decisions_matrix"].shape == (G, n_var)
        assert call["feasible_mask"].shape == (G,)
        assert call["constraint_violation"].shape == (G,)
        # AdapterMock daje wszystkie osobniki feasible — feasible_ratio liderów = 1.0
        assert call["feasible_mask"].all()


def test_history_log_decisions_match_swarm_best_pos_after_optimize(base_params):
    """
    decisions_matrix[gen=last] z history_writer musi pokrywać się z
    final swarm_best_pos (po reshape) — gwarantuje, że logujemy faktyczny
    stan algorytmu, nie offspring/eksplorację.
    """
    base_params["max_generations"] = 3
    writer = _CapturingHistoryWriter()
    adapter = _AdapterMock(n_obj=2, n_g=1)

    opt = MSFFOAOptimizer(
        **base_params,
        fitness_function=adapter,
        history_writer=writer,
    )
    opt.optimize()

    last_decisions = writer.calls[-1]["decisions_matrix"]
    expected = opt.swarm_best_pos.reshape(opt.G, -1)
    np.testing.assert_allclose(last_decisions, expected, rtol=0, atol=0,
                               err_msg="Logowane decisions w ostatniej generacji "
                                        "muszą równać się swarm_best_pos (stan algo)")


def test_swarm_best_F_G_track_elitism(base_params):
    """
    swarm_best_F i swarm_best_G aktualizują się tym samym `update_mask` co
    swarm_best_pos. Po optymalizacji mają shape (G, M) / (G, K) i są spójne
    z liderami: F[g, 0] (kolumna fitness-proxy) ≈ swarm_best_fit[g].
    """
    base_params["max_generations"] = 5
    adapter = _AdapterMock(n_obj=2, n_g=1)

    opt = MSFFOAOptimizer(**base_params, fitness_function=adapter)
    opt.optimize()

    assert opt.swarm_best_F is not None
    assert opt.swarm_best_G is not None
    assert opt.swarm_best_F.shape == (opt.G, adapter.n_obj)
    assert opt.swarm_best_G.shape == (opt.G, adapter.n_g)
    # AdapterMock: pierwsza kolumna F = fitness, więc F[:, 0] ≈ swarm_best_fit
    np.testing.assert_allclose(opt.swarm_best_F[:, 0], opt.swarm_best_fit, rtol=1e-9)


def test_static_threshold_mode_does_not_modify_threshold(base_params):
    """
    Domyślnie `threshold_ratio is None` — paperowy tryb statyczny: wartość
    `threshold` przekazana w konstruktorze NIE zmienia się w trakcie optymalizacji.
    """
    base_params["max_generations"] = 5
    initial_threshold = 42.0

    opt = MSFFOAOptimizer(
        **base_params,
        fitness_function=dummy_fitness,
        threshold=initial_threshold,
    )
    opt.optimize()

    # Brak adaptacji → threshold pozostaje dokładnie 42.0
    assert opt.threshold == initial_threshold
    assert opt.threshold_ratio is None


def test_adaptive_threshold_uses_best_feasible_swarm_fit(base_params):
    """
    Tryb adaptacyjny: `threshold = best_feasible_fit × (1 + threshold_ratio)`.
    Dla feasible AdapterMock (G=0 dla wszystkich) próg po pełnym runie
    musi być spójny z aktualnym best feasible podrojem.
    """
    base_params["max_generations"] = 4
    adapter = _AdapterMock(n_obj=2, n_g=1)
    ratio = 0.18

    opt = MSFFOAOptimizer(
        **base_params,
        fitness_function=adapter,
        threshold_ratio=ratio,
    )
    opt.optimize()

    # Wszystkie liderzy feasible (mock daje G=−1 stale)
    feasible = opt.swarm_best_fit < opt._HARD_INFEASIBLE_BASE
    assert feasible.all()
    expected_threshold = float(np.min(opt.swarm_best_fit)) * (1.0 + ratio)
    assert opt.threshold == pytest.approx(expected_threshold, rel=1e-9)


def test_adaptive_threshold_preserves_initial_when_all_infeasible(base_params):
    """
    Gdy wszystkie liderzy są infeasible (≥ HARD_INFEASIBLE_BASE), threshold
    NIE jest resetowany — zachowuje się wartość początkowa z konstruktora
    (lub ostatnia kalibracja feasible). Funkcjonalnie równoważne resetowi
    do 0 (każde infeasible ~1e6 ≫ jakikolwiek sensowny threshold), ale
    daje spójny log między strategy a optimizer.
    """
    base_params["max_generations"] = 3
    initial_threshold = 0.26  # symuluje wartość z strategy (sum(|w|) * ratio)

    class InfeasibleAdapter:
        """AdapterMock zawsze zwracający fitness ≥ Big-M (infeasible)."""
        def __init__(self):
            self.last_objectives: np.ndarray | None = None
            self.last_constraints: np.ndarray | None = None
            self.evaluator = None

        def __call__(self, pop: np.ndarray) -> np.ndarray:
            n = pop.shape[0]
            # Każdy osobnik infeasible: fitness = 1e6 + scalar
            base_fit = np.linalg.norm(pop.reshape(n, -1), axis=1)
            fit = 1e6 + base_fit  # zawsze ≥ 1e6
            self.last_objectives = np.column_stack([fit, fit * 0.5])
            # G > 0 → infeasible (per konwencję per_gen_metrics_from_FG)
            self.last_constraints = np.full((n, 1), 5.0)
            return fit

    adapter = InfeasibleAdapter()
    opt = MSFFOAOptimizer(
        **base_params,
        fitness_function=adapter,
        threshold=initial_threshold,
        threshold_ratio=0.18,
    )
    opt.optimize()

    # Wszystkie liderzy infeasible → threshold zachowuje wartość initial,
    # nie zostaje zresetowany do 0. Behavior pozostaje all GLOBAL bo
    # każde infeasible (~1e6) >> initial_threshold (0.26).
    assert (opt.swarm_best_fit >= opt._HARD_INFEASIBLE_BASE).all()
    assert opt.threshold == pytest.approx(initial_threshold, rel=1e-12)


def test_adaptive_threshold_recalibrates_per_generation(base_params):
    """
    Threshold musi się aktualizować co generację. Sprawdzamy, że po
    optymalizacji threshold jest spójny z FINAL stanem swarm_best_fit
    (a nie z pre-init wartością `threshold` przekazaną w konstruktorze).
    """
    base_params["max_generations"] = 6
    adapter = _AdapterMock(n_obj=3, n_g=2)
    ratio = 0.5

    opt = MSFFOAOptimizer(
        **base_params,
        fitness_function=adapter,
        threshold=99999.0,  # wstępna wartość — powinna być nadpisana
        threshold_ratio=ratio,
    )
    opt.optimize()

    # Final threshold odzwierciedla bieżący best feasible, nie 99999
    assert opt.threshold < 99999.0
    expected = float(np.min(opt.swarm_best_fit)) * (1.0 + ratio)
    assert opt.threshold == pytest.approx(expected, rel=1e-9)


def test_logging_change_does_not_break_seed_reproducibility(base_params):
    """
    Regresja: zmiana logowania (snapshot F/G + reshape) NIE konsumuje rng,
    więc dwa runy z tym samym seedem dają identyczny final fitness — także
    z aktywnym history_writer.
    """
    base_params["max_generations"] = 8
    params_a = base_params.copy()
    params_b = base_params.copy()
    params_a["rng"] = np.random.default_rng(123)
    params_b["rng"] = np.random.default_rng(123)

    writer_a = _CapturingHistoryWriter()
    writer_b = _CapturingHistoryWriter()
    adapter_a = _AdapterMock()
    adapter_b = _AdapterMock()

    _, fit_a = MSFFOAOptimizer(**params_a, fitness_function=adapter_a,
                               history_writer=writer_a).optimize()
    _, fit_b = MSFFOAOptimizer(**params_b, fitness_function=adapter_b,
                               history_writer=writer_b).optimize()

    assert fit_a == pytest.approx(fit_b, rel=1e-12)
