"""
Multiple Swarm Fruit Fly Optimization Algorithm (MSFOA) for 3D Trajectory Optimization.

Implementacja na podstawie:
Shi, K., Zhang, X., & Xia, S. (2020). Multiple Swarm Fruit Fly Optimization
Algorithm Based Path Planning Method for Multi-UAVs. Applied Sciences, 10(8), 2822.

Mapowanie sekcji paperu → kod (literature/MSFFOA.md):

| Paper sec.  | Implementacja                                                  |
|-------------|----------------------------------------------------------------|
| Sec. 1      | Podział populacji na G podrojów + walidacja coe1+coe2=1        |
|             | (`MSFFOAOptimizer.__init__`).                                  |
| Sec. 2 Eq.7 | Faza globalna: x_best + sin(2·random(-1,1))                    |
|             | (`optimize` → `step_global_val`).                              |
| Sec. 2 Eq.8 | Faza lokalna: x_best + R · random(-1,1)                        |
|             | (`optimize` → `step_local_val`).                               |
| Sec. 3      | POMINIĘTE: paper transformuje (X,Y) ∈ smell-space na fizyczne  |
|             | współrzędne przez S = 1/√(X²+Y²). Adaptacja 3D-trajektorii:    |
|             | operujemy bezpośrednio na fizycznych punktach kontrolnych      |
|             | wielokąta B-Spline. Skalowanie kroku (`step_global/local`)     |
|             | rekompensuje brak smell-to-physical mappingu.                  |
| Sec. 4 Eq.14| Krzyżowanie międzyrojowe `coe1 · leader_random + coe2 · X_g,i` |
|             | (`optimize` → Phase 2: `new_pop`).                             |
| Sec. 4      | Zaktualizowano (Eq. 18-19): Wprowadzono elityzm dla liderów    |
|             | roju (środek roju ulega aktualizacji tylko gdy winner jest     |
|             | ściśle lepszy od dotychczasowego best).                        |
| Logging     | Per-gen history zapisuje swarm_best_pos (G liderów po Phase-3  |
|             | elitism), nie offspring new_pop. Spójne z NSGA-III/OOA/SSA,    |
|             | które logują populację po selekcji (current state), nie        |
|             | eksplorację. Eliminuje fałszywą degradację feasible_ratio      |
|             | przy konwergencji liderów w wąską niszę feasibility (offspring |
|             | tuż za progiem nie są stanem algorytmu, lecz perturbacją).     |

PRAGMATYCZNE ADAPTACJE vs paper (udokumentowane lokalnie w kodzie):
1. Anizotropowy R (per-osi step_local) zamiast skalarnego R z paperu —
   konieczne ze względu na asymetrię świata (np. forest: X=60m, Y=600m, Z=11m).
2. Skalowanie step_global/step_local — rekompensata za brak smell-space (Sec. 3).
3. `initial_population` jako opcjonalny argument konstruktora — dla warunku
   ceteris paribus z OOA/NSGA-III (StraightLineNoiseSampling + ten sam
   SwarmOptimizationProblem).
4. `threshold` skalibrowany dynamicznie z `initial_fitness * threshold_ratio`
   w `msffoa_strategy.py` — paper podaje threshold jako parametr stały.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, TYPE_CHECKING
from contextlib import nullcontext

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from src.algorithms.abstraction.trajectory.strategies.timing_utils import TimingCollector
    from src.utils.optimization_history_writer import OptimizationHistoryWriter


class MSFFOAOptimizer:
    """Multiple Swarm Fruit Fly Optimization Algorithm for 3D trajectory tensors.

    Args:
        pop_size: Total number of fruit flies (must be divisible by n_swarms).
        n_drones: Number of drones in the swarm.
        n_inner: Number of inner waypoints per drone.
        world_min_bounds: (3,) lower world bounds [x, y, z].
        world_max_bounds: (3,) upper world bounds [x, y, z].
        start_positions: (N_drones, 3) fixed start positions.
        target_positions: (N_drones, 3) fixed target positions.
        fitness_function: Callable evaluating a (Pop_size, D, I, 3) tensor.
        max_generations: Maximum number of optimization generations (NC).
        seed: Random seed for reproducibility.
        n_swarms: Number of sub-swarms (G). Default 5.
        coe1: Weight of the parent/leader path data (coe1 + coe2 = 1). Default 0.8.
        coe2: Weight of the current fly's path data. Default 0.2.
        threshold: Initial value of the threshold dividing global and local
            search phases. If ``threshold_ratio`` is None, this value stays
            STATIC throughout optimization (paper Sec. 1 — stały próg).
        threshold_ratio: If provided, włącza ADAPTACYJNY threshold: po każdej
            generacji próg jest przeliczany jako
            ``threshold = best_feasible_swarm_fit × (1 + threshold_ratio)``.
            Semantyka: podroje w obrębie ``threshold_ratio`` od najlepszego
            feasible podroju → LOCAL (refinement); pozostałe → GLOBAL
            (exploration). Mała wartość = ostre kryterium (mało LOCAL),
            duża = liberalne. Gdy żaden podrój nie jest jeszcze feasible,
            threshold=0 → wszystkie w GLOBAL (czysta eksploracja).
            Adaptacja jest pragmatyczną adaptacją do naszego Big-M
            (``HARD_INFEASIBLE_BASE`` w soo_adapter): paperowy stały próg
            zakłada gładki krajobraz fitness, a Big-M wprowadza nieciągłość
            przy granicy feasibility, którą statyczny próg źle interpretuje.
        bounds_margin: Extra margin applied to supplied ``world_min_bounds`` /
            ``world_max_bounds``. Domyślnie 0, bo oczekujemy, że wywołujący
            dostarczy już poprawnie rozszerzone granice z
            ``SwarmOptimizationProblem`` (spójne z NSGA-III i OOA).
        step_global_frac: (3,) frakcja zakresu świata używana jako amplituda
            kroku w fazie globalnej (Eq. 7-8 paperu, faza globalna). Paper ma
            tu skalar; nasza adaptacja jest anizotropowa per-osi (XY vs Z),
            bo świat jest asymetryczny (forest 60×600×11 m). Default
            [0.010, 0.010, 0.005] dobrany empirycznie — patrz plan.md.
        step_local_frac: (3,) frakcja zakresu świata używana jako współczynnik
            R w fazie lokalnej (Eq. 9-10 paperu). Default [0.003, 0.003, 0.001]
            ≈ 30% step_global, co odpowiada paperowej intuicji „local <
            global" (eksploatacja vs eksploracja).
        initial_population: Optional (pop_size, n_drones, n_inner, 3) tensor
            z zewnętrzną populacją inicjalną. Jeśli podana, metoda
            _initialize_swarms pomija własną inicjalizację i używa tej
            populacji bezpośrednio. Zapewnia warunek ceteris paribus
            w porównaniu z innymi algorytmami (np. NSGA-III).
    """

    def __init__(
        self,
        pop_size: int,
        n_drones: int,
        n_inner: int,
        world_min_bounds: NDArray[np.float64],
        world_max_bounds: NDArray[np.float64],
        start_positions: NDArray[np.float64],
        target_positions: NDArray[np.float64],
        fitness_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        max_generations: int = 500,
        rng: np.random.Generator | int | None = None,
        n_swarms: int = 5,
        coe1: float = 0.8,
        coe2: float = 0.2,
        threshold: float = 100.0,
        threshold_ratio: Optional[float] = None,
        bounds_margin: float = 0.0,
        step_global_frac: Optional[NDArray[np.float64]] = None,
        step_local_frac: Optional[NDArray[np.float64]] = None,
        history_writer: "OptimizationHistoryWriter | None" = None,
        timing: "TimingCollector | None" = None,
        initial_population: Optional[NDArray[np.float64]] = None,
    ) -> None:
        self.rng = np.random.default_rng(rng)

        self.pop_size = pop_size
        self.n_drones = n_drones
        self.n_inner = n_inner
        self.max_generations = max_generations

        self.G = n_swarms
        self.coe1 = coe1
        self.coe2 = coe2
        self.threshold = float(threshold)
        # Adaptacyjny próg (opcjonalny). Gdy None — paperowy tryb statyczny.
        # Synchroniczne z soo_adapter.HARD_INFEASIBLE_BASE = 1e6.
        self.threshold_ratio: Optional[float] = (
            float(threshold_ratio) if threshold_ratio is not None else None
        )
        self._HARD_INFEASIBLE_BASE: float = 1e6
        self._THRESHOLD_FLOOR: float = 0.01

        # Walidacje strukturalne (paper Sec. 1: G ≥ 1, M_pop podzielne przez G)
        # MUSZĄ być przed `self.P = pop_size // n_swarms`, żeby n_swarms=0
        # nie powodował ZeroDivisionError zamiast czytelnego ValueError.
        if self.G < 1:
            raise ValueError(f"n_swarms must be ≥ 1, got {n_swarms}.")
        if self.pop_size < self.G:
            raise ValueError(
                f"Population size ({pop_size}) must be ≥ n_swarms ({self.G})."
            )
        if self.pop_size % self.G != 0:
            raise ValueError(
                f"Population size ({pop_size}) must be divisible by "
                f"number of swarms ({self.G})."
            )

        self.P = pop_size // n_swarms  # Number of flies per swarm

        # Paper Shi et al. (2020), Sec. 1: warunek coe1 + coe2 = 1.
        # Zabezpieczenie przed konfiguracją łamiącą krzyżowanie z Eq. 14
        # (np. obie wagi 0.5 + 0.5 = 1 OK; 0.7 + 0.5 = 1.2 byłoby błędne
        # bo wynikowy `new_pop` przekroczyłby konwoluty wypukłej rodziców).
        if abs((self.coe1 + self.coe2) - 1.0) > 1e-6:
            raise ValueError(
                f"MSFOA paper (Shi et al. 2020) wymaga coe1 + coe2 = 1, "
                f"otrzymano coe1={self.coe1}, coe2={self.coe2} "
                f"(suma={self.coe1 + self.coe2})."
            )

        self.world_min = np.asarray(world_min_bounds, dtype=np.float64)
        self.world_max = np.asarray(world_max_bounds, dtype=np.float64)
        self.world_size = self.world_max - self.world_min

        self.start_pos = np.asarray(start_positions, dtype=np.float64)
        self.target_pos = np.asarray(target_positions, dtype=np.float64)
        self._starts_bc = self.start_pos[np.newaxis, :, np.newaxis, :]
        self._targets_bc = self.target_pos[np.newaxis, :, np.newaxis, :]

        self.fitness_fn = fitness_function

        # Granice clippingu: przyjmujemy bezpośrednio wartości z zewnątrz (np.
        # xl/xu z SwarmOptimizationProblem). Gwarantuje to identyczną przestrzeń
        # poszukiwań co w NSGA-III i OOA — Z ograniczone do przedziału lotu,
        # XY rozszerzone o margines świata. Wewnętrzne „zabezpieczenia” w
        # rodzaju hardcoded MIN_Z_ALTITUDE zostały usunięte, bo maskowały
        # rozbieżności między algorytmami i pozwalały osobnikom wlatywać w
        # sufit/podłogę poza zakresem rozpatrywanym przez pozostałe strategie.
        self._lb = self.world_min.copy() - bounds_margin
        self._ub = self.world_max.copy() + bounds_margin

        # Anizotropowe skalowanie kroku bazujące na zróżnicowanych granicach.
        # Paper Sec. 1 traktuje R jako jeden z konfigurowalnych parametrów —
        # wystawiamy go jako step_global_frac/step_local_frac, w postaci
        # anizotropowej (per-osi) z uwagi na asymetrię świata.
        #
        # Defaults (forest-tuned, grid-search z plan.md):
        #   global ≈ 1% zakresu świata XY, 0.5% Z
        #   local  ≈ 0.3% zakresu świata XY, 0.1% Z (≈ 30% step_global)
        # Domyślne wartości z paperu (~10% zakresu świata) prowadzą do
        # kandydatów ~200× gorszych od liderów na fitness landscape
        # zdominowanym przez F_smoothness — patrz plan.md.
        default_global = np.array([0.010, 0.010, 0.005])
        default_local  = np.array([0.003, 0.003, 0.001])
        sg_frac = np.asarray(
            step_global_frac if step_global_frac is not None else default_global,
            dtype=np.float64,
        )
        sl_frac = np.asarray(
            step_local_frac if step_local_frac is not None else default_local,
            dtype=np.float64,
        )
        if sg_frac.shape != (3,) or sl_frac.shape != (3,):
            raise ValueError(
                f"step_global_frac and step_local_frac must have shape (3,), "
                f"got {sg_frac.shape} and {sl_frac.shape}."
            )
        if np.any(sg_frac <= 0) or np.any(sl_frac <= 0):
            raise ValueError(
                f"step_global_frac and step_local_frac must be strictly positive, "
                f"got {sg_frac.tolist()} and {sl_frac.tolist()}."
            )
        self.step_global = self.world_size * sg_frac
        self.step_local = self.world_size * sl_frac

        self._history_writer = history_writer

        self._local_timing = False
        if timing is None:
            try:
                from src.algorithms.abstraction.trajectory.strategies.timing_utils import TimingCollector
                self._timing = TimingCollector("MSFFOA")
                self._local_timing = True
            except ImportError:
                self._timing = None
        else:
            self._timing = timing

        self._measure = self._timing.measure if self._timing else lambda *a, **kw: nullcontext()

        self.swarm_best_pos: NDArray[np.float64] = np.empty((self.G, self.n_drones, self.n_inner, 3))
        self.swarm_best_fit: NDArray[np.float64] = np.full(self.G, np.inf)

        # Lazy-init w `_initialize_swarms` gdy znamy M, K z adaptera (None gdy
        # fitness_fn nie wystawia last_objectives/last_constraints → fallback
        # do skalarnego fitness w bloku logowania).
        self.swarm_best_F: NDArray[np.float64] | None = None
        self.swarm_best_G: NDArray[np.float64] | None = None

        self.global_best_pos: NDArray[np.float64] = np.empty((self.n_drones, self.n_inner, 3))
        self.global_best_fitness: float = np.inf

        # Zewnętrzna populacja inicjalna (opcjonalna)
        # Kształt oczekiwany: (pop_size, n_drones, n_inner, 3)
        if initial_population is not None:
            if initial_population.shape != (pop_size, n_drones, n_inner, 3):
                raise ValueError(
                    f"initial_population shape mismatch: expected "
                    f"({pop_size}, {n_drones}, {n_inner}, 3), "
                    f"got {initial_population.shape}."
                )
        self._initial_population: Optional[NDArray[np.float64]] = initial_population

    def _snapshot_FG_flat(
        self,
    ) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
        """Zwraca kopię (F, G) z ostatniej ewaluacji adaptera, jeśli wystawia
        last_objectives / last_constraints (TrajectorySOOAdapter). Dla prostych
        funkcji fitness (testy) zwraca (None, None) i wówczas blok logowania
        per-gen używa skalarnego fitness jako fallback. Snapshot trzeba wykonać
        zaraz po `_evaluate`, zanim kolejne wywołanie nadpisze cache adaptera.
        """
        raw_f = getattr(self.fitness_fn, "last_objectives", None)
        raw_g = getattr(self.fitness_fn, "last_constraints", None)
        F = np.asarray(raw_f, dtype=np.float64).copy() if raw_f is not None else None
        G = np.asarray(raw_g, dtype=np.float64).copy() if raw_g is not None else None
        return F, G

    def _initialize_swarms(self) -> None:
        """
        Initializes the G swarm leaders.

        Jeśli w konstruktorze podano `initial_population`, liderzy każdego
        z G podrojów są wybierani jako najlepsi osobnicy z odpowiadającego im
        równomiernego wycinka tej populacji (slice o rozmiarze P).

        W przeciwnym razie liderzy są inicjalizowani wewnętrznie wokół
        zaszumionej linii prostej — zachowanie oryginalne algorytmu.
        """
        if self._initial_population is not None:
            # Zewnętrzna populacja: dzielimy na G równych plasterków po P osobników
            # i wybieramy najlepszego osobnika z każdego jako lidera roju.
            # Kształt: (pop_size, n_drones, n_inner, 3) → (G, P, n_drones, n_inner, 3)
            pop_split = self._initial_population.reshape(
                self.G, self.P, self.n_drones, self.n_inner, 3
            )

            # Ewaluacja całej populacji jednym batch-callem (wydajność)
            all_fit = self._evaluate(
                self._initial_population
            ).reshape(self.G, self.P)
            # Snapshot F/G PRZED kolejnym _evaluate; potrzebne do logowania liderów.
            init_F_flat, init_G_flat = self._snapshot_FG_flat()

            # Wybór lidera każdego roju: osobnik z najniższym fitness
            best_in_slice_idx = np.argmin(all_fit, axis=1)   # Shape: (G,)
            for g in range(self.G):
                self.swarm_best_pos[g] = pop_split[g, best_in_slice_idx[g]]
                self.swarm_best_fit[g] = all_fit[g, best_in_slice_idx[g]]

            # Wyciągnięcie F/G zwycięzców z plasterków (jeśli adapter je wystawia)
            if init_F_flat is not None:
                M = init_F_flat.shape[1]
                init_F = init_F_flat.reshape(self.G, self.P, M)
                self.swarm_best_F = init_F[np.arange(self.G), best_in_slice_idx]
            if init_G_flat is not None:
                K = init_G_flat.shape[1]
                init_G = init_G_flat.reshape(self.G, self.P, K)
                self.swarm_best_G = init_G[np.arange(self.G), best_in_slice_idx]

        else:
            # Oryginalna wewnętrzna inicjalizacja (fallback gdy brak external pop)
            t_vals = np.linspace(0, 1, self.n_inner + 2)[1:-1]
            t = t_vals.reshape(1, 1, self.n_inner, 1)
            base = self._starts_bc + t * (self._targets_bc - self._starts_bc)
            base_tiled = np.tile(base, (self.G, 1, 1, 1))

            noise = self.rng.normal(0.0, 1.0, size=(self.G, self.n_drones, self.n_inner, 3))
            noise *= (self.world_size * 0.1)[np.newaxis, np.newaxis, np.newaxis, :]
            self.swarm_best_pos = self._clip_to_bounds(base_tiled + noise)
            self.swarm_best_fit = self._evaluate(self.swarm_best_pos)
            # F/G dotyczą bezpośrednio G liderów (1:1, bez selekcji).
            init_F_flat, init_G_flat = self._snapshot_FG_flat()
            self.swarm_best_F = init_F_flat  # shape (G, M) lub None
            self.swarm_best_G = init_G_flat  # shape (G, K) lub None

        # Ustalenie globalnego optimum na podstawie stanu początkowego
        best_idx = int(np.argmin(self.swarm_best_fit))
        self.global_best_pos = self.swarm_best_pos[best_idx].copy()
        self.global_best_fitness = float(self.swarm_best_fit[best_idx])

    def _clip_to_bounds(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(positions, self._lb, self._ub)

    def _evaluate(self, pop: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.fitness_fn(pop)

    def _update_adaptive_threshold(self) -> None:
        """Adaptacyjna kalibracja threshold (Eq. 5-8) — relatywna do current best.

        No-op gdy ``threshold_ratio is None`` (paperowy tryb statyczny).

        Semantyka:
            ``threshold = best_feasible_swarm_fit × (1 + threshold_ratio)``

        Konsekwencja w `is_global = swarm_best_fit > threshold`:
        - Najlepszy feasible podrój: fit = best_feasible < threshold → LOCAL
          (zawsze refinement najlepszego).
        - Podroje w obrębie ``threshold_ratio`` od best: fit ≤ threshold → LOCAL.
        - Podroje dalej: fit > threshold → GLOBAL (eksploracja).

        Gdy żaden podrój nie jest jeszcze feasible (wszystkie ≥ HARD_INFEASIBLE_BASE),
        zachowujemy bieżącą wartość ``self.threshold`` (initial z konstruktora lub
        ostatnia kalibracja feasible). Działa to poprawnie, bo każde infeasible
        fitness ~1e6 jest z góry większe od jakiejkolwiek sensownej wartości
        threshold (zarówno 0, jak i np. 0.26 z anchora `sum(|w|)`), więc wszystkie
        podroje i tak są w GLOBAL — wybór konkretnej wartości jest bez znaczenia
        dla zachowania, a zachowanie initial value daje spójny log.

        Powód adaptacji: paperowy stały próg zakłada gładki krajobraz fitness;
        nasz Big-M wprowadza skok rzędu 1e6 między infeasible a feasible, więc
        statyczny próg ustawiony pre-run źle interpretuje dynamikę krajobrazu
        feasible (zob. eksperyment 2026-05-09 — analytical anchor `sum(|w|)`
        dawał threshold poniżej osiąganego floor fitness).
        """
        if self.threshold_ratio is None:
            return  # paperowy tryb statyczny — nie ruszamy self.threshold
        feasible = self.swarm_best_fit < self._HARD_INFEASIBLE_BASE
        if feasible.any():
            best_feasible = float(np.min(self.swarm_best_fit[feasible]))
            self.threshold = max(
                best_feasible * (1.0 + self.threshold_ratio),
                self._THRESHOLD_FLOOR,
            )
        # else: zachowaj poprzednią wartość — wszystkie infeasible (≥ 1e6)
        # i tak ją przekraczają, więc all GLOBAL niezależnie od konkretnej
        # liczby. Brak resetu utrzymuje czytelny log między strategy a optimizer.

    def optimize(self) -> Tuple[NDArray[np.float64], float]:
        """Wykonaj pełną pętlę optymalizacji MSFOA przez `max_generations` iteracji.

        Returns:
            Krotka `(best_pos, best_fitness)`:
            - `best_pos` `(n_drones, n_inner, 3)` — najlepszy znaleziony
              wielobok kontrolny.
            - `best_fitness` — wartość fitnessu najlepszego rozwiązania.
        """
        try:
            with self._measure("total_optimization"):
                with self._measure("population_initialization"):
                    self._initialize_swarms()
                    self._update_adaptive_threshold()

                source = "external (ceteris paribus)" if self._initial_population is not None else "internal"
                threshold_mode = (
                    f"adaptive(ratio={self.threshold_ratio})"
                    if self.threshold_ratio is not None else "static"
                )
                print(
                    f"[MSFOA] Init complete [{source}]. Swarms (G): {self.G}, "
                    f"Flies/Swarm: {self.P}, Gens: {self.max_generations}, "
                    f"Threshold: {self.threshold:.4f} [{threshold_mode}]\n"
                    f"[MSFOA] Init Global Best: {self.global_best_fitness:.4f}"
                )

                log_interval = max(1, self.max_generations // 10)

                with self._measure("generation_loop"):
                    import time
                    for gen in range(self.max_generations):
                        gen_t0 = time.monotonic()
                        # =================================================================
                        # Phase 1: Multi-Swarm with Multi-Tasks Searching Strategy
                        # =================================================================
                        rand_vals = self.rng.uniform(
                            -1.0, 1.0,
                            size=(self.G, self.P, self.n_drones, self.n_inner, 3)
                        )

                        is_global = self.swarm_best_fit > self.threshold
                        is_global_mask = is_global[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

                        # Paper Eq. 7–8 (Shi et al. 2020):
                        #   global: X_{g,i} = x_{g,best} + sin(2 · random(-1, 1))
                        #   local:  X_{g,i} = x_{g,best} + R · random(-1, 1)
                        # Mnożnik step_global / step_local jest dodatkiem do paperowej
                        # formuły — paper operuje na bezwymiarowej smell-space (Sec. 3),
                        # my pomijamy tę warstwę i pracujemy bezpośrednio na fizycznych
                        # punktach kontrolnych, więc skalowanie jest niezbędne.
                        step_global_val = (
                            self.step_global[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
                            * np.sin(2.0 * rand_vals)
                        )
                        step_local_val = (
                            self.step_local[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
                            * rand_vals
                        )

                        steps = np.where(is_global_mask, step_global_val, step_local_val)

                        old_pop = self.swarm_best_pos[:, np.newaxis, :, :, :] + steps
                        old_pop = self._clip_to_bounds(old_pop)

                        old_fit_flat = self._evaluate(
                            old_pop.reshape(self.pop_size, self.n_drones, self.n_inner, 3)
                        )
                        # Snapshot Phase-1 F/G PRZED Phase 2 nadpisze cache adaptera.
                        old_F_flat, old_G_flat = self._snapshot_FG_flat()
                        old_fit = old_fit_flat.reshape(self.G, self.P)

                        old_best_idx = np.argmin(old_fit, axis=1)
                        old_best_pos = old_pop[np.arange(self.G), old_best_idx]
                        old_best_fit = old_fit[np.arange(self.G), old_best_idx]
                        # F/G zwycięzcy Phase 1 per swarm (jeśli adapter wystawia)
                        if old_F_flat is not None:
                            M = old_F_flat.shape[1]
                            old_best_F = old_F_flat.reshape(self.G, self.P, M)[
                                np.arange(self.G), old_best_idx
                            ]
                        else:
                            old_best_F = None
                        if old_G_flat is not None:
                            K = old_G_flat.shape[1]
                            old_best_G = old_G_flat.reshape(self.G, self.P, K)[
                                np.arange(self.G), old_best_idx
                            ]
                        else:
                            old_best_G = None

                        # =================================================================
                        # Phase 2: Competitive Strategies of Offspring
                        # =================================================================
                        rand_swarm_idx = self.rng.integers(0, self.G, size=(self.G, self.P))
                        selected_leaders = self.swarm_best_pos[rand_swarm_idx]

                        new_pop = self.coe1 * selected_leaders + self.coe2 * old_pop
                        new_pop = self._clip_to_bounds(new_pop)

                        new_fit_flat = self._evaluate(
                            new_pop.reshape(self.pop_size, self.n_drones, self.n_inner, 3)
                        )
                        new_F_flat, new_G_flat = self._snapshot_FG_flat()
                        new_fit = new_fit_flat.reshape(self.G, self.P)

                        new_best_idx = np.argmin(new_fit, axis=1)
                        new_best_pos = new_pop[np.arange(self.G), new_best_idx]
                        new_best_fit = new_fit[np.arange(self.G), new_best_idx]
                        if new_F_flat is not None:
                            M = new_F_flat.shape[1]
                            new_best_F = new_F_flat.reshape(self.G, self.P, M)[
                                np.arange(self.G), new_best_idx
                            ]
                        else:
                            new_best_F = None
                        if new_G_flat is not None:
                            K = new_G_flat.shape[1]
                            new_best_G = new_G_flat.reshape(self.G, self.P, K)[
                                np.arange(self.G), new_best_idx
                            ]
                        else:
                            new_best_G = None

                        # =================================================================
                        # Phase 3: Competition & Update (Sec. 4 paperu)
                        # =================================================================
                        win_is_new = new_best_fit < old_best_fit
                        winner_fit = np.where(win_is_new, new_best_fit, old_best_fit)
                        winner_pos = np.where(
                            win_is_new[:, np.newaxis, np.newaxis, np.newaxis],
                            new_best_pos,
                            old_best_pos,
                        )
                        # Wybór F/G zwycięzcy zgodny z `win_is_new` (broadcast po osi feature)
                        if old_best_F is not None and new_best_F is not None:
                            winner_F = np.where(win_is_new[:, np.newaxis], new_best_F, old_best_F)
                        else:
                            winner_F = None
                        if old_best_G is not None and new_best_G is not None:
                            winner_G = np.where(win_is_new[:, np.newaxis], new_best_G, old_best_G)
                        else:
                            winner_G = None

                        # POPRAWKA ZGODNA Z LITERATURĄ: Elityzm lokalny liderów roju
                        # (Eq. 18-19, Shi et al. 2020). Lider pod-roju zastępowany jest
                        # nową pozycją tylko i wyłącznie wtedy, gdy funkcja dopasowania
                        # zwycięzcy z danego kroku jest mniejsza od historycznie
                        # najlepszej funkcji dopasowania dla tego pod-roju.
                        update_mask = winner_fit < self.swarm_best_fit

                        self.swarm_best_fit = np.where(
                            update_mask, winner_fit, self.swarm_best_fit
                        )

                        update_mask_pos = update_mask[:, np.newaxis, np.newaxis, np.newaxis]
                        self.swarm_best_pos = np.where(
                            update_mask_pos, winner_pos, self.swarm_best_pos
                        )
                        # Aktualizacja persistent F/G liderów tym samym maskowaniem
                        # (zachowuje semantykę elityzmu — F/G nieaktualizowanego lidera
                        # pozostaje z poprzedniej generacji).
                        if winner_F is not None and self.swarm_best_F is not None:
                            self.swarm_best_F = np.where(
                                update_mask[:, np.newaxis], winner_F, self.swarm_best_F
                            )
                        if winner_G is not None and self.swarm_best_G is not None:
                            self.swarm_best_G = np.where(
                                update_mask[:, np.newaxis], winner_G, self.swarm_best_G
                            )

                        gen_global_idx = int(np.argmin(self.swarm_best_fit))
                        if self.swarm_best_fit[gen_global_idx] < self.global_best_fitness:
                            self.global_best_fitness = float(self.swarm_best_fit[gen_global_idx])
                            self.global_best_pos = self.swarm_best_pos[gen_global_idx].copy()

                        # Re-kalibracja adaptacyjnego threshold na podstawie
                        # bieżącego stanu podrojów po Phase-3 elitism.
                        # No-op gdy threshold_ratio is None.
                        self._update_adaptive_threshold()

                        if self._history_writer is not None:
                            # Logujemy STAN algorytmu (G liderów po Phase-3 elitism),
                            # nie offspring (new_pop) — spójne z NSGA-III/OOA/SSA, które
                            # logują populację po selekcji, nie eksplorację.
                            # Eliminuje fałszywą degradację `feasible_ratio` przy
                            # konwergencji liderów w wąską niszę feasibility (offspring
                            # tuż za progiem nie są stanem algorytmu, lecz perturbacją).
                            if self.swarm_best_F is not None:
                                obj_matrix = self.swarm_best_F
                            else:
                                obj_matrix = self.swarm_best_fit.reshape(-1, 1)

                            evaluator = getattr(self.fitness_fn, "evaluator", None)
                            n_eval = (
                                int(evaluator.individuals_evaluated)
                                if evaluator is not None
                                and hasattr(evaluator, "individuals_evaluated")
                                else (gen + 1) * self.pop_size  # fallback estymata
                            )
                            gen_elapsed = time.monotonic() - gen_t0

                            from src.utils.per_gen_metrics import per_gen_metrics_from_FG

                            self._history_writer.put_generation_data(
                                per_gen_metrics_from_FG(
                                    objectives=obj_matrix,
                                    constraints=self.swarm_best_G,
                                    decisions=self.swarm_best_pos.reshape(self.G, -1).copy(),
                                    elapsed_s=gen_elapsed,
                                    eval_count_cumulative=n_eval,
                                )
                            )

                        if (gen + 1) % log_interval == 0 or (gen + 1) == self.max_generations:
                            swarms_in_global = int(np.sum(is_global))
                            extra = (
                                f" | Threshold: {self.threshold:.4f}"
                                if self.threshold_ratio is not None else ""
                            )
                            print(
                                f"[MSFOA] Gen {(gen + 1):4d}/{self.max_generations} | "
                                f"Best Fitness: {self.global_best_fitness:.4f} | "
                                f"Swarms in Global phase: {swarms_in_global}/{self.G}"
                                f"{extra}"
                            )

                print(f"[MSFOA] Finished. Final Best Fitness: {self.global_best_fitness:.6f}")

        finally:
            if getattr(self, "_local_timing", False) and self._timing is not None:
                try:
                    import os
                    from hydra.core.hydra_config import HydraConfig
                    out_dir = HydraConfig.get().runtime.output_dir
                    self._timing.save_csv(os.path.join(out_dir, "optimization_timings.csv"))
                except Exception as e:
                    print(f"[MSFOA] Failed to save timing logs: {e}")

        return self.global_best_pos.copy(), self.global_best_fitness

    def get_best_dense_trajectory(self) -> NDArray[np.float64]:
        """Zwróć pełny wielobok kontrolny `[start, inner…, target]` najlepszego rozwiązania.

        Returns:
            `(n_drones, n_inner + 2, 3)` — wielobok gotowy do
            `generate_bspline_batch` w post-processingu.
        """
        with self._measure("dense_trajectory_reconstruction"):
            inner = self.global_best_pos[np.newaxis, :, :, :]
            starts = np.broadcast_to(self._starts_bc, (1, self.n_drones, 1, 3)).copy()
            targets = np.broadcast_to(self._targets_bc, (1, self.n_drones, 1, 3)).copy()
            sparse = np.concatenate([starts, inner, targets], axis=2)

        return sparse[0]