"""
Multi-Strategy Fruit Fly Optimization Algorithm (MSFFOA) for 3D Trajectory Optimization.

Adapts the Fruit Fly Optimization Algorithm to multi-strategy search for
collision-free drone swarm trajectory planning. The population is represented
natively as a 4D tensor of shape (Pop_size, N_drones, N_waypoints_inner, 3).

Two characteristic FOA phases are implemented:
  1. Smell-based search (Osphresis) — generates candidate positions via three
     complementary strategies (Gaussian walk, Levy flight, global-best guidance).
  2. Vision-based search — greedy selection where each fly keeps the better of
     its current and candidate position, then the swarm records the global best.

References:
    Pan, W.-T. (2012). A new Fruit Fly Optimization Algorithm.
    Multi-strategy variants: adaptive strategy allocation per generation.
"""

from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Levy flight helper
# ---------------------------------------------------------------------------

def _levy_step(shape: Tuple[int, ...], beta: float = 1.5, rng: np.random.Generator | None = None) -> NDArray:
    """Generate Levy-distributed random steps via Mantegna's algorithm.

    Args:
        shape: Output array shape.
        beta: Levy exponent in (0, 2]. Default 1.5 (standard for metaheuristics).
        rng: Numpy random generator.

    Returns:
        Array of Levy-distributed values with the requested shape.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Mantegna's formula for the step-length ratio
    from math import gamma as math_gamma

    sigma_u = (
        math_gamma(1 + beta) * np.sin(np.pi * beta / 2)
        / (math_gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    u = rng.normal(0, sigma_u, size=shape)
    v = rng.normal(0, 1.0, size=shape)

    step = u / (np.abs(v) ** (1 / beta))
    return step


def _generate_bezier_curve(
    control_points: NDArray[np.float64],
    num_samples: int = 100,
) -> NDArray[np.float64]:
    """
    Wektoryzowana generacja krzywej Beziera na podstawie punktów kontrolnych.
    Gwarantuje nieskończoną gładkość (brak ostrych kątów = 0 kary za jerk).
    
    Args:
        control_points: (Pop_size, Drones, N_control_points, 3)
    Returns:
        (Pop_size, Drones, num_samples, 3)
    """
    pop_size, n_drones, n_cp, dims = control_points.shape
    n = n_cp - 1 # Stopień krzywej
    
    # Parametr 't' od 0 do 1
    t = np.linspace(0, 1, num_samples) # (num_samples,)
    
    # Przygotowanie tablicy na wynik
    curve = np.zeros((pop_size, n_drones, num_samples, dims))
    
    from math import comb
    
    # Równanie Beziera
    for i in range(n_cp):
        # Współczynnik Bernsteina (n po i) * t^i * (1-t)^(n-i)
        bernstein = comb(n, i) * (t**i) * ((1 - t)**(n - i)) # Kształt: (num_samples,)
        
        # Broadcasting do kształtu 4D: (1, 1, num_samples, 1)
        bernstein_4d = bernstein.reshape(1, 1, num_samples, 1)
        
        # Punkty kontrolne wyciągamy jako: (Pop_size, Drones, 1, 3)
        cp_i = control_points[:, :, i, :].reshape(pop_size, n_drones, 1, dims)
        
        curve += cp_i * bernstein_4d
        
    return curve
# ---------------------------------------------------------------------------
# MSFFOA Optimizer
# ---------------------------------------------------------------------------

class MSFFOAOptimizer:
    """Multi-Strategy Fruit Fly Optimization Algorithm for 3D trajectory tensors.

    The optimizer maintains a population of candidate trajectories stored as a
    native 4D tensor ``(pop_size, n_drones, n_inner, 3)`` — only the inner
    waypoints are optimized; start and target positions are fixed endpoints.

    Three search strategies are used during the *smell-based* phase:
        S1 — **Gaussian walk**: local perturbation around each fly's position.
        S2 — **Levy flight**: heavy-tailed jumps for long-range exploration.
        S3 — **Global-best guidance**: step toward the best-known position
              with an adaptive attraction coefficient.

    The *vision-based* phase performs greedy selection (keep the better of
    current vs. candidate) and updates the swarm's global best.

    Args:
        pop_size: Number of fruit flies (population size).
        n_drones: Number of drones in the swarm.
        n_inner: Number of *inner* waypoints per drone (excluding start/target).
        n_output_samples: Number of dense waypoints for trajectory evaluation.
        world_min_bounds: (3,) lower world bounds [x, y, z].
        world_max_bounds: (3,) upper world bounds [x, y, z].
        start_positions: (N_drones, 3) fixed start positions.
        target_positions: (N_drones, 3) fixed target positions.
        fitness_function: Callable that receives a population tensor of shape
            ``(Pop_size, N_drones, N_output_samples, 3)`` (dense trajectories)
            and returns a 1D fitness array of shape ``(Pop_size,)`` (lower is better).
        max_generations: Maximum number of optimization generations.
        seed: Random seed for reproducibility.
        levy_beta: Levy exponent for Strategy 2 (default 1.5).
        sigma_min_fraction: Minimum Gaussian sigma as a fraction of world size
            (prevents search collapse). Default 0.01.
        bounds_margin: Extra margin outside world bounds allowed for waypoints.
    """

    def __init__(
        self,
        pop_size: int,
        n_drones: int,
        n_inner: int,
        n_output_samples: int,
        world_min_bounds: NDArray[np.float64],
        world_max_bounds: NDArray[np.float64],
        start_positions: NDArray[np.float64],
        target_positions: NDArray[np.float64],
        fitness_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        max_generations: int = 500,
        seed: int = 42,
        levy_beta: float = 1.5,
        sigma_min_fraction: float = 0.01,
        bounds_margin: float = 50.0,
    ) -> None:
        self.pop_size = pop_size
        self.n_drones = n_drones
        self.n_inner = n_inner
        self.n_output_samples = n_output_samples
        self.max_generations = max_generations
        self.levy_beta = levy_beta

        self.world_min = np.asarray(world_min_bounds, dtype=np.float64)
        self.world_max = np.asarray(world_max_bounds, dtype=np.float64)
        self.world_size = self.world_max - self.world_min  # (3,)

        # Fixed endpoints — broadcast-ready shapes
        self.start_pos = np.asarray(start_positions, dtype=np.float64)   # (D, 3)
        self.target_pos = np.asarray(target_positions, dtype=np.float64) # (D, 3)
        self._starts_bc = self.start_pos[np.newaxis, :, np.newaxis, :]   # (1, D, 1, 3)
        self._targets_bc = self.target_pos[np.newaxis, :, np.newaxis, :] # (1, D, 1, 3)

        self.fitness_fn = fitness_function

        # Decision-variable bounds (per-point) with Z floor
        MIN_Z_ALTITUDE = 0.5
        self._lb = self.world_min.copy() - bounds_margin
        self._ub = self.world_max.copy() + bounds_margin
        self._lb[2] = max(MIN_Z_ALTITUDE, float(self.world_min[2]))

        # Minimum Gaussian sigma to prevent search stagnation
        self._sigma_min = self.world_size * sigma_min_fraction  # (3,)

        self.rng = np.random.default_rng(seed)

        # State — populated by optimize()
        self.population: NDArray[np.float64] = np.empty(0)
        self.fitness: NDArray[np.float64] = np.empty(0)
        self.global_best_pos: NDArray[np.float64] = np.empty(0)
        self.global_best_fitness: float = np.inf

        # Strategy success tracking for adaptive allocation
        self._strategy_rewards = np.ones(3, dtype=np.float64)

    # ------------------------------------------------------------------
    # Initialization (Golden Rule #3: spatial initialization)
    # ------------------------------------------------------------------

    def _initialize_population(self) -> NDArray[np.float64]:
        """Generate the initial population by adding Gaussian noise to a
        straight-line reference trajectory.

        The noise variance is proportional to the *world dimensions*
        ``(world_max - world_min)``, ensuring adequate spatial coverage
        regardless of local obstacle sizes.

        Returns:
            (pop_size, n_drones, n_inner, 3) initial population tensor.
        """
        # Straight-line reference: evenly spaced inner waypoints
        t_vals = np.linspace(0, 1, self.n_inner + 2)[1:-1]  # exclude start/target
        t = t_vals.reshape(1, 1, self.n_inner, 1)  # (1, 1, Inner, 1)

        base = self._starts_bc + t * (self._targets_bc - self._starts_bc)
        # (1, D, Inner, 3) -> tile to (Pop, D, Inner, 3)
        population = np.tile(base, (self.pop_size, 1, 1, 1))

        # Gaussian noise with variance proportional to world dimensions
        # Scale: ~15% of world size per axis for XY, ~30% of Z range for altitude
        noise_scale = self.world_size.copy()  # (3,)
        noise_scale[0] *= 0.15  # X
        noise_scale[1] *= 0.15  # Y
        noise_scale[2] *= 0.30  # Z — allow broader altitude exploration

        noise = self.rng.normal(
            0.0, 1.0,
            size=(self.pop_size, self.n_drones, self.n_inner, 3),
        )
        noise = self._smooth_perturbation(noise) # <--- NOWE
        noise *= noise_scale[np.newaxis, np.newaxis, np.newaxis, :] 
        noise[:, :, :, 2] += (noise_scale[2] * 0.5)

        population = population + noise

        return self._clip_to_bounds(population)

    # ------------------------------------------------------------------
    # Bound enforcement
    # ------------------------------------------------------------------

    def _clip_to_bounds(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """Clip waypoints to the decision-variable bounds."""
        return np.clip(positions, self._lb, self._ub)

    # ------------------------------------------------------------------
    # Dense trajectory construction
    # ------------------------------------------------------------------

    def _build_dense(self, inner: NDArray[np.float64]) -> NDArray[np.float64]:
        """Prepend start, append target, and resample to dense trajectory.

        Args:
            inner: (Pop, D, Inner, 3) inner waypoints.

        Returns:
            (Pop, D, N_output, 3) dense trajectory.
        """
        pop = inner.shape[0]
        starts = np.broadcast_to(self._starts_bc, (pop, self.n_drones, 1, 3)).copy()
        targets = np.broadcast_to(self._targets_bc, (pop, self.n_drones, 1, 3)).copy()
        sparse = np.concatenate([starts, inner, targets], axis=2)
        return _generate_bezier_curve(sparse, self.n_output_samples)

    # ------------------------------------------------------------------
    # Fitness evaluation (batch)
    # ------------------------------------------------------------------

    def _evaluate(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the entire population in a single vectorized call.

        Args:
            population: (Pop, D, Inner, 3) inner waypoints.

        Returns:
            (Pop,) fitness values (lower is better).
        """
        dense = self._build_dense(population)
        return self.fitness_fn(dense)

    # ------------------------------------------------------------------
    # Smell-based search — three strategies
    # ------------------------------------------------------------------

    def _strategy_gaussian(
        self, positions: NDArray[np.float64], sigma: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """S1: Gaussian walk — local perturbation around each fly's position.

        Args:
            positions: (K, D, Inner, 3) current positions of the sub-population.
            sigma: (3,) per-axis standard deviations.

        Returns:
            (K, D, Inner, 3) candidate positions.
        """
        noise = self.rng.normal(0.0, 1.0, size=positions.shape)
        noise = self._smooth_perturbation(noise) # <--- NOWE
        noise *= sigma[np.newaxis, np.newaxis, np.newaxis, :]
        return self._clip_to_bounds(positions + noise)

    def _strategy_levy(
        self, positions: NDArray[np.float64], scale: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """S2: Levy flight — heavy-tailed jumps for global exploration.

        Args:
            positions: (K, D, Inner, 3).
            scale: (3,) per-axis step scaling.

        Returns:
            (K, D, Inner, 3) candidate positions.
        """
        steps = _levy_step(positions.shape, beta=self.levy_beta, rng=self.rng)
        steps = self._smooth_perturbation(steps) # <--- NOWE
        steps *= scale[np.newaxis, np.newaxis, np.newaxis, :]
        return self._clip_to_bounds(positions + steps)

    def _strategy_global_best(
        self,
        positions: NDArray[np.float64],
        global_best: NDArray[np.float64],
        attraction: float,
    ) -> NDArray[np.float64]:
        """S3: Global-best guidance — move toward the best-known solution.

        Each fly takes a step toward ``global_best`` with a random coefficient
        drawn from U(0, attraction).

        Args:
            positions: (K, D, Inner, 3).
            global_best: (D, Inner, 3) best-known trajectory.
            attraction: Upper bound for the random attraction coefficient.

        Returns:
            (K, D, Inner, 3) candidate positions.
        """
        k = positions.shape[0]
        r = self.rng.uniform(0.0, attraction, size=(k, 1, 1, 1))
        direction = global_best[np.newaxis, :, :, :] - positions
        return self._clip_to_bounds(positions + r * direction)

    # ------------------------------------------------------------------
    # Adaptive strategy allocation
    # ------------------------------------------------------------------

    def _allocate_strategies(self) -> NDArray[np.int64]:
        """Assign each fly to one of the three strategies based on accumulated
        rewards, but enforce a hard lower bound to prevent strategy collapse.
        """
        probs = self._strategy_rewards / self._strategy_rewards.sum()
        
        # TWARDE ZABEZPIECZENIE: Żadna strategia nie spada poniżej 15%
        min_prob = 0.15
        probs = np.maximum(probs, min_prob)
        
        # Ponowna normalizacja (bo dodanie minimum zaburza sumę)
        probs = probs / probs.sum()
        
        return self.rng.choice(3, size=self.pop_size, p=probs)

    def _update_rewards(
        self,
        assignments: NDArray[np.int64],
        improved: NDArray[np.bool_],
        alpha: float = 0.2
    ) -> None:
        """Update strategy rewards based on how many flies improved per strategy.
        
        FIX: Uses an exponential moving average to prevent strategy probability 
        volatility (forgetting historically good strategies due to one bad generation).
        
        Args:
            assignments: Strategy index for each fly.
            improved: Boolean array indicating if the fly found a better position.
            alpha: Learning rate for the moving average.
        """
        for s in range(3):
            mask = assignments == s
            if mask.any():
                # Reward is 1.0 (baseline) + number of improvements
                current_reward = 1.0 + float(np.sum(improved[mask]))
                # Moving average update
                self._strategy_rewards[s] = (1.0 - alpha) * self._strategy_rewards[s] + alpha * current_reward

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def optimize(self) -> Tuple[NDArray[np.float64], float]:
        """Run the full MSFFOA optimization.

        Returns:
            best_trajectory: (N_drones, N_inner, 3) best inner waypoints found.
            best_fitness: Scalar fitness of the best trajectory.
        """
        # --- Initialization ---
        self.population = self._initialize_population()
        self.fitness = self._evaluate(self.population)

        best_idx = int(np.argmin(self.fitness))
        self.global_best_pos = self.population[best_idx].copy()
        self.global_best_fitness = float(self.fitness[best_idx])

        print(
            f"[MSFFOA] Init complete. Pop: {self.pop_size}, "
            f"Gens: {self.max_generations}, Inner pts: {self.n_inner}, "
            f"Best fitness: {self.global_best_fitness:.6f}"
        )

        # --- Generation loop ---
        for gen in range(self.max_generations):
            progress = gen / max(self.max_generations - 1, 1)  # 0 -> 1

            # Adaptive parameters: sigma decays, attraction grows
            sigma = self.world_size * (0.15 * (1 - progress) + 0.01)
            sigma = np.maximum(sigma, self._sigma_min)
            
            # FIX: Zabezpieczenie Levy Scale przed spadkiem do zera (minimum 1.5% wielkości świata)
            # Inaczej w późnych epokach algorytm traci zdolność przenikania przez ściany (duże skoki)
            levy_scale = self.world_size * np.maximum(0.02 * (1 - progress), 0.005)            
            attraction = 0.3 + 0.5 * progress  # 0.3 -> 0.8

            # ---- Phase 1: Smell-based search (Osphresis) ----
            assignments = self._allocate_strategies()
            candidates = np.empty_like(self.population)

            mask_s1 = assignments == 0
            mask_s2 = assignments == 1
            mask_s3 = assignments == 2

            if mask_s1.any():
                candidates[mask_s1] = self._strategy_gaussian(
                    self.population[mask_s1], sigma,
                )
            if mask_s2.any():
                candidates[mask_s2] = self._strategy_levy(
                    self.population[mask_s2], levy_scale,
                )
            if mask_s3.any():
                candidates[mask_s3] = self._strategy_global_best(
                    self.population[mask_s3], self.global_best_pos, attraction,
                )

            # ---- Phase 2: Vision-based search (greedy selection) ----
            candidate_fitness = self._evaluate(candidates)

            improved = candidate_fitness < self.fitness
            self.population[improved] = candidates[improved]
            self.fitness[improved] = candidate_fitness[improved]

            # Update global best
            gen_best_idx = int(np.argmin(self.fitness))
            if self.fitness[gen_best_idx] < self.global_best_fitness:
                self.global_best_pos = self.population[gen_best_idx].copy()
                self.global_best_fitness = float(self.fitness[gen_best_idx])

            # Update strategy rewards for next generation
            self._update_rewards(assignments, improved)

            if (gen + 1) % max(1, self.max_generations // 10) == 0:
                n_improved = int(np.sum(improved))
                print(
                    f"[MSFFOA] Gen {gen + 1}/{self.max_generations} | "
                    f"Best: {self.global_best_fitness:.6f} | "
                    f"Improved: {n_improved}/{self.pop_size} | "
                    f"Strategy probs: [{self._strategy_rewards / self._strategy_rewards.sum()}]"
                )

        print(f"[MSFFOA] Finished. Best fitness: {self.global_best_fitness:.6f}")
        return self.global_best_pos.copy(), self.global_best_fitness

    # ------------------------------------------------------------------
    # Convenience: full trajectory from best solution
    # ------------------------------------------------------------------

    def get_best_dense_trajectory(self) -> NDArray[np.float64]:
        """Build the dense trajectory for the best solution found.

        Returns:
            (N_drones, N_output_samples, 3) dense trajectory with start/target.
        """
        inner = self.global_best_pos[np.newaxis, :, :, :]  # (1, D, Inner, 3)
        dense = self._build_dense(inner)
        return dense[0]  # (D, N_out, 3)
    
    def _smooth_perturbation(self, noise: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Przekształca niezależny szum w skorelowaną falę wzdłuż osi waypointów (axis=2).
        Zapobiega to drastycznym naruszeniom max_jerk i uniformity_std.
        """
        # Rozmiar okna wygładzania (np. 1/3 długości trajektorii)
        window_size = max(3, self.n_inner // 3)
        window = np.ones(window_size) / window_size
        
        # Aplikujemy ruchomą średnią wzdłuż osi waypointów (axis=2)
        smoothed = np.apply_along_axis(
            lambda x: np.convolve(x, window, mode='same'), 
            axis=2, 
            arr=noise
        )
        
        # Wygładzanie drastycznie zmniejsza amplitudę szumu, więc musimy ją przywrócić,
        # aby skoki Levy'ego nadal pozwalały na ucieczkę z lokalnych minimów
        std_orig = np.std(noise, axis=2, keepdims=True) + 1e-9
        std_smooth = np.std(smoothed, axis=2, keepdims=True) + 1e-9
        
        return smoothed * (std_orig / std_smooth)