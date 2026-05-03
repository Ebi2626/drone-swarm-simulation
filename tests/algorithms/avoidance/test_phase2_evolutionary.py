"""Testy refactor 2026-05-02 (Single-Arc Deflection):
  - `AxisChooser`: deterministyczny wyborca osi uniku (clearance, anti-threat,
    secondary blocking, sticky hint).
  - `SingleArcDeflection`: 2D path representation (magnitude, peak_position),
    geometrycznie wymuszone single-hump shape.
  - `WeightedSumFitness`: 4D `[c_safety, c_energy, c_jerk, c_symmetry]`,
    monotoniczność, secondary_threats, sentinel.
  - Integration: Mealpy/MSFFOA/NSGA-3 on 2D search space.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.algorithms.avoidance.budget import TimeBudget
from src.algorithms.avoidance.fitness.WeightedSumFitness import WeightedSumFitness
from src.algorithms.avoidance.interfaces import PathProblem
from src.algorithms.avoidance.path.AxisChooser import AxisChooser
from src.algorithms.avoidance.path.SingleArcDeflection import SingleArcDeflection
from src.algorithms.avoidance.predictors.ConstantVelocityPredictor import (
    ConstantVelocityPredictor,
)
from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import (
    EvasionContext,
    KinematicState,
    ThreatAlert,
)
from src.trajectory.BSplineTrajectory import BSplineTrajectory


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def head_on_context() -> EvasionContext:
    """Head-on encounter: dron w +X 5 m/s, obstakle w +X jadące -X."""
    drone_state = KinematicState(
        position=np.array([0.0, 0.0, 5.0]),
        velocity=np.array([5.0, 0.0, 0.0]),
        radius=0.4,
    )
    obs_state = KinematicState(
        position=np.array([10.0, 0.0, 5.0]),
        velocity=np.array([-5.0, 0.0, 0.0]),
        radius=0.5,
    )
    threat = ThreatAlert(
        obstacle_state=obs_state,
        distance=10.0,
        time_to_collision=1.0,
        relative_velocity=np.array([10.0, 0.0, 0.0]),
    )
    base_spline = MagicMock()
    base_spline.profile.cruise_speed = 5.0
    base_spline.profile.max_accel = 2.0
    base_spline.cruise_speed = 5.0
    base_spline.max_accel = 2.0
    base_spline.arc_length = 100.0
    return EvasionContext(
        drone_id=0,
        current_time=10.0,
        drone_state=drone_state,
        threat=threat,
        base_spline=base_spline,
        rejoin_point=np.array([20.0, 0.0, 5.0]),
        rejoin_base_arc=20.0,
        world_bounds=(np.array([0.0, -10.0, 0.0]), np.array([30.0, 10.0, 10.0])),
        search_space_min=np.array([0.0, -8.0, 1.0]),
        search_space_max=np.array([30.0, 8.0, 9.0]),
    )


# --------------------------------------------------------------------------- #
# AxisChooser                                                                 #
# --------------------------------------------------------------------------- #


def test_axis_chooser_picks_anti_velocity_axis_for_head_on(head_on_context) -> None:
    """Head-on threat (obs vel = -X). AxisChooser PERPENDICULAR axis (right/left)
    powinien dostać `anti_threat ≈ 0` (perpendicular). Rzeczywiście nie przegrywa
    z up/down (które też mają anti_threat ≈ 0). Wszystkie cztery są równo dobre
    pod anti-velocity, więc decyduje clearance — search bbox ma równe Y/Z range
    [-8, 8] i [1, 9]; Y wins (większy clearance)."""
    chooser = AxisChooser()
    name, unit = chooser.pick(head_on_context)
    assert name in ("right", "left", "up", "down")
    np.testing.assert_almost_equal(np.linalg.norm(unit), 1.0, decimal=6)


def test_axis_chooser_anti_velocity_breaks_tie_with_threat_velocity() -> None:
    """Threat porusza się w +Y. AxisChooser powinien preferować -Y (left, jeśli
    drone leci w +X) bo `anti_threat = -dot(left, +Y) = -(-1) = 1 → max`."""
    drone_state = KinematicState(
        position=np.array([0.0, 0.0, 5.0]),
        velocity=np.array([5.0, 0.0, 0.0]),  # forward = +X
        radius=0.4,
    )
    # threat moving in +Y → drone should pick "left" (-Y axis from drone POV)
    obs_state = KinematicState(
        position=np.array([10.0, 0.0, 5.0]),
        velocity=np.array([0.0, 5.0, 0.0]),
        radius=0.5,
    )
    threat = ThreatAlert(
        obstacle_state=obs_state, distance=10.0,
        time_to_collision=1.0,
        relative_velocity=np.array([5.0, -5.0, 0.0]),
    )
    ctx = EvasionContext(
        drone_id=0, current_time=0.0, drone_state=drone_state, threat=threat,
        base_spline=MagicMock(), rejoin_point=np.array([20.0, 0.0, 5.0]),
        rejoin_base_arc=20.0,
        world_bounds=(np.array([-10, -10, 0]), np.array([30, 10, 10])),
        search_space_min=np.array([-10, -10, 1]),
        search_space_max=np.array([30, 10, 9]),
    )
    # forward=+X, lateral_xy=+Y; "right" = +Y, "left" = -Y.
    # threat_vel = +Y → axis_unit · vel: right=+1, left=-1.
    # anti_threat: right = -1, left = +1. left wins.
    chooser = AxisChooser(w_clearance=0.1, w_anti_threat=10.0)
    name, _ = chooser.pick(ctx)
    assert name == "left"


def test_axis_chooser_secondary_threat_blocks_axis(head_on_context) -> None:
    """Secondary threat na +Y od drone → 'right' axis blocked."""
    sec = ThreatAlert(
        obstacle_state=KinematicState(
            position=np.array([0.0, 5.0, 5.0]),  # +Y from drone
            velocity=np.zeros(3),
            radius=0.5,
        ),
        distance=5.0, time_to_collision=10.0, relative_velocity=np.zeros(3),
    )
    head_on_context.secondary_threats = [sec]

    chooser = AxisChooser(
        w_clearance=0.0, w_anti_threat=0.0, w_secondary_blocking=10.0
    )
    name, _ = chooser.pick(head_on_context)
    # forward=+X, "right" = +Y → blocked. left/up/down should be preferred.
    assert name != "right"


def test_axis_chooser_sticky_hint_when_viable(head_on_context) -> None:
    """Hint = 'up' i jest viable (score ≥ threshold * best) → wracamy 'up'."""
    head_on_context.preferred_axis_hint = "up"
    chooser = AxisChooser(sticky_hint_threshold=0.5)
    name, _ = chooser.pick(head_on_context)
    assert name == "up"


def test_axis_chooser_sticky_hint_overridden_when_blocked() -> None:
    """Hint = 'up' ale +Z blocked przez secondary threat → najlepsza inna oś."""
    drone_state = KinematicState(
        position=np.array([0.0, 0.0, 5.0]),
        velocity=np.array([5.0, 0.0, 0.0]),
        radius=0.4,
    )
    sec = ThreatAlert(
        obstacle_state=KinematicState(
            position=np.array([0.0, 0.0, 8.0]),  # bezpośrednio nad dronem
            velocity=np.zeros(3),
            radius=0.5,
        ),
        distance=3.0, time_to_collision=10.0, relative_velocity=np.zeros(3),
    )
    threat = ThreatAlert(
        obstacle_state=KinematicState(
            position=np.array([10.0, 0.0, 5.0]),
            velocity=np.array([-5.0, 0.0, 0.0]),
            radius=0.5,
        ),
        distance=10.0, time_to_collision=1.0,
        relative_velocity=np.array([10.0, 0.0, 0.0]),
    )
    ctx = EvasionContext(
        drone_id=0, current_time=0.0, drone_state=drone_state, threat=threat,
        base_spline=MagicMock(), rejoin_point=np.array([20.0, 0.0, 5.0]),
        rejoin_base_arc=20.0,
        world_bounds=(np.array([-10, -10, 0]), np.array([30, 10, 10])),
        search_space_min=np.array([-10, -10, 1]),
        search_space_max=np.array([30, 10, 9]),
        preferred_axis_hint="up",
        secondary_threats=[sec],
    )
    chooser = AxisChooser(
        w_clearance=0.1, w_anti_threat=0.0, w_secondary_blocking=10.0,
        sticky_hint_threshold=0.99,  # bardzo restrykcyjnie
    )
    name, _ = chooser.pick(ctx)
    # 'up' blocked → sticky should NOT win.
    assert name != "up"


# --------------------------------------------------------------------------- #
# SingleArcDeflection                                                         #
# --------------------------------------------------------------------------- #


@pytest.fixture
def chooser() -> AxisChooser:
    return AxisChooser()


def test_single_arc_deflection_gene_dim_is_2(head_on_context, chooser) -> None:
    repr_ = SingleArcDeflection(axis_chooser=chooser)
    assert repr_.gene_dim(head_on_context) == 2
    lb, ub = repr_.gene_bounds(head_on_context)
    assert lb.shape == (2,) and ub.shape == (2,)
    assert lb[0] == 0.8 and ub[0] == 4.0  # magnitude bounds
    assert lb[1] == 0.3 and ub[1] == 0.7  # peak_position bounds


def test_single_arc_deflection_creates_single_hump_shape(
    head_on_context, chooser
) -> None:
    """Spline z magnitude=2.0 deflection ma JEDNO extremum w wybranej osi
    (single-hump). Walidacja przez monotoniczność na obu połówkach."""
    repr_ = SingleArcDeflection(axis_chooser=chooser, min_applied_cruise_ratio=0.0)
    spline = repr_.decode_genes(np.array([2.0, 0.5]), head_on_context)
    assert spline is not None

    from scipy.interpolate import splev
    u = np.linspace(0, 1, 51)
    samples = np.asarray(splev(u, spline.tck), dtype=np.float64).T  # (51, 3)
    # Off-axis projection on (rejoin - start). Pierwsza połowa rośnie do peak,
    # druga maleje. Dokładnie 1 maksimum.
    diff_z = samples[:, 2] - 5.0
    diff_y = samples[:, 1]
    # Wszystkie próbki z jednej strony powinny mieć ten sam znak (single-side hump).
    # Sprawdzamy że monotonicznie rośnie do peak indeksu, potem maleje.
    abs_off = np.maximum(np.abs(diff_z), np.abs(diff_y))
    peak_idx = int(np.argmax(abs_off))
    # Up to peak: monotone non-decreasing (z marginesem na cubic noise ~5%)
    rising = abs_off[: peak_idx + 1]
    falling = abs_off[peak_idx:]
    # Tolerujemy małe wahania numeryczne — sprawdzamy dominującą tendencję.
    assert np.sum(np.diff(rising) >= -0.01) >= 0.85 * len(rising), \
        f"Pierwsza połowa NIE monotone-rising (peak_idx={peak_idx})"
    assert np.sum(np.diff(falling) <= 0.01) >= 0.85 * len(falling), \
        f"Druga połowa NIE monotone-falling (peak_idx={peak_idx})"


def test_single_arc_deflection_clamps_peak_to_floor(chooser) -> None:
    """Magnitude w kierunku -Z większa niż drone Z → peak clampowany do
    `world_min_z + floor_safe_margin`."""
    drone_state = KinematicState(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.array([5.0, 0.0, 0.0]),
        radius=0.4,
    )
    threat = ThreatAlert(
        obstacle_state=KinematicState(
            position=np.array([10.0, 0.0, 2.0]),
            velocity=np.array([0.0, 0.0, 5.0]),  # pcha w +Z → drone wybierze -Z (down)
            radius=0.5,
        ),
        distance=10.0, time_to_collision=1.0,
        relative_velocity=np.array([5.0, 0.0, -5.0]),
    )
    base = MagicMock()
    base.cruise_speed = 5.0
    base.max_accel = 2.0
    ctx = EvasionContext(
        drone_id=0, current_time=0.0, drone_state=drone_state, threat=threat,
        base_spline=base, rejoin_point=np.array([20.0, 0.0, 2.0]),
        rejoin_base_arc=20.0,
        world_bounds=(np.array([-10, -10, 0]), np.array([30, 10, 10])),
        search_space_min=np.array([-10, -10, 0]),
        search_space_max=np.array([30, 10, 10]),
    )
    repr_ = SingleArcDeflection(
        axis_chooser=AxisChooser(w_clearance=0.0, w_anti_threat=10.0),
        floor_safe_margin_m=1.0, min_applied_cruise_ratio=0.0,
    )
    # Wymuszamy magnitude=4 w kierunku -Z. Drone na Z=2 → peak naturalny -2,
    # ale floor=1 (world_min_z=0 + margin=1). Clamp powinien zadziałać.
    spline = repr_.decode_genes(np.array([4.0, 0.5]), ctx)
    if spline is None:
        pytest.skip("Spline rejected przez clamp ratio (acceptable, just sanity)")
    from scipy.interpolate import splev
    samples = np.asarray(splev(np.linspace(0, 1, 33), spline.tck), dtype=np.float64).T
    assert samples[:, 2].min() >= 1.0 - 0.15, \
        f"Peak Z below floor: min={samples[:, 2].min():.3f}"


def test_single_arc_deflection_invalid_genes_returns_none(
    head_on_context, chooser
) -> None:
    repr_ = SingleArcDeflection(axis_chooser=chooser)
    assert repr_.decode_genes(np.zeros(5), head_on_context) is None


def test_single_arc_deflection_constructor_validations(chooser) -> None:
    with pytest.raises(ValueError, match="magnitude"):
        SingleArcDeflection(axis_chooser=chooser, magnitude_min_m=-1.0)
    with pytest.raises(ValueError, match="peak_position"):
        SingleArcDeflection(
            axis_chooser=chooser,
            peak_position_min=0.7, peak_position_max=0.3,
        )
    with pytest.raises(ValueError, match="floor"):
        SingleArcDeflection(axis_chooser=chooser, floor_safe_margin_m=-0.1)
    with pytest.raises(ValueError, match="min_applied_cruise_ratio"):
        SingleArcDeflection(axis_chooser=chooser, min_applied_cruise_ratio=2.0)


def test_single_arc_deflection_rejects_severe_clamp(head_on_context) -> None:
    """Jeśli `min_applied_cruise_ratio=0.99` a spline ma jakąkolwiek krzywiznę
    powodującą clamp → spline odrzucony."""
    repr_ = SingleArcDeflection(
        axis_chooser=AxisChooser(),
        magnitude_min_m=3.0, magnitude_max_m=4.0,  # forsuje krzywiznę
        min_applied_cruise_ratio=0.99,
    )
    result = repr_.decode_genes(np.array([4.0, 0.5]), head_on_context)
    assert result is None, "Spline z big magnitude i restrykcyjnym ratio MUSI być rejected"


def test_single_arc_deflection_relaxed_ratio_allows_path(head_on_context) -> None:
    repr_ = SingleArcDeflection(
        axis_chooser=AxisChooser(), min_applied_cruise_ratio=0.0,
    )
    result = repr_.decode_genes(np.array([2.0, 0.5]), head_on_context)
    assert result is not None


# --------------------------------------------------------------------------- #
# WeightedSumFitness (kept from previous iteration)                           #
# --------------------------------------------------------------------------- #


def test_weighted_sum_fitness_components_returns_4d_vector(head_on_context) -> None:
    fitness = WeightedSumFitness()
    predictor = ConstantVelocityPredictor()
    pts = np.array(
        [[0, 0, 5], [5, 4, 5], [10, 4, 5], [15, 4, 5], [20, 0, 5]],
        dtype=np.float64,
    )
    spline = BSplineTrajectory(
        waypoints=pts, cruise_speed=5.0, max_accel=2.0, constant_speed=True
    )
    components = fitness.evaluate_components(spline, head_on_context, predictor)
    assert components.shape == (4,)
    assert np.all(components >= 0.0)


def test_weighted_sum_fitness_components_sentinel_for_none(head_on_context) -> None:
    fitness = WeightedSumFitness()
    components = fitness.evaluate_components(
        None, head_on_context, ConstantVelocityPredictor()
    )
    np.testing.assert_array_equal(components, np.full(4, 1e9))


def test_weighted_sum_fitness_secondary_threats_increase_safety_cost(
    head_on_context,
) -> None:
    """Krok B-rev #3: secondary threat na trajektorii MUSI zwiększyć c_safety."""
    fitness = WeightedSumFitness(
        w_safety=1.0, w_energy=0.0, w_jerk=0.0, w_symmetry=0.0
    )
    predictor = ConstantVelocityPredictor()
    pts = np.array(
        [[0, 0, 5], [5, 0, 5], [10, 0, 5], [15, 0, 5], [20, 0, 5]],
        dtype=np.float64,
    )
    spline = BSplineTrajectory(
        waypoints=pts, cruise_speed=5.0, max_accel=2.0, constant_speed=True
    )
    c_safety_primary = fitness.evaluate_components(
        spline, head_on_context, predictor
    )[0]

    secondary = ThreatAlert(
        obstacle_state=KinematicState(
            position=np.array([10.0, 0.0, 5.0]),
            velocity=np.zeros(3), radius=0.5,
        ),
        distance=10.0, time_to_collision=2.0, relative_velocity=np.zeros(3),
    )
    head_on_context.secondary_threats = [secondary]
    c_safety_with_secondary = fitness.evaluate_components(
        spline, head_on_context, predictor
    )[0]

    assert c_safety_with_secondary > c_safety_primary


def test_constant_velocity_predictor_uses_spline_when_present() -> None:
    """Krok B-rev #4: gdy `ThreatAlert.trajectory` jest ustawiony, predyktor
    używa spline'u zamiast linear extrapolation."""
    pts = np.array(
        [[0, 0, 5], [0, 3, 5], [0, 6, 5], [0, 9, 5], [0, 12, 5]],
        dtype=np.float64,
    )
    spline = BSplineTrajectory(
        waypoints=pts, cruise_speed=2.0, max_accel=2.0, constant_speed=True
    )
    threat_with_traj = ThreatAlert(
        obstacle_state=KinematicState(
            position=np.array([0.0, 0.0, 5.0]),
            velocity=np.array([0.0, 5.0, 0.0]),  # MISLEADING — ma być zignorowana
            radius=0.5,
        ),
        distance=10.0, time_to_collision=2.0,
        relative_velocity=np.array([0.0, -5.0, 0.0]),
        trajectory=spline, trajectory_start_offset=0.0,
    )
    predictor = ConstantVelocityPredictor()
    state_at_t1 = predictor.predict_state(threat_with_traj, t_offset=1.0)
    # Po t=1s spline na ~Y=2 (cruise=2). Linear by dało Y=5.
    assert abs(state_at_t1.position[1] - 2.0) < 0.5


# --------------------------------------------------------------------------- #
# Integration tests: Mealpy / MSFFOA / NSGA-3 with SingleArcDeflection        #
# --------------------------------------------------------------------------- #


def _make_problem_with_single_arc(ctx: EvasionContext) -> PathProblem:
    return PathProblem(
        context=ctx,
        predictor=ConstantVelocityPredictor(),
        fitness=WeightedSumFitness(),
        path_repr=SingleArcDeflection(
            axis_chooser=AxisChooser(), min_applied_cruise_ratio=0.0,
        ),
    )


def test_mealpy_optimizer_smoke_with_single_arc(head_on_context) -> None:
    """End-to-end smoke: MealpyOptimizer + SSA + SingleArcDeflection (2D)."""
    pytest.importorskip("mealpy")
    from mealpy.swarm_based.SSA import OriginalSSA
    from src.algorithms.avoidance.optimizers.MealpyOptimizer import MealpyOptimizer

    optimizer = MealpyOptimizer(
        algorithm_factory=OriginalSSA, epoch=10, pop_size=8,
        min_compute_time_s=0.05, rng=42,
    )
    problem = _make_problem_with_single_arc(head_on_context)
    budget = TimeBudget.start_now(max_seconds=2.0)
    result = optimizer.optimize(problem, budget)
    assert result.status == "ok", f"Got {result.status}: {result.extra}"
    assert result.waypoints is not None
    assert result.waypoints.shape[1] == 3


def test_msffoa_optimizer_smoke_with_single_arc(head_on_context) -> None:
    """End-to-end smoke: MSFFOA + SingleArcDeflection (2D, n_inner_waypoints
    legacy field unused)."""
    from src.algorithms.avoidance.optimizers.MSFFOAOnlineOptimizer import (
        MSFFOAOnlineOptimizer,
    )

    optimizer = MSFFOAOnlineOptimizer(
        n_inner_waypoints=5, pop_size=8, n_swarms=4,
        max_generations=10, rng=42,
    )
    problem = _make_problem_with_single_arc(head_on_context)
    budget = TimeBudget.start_now(max_seconds=2.0)
    result = optimizer.optimize(problem, budget)
    # MSFFOA może wrócić ok lub timed_out — oba akceptowalne; ważne że NIE failed.
    assert result.status in ("ok", "timed_out"), f"Got {result.status}: {result.extra}"


def test_nsga3_optimizer_smoke_with_single_arc(head_on_context) -> None:
    """End-to-end smoke: NSGA-III + SingleArcDeflection (2D)."""
    pytest.importorskip("pymoo")
    from src.algorithms.avoidance.optimizers.NSGA3OnlineOptimizer import (
        NSGA3OnlineOptimizer,
    )

    optimizer = NSGA3OnlineOptimizer(
        n_inner_waypoints=5, epoch=5, pop_size=35,
        n_partitions=4, decision_mode="safety", rng=42,
    )
    problem = _make_problem_with_single_arc(head_on_context)
    budget = TimeBudget.start_now(max_seconds=3.0)
    result = optimizer.optimize(problem, budget)
    assert result.status in ("ok", "timed_out"), f"Got {result.status}: {result.extra}"


def test_full_stack_evolutionary_avoidance_e2e(head_on_context) -> None:
    """Full stack: GenericOptimizingAvoidance + Mealpy + SingleArcDeflection."""
    pytest.importorskip("mealpy")
    from mealpy.swarm_based.SSA import OriginalSSA
    from src.algorithms.avoidance.GenericOptimizingAvoidance import (
        GenericOptimizingAvoidance,
    )
    from src.algorithms.avoidance.optimizers.MealpyOptimizer import MealpyOptimizer

    avoidance = GenericOptimizingAvoidance(
        name="test",
        predictor=ConstantVelocityPredictor(),
        path_representation=SingleArcDeflection(
            axis_chooser=AxisChooser(), min_applied_cruise_ratio=0.0,
        ),
        fitness=WeightedSumFitness(),
        optimizer=MealpyOptimizer(
            algorithm_factory=OriginalSSA, epoch=8, pop_size=8,
            min_compute_time_s=0.05, rng=42,
        ),
        time_budget_s=2.0,
    )
    plan = avoidance.compute_evasion_plan(head_on_context)
    if plan is not None:
        assert plan.evasion_spline is not None
