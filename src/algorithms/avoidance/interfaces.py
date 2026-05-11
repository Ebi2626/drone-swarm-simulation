"""Cztery interfejsy abstrakcyjne (Strategy Pattern) dla planera uniku online.

Hierarchia ról:
- `IObstaclePredictor` — model przyszłej pozycji przeszkody.
- `IPathRepresentation` — konwersja waypointów / genów na `BSplineTrajectory`.
- `IFitnessEvaluator` — koszt / ocena trajektorii uniku (`axis_score` lub
  pełne `evaluate`).
- `IPathOptimizer` — silnik wybierający waypointy w zadanym budżecie czasu.

Wszystkie implementacje są wstrzykiwane przez Hydra (`_target_` w yaml-u
strategii uniku) — wymiana komponentu nie wymaga zmian w kodzie.
`PathProblem` i `OptimizationResult` to dataklasy hermetyzujące I/O
optymalizatora.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.algorithms.avoidance.ThreatAnalyzer.ThreatAnalyzer import (
    EvasionContext,
    KinematicState,
    ThreatAlert,
)
from src.algorithms.avoidance.budget import TimeBudget

if TYPE_CHECKING:
    from src.trajectory.BSplineTrajectory import BSplineTrajectory


class IObstaclePredictor(ABC):
    """Estymator stanu kinematycznego przeszkody w przyszłości.

    Implementacja produkcyjna: `ConstantVelocityPredictor` (liniowa
    ekstrapolacja `pos += vel · t`).
    """

    @abstractmethod
    def predict_state(self, threat: ThreatAlert, t_offset: float) -> KinematicState:
        """Zwróć przewidywany stan przeszkody w czasie `t_offset` od teraz.

        :param threat: aktualna obserwacja zagrożenia (z lidaru → ThreatAnalyzer).
        :param t_offset: horyzont predykcji [s].
        """

    @abstractmethod
    def time_to_collision(
        self,
        drone_state: KinematicState,
        threat: ThreatAlert,
    ) -> float:
        """Estymuj czas do kolizji przy zachowaniu obecnych prędkości.

        :return: TTC w sekundach; +inf jeśli zagrożenie się oddala.
        """


class IPathRepresentation(ABC):
    """Reprezentacja trajektorii ucieczkowej.

    Dwa tryby:
    - `waypoints_to_spline(...)` — wejście = surowe waypointy 3D, wyjście =
      wygładzony BSpline 3D (impl. `BSplineSmoother`).
    - `decode_genes(...)` — wejście = wektor decyzji optymalizatora
      ewolucyjnego, wyjście = BSpline (impl. `SingleArcDeflection`).
      Kontrakt rozszerzony o `gene_dim(ctx)` i `gene_bounds(ctx)` — używane
      przez mealpy/pymoo do budowy problemu.
    """

    @abstractmethod
    def waypoints_to_spline(
        self,
        waypoints: NDArray[np.float64],
        context: EvasionContext,
        *,
        axis_name: str | None = None,
    ) -> "BSplineTrajectory | None":
        """Zbuduj wykonalny BSpline z surowych waypointów.

        :param axis_name: opcjonalny hint o wybranej osi uniku ("up"/"down"/
            "right"/"left") — `BSplineSmoother` używa go do Z-linearyzacji
            przy osi lateralnych (zapobiega Z-oscylacjom).
        :return: BSpline lub `None` jeśli budowa się nie powiodła
                 (zdegenerowana sekwencja, naruszenie warunków brzegowych).
        """

    def decode_genes(
        self,
        genes: NDArray[np.float64],
        context: EvasionContext,
    ) -> "BSplineTrajectory | None":
        """Zdekoduj wektor genów (decyzje optymalizatora ewolucyjnego) na BSpline.

        Domyślnie `NotImplementedError` — implementacje legacy oparte tylko
        na `waypoints_to_spline` nie potrzebują dekodera genów.

        :param genes: wektor decyzji o długości `gene_dim(context)`. Geometria
                      kodowania jest sprawą konkretnej implementacji.
        :return: BSpline lub `None` przy niepowodzeniu (degeneracja, naruszenie
                 granic). Optimizer powinien w fitness penalizować takie wyniki.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.decode_genes() — używane wyłącznie przez "
            f"reprezentacje wspierające evolutionary search."
        )

    def gene_dim(self, context: EvasionContext) -> int:
        """Wymiar wektora decyzji dla danego kontekstu (zazwyczaj zależy od
        liczby inner waypointów × liczby zmiennych osiowych).

        Defaultowo `NotImplementedError` — patrz `decode_genes` rationale.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.gene_dim() — używane wyłącznie przez "
            f"reprezentacje evolutionary."
        )

    def gene_bounds(
        self,
        context: EvasionContext,
    ) -> "tuple[NDArray[np.float64], NDArray[np.float64]]":
        """Granice (lb, ub) wektora decyzji dla danego kontekstu.

        Wynik MUSI mieć shape `(gene_dim,)` × 2. Używane przez mealpy `FloatVar`
        oraz pymoo `Problem(xl=…, xu=…)`.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.gene_bounds() — używane wyłącznie przez "
            f"reprezentacje evolutionary."
        )


class IFitnessEvaluator(ABC):
    """Funkcja kosztu / oceny wariantu trajektorii uniku.

    Implementacje:
    - `AxisBiasFitness` — używa tylko `axis_score` (heurystyka wyboru osi
      uniku right/left/up/down).
    - `WeightedSumFitness` — pełne `evaluate(spline, ctx, predictor)` dla SOO
      (SSA, OOA, MSFOA Online) oraz `evaluate_components` zwracające 4-wektor
      `[c_safety, c_energy, c_jerk, c_symmetry]` dla NSGA-III multi-obj.

    Atrybuty bias (`bias_preferred`, `bias_perpendicular`, `bias_oppose`)
    parametryzują heurystyczne wagi osi.
    """

    bias_preferred: float = 1.0
    bias_perpendicular: float = 1.0
    bias_oppose: float = 1.0

    @abstractmethod
    def axis_score(
        self,
        axis_name: str,
        axis_dir: NDArray[np.float64],
        space: float,
        obs_vel_hat: NDArray[np.float64] | None,
        order_score: float,
    ) -> float:
        """Score osi uniku — wyższy = lepszy.

        :param axis_name: jedna z `{"up", "down", "right", "left"}`.
        :param axis_dir: znormalizowany wektor kierunku w 3D.
        :param space: dostępna przestrzeń wzdłuż osi [m] (do ściany / sufitu / podłogi).
        :param obs_vel_hat: znormalizowany wektor prędkości przeszkody lub `None`
                            (`None` gdy |v_obs| poniżej progu szumu — wtedy
                            anty-prędkościowy bias nieaktywny).
        :param order_score: tie-break z `prefer_axis_order` w configu (0…1).
        """

    def evaluate(
        self,
        candidate: "BSplineTrajectory | None",
        context: EvasionContext,
        predictor: IObstaclePredictor,
    ) -> float:
        """Skalarne fitness pełnego BSplinu (lower = better).

        Implementacja MUSI obsłużyć `candidate is None` — niezdekodowalne geny
        z `IPathRepresentation.decode_genes` powinny dostać ekstremalny koszt
        (np. 1e9), by populacja optymalizatora wybrakowała takie osobniki.

        Domyślnie `NotImplementedError` — implementacja w `WeightedSumFitness`.
        Heurystyki czysto axis-based (`AxisBiasFitness`) nie potrzebują
        skalarnego fitness pełnej trajektorii.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.evaluate() — skalarne fitness BSplinu "
            f"implementuje `WeightedSumFitness`."
        )

    def evaluate_components(
        self,
        candidate: "BSplineTrajectory | None",
        context: EvasionContext,
        predictor: IObstaclePredictor,
    ) -> NDArray[np.float64]:
        """Wektor surowych składników kosztu (przed wagami) — domyślnie 4D
        `[c_safety, c_energy, c_jerk, c_symmetry]`. Używane przez
        `NSGA3OnlineOptimizer` do multi-objective optymalizacji.

        Domyślnie `NotImplementedError` — `WeightedSumFitness` nadpisuje.
        Dla niezdekodowalnych kandydatów MUSI zwrócić wektor sentinelowy
        (np. `[1e9]*4`), tak by optymalizator wybrakował osobnika.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.evaluate_components() — multi-obj fitness "
            f"implementuje `WeightedSumFitness`. SOO używają `evaluate()`."
        )


@dataclass(slots=True)
class PathProblem:
    """Pakiet danych przekazywany do `IPathOptimizer.optimize`.

    Hermetyzuje całe „pytanie" które optimizer ma rozwiązać. Optimizer nie
    powinien sięgać do żadnego globalnego stanu — wszystko ma być tutaj.
    """

    context: EvasionContext
    predictor: IObstaclePredictor
    fitness: IFitnessEvaluator
    path_repr: IPathRepresentation
    initial_population: NDArray[np.float64] | None = None
    """Pre-wygenerowana populacja początkowa (pop_size, gene_dim).

    Gwarantuje warunek ceteris paribus: wszystkie online optimizery
    (MSFOA, SSA, OOA, NSGA-III) startują z identycznego zbioru osobników
    niezależnie od wewnętrznego backendu PRNG frameworka (mealpy PCG64,
    pymoo MT19937, custom Generator).

    Generowana przez ``GenericOptimizingAvoidance`` z ``U(lb, ub)`` na
    wspólnym ``np.random.Generator``.  ``None`` → optimizer używa
    własnej domyślnej inicjalizacji (backward-compatible).
    """


@dataclass(slots=True)
class OptimizationResult:
    """Wynik `IPathOptimizer.optimize`.

    `waypoints` może być `None` przy `status="timed_out"` lub `"failed"`.
    `extra` to tablica diagnostyczna per-optimizer (np. `axis_chosen`,
    `iterations`, `fallback_used`) — wykorzystywana przez `EvasionPlan`
    do logowania.
    """

    waypoints: NDArray[np.float64] | None
    elapsed_s: float
    status: Literal["ok", "timed_out", "failed"]
    extra: dict[str, Any] = field(default_factory=dict)


class IPathOptimizer(ABC):
    """Silnik optymalizujący ścieżkę uniku w zadanym budżecie czasu.

    Kontrakt budżetu (TWARDY):
      - `optimize()` MUSI honorować `budget` przez kooperacyjne wołania
        `budget.check_or_raise()` w hot-loopie (granularność per-optimizer).
      - W razie `BudgetExceeded` MUSI zwrócić `OptimizationResult(status="timed_out")`
        — można zwrócić częściowy `best_so_far` lub `None`.
      - Optimizer NIE odpowiada za zewnętrzny SIGALRM hard-kill — to zapewnia
        wrapper `GenericOptimizingAvoidance.compute_evasion_plan`.

    Implementacje produkcyjne: NSGA3/MSFFOA/SSA/OOA Online — z populacjami
    10–30, max 5–15 generacji, by zmieścić się w ~1 s budżetu.
    """

    @property
    def population_size(self) -> int:
        """Rozmiar populacji wymagany przez optimizer.

        Zwracane przez każdy ewolucyjny optimizer (MSFOA, SSA, OOA, NSGA-III).
        ``GenericOptimizingAvoidance`` używa tej wartości do pre-generacji
        ``PathProblem.initial_population`` o wymiarze ``(population_size, gene_dim)``.

        Wartość ``0`` oznacza, że optimizer nie wymaga pre-generowanej populacji
        (np. deterministyczny A*, hill-climbing).
        """
        return 0

    @abstractmethod
    def optimize(
        self,
        problem: PathProblem,
        budget: TimeBudget,
    ) -> OptimizationResult:
        """Wybierz najlepsze waypointy uniku w ramach `budget`.

        Args:
            problem: Pakiet danych (kontekst, predyktor, fitness, reprezentacja
                ścieżki, opcjonalna pre-wygenerowana populacja).
            budget: Kooperacyjny limit czasu — sprawdzaj
                `budget.check_or_raise()` w hot-loopie.

        Returns:
            `OptimizationResult` z `waypoints` (`None` przy `timed_out`/`failed`),
            `elapsed_s`, `status` i diagnostyką w `extra`.
        """
        ...
