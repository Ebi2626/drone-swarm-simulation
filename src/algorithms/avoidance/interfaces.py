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


# --------------------------------------------------------------------------- #
# 4 abstrakcyjne interfejsy z planu (Strategy Pattern, Hydra-instancjowalne)  #
# --------------------------------------------------------------------------- #


class IObstaclePredictor(ABC):
    """Estymator stanu kinematycznego przeszkody w przyszłości.

    Faza 1 (`ConstantVelocityPredictor`): liniowa ekstrapolacja `pos += vel * t`.
    Faza 2+: warianty z modelami wyższego rzędu (Kalman, IMM) — wymienialne
    w yaml bez zmian w optimizerze.
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

    Faza 1 (`BSplineSmoother`): wejście = surowe waypointy 3D z optimizera
    (typowo z A*); wyjście = wygładzony BSpline 3D, gotowy dla `EvasionPlan`.

    Faza 2 (`BSplineYZGenes`): dekoder genów (`decode_genes`) — wektor decyzji
    (Δy, Δz) osi YZ → BSpline, X liniowo interpolowane od start do rejoin.
    Kontrakt rozszerzony o `gene_dim(ctx)` i `gene_bounds(ctx)` — używane przez
    optymalizatory ewolucyjne przy budowie problemu (mealpy/pymoo bounds).
    """

    @abstractmethod
    def waypoints_to_spline(
        self,
        waypoints: NDArray[np.float64],
        context: EvasionContext,
        *,
        axis_name: str | None = None,
    ) -> "BSplineTrajectory | None":
        """Zbuduj wykonalny BSpline z surowych waypointów (ścieżka A*).

        :param axis_name: opcjonalny hint o wybranej osi uniku ("up"/"down"/
            "right"/"left") — np. `BSplineSmoother` używa go do Z-linearyzacji
            przy axis lateralnych (regresja Fazy 8.1). `BSplineYZGenes` ignoruje.
        :return: BSpline lub `None` jeśli budowa się nie powiodła
                 (np. zdegenerowana sekwencja punktów, naruszenie warunków
                 brzegowych przy rejoin-point).
        """

    # --------- Kontrakt evolutionary (Faza 2) — domyślnie NotImplementedError --------- #

    def decode_genes(
        self,
        genes: NDArray[np.float64],
        context: EvasionContext,
    ) -> "BSplineTrajectory | None":
        """Zdekoduj wektor genów (decyzje optymalizatora ewolucyjnego) na BSpline.

        Domyślnie nie zaimplementowane — `BSplineSmoother` (używany przez
        `AStarOptimizer`) nie ma sensu dla genów. `BSplineYZGenes` w Fazie 2
        nadpisuje tę metodę.

        :param genes: wektor decyzji o długości `gene_dim(context)`. Geometria
                      kodowania (np. (Δy_1, …, Δy_K, Δz_1, …, Δz_K) lub interleaved)
                      jest sprawą konkretnej implementacji.
        :return: BSpline lub `None` przy niepowodzeniu (np. degeneracja, naruszenie
                 granic). Optimizer powinien w fitness penalizować takie wyniki.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.decode_genes() — używane wyłącznie przez "
            f"optymalizatory ewolucyjne (Faza 2). AStar nie używa."
        )

    def gene_dim(self, context: EvasionContext) -> int:
        """Wymiar wektora decyzji dla danego kontekstu (zazwyczaj zależy od
        liczby inner waypointów × liczby zmiennych osiowych).

        Defaultowo `NotImplementedError` — patrz `decode_genes` rationale.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.gene_dim() — używane wyłącznie przez "
            f"optymalizatory ewolucyjne (Faza 2). AStar nie używa."
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
            f"optymalizatory ewolucyjne (Faza 2). AStar nie używa."
        )


class IFitnessEvaluator(ABC):
    """Funkcja kosztu / oceny wariantu trajektorii uniku.

    Faza 1 (`AxisBiasFitness`): wykorzystywany jest tylko `axis_score` —
    heurystyka wyboru półpłaszczyzny uniku (right/left/up/down) zgodna
    z `AStarOptimizer` (preferowana oś + bias anty-prędkościowy).

    Faza 2 (`WeightedSumFitness`) dodaje `evaluate(spline, ctx, predictor)` —
    skalarna funkcja kosztu dla optymalizatorów ewolucyjnych (SSA, OOA, MSFOA),
    oraz `evaluate_components` zwracające 4-wektor `[c_safety, c_energy, c_jerk,
    c_symmetry]` dla NSGA-III multi-obj.

    Atrybuty bias (`bias_preferred`, `bias_perpendicular`, `bias_oppose`) są
    częścią kontraktu — używane przez `AStarOptimizer` do parametryzacji
    `_BudgetAwareGridSearch`. Domyślne wartości neutralne (1.0).
    """

    # Domyślne biasy osiowe — używane przez `AStarOptimizer._BudgetAwareGridSearch`.
    # Konkretne implementacje (`AxisBiasFitness`) przesłaniają tymi z yaml.
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

        Domyślnie nie zaimplementowane — używane wyłącznie przez optymalizatory
        ewolucyjne (Faza 2). AStar wybiera punkty kontrolne na podstawie
        biasów osi (`axis_score`) i w pętli expansion nie potrzebuje fitness
        całej trajektorii.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.evaluate() — fitness skalar dla BSplinu "
            f"jest częścią Fazy 2 (NSGA-III/MSFFOA/SSA/OOA). AStar nie używa."
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

        Domyślnie nie zaimplementowane — `WeightedSumFitness` (Faza 2) nadpisuje.
        Dla niezdekodowalnych kandydatów MUSI zwrócić wektor sentinelowy
        (np. `[1e9]*4`), tak by optymalizator wybrakował osobnika.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.evaluate_components() — multi-obj fitness "
            f"jest częścią Fazy 2 (NSGA-III). SSA/OOA/MSFOA używają `evaluate()`."
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

    Faza 1: tylko `AStarOptimizer`. Faza 2: NSGA3/MSFFOA/SSA/OOA Online — z
    populacjami 10–30, max 5–15 generacji, by zmieścić się w 1 s.
    """

    @abstractmethod
    def optimize(
        self,
        problem: PathProblem,
        budget: TimeBudget,
    ) -> OptimizationResult:
        ...
