"""Common-contract dataclasses dla logowania metryk optymalizacji online.

Każdy z 4 algorytmów avoidance (`NSGA3Avoidance`, `OOAAvoidance`,
`SSAAvoidance`, `MSFFOAAvoidance`) MUSI produkować rekordy zgodne z
`OnlineOptimizationRecord` po każdym wywołaniu `compute_evasion_plan`.
Zapewnia to porównywalność metryki w pracy magisterskiej (per-trigger
summary) oraz `ConvergenceSample` (long-format, per-generation fitness
trace) dla wykresów konwergencji.

Pliki wyjściowe runa (zapisywane przez `SimulationLogger.save()`):
  - `online_optimization.csv` — N wierszy = liczba triggerów uniku.
  - `convergence_traces.csv` — N×G wierszy (per generacja w trace).

Outcome (`pos_err_at_rejoin_m` etc.) jest wypełniane PÓŹNIEJ — przez
`update_online_optimization_outcome` po BLEND_END / collision.

Reference: standard "online optimization benchmark logging" w pracach typu
Mehdi 2017, Bing 2018 — zawsze raportowane: `evaluations_completed`,
`wallclock_s`, `best_fitness`, `success_rate`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List


# Sentinel'e dla outcome (pre-rejoin/collision).
OUTCOME_PENDING = "pending"
OUTCOME_REJOINED_OK = "rejoined_ok"
OUTCOME_COLLIDED_GROUND = "collided_ground"
OUTCOME_COLLIDED_DRONE = "collided_drone"
OUTCOME_COLLIDED_OBSTACLE = "collided_obstacle"
OUTCOME_NEVER_REJOINED = "never_rejoined"


# Sentinel'e dla niewypełnionych pól (NIE używamy None — pandas/csv lepiej
# obsługuje float NaN i puste stringi).
_NAN = float("nan")
_EMPTY = ""


@dataclass
class OnlineOptimizationRecord:
    """Per-trigger summary online avoidance — pól identyfikacja + status + outcome.

    Wypełniany dwustopniowo:
    1. Przy `plan_built` (z `BaseAvoidance.compute_evasion_plan`) —
       identyfikacja, grupa A (optimizer summary), grupa B (decision).
       Outcome inicjalizowany na `OUTCOME_PENDING`.
    2. Przy BLEND_END / collision (`update_online_optimization_outcome`) —
       wypełnia grupę D z obserwowanego rezultatu.

    Klucz `(drone_id, trigger_time)` umożliwia update wiersza w buforze.
    """
    # === Identyfikacja ===
    run_id: str
    drone_id: int
    trigger_time: float                 # PK component
    algorithm: str                      # SSA, OOA, MSFOA, NSGA3

    # === Grupa A — optimizer summary ===
    status: str                         # ok / timed_out / failed
    reason: str                         # ok / no_feasible / budget_exceeded / ...
    best_fitness: float
    evaluations_completed: int
    generations_completed: int
    wallclock_s: float
    time_budget_s: float                # config copy (dla weryfikacji)

    # === Grupa B — decision (po plan_built; jeśli plan is None, pola sentinel) ===
    chosen_axis: str = _EMPTY           # right/left/up/down/none
    plan_waypoints_json: str = _EMPTY   # JSON list of [x,y,z]
    plan_total_duration_s: float = _NAN
    plan_arc_length_m: float = _NAN

    # === Grupa D — outcome (wypełniane PÓŹNIEJ) ===
    outcome: str = OUTCOME_PENDING
    pos_err_at_rejoin_m: float = _NAN
    vel_err_at_rejoin_mps: float = _NAN
    time_to_rejoin_s: float = _NAN


@dataclass
class ConvergenceSample:
    """Pojedyncza próbka konwergencji (1 generacja 1 triggera) w formacie long-form.

    Klucz kompozytowy `(drone_id, trigger_time, generation)`; FK do
    `OnlineOptimizationRecord` przez `(drone_id, trigger_time)`.
    """
    run_id: str
    drone_id: int
    trigger_time: float                 # FK
    algorithm: str
    generation: int                     # 0-indexed
    best_fitness: float


def online_record_headers() -> List[str]:
    """Lista nazw kolumn `online_optimization.csv` (zgodna z dataclass)."""
    return [f.name for f in fields(OnlineOptimizationRecord)]


def convergence_sample_headers() -> List[str]:
    """Lista nazw kolumn `convergence_traces.csv`."""
    return [f.name for f in fields(ConvergenceSample)]


def record_to_dict(record: OnlineOptimizationRecord | ConvergenceSample) -> Dict[str, Any]:
    """Konwersja dataclass → dict (kompatybilny z `csv.DictWriter` i pandas)."""
    return asdict(record)
