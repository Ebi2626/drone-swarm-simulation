from __future__ import annotations

import logging
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass


logger = logging.getLogger(__name__)


class BudgetExceeded(Exception):
    """Cooperative budget exhaustion — raised by `TimeBudget.check_or_raise`.

    Optimizers powinny tę wyjątek łapać u góry własnego `optimize()` i zwracać
    `OptimizationResult(status="timed_out", waypoints=None | best_so_far)`.
    """


class HardDeadlineExceeded(Exception):
    """Outer SIGALRM circuit breaker — bezpieczeństwo na wypadek gdy cooperative
    check zawiedzie (bug w optimizerze, deadlock w native code, mealpy ignoruje
    `max_time` itd.). Nigdy nie powinno się zdarzyć w normalnym przebiegu.
    """


@dataclass(slots=True)
class TimeBudget:
    """Kooperacyjny budżet czasu dla iteracyjnych optymalizatorów.

    Kontrakt:
      - `optimize()` woła `budget.check_or_raise()` co N iteracji (granularność
        zdefiniowana w configu — `budget_check_every_n_nodes` dla AStara, natywne
        `max_time` dla mealpy).
      - Sprawdzenie kosztuje ~50 ns (jeden odczyt `perf_counter` + odejmowanie),
        więc można dorzucić bez obawy o hot-loop.

    Ten obiekt jest *kooperacyjny* — wymaga współpracy optymalizatora. Twardy
    sufit gwarantujący przerwanie w razie buga zapewnia `hard_deadline()`
    (SIGALRM) na poziomie wrappera `GenericOptimizingAvoidance`.
    """

    max_seconds: float
    start: float

    @classmethod
    def start_now(cls, max_seconds: float) -> "TimeBudget":
        return cls(max_seconds=float(max_seconds), start=time.perf_counter())

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    @property
    def remaining(self) -> float:
        return self.max_seconds - self.elapsed

    def exceeded(self) -> bool:
        return self.elapsed >= self.max_seconds

    def check_or_raise(self) -> None:
        if self.elapsed >= self.max_seconds:
            raise BudgetExceeded(
                f"Cooperative budget {self.max_seconds:.3f}s exceeded "
                f"(elapsed={self.elapsed:.3f}s)"
            )


@contextmanager
def hard_deadline(seconds: float):
    """Outer circuit breaker oparty o `SIGALRM`.

    Kontrakt:
      - Ustawia `setitimer(ITIMER_REAL, seconds)` w momencie wejścia w blok.
      - Po przekroczeniu czasu sygnał odpala handler → `HardDeadlineExceeded`.
      - Po wyjściu z bloku timer kasowany (`setitimer(ITIMER_REAL, 0)`),
        oryginalny handler `SIGALRM` przywracany — bezpieczne zagnieżdżanie.

    Ograniczenia:
      - SIGALRM działa tylko w **main-thread procesu**. W naszej topologii
        (subprocess wrapper z Kroku 5c) avoidance leci w main-thread workera,
        więc OK. Per-proces — 6 workerów ma 6 niezależnych alarmów, brak
        crosstalk.
      - Long-running native code (numba @njit bez powrotu do Pythona) opóźnia
        odpalenie sygnału aż do powrotu do interpretera. Nasze kernele są
        single-step (μs), więc w praktyce nie problem.
      - `seconds <= 0` lub brak `signal.SIGALRM` (Windows) → no-op, blok wykonuje
        się bez ograniczenia. Logujemy WARNING raz (na pierwsze takie wywołanie).

    Użycie::

        with hard_deadline(1.5):
            result = optimizer.optimize(problem, budget=TimeBudget.start_now(1.0))
    """
    if seconds <= 0.0:
        yield
        return

    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        # Windows / platforma bez SIGALRM. Cooperative pozostaje single defense line.
        if not getattr(hard_deadline, "_warned_no_sigalrm", False):
            logger.warning(
                "hard_deadline: SIGALRM niedostępny na tej platformie — twardy "
                "sufit czasowy wyłączony, polegamy wyłącznie na cooperative checkach."
            )
            hard_deadline._warned_no_sigalrm = True  # type: ignore[attr-defined]
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001
        raise HardDeadlineExceeded(
            f"Hard deadline {seconds:.3f}s exceeded — circuit breaker tripped"
        )

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        # Anuluj timer ZANIM przywrócimy stary handler — inaczej race window,
        # w którym alarm może odpalić oryginalny handler (np. domyślny Pythonowy
        # KeyboardInterrupt-equivalent) zanim go zdejmiemy.
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
