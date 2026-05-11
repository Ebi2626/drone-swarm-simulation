"""Dwuwarstwowy budżet czasu dla optymalizatorów online avoidance.

`TimeBudget` to kooperacyjny limit (optymalizator dobrowolnie sprawdza
`check_or_raise()`); `hard_deadline` jest zewnętrznym wyłącznikiem opartym
o `SIGALRM` na wypadek, gdy kooperacja zawiedzie.
"""
from __future__ import annotations

import logging
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass


logger = logging.getLogger(__name__)


class BudgetExceeded(Exception):
    """Wyjątek wyrzucany przez `TimeBudget.check_or_raise` po przekroczeniu limitu.

    Optymalizatory powinny łapać go u góry `optimize()` i zwracać
    `OptimizationResult(status="timed_out", waypoints=None | best_so_far)`.
    """


class HardDeadlineExceeded(Exception):
    """Wyjątek z handlera `SIGALRM` w `hard_deadline` — twardy wyłącznik czasowy.

    Sygnał, że limit kooperacyjny zawiódł (bug w optymalizatorze, deadlock
    w kodzie natywnym, biblioteka ignoruje `max_time`). Normalny przebieg
    nigdy nie powinien tu trafić.
    """


@dataclass(slots=True)
class TimeBudget:
    """Kooperacyjny budżet czasu dla iteracyjnych optymalizatorów.

    Optymalizator wywołuje `check_or_raise()` co N iteracji (granularność
    zdefiniowana w jego konfiguracji). Pojedyncze sprawdzenie kosztuje
    ~50 ns, można je wstawiać w hot-loopie bez wpływu na wydajność.
    Twardy sufit czasowy zapewnia `hard_deadline()` (SIGALRM) na poziomie
    `GenericOptimizingAvoidance`.

    Pola:
        max_seconds: Limit czasu od `start` [s].
        start: Znacznik startu (`time.perf_counter()`).
    """

    max_seconds: float
    start: float

    @classmethod
    def start_now(cls, max_seconds: float) -> "TimeBudget":
        """Utwórz budżet zaczynający odliczanie od chwili wywołania.

        Args:
            max_seconds: Limit czasu [s].
        """
        return cls(max_seconds=float(max_seconds), start=time.perf_counter())

    @property
    def elapsed(self) -> float:
        """Czas wallclock od `start` [s]."""
        return time.perf_counter() - self.start

    @property
    def remaining(self) -> float:
        """Pozostały budżet [s]; może być ujemny po przekroczeniu limitu."""
        return self.max_seconds - self.elapsed

    def exceeded(self) -> bool:
        """`True` gdy `elapsed ≥ max_seconds`."""
        return self.elapsed >= self.max_seconds

    def check_or_raise(self) -> None:
        """Wyrzuć `BudgetExceeded`, jeśli budżet jest wyczerpany."""
        if self.elapsed >= self.max_seconds:
            raise BudgetExceeded(
                f"Cooperative budget {self.max_seconds:.3f}s exceeded "
                f"(elapsed={self.elapsed:.3f}s)"
            )


@contextmanager
def hard_deadline(seconds: float):
    """Twardy wyłącznik czasowy oparty o `SIGALRM` (zewnętrzna warstwa budżetu).

    Po przekroczeniu `seconds` handler sygnału wyrzuca `HardDeadlineExceeded`.
    Timer jest kasowany po wyjściu z bloku, oryginalny handler `SIGALRM`
    przywracany — bezpieczne zagnieżdżanie.

    Args:
        seconds: Limit czasu [s]; `≤ 0` ⇒ blok bez ograniczenia.

    Ograniczenia:
        - `SIGALRM` działa tylko w wątku głównym procesu (akceptowalne, bo
          unik leci w wątku głównym workera).
        - Długo wykonujący się kod natywny (numba `@njit` bez powrotu do
          interpretera) opóźni odpalenie sygnału do powrotu z natywki.
        - Brak `SIGALRM`/`setitimer` (Windows) ⇒ blok wykonuje się bez
          ograniczenia; ostrzeżenie logowane raz.

    Raises:
        HardDeadlineExceeded: Po przekroczeniu czasu wewnątrz bloku.

    Przykład:
        ```python
        with hard_deadline(1.5):
            result = optimizer.optimize(problem, budget=TimeBudget.start_now(1.0))
        ```
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
