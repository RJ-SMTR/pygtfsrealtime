import logging
import threading
import time
from collections.abc import Callable
from typing import Generic, TypeVar

from pygtfsrealtime.exceptions import FatalConfigurationError
from pygtfsrealtime.settings import LoopSchedule

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SnapshotStore(Generic[T]):
    """Holds the current value of one published snapshot behind a narrow lock.

    The lock only ever protects the reference swap/read (one line) - never the
    work that produces or consumes the snapshot - so producer and consumers
    can run concurrently without blocking each other beyond that single
    assignment.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: T | None = None

    def get(self) -> T | None:
        with self._lock:
            return self._value

    def set(self, value: T) -> None:
        with self._lock:
            self._value = value


def run_periodic(
    schedule: LoopSchedule,
    work_fn: Callable[[], None],
    stop_event: threading.Event | None = None,
) -> None:
    """Run `work_fn` forever on a fixed schedule, blocking the calling thread.

    Args:
        schedule: interval and accounting/missed-deadline behavior.
        work_fn: called once per cycle; exceptions are logged and swallowed
            so one bad cycle doesn't kill the loop, except
            FatalConfigurationError, which is let through so it can stop the
            whole engine instead.
        stop_event: optional; omit it to run with no external stop support.
            When given, sleeping goes through stop_event.wait(timeout=x)
            instead of time.sleep(x), so a stop request interrupts an
            in-progress sleep immediately instead of waiting out the full
            interval; the top of the loop also checks stop_event before
            starting another cycle, so a stop requested while work_fn is
            running is honored on the next iteration instead of starting one
            more cycle first.
    """

    def _sleep(seconds: float) -> None:
        if stop_event is not None:
            stop_event.wait(timeout=seconds)
        else:
            time.sleep(seconds)

    next_run = time.time()
    while True:
        if stop_event is not None and stop_event.is_set():
            return

        try:
            work_fn()
        except FatalConfigurationError:
            raise
        except Exception:
            logger.exception("Periodic loop cycle raised an exception")

        if schedule.accounting_mode == "fixed_delay":
            # Next tick is always "now + interval", computed after work_fn
            # returns - execution time is never absorbed, so a deadline can
            # never be missed here (on_missed_deadline doesn't apply).
            _sleep(schedule.interval)
            continue

        next_run += schedule.interval
        sleep_for = next_run - time.time()
        if sleep_for > 0:
            _sleep(sleep_for)
            continue

        if schedule.on_missed_deadline == "immediate":
            next_run = time.time()
        elif schedule.on_missed_deadline == "wait_full_interval":
            _sleep(schedule.interval)
            next_run = time.time()
        else:  # skip_to_next_tick
            now = time.time()
            missed_ticks = (now - next_run) // schedule.interval + 1
            next_run += missed_ticks * schedule.interval
            _sleep(next_run - now)


def run_conditional(
    schedule: LoopSchedule,
    work_fn: Callable[[], float],
    wake_event: threading.Event,
    stop_event: threading.Event | None = None,
) -> None:
    """Sibling of run_periodic for a loop whose next-run delay is computed
    fresh each cycle from data the cycle itself just produced (e.g.
    TripWindowLoop's window_end), instead of being a constant tick.

    Args:
        schedule: `schedule.interval` only seeds the very first cycle,
            before work_fn has run once - every cycle after that, the sleep
            duration comes from work_fn's own return value.
            `schedule.accounting_mode`/`schedule.on_missed_deadline` keep
            their run_periodic meaning, applied to that dynamic interval.
            `schedule` is only ever read here, never written, so the same
            `LoopSchedule` instance a domain function reads for an unrelated
            purpose (e.g. `build_trips_snapshot` reading
            `settings.trip_window_loop_schedule.interval` as a window
            length) can be passed in safely.
        work_fn: called once per cycle, returning the number of seconds
            until the next cycle should run; exceptions are logged and
            swallowed so one bad cycle doesn't kill the loop, except
            FatalConfigurationError, which is let through so it can stop the
            whole engine instead.
        wake_event: required (not optional like run_periodic's stop_event) -
            a conditional loop only exists because something needs to wake
            it early; if that need doesn't apply, use run_periodic instead.
        stop_event: optional; omit it to run with no external stop support.
            Sleeping still goes through wake_event - a caller that wants
            stop() to interrupt an in-progress sleep immediately must also
            .set() wake_event itself (e.g. GTFSRealtimeEngine.stop() does
            both). stop_event is only checked at the top of the loop, before
            calling work_fn again.
    """

    def _sleep(seconds: float) -> None:
        if wake_event.wait(timeout=seconds):
            wake_event.clear()

    interval = schedule.interval
    next_run = time.time()
    while True:
        if stop_event is not None and stop_event.is_set():
            return

        try:
            interval = work_fn()
        except FatalConfigurationError:
            raise
        except Exception:
            logger.exception("Conditional loop cycle raised an exception")

        if schedule.accounting_mode == "fixed_delay":
            _sleep(interval)
            continue

        next_run += interval
        sleep_for = next_run - time.time()
        if sleep_for > 0:
            _sleep(sleep_for)
            continue

        if schedule.on_missed_deadline == "immediate":
            next_run = time.time()
        elif schedule.on_missed_deadline == "wait_full_interval":
            _sleep(interval)
            next_run = time.time()
        else:  # skip_to_next_tick
            now = time.time()
            missed_ticks = (now - next_run) // interval + 1
            next_run += missed_ticks * interval
            _sleep(next_run - now)
