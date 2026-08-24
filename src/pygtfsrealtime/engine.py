import io
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pyproj

from pygtfsrealtime.contracts import validate_cache_result, validate_gps_data_result
from pygtfsrealtime.exceptions import FatalConfigurationError
from pygtfsrealtime.models import GPSEntry
from pygtfsrealtime.realtime.fsm import TransitionCallback
from pygtfsrealtime.realtime.ingest import GPSIngester
from pygtfsrealtime.realtime.loop import FSMLoop
from pygtfsrealtime.runner import SnapshotStore, run_conditional, run_periodic
from pygtfsrealtime.schedule.ingest import GTFSScheduleIngester
from pygtfsrealtime.schedule.loop import GTFSScheduleLoop
from pygtfsrealtime.schedule.snapshot import GtfsSnapshot
from pygtfsrealtime.settings import Settings
from pygtfsrealtime.trip_window.compute import TripsSnapshot
from pygtfsrealtime.trip_window.loop import TripWindowLoop

logger = logging.getLogger(__name__)


def _resolve_gtfs_callback(
    gtfs_schedule: Callable[[], io.BytesIO] | str | Path,
) -> Callable[[], io.BytesIO]:
    """Normalize the `gtfs_schedule` constructor argument to a callback.

    Args:
        gtfs_schedule: either a callback returning a file, or a local file
            path.

    Returns:
        A callback returning a file, ready for `GTFSScheduleIngester`.

    Raises:
        FileNotFoundError: if `gtfs_schedule` is a path and it doesn't exist
            - checked right here (construction time), not discovered missing
            inside a background thread's first cycle.
    """
    if callable(gtfs_schedule):
        return gtfs_schedule

    path = Path(gtfs_schedule)
    if not path.is_file():
        raise FileNotFoundError(f"gtfs_schedule path does not exist: {path}")
    return lambda: GTFSScheduleIngester.read_gtfs_schedule_locally(path)


class GTFSRealtimeEngine:
    """Public entry point of pygtfsrealtime: wires GTFSScheduleLoop,
    TripWindowLoop, and FSMLoop into one runnable unit. Construct it with
    your 5 callbacks (gtfs_schedule, ingest_gps_data, publish_protobuf, and
    the optional set_cache/get_cache pair), then call `run()` to block until
    interrupted, or `start()`/`stop()` for programmatic control (tests, an
    app with its own signal handling, etc).

    Validates what it can eagerly, at construction time, rather than letting
    it fail silently inside a background thread later - run_periodic/
    run_conditional swallow every work_fn exception into a log line, which an
    operator watching only stdout/exit codes could easily miss.
    """

    def __init__(
        self,
        gtfs_schedule: Callable[[], io.BytesIO] | str | Path,
        ingest_gps_data: Callable[[], list[GPSEntry]],
        publish_protobuf: Callable[[bytes], None],
        set_cache: Callable[[bytes], None] | None = None,
        get_cache: Callable[[], bytes | None] | None = None,
        on_transition: TransitionCallback | None = None,
        settings: Settings | None = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        """Build the engine and validate what's known at construction time.

        Args:
            gtfs_schedule: a callback returning a file, or a local file path.
            ingest_gps_data: called each FSM cycle; must return a
                `list[GPSEntry]`.
            publish_protobuf: called each FSM cycle with the serialized
                GTFS-RT `bytes`.
            set_cache: optional; persists the FSM/vehicle-window state
                between cycles.
            get_cache: optional; restores that state on startup. Must be
                given together with `set_cache` for the cache to round-trip.
            on_transition: optional; called once per completed FSM cycle
                (not when a cycle is skipped for lack of a TripsSnapshot)
                with the list of every vehicle's `TransitionEvent`
                (`vehicle_id, old_state, new_state, reason, observation`)
                from that cycle - possibly empty if no vehicle reported -
                e.g. to alert on a specific `TransitionReason` such as
                `OFF_PATH`.
            settings: library configuration; defaults to `Settings()`.
            now_fn: clock injection point, mainly for tests.

        Raises:
            TypeError: if `ingest_gps_data`/`publish_protobuf`/`on_transition`
                aren't callable, or if
                `settings.trip_matching`/`settings.projection` is unset or
                invalid - checked here rather than deep inside a background
                thread's first cycle, where
                `run_periodic`/`run_conditional`'s blanket exception logging
                could otherwise hide it.
        """
        for name, func in (
            ("ingest_gps_data", ingest_gps_data),
            ("publish_protobuf", publish_protobuf),
        ):
            if not callable(func):
                raise TypeError(f"{name} must be a function, received: {func!r}")
        if on_transition is not None and not callable(on_transition):
            raise TypeError(f"on_transition must be a function, received: {on_transition!r}")
        fetch_gtfs_schedule = _resolve_gtfs_callback(gtfs_schedule)

        self.settings = settings if settings is not None else Settings()
        if self.settings.trip_matching is None:
            raise TypeError(
                "settings.trip_matching is not set - construct one with "
                "pygtfsrealtime.MatchingStrategy(key=...) and pass it as "
                "Settings(trip_matching=...) before constructing GTFSRealtimeEngine."
            )
        if self.settings.projection is None:
            raise TypeError(
                "settings.projection is not set - pass a valid UTM CRS identifier, "
                'e.g. Settings(projection="EPSG:32723"), before constructing '
                "GTFSRealtimeEngine."
            )
        try:
            pyproj.CRS(self.settings.projection)
        except pyproj.exceptions.CRSError as exc:
            raise TypeError(
                f"settings.projection={self.settings.projection!r} is not a valid CRS: {exc}"
            ) from exc
        if (set_cache is None) != (get_cache is None):
            logger.warning(
                "Only one of set_cache/get_cache was provided - the cache will never round-trip."
            )

        self.now_fn = now_fn
        self._stop_event = threading.Event()
        self._new_gtfs_event = threading.Event()
        self._gtfs_snapshot_store: SnapshotStore[GtfsSnapshot] = SnapshotStore()
        self._trips_snapshot_store: SnapshotStore[TripsSnapshot] = SnapshotStore()

        # ingester carries Settings (projection, buffer/zone thresholds) and
        # does no I/O itself at construction time - only fetch()/ingest()
        # (called from the GTFS loop's own thread) actually touch the network.
        ingester = GTFSScheduleIngester(callback=fetch_gtfs_schedule, settings=self.settings)

        def wrapped_ingest_gps_data():
            return validate_gps_data_result(ingest_gps_data())

        gps_ingester = GPSIngester(callback=wrapped_ingest_gps_data, settings=self.settings)
        wrapped_get_cache = (lambda: validate_cache_result(get_cache())) if get_cache else None

        self._gtfs_loop = GTFSScheduleLoop(
            ingester, self._gtfs_snapshot_store, self._new_gtfs_event
        )
        self._trip_window_loop = TripWindowLoop(
            self.settings, self._gtfs_snapshot_store, self._trips_snapshot_store, now_fn=self.now_fn
        )
        self._fsm_loop = FSMLoop(
            self.settings,
            self._trips_snapshot_store,
            gps_ingester,
            publish_protobuf,
            get_cache=wrapped_get_cache,
            set_cache=set_cache,
            on_transition=on_transition,
            now_fn=self.now_fn,
        )
        self._threads: list[threading.Thread] = []
        self._fatal_exception: BaseException | None = None

    def _run_loop(self, runner_fn: Callable[..., None], *args) -> None:
        """Thread target wrapper: runs `runner_fn(*args)` (run_periodic or
        run_conditional), and if it exits via `FatalConfigurationError` -
        which those runners deliberately let through instead of swallowing -
        stops every loop rather than leaving this one thread dead while its
        siblings keep running against stale/absent state.
        """
        try:
            runner_fn(*args)
        except FatalConfigurationError as exc:
            logger.error(
                "%s hit a fatal configuration error - stopping the engine",
                threading.current_thread().name,
            )
            self._fatal_exception = exc
            self._stop_event.set()
            self._new_gtfs_event.set()

    def start(self) -> None:
        """Spawn one background thread per cycle and return immediately.

        Raises:
            RuntimeError: if the engine is already running.
        """
        if self._threads:
            raise RuntimeError("GTFSRealtimeEngine.start() called while already running")

        self._stop_event.clear()
        self._fatal_exception = None
        self._threads = [
            threading.Thread(
                target=self._run_loop,
                args=(
                    run_periodic,
                    self.settings.gtfs_loop_schedule,
                    self._gtfs_loop.run_once,
                    self._stop_event,
                ),
                name="pygtfsrealtime-gtfs-loop",
                daemon=True,
            ),
            threading.Thread(
                target=self._run_loop,
                args=(
                    run_conditional,
                    self.settings.trip_window_loop_schedule,
                    self._trip_window_loop.next_interval,
                    self._new_gtfs_event,
                    self._stop_event,
                ),
                name="pygtfsrealtime-trip-window-loop",
                daemon=True,
            ),
            threading.Thread(
                target=self._run_loop,
                args=(
                    run_periodic,
                    self.settings.fsm_loop_schedule,
                    self._fsm_loop.run_once,
                    self._stop_event,
                ),
                name="pygtfsrealtime-fsm-loop",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Signal every loop to stop and join its thread.

        Safe to call before start() or more than once.

        Args:
            timeout: max seconds to wait per thread. A Python thread can't
                be forcibly killed - if a thread is still alive after
                `timeout`, this logs a warning and returns rather than
                blocking forever or pretending termination is guaranteed.
        """
        self._stop_event.set()
        self._new_gtfs_event.set()  # unblocks the trip-window loop's in-progress sleep
        for thread in self._threads:
            thread.join(timeout)
            if thread.is_alive():
                logger.warning("%s did not stop within timeout=%r", thread.name, timeout)
        self._threads = []

    def run(self) -> None:
        """Starts every loop and blocks until interrupted (Ctrl+C) or a loop
        hits a fatal configuration error, then stops them. The simplest way
        to run pygtfsrealtime as a standalone process.

        Raises:
            FatalConfigurationError: if a background loop discovers required
                configuration is missing/invalid only after it started
                running (e.g. the GTFS feed's timezone - see
                pygtfsrealtime.schedule.snapshot.resolve_gtfs_timezone) -
                re-raised here, on the main thread, instead of being left to
                retry forever inside its own background thread.
        """
        self.start()
        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
        if self._fatal_exception is not None:
            raise self._fatal_exception
