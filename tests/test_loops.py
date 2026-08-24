import io
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from pygtfsrealtime.models import GPSEntry
from pygtfsrealtime.pb.gtfs_realtime_pb2 import FeedMessage  # type: ignore[attr-defined]
from pygtfsrealtime.realtime.fsm import VehicleState
from pygtfsrealtime.realtime.ingest import GPSIngester
from pygtfsrealtime.realtime.loop import FSMLoop
from pygtfsrealtime.runner import SnapshotStore, run_conditional, run_periodic
from pygtfsrealtime.schedule.ingest import GTFSScheduleIngester
from pygtfsrealtime.schedule.loop import GTFSScheduleLoop, _hash_raw_gtfs
from pygtfsrealtime.schedule.snapshot import GtfsSnapshot, build_gtfs_snapshot
from pygtfsrealtime.settings import LoopSchedule, MatchingStrategy, Settings
from pygtfsrealtime.trip_window.compute import TripsSnapshot
from pygtfsrealtime.trip_window.loop import TripWindowLoop
from tests.gtfs_data import build_gtfs_zip


class FakeIngester:
    """Stands in for GTFSScheduleIngester so GTFSScheduleLoop's own hash-skip/
    publish/event control flow can be tested without re-exercising the real
    parse/validate pipeline (already covered by tests/test_ingest.py).
    """

    def __init__(self, gtfs_files: dict):
        self.raw_schedule = b"v1"
        self.gtfs_files = gtfs_files
        self.fetch_calls = 0
        self.ingest_calls = 0
        self.settings = Settings()

    def fetch(self) -> bytes:
        self.fetch_calls += 1
        return self.raw_schedule

    def ingest(self, raw_schedule: bytes) -> dict:
        self.ingest_calls += 1
        return self.gtfs_files


@pytest.fixture
def gtfs_files() -> dict:
    ingester = GTFSScheduleIngester(
        callback=lambda: io.BytesIO(), settings=Settings(projection="EPSG:32723")
    )
    return ingester.ingest(build_gtfs_zip())


@pytest.fixture
def gtfs_snapshot(gtfs_files) -> GtfsSnapshot:
    return build_gtfs_snapshot(gtfs_files, "hash1", Settings())


# --- SnapshotStore ------------------------------------------------------------


def test_snapshot_store_get_returns_none_before_first_set():
    store = SnapshotStore()
    assert store.get() is None


def test_snapshot_store_set_then_get_returns_the_value():
    store = SnapshotStore()
    store.set("snapshot-1")
    assert store.get() == "snapshot-1"


# --- GTFSScheduleLoop.run_once -------------------------------------------------


def test_run_once_publishes_snapshot_and_sets_event_on_first_cycle(gtfs_files):
    ingester = FakeIngester(gtfs_files)
    store = SnapshotStore()
    event = threading.Event()
    loop = GTFSScheduleLoop(ingester, store, event)

    loop.run_once()

    assert ingester.ingest_calls == 1
    snapshot = store.get()
    assert isinstance(snapshot, GtfsSnapshot)
    assert snapshot.gtfs_hash == _hash_raw_gtfs(ingester.raw_schedule)
    assert event.is_set()


def test_run_once_skips_republish_when_raw_schedule_is_unchanged(gtfs_files):
    ingester = FakeIngester(gtfs_files)
    store = SnapshotStore()
    event = threading.Event()
    loop = GTFSScheduleLoop(ingester, store, event)
    loop.run_once()
    first_snapshot = store.get()
    event.clear()

    loop.run_once()

    assert ingester.fetch_calls == 2
    assert ingester.ingest_calls == 1
    assert store.get() is first_snapshot
    assert not event.is_set()


def test_run_once_republishes_and_sets_event_when_raw_schedule_changes(gtfs_files):
    ingester = FakeIngester(gtfs_files)
    store = SnapshotStore()
    event = threading.Event()
    loop = GTFSScheduleLoop(ingester, store, event)
    loop.run_once()
    first_snapshot = store.get()
    event.clear()
    ingester.raw_schedule = b"v2"

    loop.run_once()

    assert ingester.ingest_calls == 2
    second_snapshot = store.get()
    assert second_snapshot is not first_snapshot
    assert second_snapshot.gtfs_hash == _hash_raw_gtfs(b"v2")
    assert event.is_set()


# --- run_periodic ---------------------------------------------------------------


class _StopLoop(BaseException):
    """Raised by test doubles to escape run_periodic's `while True` - a
    BaseException (not Exception) so it isn't swallowed by run_periodic's own
    `except Exception` around work_fn().
    """


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeEvent:
    """Stands in for threading.Event in run_conditional tests - .wait(timeout)
    advances the shared FakeClock the same way the fake_sleep closures above
    advance it for run_periodic, records every timeout it was asked to wait
    for, and (unless `stop_after_wait`) pops its return value from
    `wait_results` (defaulting to False - a plain timeout - once exhausted).

    `stop_after_wait=True` (the default) raises _StopLoop right after
    recording/advancing - mirrors the run_periodic tests that only care about
    a single sleep call's duration. Tests that need the loop to survive past
    one wait() call (multiple cycles, wake/clear behavior) pass
    `stop_after_wait=False` and escape via work_fn raising _StopLoop instead.
    """

    def __init__(
        self,
        clock: FakeClock,
        stop_after_wait: bool = True,
        wait_results: list[bool] | None = None,
    ):
        self.clock = clock
        self.stop_after_wait = stop_after_wait
        self._wait_results = list(wait_results) if wait_results is not None else []
        self.wait_calls: list[float] = []
        self.clear_calls = 0

    def wait(self, timeout: float) -> bool:
        self.wait_calls.append(timeout)
        self.clock.advance(timeout)
        if self.stop_after_wait:
            raise _StopLoop()
        return self._wait_results.pop(0) if self._wait_results else False

    def clear(self) -> None:
        self.clear_calls += 1


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr("pygtfsrealtime.runner.time.time", clock.time)
    return clock


def test_run_periodic_fixed_rate_sleeps_interval_minus_execution_time(clock, monkeypatch):
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)
        raise _StopLoop()

    monkeypatch.setattr("pygtfsrealtime.runner.time.sleep", fake_sleep)

    def work_fn():
        clock.advance(2)

    schedule = LoopSchedule(interval=10, accounting_mode="fixed_rate")
    with pytest.raises(_StopLoop):
        run_periodic(schedule, work_fn)

    assert sleeps == [8]


def test_run_periodic_immediate_runs_next_cycle_without_sleeping_after_a_miss(clock, monkeypatch):
    sleeps = []
    monkeypatch.setattr(
        "pygtfsrealtime.runner.time.sleep",
        lambda seconds: (sleeps.append(seconds), clock.advance(seconds)),
    )

    calls = 0

    def work_fn():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise _StopLoop()
        clock.advance(15)  # longer than interval=10 - misses the deadline

    schedule = LoopSchedule(
        interval=10, accounting_mode="fixed_rate", on_missed_deadline="immediate"
    )
    with pytest.raises(_StopLoop):
        run_periodic(schedule, work_fn)

    assert calls == 2
    assert sleeps == []


def test_run_periodic_wait_full_interval_sleeps_the_full_interval_after_a_miss(clock, monkeypatch):
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)
        raise _StopLoop()

    monkeypatch.setattr("pygtfsrealtime.runner.time.sleep", fake_sleep)

    def work_fn():
        clock.advance(15)  # longer than interval=10 - misses the deadline

    schedule = LoopSchedule(
        interval=10, accounting_mode="fixed_rate", on_missed_deadline="wait_full_interval"
    )
    with pytest.raises(_StopLoop):
        run_periodic(schedule, work_fn)

    assert sleeps == [10]


def test_run_periodic_skip_to_next_tick_realigns_to_the_original_grid_after_a_miss(
    clock, monkeypatch
):
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)
        raise _StopLoop()

    monkeypatch.setattr("pygtfsrealtime.runner.time.sleep", fake_sleep)

    def work_fn():
        clock.advance(15)  # longer than interval=10 - misses the deadline by 5s

    schedule = LoopSchedule(
        interval=10, accounting_mode="fixed_rate", on_missed_deadline="skip_to_next_tick"
    )
    with pytest.raises(_StopLoop):
        run_periodic(schedule, work_fn)

    # Original grid: tick at +10. Missed by 5s (work finished at +15), so the
    # next grid-aligned tick is +20 - sleep(5), not a full interval(10).
    assert sleeps == [5]


def test_run_periodic_fixed_delay_always_sleeps_the_full_interval(clock, monkeypatch):
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)
        raise _StopLoop()

    monkeypatch.setattr("pygtfsrealtime.runner.time.sleep", fake_sleep)

    def work_fn():
        clock.advance(15)  # would miss a fixed_rate deadline - irrelevant here

    schedule = LoopSchedule(interval=10, accounting_mode="fixed_delay")
    with pytest.raises(_StopLoop):
        run_periodic(schedule, work_fn)

    assert sleeps == [10]


def test_run_periodic_logs_and_continues_when_work_fn_raises(clock, monkeypatch, caplog):
    monkeypatch.setattr("pygtfsrealtime.runner.time.sleep", lambda seconds: clock.advance(seconds))

    calls = 0

    def work_fn():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("boom")
        raise _StopLoop()

    schedule = LoopSchedule(interval=10)
    with caplog.at_level("ERROR", logger="pygtfsrealtime.runner"):
        with pytest.raises(_StopLoop):
            run_periodic(schedule, work_fn)

    assert calls == 2
    assert any(
        "Periodic loop cycle raised an exception" in record.message for record in caplog.records
    )


# --- run_periodic stop_event -----------------------------------------------------


def test_run_periodic_returns_immediately_when_stop_event_already_set():
    stop_event = threading.Event()
    stop_event.set()
    calls = 0

    def work_fn():
        nonlocal calls
        calls += 1

    run_periodic(LoopSchedule(interval=10), work_fn, stop_event=stop_event)

    assert calls == 0


def test_run_periodic_stop_event_interrupts_an_in_progress_sleep():
    stop_event = threading.Event()

    def work_fn():
        pass

    schedule = LoopSchedule(interval=10)  # long enough that "waited it out" would fail the test
    thread = threading.Thread(
        target=run_periodic, args=(schedule, work_fn, stop_event), daemon=True
    )
    thread.start()
    time.sleep(0.05)
    stop_event.set()
    thread.join(timeout=2)

    assert not thread.is_alive()


# --- run_conditional ------------------------------------------------------------


def test_run_conditional_fixed_rate_uses_the_dynamic_interval_from_work_fn(clock):
    event = FakeEvent(clock, stop_after_wait=False)
    intervals = iter([10, 20])
    calls = 0

    def work_fn():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise _StopLoop()
        return next(intervals)

    schedule = LoopSchedule(interval=999, accounting_mode="fixed_rate")
    with pytest.raises(_StopLoop):
        run_conditional(schedule, work_fn, event)

    assert event.wait_calls == [10, 20]


def test_run_conditional_fixed_delay_sleeps_the_returned_value(clock):
    event = FakeEvent(clock, stop_after_wait=False)
    calls = 0

    def work_fn():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise _StopLoop()
        return 15

    schedule = LoopSchedule(interval=999, accounting_mode="fixed_delay")
    with pytest.raises(_StopLoop):
        run_conditional(schedule, work_fn, event)

    assert event.wait_calls == [15]


def test_run_conditional_immediate_runs_next_cycle_without_sleeping_after_a_miss(clock):
    event = FakeEvent(clock, stop_after_wait=False)
    calls = 0

    def work_fn():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise _StopLoop()
        clock.advance(15)  # execution takes longer than the returned interval
        return 10

    schedule = LoopSchedule(
        interval=999, accounting_mode="fixed_rate", on_missed_deadline="immediate"
    )
    with pytest.raises(_StopLoop):
        run_conditional(schedule, work_fn, event)

    assert calls == 2
    assert event.wait_calls == []


def test_run_conditional_wait_full_interval_sleeps_the_dynamic_interval_after_a_miss(clock):
    event = FakeEvent(clock)

    def work_fn():
        clock.advance(15)
        return 10

    schedule = LoopSchedule(
        interval=999, accounting_mode="fixed_rate", on_missed_deadline="wait_full_interval"
    )
    with pytest.raises(_StopLoop):
        run_conditional(schedule, work_fn, event)

    # The "full interval" slept is the just-returned dynamic value (10), not
    # the static schedule.interval (999).
    assert event.wait_calls == [10]


def test_run_conditional_skip_to_next_tick_realigns_using_the_dynamic_interval(clock):
    event = FakeEvent(clock)

    def work_fn():
        clock.advance(15)
        return 10

    schedule = LoopSchedule(
        interval=999, accounting_mode="fixed_rate", on_missed_deadline="skip_to_next_tick"
    )
    with pytest.raises(_StopLoop):
        run_conditional(schedule, work_fn, event)

    # Grid step is the dynamic interval (10), not schedule.interval (999).
    # next_run was at +10, missed by 5s (work finished at +15) - next
    # grid-aligned tick is +20 - sleep(5).
    assert event.wait_calls == [5]


def test_run_conditional_clears_the_event_when_woken_early(clock):
    event = FakeEvent(clock, stop_after_wait=False, wait_results=[True])
    calls = 0

    def work_fn():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise _StopLoop()
        return 10

    schedule = LoopSchedule(interval=999, accounting_mode="fixed_rate")
    with pytest.raises(_StopLoop):
        run_conditional(schedule, work_fn, event)

    assert event.clear_calls == 1


def test_run_conditional_does_not_clear_the_event_on_a_plain_timeout(clock):
    event = FakeEvent(clock, stop_after_wait=False, wait_results=[False])
    calls = 0

    def work_fn():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise _StopLoop()
        return 10

    schedule = LoopSchedule(interval=999, accounting_mode="fixed_rate")
    with pytest.raises(_StopLoop):
        run_conditional(schedule, work_fn, event)

    assert event.clear_calls == 0


def test_run_conditional_logs_and_continues_when_work_fn_raises(clock, caplog):
    event = FakeEvent(clock, stop_after_wait=False)
    calls = 0

    def work_fn():
        nonlocal calls
        calls += 1
        if calls == 1:
            return 10
        if calls == 2:
            raise ValueError("boom")
        raise _StopLoop()

    schedule = LoopSchedule(interval=999, accounting_mode="fixed_rate")
    with caplog.at_level("ERROR", logger="pygtfsrealtime.runner"):
        with pytest.raises(_StopLoop):
            run_conditional(schedule, work_fn, event)

    assert calls == 3
    # The interval used for the cycle after the failure is the last
    # successfully-returned one (10) - the failing cycle never overwrote it.
    assert event.wait_calls == [10, 10]
    assert any(
        "Conditional loop cycle raised an exception" in record.message for record in caplog.records
    )


def test_run_conditional_never_mutates_the_schedule_it_receives(clock):
    event = FakeEvent(clock, stop_after_wait=False)
    calls = 0

    def work_fn():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise _StopLoop()
        return 5

    schedule = LoopSchedule(
        interval=999, accounting_mode="fixed_rate", on_missed_deadline="immediate"
    )
    with pytest.raises(_StopLoop):
        run_conditional(schedule, work_fn, event)

    assert schedule.interval == 999
    assert schedule.accounting_mode == "fixed_rate"
    assert schedule.on_missed_deadline == "immediate"


# --- run_conditional stop_event ---------------------------------------------------


def test_run_conditional_returns_immediately_when_stop_event_already_set():
    stop_event = threading.Event()
    stop_event.set()
    wake_event = threading.Event()
    calls = 0

    def work_fn():
        nonlocal calls
        calls += 1
        return 10

    run_conditional(LoopSchedule(interval=10), work_fn, wake_event, stop_event=stop_event)

    assert calls == 0


def test_run_conditional_stop_event_interrupts_an_in_progress_sleep():
    stop_event = threading.Event()
    wake_event = threading.Event()

    def work_fn():
        return 10  # long enough that "waited it out" would fail the test

    schedule = LoopSchedule(interval=999)
    thread = threading.Thread(
        target=run_conditional, args=(schedule, work_fn, wake_event, stop_event), daemon=True
    )
    thread.start()
    time.sleep(0.05)
    stop_event.set()
    wake_event.set()  # unblocks the in-progress wake_event.wait() - see run_conditional's docstring
    thread.join(timeout=2)

    assert not thread.is_alive()


# --- TripWindowLoop.run_once -----------------------------------------------------


def test_trip_window_loop_run_once_publishes_a_trips_snapshot_when_gtfs_is_available(
    gtfs_snapshot,
):
    gtfs_store = SnapshotStore()
    gtfs_store.set(gtfs_snapshot)
    trips_store = SnapshotStore()
    now = datetime(2026, 1, 1, 5, 0, 0, tzinfo=UTC)
    loop = TripWindowLoop(Settings(), gtfs_store, trips_store, now_fn=lambda: now)

    result = loop.run_once()

    assert isinstance(result, TripsSnapshot)
    assert trips_store.get() is result
    assert result.gtfs_hash == gtfs_snapshot.gtfs_hash


def test_trip_window_loop_run_once_returns_none_when_no_gtfs_snapshot_yet():
    gtfs_store = SnapshotStore()
    trips_store = SnapshotStore()
    loop = TripWindowLoop(Settings(), gtfs_store, trips_store)

    result = loop.run_once()

    assert result is None
    assert trips_store.get() is None


# --- TripWindowLoop.next_interval -------------------------------------------------


def test_trip_window_loop_next_interval_matches_window_end_minus_margin(gtfs_snapshot):
    gtfs_store = SnapshotStore()
    gtfs_store.set(gtfs_snapshot)
    trips_store = SnapshotStore()
    now = datetime(2026, 1, 1, 5, 0, 0, tzinfo=UTC)
    settings = Settings()
    loop = TripWindowLoop(settings, gtfs_store, trips_store, now_fn=lambda: now)

    interval = loop.next_interval()

    # Same `now` fed to both build_trips_snapshot and the interval math, so
    # window_end - margin - now collapses exactly to window_length.
    assert interval == pytest.approx(settings.trip_window_loop_schedule.interval)


def test_trip_window_loop_next_interval_falls_back_to_gtfs_check_cadence_when_no_snapshot():
    gtfs_store = SnapshotStore()
    trips_store = SnapshotStore()
    settings = Settings()
    loop = TripWindowLoop(settings, gtfs_store, trips_store)

    interval = loop.next_interval()

    assert interval == settings.gtfs_loop_schedule.interval


def test_trip_window_loop_never_mutates_settings_trip_window_loop_schedule(gtfs_snapshot):
    gtfs_store = SnapshotStore()
    gtfs_store.set(gtfs_snapshot)
    trips_store = SnapshotStore()
    settings = Settings()
    original_interval = settings.trip_window_loop_schedule.interval
    now = datetime(2026, 1, 1, 5, 0, 0, tzinfo=UTC)
    loop = TripWindowLoop(settings, gtfs_store, trips_store, now_fn=lambda: now)

    loop.next_interval()
    loop.next_interval()

    assert settings.trip_window_loop_schedule.interval == original_interval


# --- TripWindowLoop + run_conditional integration ---------------------------------


def test_trip_window_loop_run_conditional_integration_uses_dynamic_window_interval(
    gtfs_snapshot, clock
):
    gtfs_store = SnapshotStore()
    gtfs_store.set(gtfs_snapshot)
    trips_store = SnapshotStore()
    settings = Settings()
    now = datetime(2026, 1, 1, 5, 0, 0, tzinfo=UTC)
    loop = TripWindowLoop(settings, gtfs_store, trips_store, now_fn=lambda: now)
    event = FakeEvent(clock)

    with pytest.raises(_StopLoop):
        run_conditional(settings.trip_window_loop_schedule, loop.next_interval, event)

    assert trips_store.get() is not None
    assert event.wait_calls == [pytest.approx(settings.trip_window_loop_schedule.interval)]


def test_trip_window_loop_run_conditional_rebuilds_when_event_fires_early(gtfs_snapshot, clock):
    gtfs_store = SnapshotStore()
    gtfs_store.set(gtfs_snapshot)
    trips_store = SnapshotStore()
    settings = Settings()
    now = datetime(2026, 1, 1, 5, 0, 0, tzinfo=UTC)
    loop = TripWindowLoop(settings, gtfs_store, trips_store, now_fn=lambda: now)
    event = FakeEvent(clock, stop_after_wait=False, wait_results=[True, False])
    calls = 0

    def next_interval_and_count():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise _StopLoop()
        return loop.next_interval()

    with pytest.raises(_StopLoop):
        run_conditional(settings.trip_window_loop_schedule, next_interval_and_count, event)

    assert calls == 3
    assert event.clear_calls == 1


# --- FSMLoop ------------------------------------------------------------

FSM_NOW = datetime(2026, 1, 1, 5, 0, 0, tzinfo=UTC)


def _active_trips_snapshot(now: datetime = FSM_NOW) -> TripsSnapshot:
    """A minimal, hand-built TripsSnapshot with one trip (T1) active at `now`.

    Coordinates are plain (lon, lat)-shaped numbers, not real geography -
    settings.projection is set to "EPSG:4326" in these tests (see
    _fsm_settings) so GPSIngester.ingest's lat/lon -> projection
    reprojection is a no-op and VehicleReport.point.x/y equal GPSEntry.longitude/
    latitude directly, keeping the geometry math trivial to reason about.
    """
    row = {
        "trip_id": "T1",
        "start_dt": now - timedelta(minutes=5),
        "end_dt": now + timedelta(hours=1),
        "route_id": "R1",
        "route_short_name": "100",
        "direction_id": 0,
        "start_zone": Point(0, 0).buffer(0.05),
        "start_zone_source": "stop",
        "end_zone": Point(1, 0).buffer(0.05),
        "end_zone_source": "stop",
        "shape_geometry": LineString([(0, 0), (1, 0)]),
    }
    trips = gpd.GeoDataFrame([row], geometry="shape_geometry")
    trips = trips.set_index(["trip_id", "start_dt"], drop=False)
    return TripsSnapshot(
        trips=trips,
        window_start=pd.Timestamp(now),
        window_end=pd.Timestamp(now + timedelta(hours=8)),
        gtfs_hash="h1",
    )


def _fsm_settings() -> Settings:
    # mode="strict": every fixture here has exactly one candidate trip (T1),
    # and none of these tests exercise matching-score internals - "strict"
    # passes that sole candidate through unconditionally, same effective
    # behavior these tests always relied on. "progress_match" (the default)
    # would score this fixture's fixed shape/timing against its
    # acceptance_margin, which isn't what these FSM-transition tests are
    # about.
    return Settings(
        projection="EPSG:4326",
        trip_matching=MatchingStrategy(key="route_short_name,direction_id", mode="strict"),
    )


def _ingester(callback: Callable[[], list[GPSEntry]]) -> GPSIngester:
    return GPSIngester(callback=callback, settings=_fsm_settings())


def _matched_gps_entry(vehicle_id: str = "V1", now: datetime = FSM_NOW) -> GPSEntry:
    # On the T1 shape (LineString (0,0)-(1,0)), far from either terminal zone
    # (buffer radius 0.05 around each endpoint) - matches route/direction, on
    # path, not at a terminal, so a FREE vehicle goes BUSY this cycle.
    return GPSEntry(
        vehicle_id=vehicle_id,
        latitude=0.0,
        longitude=0.5,
        datetime=now,
        route_short_name="100",
        direction_id=0,
    )


def _unmatched_gps_entry(vehicle_id: str = "V2", now: datetime = FSM_NOW) -> GPSEntry:
    return GPSEntry(
        vehicle_id=vehicle_id,
        latitude=0.0,
        longitude=0.5,
        datetime=now,
        route_short_name="999",
        direction_id=0,
    )


class _FakeCacheBackend:
    def __init__(self):
        self.data: bytes | None = None

    def set_cache(self, data: bytes) -> None:
        self.data = data

    def get_cache(self) -> bytes | None:
        return self.data


def _parse_feed(data: bytes) -> FeedMessage:
    feed = FeedMessage()
    feed.ParseFromString(data)
    return feed


def test_fsm_loop_run_once_publishes_matched_vehicle():
    trips_store = SnapshotStore()
    trips_store.set(_active_trips_snapshot())
    published: list[bytes] = []

    loop = FSMLoop(
        _fsm_settings(),
        trips_store,
        ingester=_ingester(lambda: [_matched_gps_entry()]),
        publish_protobuf=published.append,
        now_fn=lambda: FSM_NOW,
    )
    loop.run_once()

    assert len(published) == 1
    feed = _parse_feed(published[0])
    assert len(feed.entity) == 1
    assert feed.entity[0].id == "V1"
    assert feed.entity[0].vehicle.trip.trip_id == "T1"
    assert feed.entity[0].vehicle.trip.route_id == "R1"
    assert feed.entity[0].vehicle.position.longitude == pytest.approx(0.5)


def test_fsm_loop_run_once_omits_vehicle_with_no_matching_trip():
    trips_store = SnapshotStore()
    trips_store.set(_active_trips_snapshot())
    published: list[bytes] = []

    loop = FSMLoop(
        _fsm_settings(),
        trips_store,
        ingester=_ingester(lambda: [_unmatched_gps_entry()]),
        publish_protobuf=published.append,
        now_fn=lambda: FSM_NOW,
    )
    loop.run_once()

    assert len(published) == 1
    feed = _parse_feed(published[0])
    assert len(feed.entity) == 0


def test_fsm_loop_on_transition_fires_once_per_cycle_with_a_batch():
    trips_store = SnapshotStore()
    trips_store.set(_active_trips_snapshot())
    batches: list[list] = []

    loop = FSMLoop(
        _fsm_settings(),
        trips_store,
        ingester=_ingester(lambda: [_matched_gps_entry()]),
        publish_protobuf=lambda data: None,
        on_transition=batches.append,
        now_fn=lambda: FSM_NOW,
    )
    loop.run_once()

    assert len(batches) == 1
    assert len(batches[0]) == 1
    event = batches[0][0]
    assert event.vehicle_id == "V1"
    assert event.old_state == VehicleState.FREE
    assert event.new_state == VehicleState.BUSY


def test_fsm_loop_on_transition_batches_multiple_vehicles_into_one_call():
    trips_store = SnapshotStore()
    trips_store.set(_active_trips_snapshot())
    batches: list[list] = []

    loop = FSMLoop(
        _fsm_settings(),
        trips_store,
        ingester=_ingester(
            lambda: [_matched_gps_entry(vehicle_id="V1"), _matched_gps_entry(vehicle_id="V2")]
        ),
        publish_protobuf=lambda data: None,
        on_transition=batches.append,
        now_fn=lambda: FSM_NOW,
    )
    loop.run_once()

    assert len(batches) == 1
    assert {event.vehicle_id for event in batches[0]} == {"V1", "V2"}


def test_fsm_loop_on_transition_fires_with_empty_list_when_no_vehicles_report():
    trips_store = SnapshotStore()
    trips_store.set(_active_trips_snapshot())
    batches: list[list] = []

    loop = FSMLoop(
        _fsm_settings(),
        trips_store,
        ingester=_ingester(lambda: []),
        publish_protobuf=lambda data: None,
        on_transition=batches.append,
        now_fn=lambda: FSM_NOW,
    )
    loop.run_once()

    assert batches == [[]]


def test_fsm_loop_on_transition_not_called_when_cycle_is_skipped():
    trips_store = SnapshotStore()  # never .set() - no TripsSnapshot published yet
    batches: list[list] = []

    loop = FSMLoop(
        _fsm_settings(),
        trips_store,
        ingester=_ingester(lambda: [_matched_gps_entry()]),
        publish_protobuf=lambda data: None,
        on_transition=batches.append,
        now_fn=lambda: FSM_NOW,
    )
    loop.run_once()

    assert batches == []


def test_fsm_loop_run_once_skips_cycle_when_trips_snapshot_missing():
    trips_store = SnapshotStore()  # never .set() - no cycle-2 publish yet
    published: list[bytes] = []
    fetched = False

    def ingest_gps_data():
        nonlocal fetched
        fetched = True
        return [_matched_gps_entry()]

    loop = FSMLoop(
        _fsm_settings(),
        trips_store,
        ingester=_ingester(ingest_gps_data),
        publish_protobuf=published.append,
        now_fn=lambda: FSM_NOW,
    )
    loop.run_once()  # must not raise

    # No TripsSnapshot yet - skip the cycle entirely rather than publishing a
    # feed built from an empty match index, and don't poll GPS for a cycle
    # that has nothing to match against.
    assert not fetched
    assert published == []


def test_fsm_loop_cache_roundtrip_restores_busy_vehicle_to_the_same_instance():
    trips_store = SnapshotStore()
    trips_store.set(_active_trips_snapshot())
    backend = _FakeCacheBackend()

    first = FSMLoop(
        _fsm_settings(),
        trips_store,
        ingester=_ingester(lambda: [_matched_gps_entry()]),
        publish_protobuf=lambda data: None,
        get_cache=backend.get_cache,
        set_cache=backend.set_cache,
        now_fn=lambda: FSM_NOW,
    )
    first.run_once()
    assert first._fsms["V1"].state == VehicleState.BUSY
    assert backend.data is not None

    second = FSMLoop(
        _fsm_settings(),
        trips_store,
        ingester=_ingester(lambda: []),
        publish_protobuf=lambda data: None,
        get_cache=backend.get_cache,
        set_cache=backend.set_cache,
        now_fn=lambda: FSM_NOW,
    )
    second.run_once()  # cache load is deferred to the first run_once() call

    assert second._fsms["V1"].state == VehicleState.BUSY
    assert second._fsms["V1"].current_trip.trip_id == "T1"


def test_fsm_loop_on_transition_fires_for_a_cache_restored_fsm():
    trips_store = SnapshotStore()
    trips_store.set(_active_trips_snapshot())
    backend = _FakeCacheBackend()

    first = FSMLoop(
        _fsm_settings(),
        trips_store,
        ingester=_ingester(lambda: [_matched_gps_entry()]),
        publish_protobuf=lambda data: None,
        get_cache=backend.get_cache,
        set_cache=backend.set_cache,
        now_fn=lambda: FSM_NOW,
    )
    first.run_once()
    assert first._fsms["V1"].state == VehicleState.BUSY

    batches: list[list] = []
    second_now = FSM_NOW + timedelta(seconds=30)
    second = FSMLoop(
        _fsm_settings(),
        trips_store,
        ingester=_ingester(lambda: [_matched_gps_entry(now=second_now)]),
        publish_protobuf=lambda data: None,
        get_cache=backend.get_cache,
        set_cache=backend.set_cache,
        on_transition=batches.append,
        now_fn=lambda: second_now,
    )
    second.run_once()  # cache load (see load_cache) must still be captured this cycle

    assert len(batches) == 1
    assert len(batches[0]) == 1
    assert batches[0][0].vehicle_id == "V1"


def test_fsm_loop_publishes_cache_restored_busy_vehicle_that_reports_gps_again():
    """Regression guard: a cache-restored current_trip is looked up via
    active_trips.loc[trip_key] (pygtfsrealtime.realtime.loop.FSMLoop._trip_resolver), which
    returns a pandas Series - unlike the itertuples() row match_vehicle
    produces during live matching. Checking its truthiness directly (e.g.
    `not fsm.current_trip`) raises "the truth value of a Series is
    ambiguous", and since run_periodic swallows every work_fn exception into
    a log line, that crash would be silent: nothing would get published or
    re-cached for a cycle where a cache-restored BUSY vehicle also reports
    fresh GPS. A simple cache-roundtrip test whose second run_once() call
    reports no GPS at all wouldn't catch this, since build_pb only iterates
    the current cycle's vehicles.
    """
    trips_store = SnapshotStore()
    trips_store.set(_active_trips_snapshot())
    backend = _FakeCacheBackend()

    first = FSMLoop(
        _fsm_settings(),
        trips_store,
        ingester=_ingester(lambda: [_matched_gps_entry()]),
        publish_protobuf=lambda data: None,
        get_cache=backend.get_cache,
        set_cache=backend.set_cache,
        now_fn=lambda: FSM_NOW,
    )
    first.run_once()
    assert first._fsms["V1"].state == VehicleState.BUSY

    # A later cycle with the vehicle having moved - reusing the exact same
    # position/timestamp as `first` would make ObservationWindow see a
    # zero-diagonal window and call it stationary, sending the FSM back to
    # FREE for an unrelated reason before build_pb ever runs. The jump is far
    # larger than settings.stationary_threshold.distance's default (30) -
    # _fsm_settings() uses EPSG:4326 (raw degrees, see _active_trips_snapshot)
    # to keep the geometry math trivial, so on_path/at_terminal (both compared
    # against equally degree-scale thresholds) don't matter here anyway: BUSY
    # only re-checks stationary/at_terminal/trip_duration_exceeded.
    second_now = FSM_NOW + timedelta(seconds=30)
    moved_entry = GPSEntry(
        vehicle_id="V1",
        latitude=40.0,
        longitude=0.5,
        datetime=second_now,
        route_short_name="100",
        direction_id=0,
    )

    published: list[bytes] = []
    second = FSMLoop(
        _fsm_settings(),
        trips_store,
        ingester=_ingester(lambda: [moved_entry]),
        publish_protobuf=published.append,
        get_cache=backend.get_cache,
        set_cache=backend.set_cache,
        now_fn=lambda: second_now,
    )
    second.run_once()  # must not raise despite current_trip being a .loc[] Series

    assert len(published) == 1
    feed = _parse_feed(published[0])
    assert len(feed.entity) == 1
    assert feed.entity[0].id == "V1"
    assert feed.entity[0].vehicle.trip.trip_id == "T1"


def test_fsm_loop_defers_cache_load_until_a_trips_snapshot_exists():
    """Regression guard: FSMLoop is constructed before cycle 1/2 have ever
    run in a real GTFSRealtimeEngine, so trips_snapshot_store is empty at
    construction time. Loading the cache unconditionally in __init__ would
    permanently resolve every cached BUSY vehicle's trip to None and
    silently demote it to FREE. To avoid that, the cache load retries each
    run_once() call until a TripsSnapshot actually exists, matching the same
    "no snapshot yet - retry next cycle" pattern TripWindowLoop already uses
    for GtfsSnapshot.
    """
    backend = _FakeCacheBackend()

    seed_trips_store = SnapshotStore()
    seed_trips_store.set(_active_trips_snapshot())
    seed = FSMLoop(
        _fsm_settings(),
        seed_trips_store,
        ingester=_ingester(lambda: [_matched_gps_entry()]),
        publish_protobuf=lambda data: None,
        set_cache=backend.set_cache,
        now_fn=lambda: FSM_NOW,
    )
    seed.run_once()
    assert seed._fsms["V1"].state == VehicleState.BUSY
    assert backend.data is not None

    trips_store = SnapshotStore()  # empty - cycle 2 hasn't run yet
    loop = FSMLoop(
        _fsm_settings(),
        trips_store,
        ingester=_ingester(lambda: []),
        publish_protobuf=lambda data: None,
        get_cache=backend.get_cache,
        now_fn=lambda: FSM_NOW,
    )
    assert not loop._cache_loaded

    loop.run_once()  # trips_snapshot_store still empty - cache not loadable yet
    assert not loop._cache_loaded
    assert loop._fsms == {}

    trips_store.set(_active_trips_snapshot())  # cycle 2 publishes
    loop.run_once()

    assert loop._cache_loaded
    assert loop._fsms["V1"].state == VehicleState.BUSY
    assert loop._fsms["V1"].current_trip.trip_id == "T1"
