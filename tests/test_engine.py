import io
import time
from datetime import UTC, datetime

import pytest

from pygtfsrealtime.engine import GTFSRealtimeEngine
from pygtfsrealtime.models import GPSEntry
from pygtfsrealtime.pb.gtfs_realtime_pb2 import FeedMessage  # type: ignore[attr-defined]
from pygtfsrealtime.settings import LoopSchedule, MatchingStrategy, Settings
from tests.gtfs_data import build_gtfs_zip

# Real Rio-scale coordinates (reused/adjusted from tests/gtfs_data.py's
# VALID_GTFS_CSV) so the default projection (EPSG:32723, meters) gives
# realistic terminal-buffer/shape-distance geometry without having to
# override any GeometryThreshold - a vehicle at the shape's midpoint sits
# ~750m from each stop (well past the default 250m terminal buffer) and
# exactly on the shape line (well within the default 30m path distance).
GTFS_OVERRIDES = {
    "stops.txt": (
        "stop_id,stop_lat,stop_lon,parent_station\nST1,-22.9,-43.2,\nST2,-22.91,-43.21,\n"
    ),
    "stop_times.txt": (
        "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
        "T1,ST1,1,05:00:00,05:00:00\n"
        "T1,ST2,2,06:00:00,06:00:00\n"
    ),
    # No frequencies for T1 - stop_times.txt alone is its one instance,
    # avoids needing to reason about frequency expansion in this test.
    "frequencies.txt": "trip_id,start_time,end_time,headway_secs\n",
}

# 2026-01-01 is a Thursday - calendar.txt already marks thursday=1, so S1 is
# active without needing calendar_dates.txt's exception. Falls inside the
# trip's 05:00-06:00 window (already running, not just starting).
NOW = datetime(2026, 1, 1, 5, 30, 0, tzinfo=UTC)


def _fetch_gtfs_zip(overrides: dict[str, str] | None = None):
    # GTFSScheduleIngester.fetch() calls .read() on whatever the
    # callback returns - a bare bytes object (what build_gtfs_zip returns)
    # doesn't have that method, needs wrapping in a file-like object.
    return io.BytesIO(build_gtfs_zip(overrides))


def _fast_settings(**overrides) -> Settings:
    return Settings(
        gtfs_loop_schedule=LoopSchedule(interval=0.02),
        trip_window_loop_schedule=LoopSchedule(interval=3600),
        fsm_loop_schedule=LoopSchedule(interval=0.02),
        projection="EPSG:32723",
        trip_matching=MatchingStrategy(key="route_short_name,direction_id"),
        **overrides,
    )


def _matched_gps_entry() -> GPSEntry:
    return GPSEntry(
        vehicle_id="V1",
        latitude=-22.905,
        longitude=-43.205,
        datetime=NOW,
        route_short_name="100",
        direction_id=0,
    )


def _wait_until(predicate, timeout: float = 3.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# --- constructor validation --------------------------------------------------


def test_engine_raises_when_ingest_gps_data_is_not_callable():
    with pytest.raises(TypeError):
        GTFSRealtimeEngine(
            gtfs_schedule=_fetch_gtfs_zip,
            ingest_gps_data="not a function",
            publish_protobuf=lambda data: None,
            settings=_fast_settings(),
        )


def test_engine_raises_when_publish_protobuf_is_not_callable():
    with pytest.raises(TypeError):
        GTFSRealtimeEngine(
            gtfs_schedule=_fetch_gtfs_zip,
            ingest_gps_data=lambda: [],
            publish_protobuf="not a function",
            settings=_fast_settings(),
        )


def test_engine_raises_when_on_transition_is_not_callable():
    with pytest.raises(TypeError):
        GTFSRealtimeEngine(
            gtfs_schedule=_fetch_gtfs_zip,
            ingest_gps_data=lambda: [],
            publish_protobuf=lambda data: None,
            on_transition="not a function",
            settings=_fast_settings(),
        )


def test_engine_raises_when_gtfs_schedule_path_does_not_exist():
    with pytest.raises(FileNotFoundError):
        GTFSRealtimeEngine(
            gtfs_schedule="/no/such/file.zip",
            ingest_gps_data=lambda: [],
            publish_protobuf=lambda data: None,
            settings=_fast_settings(),
        )


def test_engine_raises_when_trip_matching_is_not_set():
    with pytest.raises(TypeError):
        GTFSRealtimeEngine(
            gtfs_schedule=_fetch_gtfs_zip,
            ingest_gps_data=lambda: [],
            publish_protobuf=lambda data: None,
            settings=Settings(),  # trip_matching left unset
        )


# --- end-to-end wiring -----------------------------------------------------


def _feed_has_entities(data: bytes) -> bool:
    feed = FeedMessage()
    feed.ParseFromString(data)
    return len(feed.entity) > 0


def test_engine_start_publishes_a_real_feed_and_stop_terminates_every_thread():
    published: list[bytes] = []
    engine = GTFSRealtimeEngine(
        gtfs_schedule=lambda: _fetch_gtfs_zip(GTFS_OVERRIDES),
        ingest_gps_data=lambda: [_matched_gps_entry()],
        publish_protobuf=published.append,
        settings=_fast_settings(),
        now_fn=lambda: NOW,
    )

    engine.start()
    threads = list(engine._threads)
    try:
        # The FSM cycle publishes every tick regardless of whether it has any
        # matched vehicle yet (empty feed while cycle 1/2 haven't produced a
        # TripsSnapshot) - wait for a publish that actually carries an
        # entity, not just any publish call. now_fn is frozen for
        # determinism, so the vehicle's position never moves between polls:
        # once BUSY, the next tick sees it as stationary and flips back to
        # FREE, oscillating every cycle rather than staying BUSY. Scan the
        # whole publish history for one with a matched entity instead of
        # assuming the last call landed on it.
        assert _wait_until(lambda: any(_feed_has_entities(data) for data in published)), (
            "publish_protobuf never published a feed with a matched vehicle"
        )
    finally:
        engine.stop(timeout=2)

    assert all(not t.is_alive() for t in threads)

    matched = [data for data in published if _feed_has_entities(data)]
    feed = FeedMessage()
    feed.ParseFromString(matched[0])
    assert len(feed.entity) == 1
    assert feed.entity[0].vehicle.trip.trip_id == "T1"


def test_engine_on_transition_fires_during_a_real_cycle():
    batches: list[list] = []
    engine = GTFSRealtimeEngine(
        gtfs_schedule=lambda: _fetch_gtfs_zip(GTFS_OVERRIDES),
        ingest_gps_data=lambda: [_matched_gps_entry()],
        publish_protobuf=lambda data: None,
        on_transition=batches.append,
        settings=_fast_settings(),
        now_fn=lambda: NOW,
    )

    engine.start()
    try:
        # on_transition fires once per completed cycle, possibly with an
        # empty list (no vehicle reported yet) before the first real match -
        # wait for a batch that actually contains the vehicle's transition.
        assert _wait_until(lambda: any(batch for batch in batches)), (
            "on_transition never fired with a non-empty batch"
        )
    finally:
        engine.stop(timeout=2)

    matched_batch = next(batch for batch in batches if batch)
    assert matched_batch[0].vehicle_id == "V1"


def test_engine_start_raises_when_already_running():
    engine = GTFSRealtimeEngine(
        gtfs_schedule=lambda: _fetch_gtfs_zip(GTFS_OVERRIDES),
        ingest_gps_data=lambda: [],
        publish_protobuf=lambda data: None,
        settings=_fast_settings(),
        now_fn=lambda: NOW,
    )
    engine.start()
    try:
        with pytest.raises(RuntimeError):
            engine.start()
    finally:
        engine.stop(timeout=2)


def test_engine_stop_is_safe_before_start_and_when_called_twice():
    engine = GTFSRealtimeEngine(
        gtfs_schedule=lambda: _fetch_gtfs_zip(GTFS_OVERRIDES),
        ingest_gps_data=lambda: [],
        publish_protobuf=lambda data: None,
        settings=_fast_settings(),
        now_fn=lambda: NOW,
    )
    engine.stop()  # never started - must not raise

    engine.start()
    engine.stop(timeout=2)
    engine.stop(timeout=2)  # already stopped - must not raise
