from datetime import datetime, timedelta

from shapely.geometry import LineString, Point

from pygtfsrealtime.realtime.ingest import VehicleReport
from pygtfsrealtime.realtime.observations import (
    ObservationWindow,
    build_observation,
    is_close_to_path,
    is_in_terminal_zone,
    signal_lost,
    trip_duration_exceeded,
)
from pygtfsrealtime.settings import GeometryThreshold, Settings, StationaryThreshold

T0 = datetime(2026, 1, 1, 12, 0, 0)


def _vehicle(dt: datetime, x: float, y: float) -> VehicleReport:
    return VehicleReport(vehicle_id="V1", datetime=dt, point=Point(x, y))


# --- ObservationWindow.is_stationary --------------------------------------


def test_is_stationary_true_when_window_empty():
    window = ObservationWindow()
    assert (
        window.is_stationary(StationaryThreshold(distance=30, interval=timedelta(minutes=15)))
        is True
    )


def test_is_stationary_true_when_points_within_distance():
    window = ObservationWindow()
    window.push(_vehicle(T0, 0, 0), T0, timedelta(minutes=15))
    window.push(
        _vehicle(T0 + timedelta(minutes=1), 5, 5), T0 + timedelta(minutes=1), timedelta(minutes=15)
    )
    threshold = StationaryThreshold(distance=30, interval=timedelta(minutes=15))
    assert window.is_stationary(threshold) is True


def test_is_stationary_false_when_points_far_apart():
    window = ObservationWindow()
    window.push(_vehicle(T0, 0, 0), T0, timedelta(minutes=15))
    window.push(
        _vehicle(T0 + timedelta(minutes=1), 1000, 1000),
        T0 + timedelta(minutes=1),
        timedelta(minutes=15),
    )
    threshold = StationaryThreshold(distance=30, interval=timedelta(minutes=15))
    assert window.is_stationary(threshold) is False


def test_window_evicts_entries_older_than_interval():
    window = ObservationWindow()
    interval = timedelta(minutes=15)
    window.push(_vehicle(T0, 0, 0), T0, interval)
    later = T0 + timedelta(minutes=20)
    # a fresh, nearby observation - old far-away point should have been evicted
    window.push(_vehicle(later, 2000, 2000), later, interval)
    threshold = StationaryThreshold(distance=30, interval=interval)
    assert window.is_stationary(threshold) is True


def test_window_does_not_append_stale_observation():
    window = ObservationWindow()
    interval = timedelta(minutes=15)
    stale = _vehicle(T0 - timedelta(hours=1), 0, 0)
    window.push(stale, T0, interval)
    # nothing within the interval was ever added, so the window is empty
    assert window.is_stationary(StationaryThreshold(distance=30, interval=interval)) is True


# --- ObservationWindow entries seeding/reading (pygtfsrealtime.realtime.cache) --


def test_window_defaults_to_empty_entries():
    assert ObservationWindow().entries() == ()


def test_window_can_be_seeded_with_prior_entries():
    seed = ((T0, 1.0, 2.0), (T0 + timedelta(seconds=30), 1.5, 2.5))
    window = ObservationWindow(seed)
    assert window.entries() == seed


def test_window_entries_reflects_push_and_eviction():
    window = ObservationWindow()
    interval = timedelta(minutes=15)
    window.push(_vehicle(T0, 1, 2), T0, interval)
    assert window.entries() == ((T0, 1.0, 2.0),)

    later = T0 + timedelta(minutes=20)
    window.push(_vehicle(later, 3, 4), later, interval)
    assert window.entries() == ((later, 3.0, 4.0),)


# --- is_close_to_path / is_in_terminal_zone --------------------------------


def test_is_close_to_path_distance_mode():
    line = LineString([(0, 0), (100, 0)])
    threshold = GeometryThreshold(distance=10, mode="distance")
    assert is_close_to_path(Point(50, 5), line, threshold) is True
    assert is_close_to_path(Point(50, 50), line, threshold) is False


def test_is_close_to_path_buffer_mode_expects_prebuilt_zone():
    zone = Point(0, 0).buffer(10)
    threshold = GeometryThreshold(distance=10, mode="buffer")
    assert is_close_to_path(Point(5, 0), zone, threshold) is True
    assert is_close_to_path(Point(50, 0), zone, threshold) is False


def test_is_in_terminal_zone_matches_either_end():
    start_zone = Point(0, 0).buffer(10)
    end_zone = Point(1000, 0).buffer(10)
    threshold = GeometryThreshold(distance=10, mode="buffer")
    assert is_in_terminal_zone(Point(1000, 5), start_zone, threshold, end_zone, threshold) is True
    assert is_in_terminal_zone(Point(500, 0), start_zone, threshold, end_zone, threshold) is False


# --- trip_duration_exceeded / signal_lost ----------------------------------


def test_trip_duration_exceeded():
    threshold = timedelta(hours=3)
    assert trip_duration_exceeded(T0, T0 + timedelta(hours=2), threshold) is False
    assert trip_duration_exceeded(T0, T0 + timedelta(hours=3), threshold) is True


def test_signal_lost():
    threshold = timedelta(minutes=5)
    assert signal_lost(T0, T0 + timedelta(minutes=4), threshold) is False
    assert signal_lost(T0, T0 + timedelta(minutes=5), threshold) is True


# --- build_observation -----------------------------------------------------


class _FakeTrip:
    def __init__(
        self,
        shape_geometry,
        start_zone,
        end_zone,
        start_zone_source="terminal",
        end_zone_source="terminal",
    ):
        self.trip_id = "T1"
        self.start_dt = T0
        self.route_id = "R1"
        self.direction_id = 0
        self.shape_geometry = shape_geometry
        self.start_zone = start_zone
        self.start_zone_source = start_zone_source
        self.end_zone = end_zone
        self.end_zone_source = end_zone_source


def _settings(**overrides) -> Settings:
    return Settings(**overrides)


def test_build_observation_no_trip_is_off_path_and_not_at_terminal():
    window = ObservationWindow()
    vehicle = _vehicle(T0, 0, 0)
    observation = build_observation(vehicle, window, None, _settings(), T0)
    assert observation.trip is None
    assert observation.on_path is False
    assert observation.at_terminal is False


def test_build_observation_with_trip_evaluates_geometry_and_signal():
    settings = _settings(
        shape_geometry=GeometryThreshold(distance=10, mode="distance"),
        terminal_geometry=GeometryThreshold(distance=10, mode="buffer"),
    )
    trip = _FakeTrip(
        shape_geometry=LineString([(0, 0), (100, 0)]),
        start_zone=Point(0, 0).buffer(10),
        end_zone=Point(100, 0).buffer(10),
    )
    window = ObservationWindow()
    vehicle = _vehicle(T0, 5, 0)

    observation = build_observation(vehicle, window, trip, settings, T0)

    assert observation.on_path is True
    assert observation.at_terminal is True  # near the start zone
    assert observation.signal_lost is False


def test_build_observation_flags_signal_lost_when_stale():
    settings = _settings()
    window = ObservationWindow()
    vehicle = _vehicle(T0, 0, 0)
    now = T0 + settings.signal_loss_threshold

    observation = build_observation(vehicle, window, None, settings, now)

    assert observation.signal_lost is True
