from datetime import UTC, datetime, timedelta

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from pygtfsrealtime.realtime.ingest import VehicleReport
from pygtfsrealtime.realtime.matching import (
    build_trip_match_index,
    filter_active_now,
    match_vehicle,
    select_candidate,
    vehicle_match_key,
)
from pygtfsrealtime.settings import GeometryThreshold, MatchingStrategy, Settings

T0 = datetime(2026, 1, 1, 5, 0, 0, tzinfo=UTC)

# A 100-unit-long straight shape, shared by default across trip rows below -
# vehicle.point.x doubles as "distance traveled along the shape" for these
# tests (progress_match projects onto it via LineString.project).
SHAPE = LineString([(0, 0), (100, 0)])


def _trip_row(
    trip_id: str,
    route_id: str = "R1",
    route_short_name: str = "100",
    direction_id: int = 0,
    start_dt: datetime = T0,
    end_dt: datetime = T0 + timedelta(hours=1),
    shape_geometry: LineString = SHAPE,
) -> dict:
    return {
        "trip_id": trip_id,
        "route_id": route_id,
        "route_short_name": route_short_name,
        "direction_id": direction_id,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "shape_geometry": shape_geometry,
    }


def _trips(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _vehicle(point: Point | None = None, **kwargs) -> VehicleReport:
    return VehicleReport(point=point if point is not None else Point(0, 0), **kwargs)


def _settings(shape_distance: int = 30) -> Settings:
    return Settings(shape_geometry=GeometryThreshold(distance=shape_distance, mode="distance"))


def _trips_snapshot_shaped(*rows: dict) -> gpd.GeoDataFrame:
    """Same index shape as the real TripsSnapshot.trips (MultiIndex
    (trip_id, start_dt), drop=False - see trip_window.py). This shape makes
    active_trips.sort_values("start_dt") raise ValueError ("both an index
    level and a column label, which is ambiguous"), since a plain RangeIndex
    has no such collision - the plain pd.DataFrame `_trips()` helper above
    never exercises this case.
    """
    trips = pd.DataFrame(list(rows))
    return trips.set_index(["trip_id", "start_dt"], drop=False)


# --- filter_active_now -----------------------------------------------------


def test_filter_active_now_keeps_trip_spanning_now():
    trips = _trips(_trip_row("T1", start_dt=T0, end_dt=T0 + timedelta(minutes=30)))
    now = T0 + timedelta(minutes=10)
    result = filter_active_now(trips, now)
    assert list(result["trip_id"]) == ["T1"]


def test_filter_active_now_drops_trip_not_yet_started():
    trips = _trips(
        _trip_row("T1", start_dt=T0 + timedelta(minutes=5), end_dt=T0 + timedelta(minutes=30))
    )
    result = filter_active_now(trips, T0)
    assert result.empty


def test_filter_active_now_drops_trip_already_ended():
    trips = _trips(_trip_row("T1", start_dt=T0, end_dt=T0 + timedelta(minutes=5)))
    result = filter_active_now(trips, T0 + timedelta(minutes=10))
    assert result.empty


def test_filter_active_now_boundaries_are_inclusive():
    trips = _trips(_trip_row("T1", start_dt=T0, end_dt=T0 + timedelta(minutes=30)))
    assert list(filter_active_now(trips, T0)["trip_id"]) == ["T1"]
    assert list(filter_active_now(trips, T0 + timedelta(minutes=30))["trip_id"]) == ["T1"]


# --- vehicle_match_key ------------------------------------------------------


def test_vehicle_match_key_reads_the_configured_columns():
    vehicle = _vehicle(route_short_name="100", direction_id=0)
    assert vehicle_match_key(vehicle, "route_short_name,direction_id") == ("100", 0)
    assert vehicle_match_key(vehicle, "route_short_name") == ("100",)


# --- build_trip_match_index -------------------------------------------------


def test_build_trip_match_index_groups_by_key_and_orders_by_start_dt():
    trips = _trips(
        _trip_row("T2", start_dt=T0 + timedelta(minutes=10), end_dt=T0 + timedelta(hours=1)),
        _trip_row("T1", start_dt=T0, end_dt=T0 + timedelta(hours=1)),
    )
    index = build_trip_match_index(trips, "route_short_name,direction_id")
    candidates = index[("100", 0)]
    assert [c.trip_id for c in candidates] == ["T1", "T2"]


def test_build_trip_match_index_separates_different_keys():
    trips = _trips(
        _trip_row("T1", route_short_name="100", direction_id=0),
        _trip_row("T2", route_short_name="100", direction_id=1),
        _trip_row("T3", route_short_name="200", direction_id=0),
    )
    index = build_trip_match_index(trips, "route_short_name,direction_id")
    assert set(index.keys()) == {("100", 0), ("100", 1), ("200", 0)}


def test_build_trip_match_index_does_not_raise_on_the_real_trips_snapshot_index_shape():
    """active_trips indexed like the real TripsSnapshot.trips (MultiIndex
    (trip_id, start_dt), drop=False) must not raise ValueError ("'start_dt'
    is both an index level and a column label, which is ambiguous") from
    sort_values("start_dt"). Two instances sharing the same trip_id
    (frequencies.txt expansion) with different start_dt must also come back
    as two distinct, correctly-attributed candidates - not merged or
    cross-contaminated by dropping the index.
    """
    trips = _trips_snapshot_shaped(
        _trip_row("T1", start_dt=T0, end_dt=T0 + timedelta(hours=1)),
        _trip_row("T1", start_dt=T0 + timedelta(hours=1), end_dt=T0 + timedelta(hours=2)),
    )

    index = build_trip_match_index(trips, "route_short_name,direction_id")

    candidates = index[("100", 0)]
    assert [(c.trip_id, c.start_dt) for c in candidates] == [
        ("T1", T0),
        ("T1", T0 + timedelta(hours=1)),
    ]


# --- select_candidate: strict ------------------------------------------------


def test_select_candidate_strict_requires_exactly_one():
    strict = MatchingStrategy(key="trip_id", mode="strict")
    settings = _settings()
    # candidates/vehicle/settings/claimed_by beyond `candidates` itself are
    # never touched by the "strict" branch - None/{} stand in for "unused".
    assert select_candidate([], strict, T0, None, settings, {}) is None
    assert select_candidate(["only"], strict, T0, None, settings, {}) == "only"
    assert select_candidate(["a", "b"], strict, T0, None, settings, {}) is None


def test_select_candidate_strict_ignores_claimed_by_and_acceptance_margin():
    """ "strict" is unaffected by progress_match-only params: a sole candidate
    is returned even though it's already claimed by another vehicle (which
    would exclude it under progress_match with allow_shared_trip=False).
    """
    strict = MatchingStrategy(key="trip_id", mode="strict")
    claimed_by = {("T1", T0): "other-vehicle"}

    result = select_candidate(["only"], strict, T0, None, _settings(), claimed_by)

    assert result == "only"


# --- select_candidate: progress_match ----------------------------------------


def _row(trip_id: str = "T1", **overrides) -> pd.Series:
    """A single trip row, itertuples()-shaped like build_trip_match_index's
    candidates, via a one-row DataFrame (mirrors _trips_snapshot_shaped's
    approach for a single candidate)."""
    frame = _trips(_trip_row(trip_id, **overrides))
    # pandas-stubs types itertuples()'s Iterator against a narrower Protocol
    # than what next() actually expects.
    return next(frame.itertuples())  # type: ignore[arg-type]


def test_select_candidate_progress_match_picks_the_candidate_with_lowest_deviation():
    now = T0 + timedelta(minutes=30)
    # A: 1h trip starting at T0 -> time_fraction=0.5 at `now`. Vehicle at
    # x=45 on a 100-unit shape -> shape_fraction=0.45. deviation=0.05.
    candidate_a = _row(trip_id="A", start_dt=T0, end_dt=T0 + timedelta(hours=1))
    # B: 1h trip that has already fully elapsed by `now` -> time_fraction
    # clamps to 1.0. Same vehicle position -> shape_fraction=0.45.
    # deviation=0.55, worse than A.
    candidate_b = _row(
        trip_id="B", start_dt=T0 - timedelta(minutes=30), end_dt=T0 + timedelta(minutes=30)
    )
    vehicle = _vehicle(point=Point(45, 0))
    strategy = MatchingStrategy(key="trip_id")

    result = select_candidate([candidate_a, candidate_b], strategy, now, vehicle, _settings(), {})

    assert result.trip_id == "A"


def test_select_candidate_progress_match_excludes_candidate_outside_shape_threshold():
    now = T0 + timedelta(minutes=30)
    candidate = _row(start_dt=T0, end_dt=T0 + timedelta(hours=1))
    # 1000 units away from the shape - settings.shape_geometry.distance
    # defaults to 30, so this is nowhere close, regardless of how well the
    # time-based fraction would otherwise line up.
    vehicle = _vehicle(point=Point(50, 1000))
    strategy = MatchingStrategy(key="trip_id")

    result = select_candidate([candidate], strategy, now, vehicle, _settings(), {})

    assert result is None


def test_select_candidate_progress_match_excludes_candidate_over_acceptance_margin():
    now = T0  # time_fraction = 0.0 (trip just started)
    candidate = _row(start_dt=T0, end_dt=T0 + timedelta(hours=1))
    vehicle = _vehicle(point=Point(50, 0))  # shape_fraction = 0.5, deviation = 0.5
    strategy = MatchingStrategy(key="trip_id")  # default acceptance_margin=0.2

    result = select_candidate([candidate], strategy, now, vehicle, _settings(), {})

    assert result is None


def test_select_candidate_progress_match_excludes_claimed_trip_when_shared_trip_disallowed():
    now = T0 + timedelta(minutes=30)
    candidate = _row(trip_id="T1", start_dt=T0, end_dt=T0 + timedelta(hours=1))
    vehicle = _vehicle(point=Point(50, 0))  # shape_fraction=0.5, deviation=0.0
    strategy = MatchingStrategy(key="trip_id", allow_shared_trip=False)
    claimed_by = {("T1", T0): "other-vehicle"}

    result = select_candidate([candidate], strategy, now, vehicle, _settings(), claimed_by)

    assert result is None


def test_select_candidate_progress_match_includes_claimed_trip_when_shared_trip_allowed():
    now = T0 + timedelta(minutes=30)
    candidate = _row(trip_id="T1", start_dt=T0, end_dt=T0 + timedelta(hours=1))
    vehicle = _vehicle(point=Point(50, 0))
    strategy = MatchingStrategy(key="trip_id", allow_shared_trip=True)
    claimed_by = {("T1", T0): "other-vehicle"}

    result = select_candidate([candidate], strategy, now, vehicle, _settings(), claimed_by)

    assert result.trip_id == "T1"


def test_select_candidate_progress_match_tie_break_prefers_unclaimed_over_claimed():
    now = T0 + timedelta(minutes=30)
    # Both candidates: identical start/end and shape, so identical deviation
    # (0.0) - only "claimed" differs.
    claimed = _row(trip_id="CLAIMED", start_dt=T0, end_dt=T0 + timedelta(hours=1))
    unclaimed = _row(trip_id="UNCLAIMED", start_dt=T0, end_dt=T0 + timedelta(hours=1))
    vehicle = _vehicle(point=Point(50, 0))
    strategy = MatchingStrategy(key="trip_id", allow_shared_trip=True)
    claimed_by = {("CLAIMED", T0): "other-vehicle"}

    result = select_candidate([claimed, unclaimed], strategy, now, vehicle, _settings(), claimed_by)

    assert result.trip_id == "UNCLAIMED"


def test_select_candidate_progress_match_tie_break_prefers_earlier_start_dt():
    now = T0
    # Both candidates: same 1h duration and same vehicle position
    # (shape_fraction=0.5), but start_dt chosen so |shape_fraction -
    # time_fraction| is identical (0.1) for both, via symmetric offsets
    # around the vehicle's shape_fraction.
    earlier = _row(
        trip_id="EARLIER",
        start_dt=now - timedelta(minutes=36),
        end_dt=now - timedelta(minutes=36) + timedelta(hours=1),
    )
    later = _row(
        trip_id="LATER",
        start_dt=now - timedelta(minutes=24),
        end_dt=now - timedelta(minutes=24) + timedelta(hours=1),
    )
    vehicle = _vehicle(point=Point(50, 0))  # shape_fraction = 0.5
    strategy = MatchingStrategy(key="trip_id")

    result = select_candidate([later, earlier], strategy, now, vehicle, _settings(), {})

    assert result.trip_id == "EARLIER"


def test_select_candidate_progress_match_excludes_zero_duration_trip():
    now = T0
    candidate = _row(start_dt=T0, end_dt=T0)
    vehicle = _vehicle(point=Point(50, 0))
    strategy = MatchingStrategy(key="trip_id")

    result = select_candidate([candidate], strategy, now, vehicle, _settings(), {})

    assert result is None


def test_select_candidate_progress_match_excludes_zero_length_shape():
    now = T0 + timedelta(minutes=30)
    degenerate_shape = LineString([(5, 5), (5, 5)])
    candidate = _row(start_dt=T0, end_dt=T0 + timedelta(hours=1), shape_geometry=degenerate_shape)
    vehicle = _vehicle(point=Point(5, 5))  # distance 0 from the shape
    strategy = MatchingStrategy(key="trip_id")

    result = select_candidate([candidate], strategy, now, vehicle, _settings(), {})

    assert result is None


def test_select_candidate_progress_match_returns_none_for_empty_candidates():
    strategy = MatchingStrategy(key="trip_id")
    assert select_candidate([], strategy, T0, _vehicle(), _settings(), {}) is None


# --- MatchingStrategy defaults/validation ------------------------------------


def test_matching_strategy_defaults_to_progress_match():
    strategy = MatchingStrategy(key="trip_id")

    assert strategy.mode == "progress_match"
    assert strategy.acceptance_margin == 0.2
    assert strategy.allow_shared_trip is False


def test_matching_strategy_acceptance_margin_out_of_range_falls_back_to_default():
    assert MatchingStrategy(key="trip_id", acceptance_margin=1.5).acceptance_margin == 0.2
    assert MatchingStrategy(key="trip_id", acceptance_margin=-0.1).acceptance_margin == 0.2


# --- match_vehicle --------------------------------------------------------------


def test_match_vehicle_strict_matches_the_unambiguous_candidate():
    trips = _trips(_trip_row("T1"))
    index = build_trip_match_index(trips, "route_short_name,direction_id")
    vehicle = _vehicle(route_short_name="100", direction_id=0)
    strategy = MatchingStrategy(key="route_short_name,direction_id", mode="strict")

    result = match_vehicle(vehicle, index, strategy, T0, _settings(), {})

    assert result.trip_id == "T1"


def test_match_vehicle_strict_returns_none_when_ambiguous():
    trips = _trips(_trip_row("T1"), _trip_row("T2"))
    index = build_trip_match_index(trips, "route_short_name,direction_id")
    vehicle = _vehicle(route_short_name="100", direction_id=0)
    strategy = MatchingStrategy(key="route_short_name,direction_id", mode="strict")

    assert match_vehicle(vehicle, index, strategy, T0, _settings(), {}) is None


def test_match_vehicle_returns_none_when_no_candidate_for_the_key():
    index: dict = {}
    vehicle = _vehicle(route_short_name="999", direction_id=0)
    strategy = MatchingStrategy(key="route_short_name,direction_id")

    assert match_vehicle(vehicle, index, strategy, T0, _settings(), {}) is None


def test_match_vehicle_progress_match_resolves_the_correct_instance_among_shared_trip_ids():
    """Same shape as the real TripsSnapshot.trips index: with two
    frequency-expanded instances of the same trip_id active at once,
    "progress_match" must return the one whose elapsed-time fraction (given
    its OWN start_dt/end_dt) actually agrees with the vehicle's shape
    progress - not just any instance sharing that trip_id - with ITS OWN
    start_dt attached, not the other instance's.
    """
    trips = _trips_snapshot_shaped(
        _trip_row("T1", start_dt=T0 + timedelta(hours=1), end_dt=T0 + timedelta(hours=2)),
        _trip_row("T1", start_dt=T0, end_dt=T0 + timedelta(hours=1)),
    )
    index = build_trip_match_index(trips, "route_short_name,direction_id")
    # 50min into the T0 instance (time_fraction=0.833), 10min before the
    # T0+1h instance even starts (time_fraction clamps to 0.0). Vehicle at
    # x=80 on the shared 100-unit shape -> shape_fraction=0.8, which agrees
    # with the T0 instance (deviation ~0.033), not the T0+1h one
    # (deviation=0.8).
    vehicle = _vehicle(route_short_name="100", direction_id=0, point=Point(80, 0))
    strategy = MatchingStrategy(key="route_short_name,direction_id")
    now = T0 + timedelta(minutes=50)

    result = match_vehicle(vehicle, index, strategy, now, _settings(), {})

    assert result.trip_id == "T1"
    assert result.start_dt == T0
