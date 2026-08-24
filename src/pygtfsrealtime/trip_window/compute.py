import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import geopandas as gpd
import pandas as pd

from pygtfsrealtime.schedule.snapshot import GtfsSnapshot
from pygtfsrealtime.settings import Settings

logger = logging.getLogger(__name__)

_WEEKDAY_COLUMNS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


@dataclass(frozen=True)
class TripsSnapshot:
    """The rolling window of trip instances currently active or upcoming,
    as published by `TripWindowLoop` each time it rebuilds.

    Attributes:
        trips: one row per trip instance, indexed by the composite
            MultiIndex (trip_id, start_dt) with drop=False. A GeoDataFrame
            with shape_geometry as the active geometry column (mirrors
            shapes.txt: geometry=full path is primary, start_zone/end_zone
            are secondary plain object columns, alongside start_zone_source/
            end_zone_source recording which of "stop"/"terminal"/"shape"
            resolved each one - see
            pygtfsrealtime.schedule.compute.build_trip_endpoints).
        window_start: the window's start timestamp.
        window_end: the window's end timestamp.
        gtfs_hash: hash of the GtfsSnapshot this window was built from.
    """

    trips: gpd.GeoDataFrame
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    gtfs_hash: str


def resolve_active_service_ids(
    calendar: pd.DataFrame, calendar_dates: pd.DataFrame, service_date: date
) -> set[str]:
    """Which service_ids run on a given calendar date: calendar.txt's regular
    weekly pattern (day-of-week + start_date/end_date), adjusted by
    calendar_dates.txt's day-specific exceptions (exception_type 1 adds a
    service not otherwise scheduled that day, 2 removes one that is).

    Args:
        calendar: calendar.txt.
        calendar_dates: calendar_dates.txt.
        service_date: the calendar date to resolve.

    Returns:
        The set of active service_ids for `service_date`.
    """
    weekday_column = _WEEKDAY_COLUMNS[service_date.weekday()]
    regular = set(
        calendar.loc[
            calendar[weekday_column]
            & (calendar["start_date"] <= service_date)
            & (service_date <= calendar["end_date"]),
            "service_id",
        ]
    )

    day_exceptions = calendar_dates[calendar_dates["date"] == service_date]
    added = set(day_exceptions.loc[day_exceptions["exception_type"] == 1, "service_id"])
    removed = set(day_exceptions.loc[day_exceptions["exception_type"] == 2, "service_id"])

    return (regular - removed) | added


def candidate_service_dates(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    max_offset: pd.Timedelta,
    max_lookback_days: int = 3,
) -> list[date]:
    """Which service days could contribute a trip instance overlapping
    [window_start, window_end).

    A service day's stop_times/frequencies offsets can exceed 24h (GTFS's way of
    expressing past-midnight trips), so a trip anchored to a service day before
    window_start's calendar date can still land inside the window -
    lookback_days is derived from the feed's own largest offset rather than
    hardcoded to a fixed number of days. No forward lookahead past
    window_end's date is needed: a service day's earliest possible instance
    starts at its own midnight.

    `max_lookback_days` (see Settings.trip_window_max_lookback_days) caps
    lookback_days so a corrupted feed's absurd offset can't blow up the range
    of calendar days checked.

    Args:
        window_start: the trip window's start timestamp.
        window_end: the trip window's end timestamp.
        max_offset: the feed's largest stop_times/frequencies offset.
        max_lookback_days: cap on how many days back to look.

    Returns:
        The candidate service dates, oldest first.
    """
    lookback_days = max(1, math.ceil(max_offset / pd.Timedelta(days=1)))
    if lookback_days > max_lookback_days:
        logger.warning(
            "trip_window: feed's largest offset (%s) implies %d lookback day(s), "
            "clamping to %d - check for corrupted *_time/headway_secs data",
            max_offset,
            lookback_days,
            max_lookback_days,
        )
        lookback_days = max_lookback_days

    start_date = window_start.date() - timedelta(days=lookback_days)
    end_date = window_end.date()
    return [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]


def expand_frequencies_instances(
    frequencies: pd.DataFrame, stop_times: pd.DataFrame
) -> pd.DataFrame:
    """Expand frequencies.txt into one row per concrete departure instance.

    Each instance's end_offset comes from the ORIGINAL trip's own duration
    (stop_times.txt's arrival_time - departure_time, already linearized), never
    from headway_secs - a trip's running time and how often it repeats are
    unrelated quantities in GTFS, and must never be conflated
    (inst_end = inst_start + headway is wrong).

    Merge is how="inner" with no defensive fallback: reconcile_trip_coverage
    already guarantees every surviving frequencies.txt trip_id is present in
    stop_times.txt.

    Args:
        frequencies: frequencies.txt, restricted to active trips.
        stop_times: stop_times.txt, already linearized, restricted to
            active trips.

    Returns:
        One row per departure instance, with trip_id/start_offset/end_offset.
    """
    duration = stop_times["arrival_time"] - stop_times["departure_time"]
    freq = frequencies.merge(
        duration.rename("duration"), left_on="trip_id", right_index=True, how="inner"
    ).reset_index(drop=True)

    n_trips = (freq["end_time"] - freq["start_time"]) // freq["headway_secs"] + 1
    expanded = freq.loc[freq.index.repeat(n_trips)].copy()
    k = expanded.groupby(level=0).cumcount()
    expanded["start_offset"] = expanded["start_time"] + k * expanded["headway_secs"]
    expanded["end_offset"] = expanded["start_offset"] + expanded["duration"]

    return expanded[["trip_id", "start_offset", "end_offset"]].reset_index(drop=True)


def stop_times_only_instances(stop_times: pd.DataFrame, frequencies: pd.DataFrame) -> pd.DataFrame:
    """One instance per trip that has no frequencies.txt row - its stop_times.txt
    departure/arrival (already linearized) already are its one and only window,
    no expansion needed.

    Args:
        stop_times: stop_times.txt, already linearized, restricted to
            active trips.
        frequencies: frequencies.txt, restricted to active trips.

    Returns:
        One row per trip, with trip_id/start_offset/end_offset.
    """
    has_frequencies = set(frequencies["trip_id"])
    only = stop_times[~stop_times.index.isin(has_frequencies)]
    return pd.DataFrame(
        {
            "trip_id": only["trip_id"],
            "start_offset": only["departure_time"],
            "end_offset": only["arrival_time"],
        }
    ).reset_index(drop=True)


def anchor_offsets_to_service_day(
    offsets: pd.DataFrame, service_day: date, timezone: ZoneInfo
) -> pd.DataFrame:
    """Anchor trip_id/start_offset/end_offset (Timedelta, possibly >24h) to a
    specific service day's absolute datetime. Timedelta + Timestamp already
    handles offsets past midnight without any manual day-splitting.

    Args:
        offsets: trip_id/start_offset/end_offset rows.
        service_day: the calendar date the offsets are anchored to.
        timezone: the feed's resolved timezone.

    Returns:
        trip_id/start_dt/end_dt rows.
    """
    day_start = pd.Timestamp(service_day, tz=timezone)
    result = offsets.copy()
    result["start_dt"] = day_start + result["start_offset"]
    result["end_dt"] = day_start + result["end_offset"]
    return result[["trip_id", "start_dt", "end_dt"]]


def build_service_day_instances(
    gtfs_files: dict[str, pd.DataFrame],
    service_day: date,
    active_service_ids: set[str],
    timezone: ZoneInfo,
) -> pd.DataFrame:
    """All trip instances (frequency-expanded + stop-times-only) anchored to one
    service day, restricted to that day's active service_ids.

    Args:
        gtfs_files: the ingested GTFS files.
        service_day: the calendar date to build instances for.
        active_service_ids: service_ids running on `service_day`.
        timezone: the feed's resolved timezone.

    Returns:
        trip_id/start_dt/end_dt rows for every active instance that day.
    """
    trips = gtfs_files["trips.txt"]
    active_trip_ids = set(trips.loc[trips["service_id"].isin(active_service_ids), "trip_id"])

    frequencies = gtfs_files["frequencies.txt"]
    frequencies = frequencies[frequencies["trip_id"].isin(active_trip_ids)]

    stop_times = gtfs_files["stop_times.txt"]
    active_stop_times = stop_times[stop_times.index.isin(active_trip_ids)]

    freq_instances = expand_frequencies_instances(frequencies, active_stop_times)
    stop_time_instances = stop_times_only_instances(active_stop_times, frequencies)
    combined = pd.concat([freq_instances, stop_time_instances], ignore_index=True)

    return anchor_offsets_to_service_day(combined, service_day, timezone)


def assemble_trip_window_columns(
    instances: pd.DataFrame, gtfs_files: dict[str, pd.DataFrame]
) -> gpd.GeoDataFrame:
    """Join each trip instance to its route/shape metadata and its already-resolved
    start_zone/end_zone (plus which source - stop/terminal/shape - each one actually
    came from) - those were embedded onto trips.txt at ingest time (see
    pygtfsrealtime.schedule.compute.build_trip_endpoints), so no geometry is looked up
    or computed here, only carried through. The source columns are what let a
    consumer (pygtfsrealtime.realtime.observations.build_observation) apply the
    matching GeometryThreshold per endpoint instead of guessing one.

    Returned as a GeoDataFrame with shape_geometry as the active geometry column
    - same convention as shapes.txt itself (geometry=full path is primary,
    start_zone/end_zone stay plain object columns, same as shapes.txt's own
    zone/start_zone/end_zone) - so this carries a real .crs and supports
    .sindex/vectorized geopandas operations instead of just holding shapely
    objects in an untracked plain column.

    Args:
        instances: trip_id/start_dt/end_dt rows.
        gtfs_files: the ingested GTFS files, including trips.txt/routes.txt/
            shapes.txt.

    Returns:
        `instances` joined to route/shape metadata, start_zone/end_zone, and
        start_zone_source/end_zone_source, as a GeoDataFrame.
    """
    trips = gtfs_files["trips.txt"]
    routes = gtfs_files["routes.txt"]
    shapes = gtfs_files["shapes.txt"]

    trip_columns = trips[
        [
            "trip_id",
            "route_id",
            "direction_id",
            "shape_id",
            "start_zone",
            "start_zone_source",
            "end_zone",
            "end_zone_source",
        ]
    ]
    result = instances.merge(trip_columns, on="trip_id", how="left")
    result = result.merge(routes[["route_id", "route_short_name"]], on="route_id", how="left")
    result["shape_geometry"] = result["shape_id"].map(shapes["geometry"])

    result = result[
        [
            "trip_id",
            "start_dt",
            "end_dt",
            "route_id",
            "route_short_name",
            "direction_id",
            "start_zone",
            "start_zone_source",
            "end_zone",
            "end_zone_source",
            "shape_geometry",
        ]
    ]
    return gpd.GeoDataFrame(result, geometry="shape_geometry", crs=shapes.crs)


def build_window_trip_instances(
    gtfs_files: dict[str, pd.DataFrame],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    timezone: ZoneInfo,
    max_lookback_days: int = 3,
) -> gpd.GeoDataFrame:
    """Every trip instance overlapping [window_start, window_end) - a trip already
    running at window_start (start_dt < window_start < end_dt) is included, not
    just trips starting after window_start, so a vehicle mid-trip at rebuild time
    still has an instance to match against.

    Args:
        gtfs_files: the ingested GTFS files.
        window_start: the trip window's start timestamp.
        window_end: the trip window's end timestamp.
        timezone: the feed's resolved timezone.
        max_lookback_days: cap on how many days back to look for candidate
            service days.

    Returns:
        Every overlapping trip instance, indexed by (trip_id, start_dt).
    """
    frequencies = gtfs_files["frequencies.txt"]
    stop_times = gtfs_files["stop_times.txt"]

    max_offset = pd.Timedelta(0)
    if not frequencies.empty:
        max_offset = max(max_offset, frequencies["end_time"].max())
    if not stop_times.empty:
        max_offset = max(max_offset, stop_times["arrival_time"].max())

    calendar = gtfs_files["calendar.txt"]
    calendar_dates = gtfs_files["calendar_dates.txt"]

    day_frames = []
    for service_day in candidate_service_dates(
        window_start, window_end, max_offset, max_lookback_days
    ):
        active_service_ids = resolve_active_service_ids(calendar, calendar_dates, service_day)
        if not active_service_ids:
            continue
        day_frames.append(
            build_service_day_instances(gtfs_files, service_day, active_service_ids, timezone)
        )

    instances = (
        pd.concat(day_frames, ignore_index=True)
        if day_frames
        else pd.DataFrame(columns=["trip_id", "start_dt", "end_dt"])
    )

    overlapping = instances[
        (instances["start_dt"] < window_end) & (instances["end_dt"] > window_start)
    ]

    result = assemble_trip_window_columns(overlapping, gtfs_files)
    return result.set_index(["trip_id", "start_dt"], drop=False)


def _localize(now: datetime, timezone: ZoneInfo) -> pd.Timestamp:
    """Convert/localize `now` to a tz-aware Timestamp in `timezone`."""
    timestamp = pd.Timestamp(now)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)


def build_trips_snapshot(
    gtfs_snapshot: GtfsSnapshot, settings: Settings, now: datetime
) -> TripsSnapshot:
    """Build the rolling trip window - settings.trip_window_loop_schedule.interval
    (the window length) plus settings.trip_window_margin (safety tail) - starting
    at `now`, from an already-ingested GtfsSnapshot. No geometry/buffer
    computation happens here - that's all done once per GTFS refresh, not once
    per window rebuild.

    Anchored to gtfs_snapshot.resolved_timezone (resolved once per GTFS refresh
    from agency.txt, falling back to settings.timezone - see
    pygtfsrealtime.schedule.snapshot.resolve_gtfs_timezone), not settings.timezone
    directly - the feed's own timezone is what its stop_times/frequencies
    offsets are actually anchored to.

    Args:
        gtfs_snapshot: the current GTFS static feed state.
        settings: window length/margin/lookback configuration.
        now: the current cycle's timestamp; the window's start.

    Returns:
        The rebuilt `TripsSnapshot`.
    """
    timezone = gtfs_snapshot.resolved_timezone
    window_start = _localize(now, timezone)
    window_length = timedelta(seconds=settings.trip_window_loop_schedule.interval)
    window_end = window_start + window_length + settings.trip_window_margin

    trips = build_window_trip_instances(
        gtfs_snapshot.gtfs_files,
        window_start,
        window_end,
        timezone,
        settings.trip_window_max_lookback_days,
    )

    return TripsSnapshot(
        trips=trips,
        window_start=window_start,
        window_end=window_end,
        gtfs_hash=gtfs_snapshot.gtfs_hash,
    )
