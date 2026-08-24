import logging
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd
from shapely.geometry.base import BaseGeometry

from pygtfsrealtime.exceptions import FatalConfigurationError
from pygtfsrealtime.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GtfsSnapshot:
    """The GTFS static feed's fully validated, geometry/zone-augmented state,
    as published by `GTFSScheduleLoop` each time the feed changes.

    Attributes:
        gtfs_files: one DataFrame per GTFS file, cleaned and augmented with
            projected geometry/zone columns (see
            pygtfsrealtime.schedule.compute).
        trip_terminal_zones: each trip_id's (start, end) proximity zone.
        resolved_timezone: the timezone the feed's un-timezoned times should
            be interpreted in.
        gtfs_hash: hash of the raw feed bytes this snapshot was built from.
    """

    gtfs_files: dict[str, pd.DataFrame]
    trip_terminal_zones: dict[str, tuple[BaseGeometry, BaseGeometry]]
    resolved_timezone: ZoneInfo
    gtfs_hash: str


def _resolve_stop_zone(
    stop_id: str, stops: pd.DataFrame, terminals: pd.DataFrame
) -> BaseGeometry | None:
    """A stop's proximity zone: its parent terminal's zone if it has one,
    else its own zone; None if the stop itself isn't known.
    """
    if stop_id not in stops.index:
        return None

    parent_station = stops.loc[stop_id, "parent_station"]
    terminal_id = parent_station if isinstance(parent_station, str) and parent_station else stop_id

    if terminal_id in terminals.index:
        return terminals.loc[terminal_id, "zone"]
    if terminal_id in stops.index:
        return stops.loc[terminal_id, "zone"]
    return None


def build_trip_terminal_zones(
    gtfs_files: dict[str, pd.DataFrame],
) -> dict[str, tuple[BaseGeometry, BaseGeometry]]:
    """Map each trip_id to its (start, end) proximity zone.

    Primary source is stop_times.txt (already linearized to one row per trip_id, by
    pygtfsrealtime.schedule.compute.linearize_stop_times), resolving each endpoint's
    stop to its parent terminal's zone when it has one, else the stop's own zone.
    Trips stop_times.txt couldn't give a usable pair for (missing/incomplete rows,
    dropped earlier in the ingest pipeline) fall back to their shape's
    start_zone/end_zone via trips.txt's shape_id.

    Args:
        gtfs_files: the ingested GTFS files, already geometry/zone-augmented.

    Returns:
        A dict mapping each trip_id to its (start_zone, end_zone) pair.
    """
    stops = gtfs_files["stops.txt"]
    terminals = gtfs_files["terminals"]
    shapes = gtfs_files["shapes.txt"]
    trips = gtfs_files["trips.txt"].set_index("trip_id")
    stop_times = gtfs_files["stop_times.txt"]

    zones: dict[str, tuple[BaseGeometry, BaseGeometry]] = {}
    for row in stop_times.itertuples():
        zone_first = _resolve_stop_zone(str(row.first_stop_id), stops, terminals)
        zone_last = _resolve_stop_zone(str(row.last_stop_id), stops, terminals)
        if zone_first is not None and zone_last is not None:
            zones[str(row.trip_id)] = (zone_first, zone_last)

    for trip_id in set(trips.index) - zones.keys():
        shape_id = trips.loc[trip_id, "shape_id"]
        if shape_id not in shapes.index:
            continue
        zones[str(trip_id)] = (
            shapes.loc[shape_id, "start_zone"],
            shapes.loc[shape_id, "end_zone"],
        )

    return zones


def resolve_gtfs_timezone(gtfs_files: dict[str, pd.DataFrame], settings: Settings) -> ZoneInfo:
    """Resolve which timezone the feed's un-timezoned times (stop_times.txt,
    frequencies.txt) should be interpreted in.

    agency.txt is the primary source - the GTFS spec requires every agency in a
    feed to share one agency_timezone, so a single distinct value there is an
    unambiguous answer. If agency.txt gives more than one distinct value (a
    spec violation) or none at all (e.g. every row failed validation), that's
    logged and settings.timezone is used instead.

    Args:
        gtfs_files: the ingested GTFS files, including agency.txt.
        settings: fallback timezone source when agency.txt is ambiguous.

    Returns:
        The resolved timezone.

    Raises:
        FatalConfigurationError: if agency.txt has no usable single timezone
            and settings.timezone isn't set either - there'd be no way left
            to know which timezone the feed's times are in, so this raises
            (and propagates through run_periodic to stop the whole engine)
            rather than silently guessing (e.g. defaulting to UTC).
    """
    agency_timezones = gtfs_files["agency.txt"]["agency_timezone"]
    distinct_keys = sorted({zone.key for zone in agency_timezones})

    if len(distinct_keys) == 1:
        return agency_timezones.iloc[0]

    if distinct_keys:
        logger.warning(
            "agency.txt: %d conflicting agency_timezone value(s) found %s - the "
            "GTFS spec requires every agency in a feed to share the same "
            "timezone. Falling back to Settings.timezone.",
            len(distinct_keys),
            distinct_keys,
        )
    else:
        logger.warning(
            "agency.txt: no usable agency_timezone value found. Falling back to Settings.timezone."
        )

    if settings.timezone is not None:
        return settings.timezone

    raise FatalConfigurationError(
        "Could not determine the GTFS feed's timezone: agency.txt has no usable "
        "single agency_timezone and Settings.timezone isn't set."
    )


def build_gtfs_snapshot(
    gtfs_files: dict[str, pd.DataFrame], gtfs_hash: str, settings: Settings
) -> GtfsSnapshot:
    """Assemble a `GtfsSnapshot` from a cycle's ingested GTFS files.

    Args:
        gtfs_files: the ingested, geometry/zone-augmented GTFS files.
        gtfs_hash: hash of the raw feed bytes this snapshot is built from.
        settings: fallback timezone source (see resolve_gtfs_timezone).

    Returns:
        The assembled `GtfsSnapshot`.
    """
    return GtfsSnapshot(
        gtfs_files=gtfs_files,
        trip_terminal_zones=build_trip_terminal_zones(gtfs_files),
        resolved_timezone=resolve_gtfs_timezone(gtfs_files, settings),
        gtfs_hash=gtfs_hash,
    )
