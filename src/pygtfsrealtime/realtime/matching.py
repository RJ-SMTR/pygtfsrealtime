from datetime import datetime, timedelta

import geopandas as gpd

from pygtfsrealtime.realtime.ingest import VehicleReport
from pygtfsrealtime.settings import (
    MATCH_KEY_COLUMNS,
    MatchingStrategy,
    MatchKey,
    Settings,
)


def filter_active_now(trips: gpd.GeoDataFrame, now: datetime) -> gpd.GeoDataFrame:
    """Which TripsSnapshot.trips rows are running right now - a cheap boolean
    mask, recomputed every cycle (unlike the window itself, rebuilt only
    every ~8h) since "active now" changes every tick.

    Args:
        trips: the current TripsSnapshot's trips.
        now: the current cycle's timestamp.

    Returns:
        The subset of `trips` whose [start_dt, end_dt] covers `now`.
    """
    return trips[(trips["start_dt"] <= now) & (now <= trips["end_dt"])]


def vehicle_match_key(vehicle: VehicleReport, key: MatchKey) -> tuple:
    """The candidate-index lookup key for a vehicle, under the given MatchKey.

    Assumes vehicle already has every field `key` needs - GPSIngester.ingest
    (pygtfsrealtime.realtime.ingest) drops observations missing them at
    ingestion, so this is a plain attribute read, not a revalidation.

    Args:
        vehicle: the vehicle to key.
        key: which field(s) to read (see MatchKey).

    Returns:
        A tuple of the vehicle's values for `key`'s column(s).
    """
    return tuple(getattr(vehicle, column) for column in MATCH_KEY_COLUMNS[key])


def build_trip_match_index(active_trips: gpd.GeoDataFrame, key: MatchKey) -> dict[tuple, list]:
    """Group active_trips' rows by `key`'s column(s), once per cycle.

    Each group is ordered by start_dt ascending, so mode="progress_match"
    (see select_candidate) breaks ties deterministically toward the earlier
    start_dt instead of depending on active_trips' incidental row order.

    Args:
        active_trips: the trips currently active (see filter_active_now).
        key: which field(s) to group by (see MatchKey).

    Returns:
        A dict mapping each group key to its list of candidate trip rows.
    """
    columns = list(MATCH_KEY_COLUMNS[key])
    # TripsSnapshot.trips is indexed by (trip_id, start_dt) with drop=False,
    # so "start_dt" is both an index level and a column - sort_values("start_dt")
    # would raise "ambiguous" without dropping the index first. The index
    # itself is never read below (itertuples()'s .Index isn't used - trip_id/
    # start_dt are read as columns via getattr), so dropping it is safe.
    ordered = active_trips.reset_index(drop=True).sort_values("start_dt")
    index: dict[tuple, list] = {}
    for row in ordered.itertuples():
        group_key = tuple(getattr(row, column) for column in columns)
        index.setdefault(group_key, []).append(row)
    return index


def _time_fraction(candidate, now: datetime) -> float | None:
    """Fraction of `candidate`'s scheduled duration elapsed as of `now`,
    clamped to [0, 1]. None for a degenerate zero/negative-duration trip -
    excluded rather than dividing by zero.
    """
    duration = candidate.end_dt - candidate.start_dt
    if duration <= timedelta(0):
        return None
    fraction = (now - candidate.start_dt) / duration
    return min(1.0, max(0.0, fraction))


def _shape_fraction(vehicle: VehicleReport, candidate, shape_threshold) -> float | None:
    """Fraction of `candidate`'s shape already traveled, found by projecting
    `vehicle`'s point onto the shape (shapely LineString.project - linear
    referencing, not a proximity check).

    None if the vehicle is farther than shape_threshold.distance from the
    shape - the raw point-to-line distance is used regardless of
    shape_threshold.mode ("buffer" is a boolean-check evaluation strategy;
    linear referencing needs the raw LineString and a raw distance value
    either way) - or if the shape has zero length (degenerate, would divide
    by zero).
    """
    shape = candidate.shape_geometry
    if shape.length <= 0:
        return None
    if vehicle.point.distance(shape) > shape_threshold.distance:
        return None
    return shape.project(vehicle.point) / shape.length


def _select_progress_match(
    candidates: list,
    vehicle: VehicleReport,
    strategy: MatchingStrategy,
    now: datetime,
    settings: Settings,
    claimed_by: dict[tuple[str, datetime], str],
):
    """Pick the candidate whose shape-progress fraction best agrees with its
    elapsed-time fraction (see MatchSelectionMode's "progress_match" comment
    in settings.py for the full rationale).

    A candidate survives only if both fractions are computable, their
    deviation is within strategy.acceptance_margin, and it isn't already
    claimed by another vehicle this cycle (unless strategy.allow_shared_trip).
    Among survivors, the smallest deviation wins; ties are broken first
    toward the not-yet-claimed candidate, then toward the earlier start_dt.
    """
    scored = []
    for candidate in candidates:
        time_fraction = _time_fraction(candidate, now)
        if time_fraction is None:
            continue
        shape_fraction = _shape_fraction(vehicle, candidate, settings.shape_geometry)
        if shape_fraction is None:
            continue
        deviation = abs(shape_fraction - time_fraction)
        if deviation > strategy.acceptance_margin:
            continue
        already_claimed = claimed_by.get((candidate.trip_id, candidate.start_dt)) is not None
        if already_claimed and not strategy.allow_shared_trip:
            continue
        scored.append((deviation, already_claimed, candidate.start_dt, candidate))

    if not scored:
        return None
    return min(scored, key=lambda entry: entry[:3])[3]


def select_candidate(
    candidates: list,
    strategy: MatchingStrategy,
    now: datetime,
    vehicle: VehicleReport,
    settings: Settings,
    claimed_by: dict[tuple[str, datetime], str],
):
    """Pick one trip row among candidates sharing a match key, or None.

    Extension point for a future score/ML-based mode: a new
    MatchSelectionMode value plus a branch here, without touching
    vehicle_match_key/build_trip_match_index/match_vehicle.

    Args:
        candidates: trip rows sharing a match key, ordered by start_dt.
        strategy: the matching criterion and selection mode - mode="strict"
            only reads strategy.mode; mode="progress_match" also reads
            strategy.acceptance_margin/allow_shared_trip.
        now: the current cycle's timestamp.
        vehicle: the vehicle being matched - only used by mode="progress_match",
            to project its GPS point onto each candidate's shape.
        settings: proximity thresholds - only used by mode="progress_match".
        claimed_by: (trip_id, start_dt) -> vehicle_id for every trip
            instance a BUSY vehicle is already on this cycle - only used by
            mode="progress_match" (see MatchingStrategy.allow_shared_trip).

    Returns:
        The selected trip row, or None if no candidate qualifies.
    """
    if not candidates:
        return None
    if strategy.mode == "strict":
        return candidates[0] if len(candidates) == 1 else None
    return _select_progress_match(candidates, vehicle, strategy, now, settings, claimed_by)


def match_vehicle(
    vehicle: VehicleReport,
    index: dict[tuple, list],
    strategy: MatchingStrategy,
    now: datetime,
    settings: Settings,
    claimed_by: dict[tuple[str, datetime], str],
):
    """The active trip (a TripsSnapshot.trips row) matching `vehicle` under
    `strategy`, or None. `index` is built once per cycle via
    build_trip_match_index, not once per vehicle.

    Args:
        vehicle: the vehicle to match.
        index: this cycle's candidate index (see build_trip_match_index).
        strategy: the matching criterion and selection mode.
        now: the current cycle's timestamp.
        settings: proximity thresholds - only used by mode="progress_match".
        claimed_by: (trip_id, start_dt) -> vehicle_id for every trip
            instance a BUSY vehicle is already on this cycle.

    Returns:
        The matched trip row, or None.
    """
    group_key = vehicle_match_key(vehicle, strategy.key)
    candidates = index.get(group_key, [])
    return select_candidate(candidates, strategy, now, vehicle, settings, claimed_by)
