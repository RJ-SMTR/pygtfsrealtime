import logging
from typing import Literal

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiPoint, Point
from shapely.geometry.base import BaseGeometry

from pygtfsrealtime.settings import (
    GeometryThreshold,
    Settings,
    TerminalGeometryThreshold,
    TripEndpointSource,
)

logger = logging.getLogger(__name__)


def build_stops_geometry(stops: pd.DataFrame, projection: str) -> gpd.GeoDataFrame:
    """stop_lat/stop_lon (already validated float columns) -> a GeoDataFrame indexed by
    stop_id, with a single projected Point geometry column replacing the raw coordinates.

    The same semantic-type idea as GTFSColumnSchema.to_semantic_type, one level up: what's
    safe to validate as independent bounded floats isn't the shape the rest of the program
    should work with, which is a single geographic point.

    Args:
        stops: stops.txt, with validated stop_lat/stop_lon columns.
        projection: the target UTM CRS.

    Returns:
        A GeoDataFrame indexed by stop_id, with a projected Point geometry.
    """
    gdf = gpd.GeoDataFrame(
        stops[["stop_id", "parent_station"]],
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    ).to_crs(projection)
    return gdf.set_index("stop_id", drop=False)


def build_shapes_geometry(shapes: pd.DataFrame, projection: str) -> gpd.GeoDataFrame:
    """shape_pt_lat/shape_pt_lon (already validated, one row per point) -> a GeoDataFrame
    indexed by shape_id, with one projected LineString per shape replacing the raw,
    per-point rows.

    A LineString needs at least 2 coordinates, so a shape_id with a single point can't
    become one - those are dropped and logged, same as any other row a validator can't
    make sense of.

    Args:
        shapes: shapes.txt, one row per (shape_id, shape_pt_sequence).
        projection: the target UTM CRS.

    Returns:
        A GeoDataFrame indexed by shape_id, with a projected LineString
        geometry.
    """
    ordered = shapes.sort_values(["shape_id", "shape_pt_sequence"])

    point_counts = ordered.groupby("shape_id").size()
    too_short = point_counts[point_counts < 2].index
    if len(too_short):
        logger.warning(
            "shapes.txt: dropping %d shape_id(s) with fewer than 2 points: %s",
            len(too_short),
            sorted(too_short),
        )
        ordered = ordered[~ordered["shape_id"].isin(too_short)]

    # pandas-stubs doesn't model `include_groups` in DataFrameGroupBy.apply's
    # overloads, so it can't match this call to any of them.
    lines = ordered.groupby("shape_id").apply(  # type: ignore[call-overload]
        lambda df: LineString(zip(df["shape_pt_lon"], df["shape_pt_lat"], strict=True)),
        include_groups=False,
    )
    gdf = gpd.GeoDataFrame(
        {"shape_id": lines.index, "geometry": lines.values}, crs="EPSG:4326"
    ).to_crs(projection)
    return gdf.set_index("shape_id", drop=False)


def build_gtfs_schedule_geometries(
    gtfs_files: dict[str, pd.DataFrame], projection: str
) -> dict[str, pd.DataFrame]:
    """Replace stops.txt/shapes.txt's raw lat/lon columns with projected geometry, keyed by
    their respective id - the last step of the semantic-type pipeline that
    pygtfsrealtime.schedule.validate.validate_gtfs_schedule_types starts.

    Args:
        gtfs_files: the ingested GTFS files, including stops.txt/shapes.txt.
        projection: the target UTM CRS.

    Returns:
        The same files, with stops.txt/shapes.txt's lat/lon columns replaced
        by geometry.
    """
    result = dict(gtfs_files)
    result["stops.txt"] = build_stops_geometry(gtfs_files["stops.txt"], projection)
    result["shapes.txt"] = build_shapes_geometry(gtfs_files["shapes.txt"], projection)
    return result


def linearize_stop_times(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Collapse stop_times.txt from one row per (trip_id, stop_sequence) into one row per
    trip_id, keeping only each trip's first and last stop and discarding every
    intermediate one — the endpoints are what the rest of this library actually needs
    (where a trip starts/ends, and when).

    Deliberately not groupby("trip_id").agg(..., "first"/"last") for the time columns:
    those pandas aggregations skip NaN by default, so with arrival_time/departure_time
    now nullable they'd silently return an intermediate stop's time instead of the first/
    last stop's own (possibly missing) one. Pulling the whole first/last row by
    stop_sequence keeps the columns anchored to the actual endpoint stop.

    Args:
        stop_times: stop_times.txt, one row per (trip_id, stop_sequence).

    Returns:
        One row per trip_id, with first_stop_id/departure_time and
        last_stop_id/arrival_time columns.
    """
    ordered = stop_times.sort_values(["trip_id", "stop_sequence"])
    first = ordered.drop_duplicates(subset="trip_id", keep="first").set_index("trip_id")
    last = ordered.drop_duplicates(subset="trip_id", keep="last").set_index("trip_id")

    result = pd.DataFrame(
        {
            "first_stop_id": first["stop_id"],
            "departure_time": first["departure_time"],
            "last_stop_id": last["stop_id"],
            "arrival_time": last["arrival_time"],
        }
    )
    return result.reset_index().set_index("trip_id", drop=False)


def drop_stop_times_missing_endpoints(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Drop trips (post-linearize_stop_times) missing a departure_time or arrival_time at
    either endpoint.

    Per-stop, either being blank is legitimate (only the first/last stop_time of a trip
    is required to have one) - but once linearized to one row per trip, a trip missing
    either has no usable start or end time, so keeping it would just push the same
    "which value is real" problem downstream to whoever tries to build its time window.

    Args:
        stop_times: stop_times.txt, already linearized to one row per trip.

    Returns:
        The same rows, with any trip missing an endpoint time dropped.
    """
    total = len(stop_times)
    complete = stop_times.dropna(subset=["departure_time", "arrival_time"])

    dropped = total - len(complete)
    if dropped:
        logger.warning(
            "stop_times.txt: dropped %d/%d trip(s) missing a departure_time or "
            "arrival_time at an endpoint",
            dropped,
            total,
        )

    return complete


def reconcile_shape_coverage(
    gtfs_files: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Keep only shape_ids present in both trips.txt and shapes.txt - a trip referencing a
    shape_id with no geometry can't be matched against a path, and a shape geometry no trip
    references is dead weight nobody will ever look up.

    Works on shapes.txt either before or after build_shapes_geometry: raw (one row per point,
    shape_id only a column) or already built into a GeoDataFrame (one row per shape, shape_id
    both the index and a column) - either way "shape_id" is filtered as a column, so calling
    this twice is meant to be cheap and safe. Run once before geometry-building to avoid
    reprojecting/building LineStrings for shape_ids no trip will ever reference, and once
    again after, since build_shapes_geometry can itself drop a shape_id post-filter (fewer
    than 2 points) - the only way to catch that is after geometry actually gets built.

    Args:
        gtfs_files: the ingested GTFS files, including trips.txt/shapes.txt.

    Returns:
        The same files, with unmatched trips.txt/shapes.txt rows dropped.
    """
    trips = gtfs_files["trips.txt"]
    shapes = gtfs_files["shapes.txt"]

    common_shape_ids = set(trips["shape_id"]) & set(shapes["shape_id"])

    total_trips, total_shapes = len(trips), len(shapes)

    trips = trips[trips["shape_id"].isin(common_shape_ids)]
    shapes = shapes[shapes["shape_id"].isin(common_shape_ids)]

    dropped_trips = total_trips - len(trips)
    if dropped_trips:
        logger.warning(
            "trips.txt: dropped %d/%d row(s) with no matching shapes.txt shape_id",
            dropped_trips,
            total_trips,
        )

    dropped_shapes = total_shapes - len(shapes)
    if dropped_shapes:
        logger.warning(
            "shapes.txt: dropped %d/%d row(s) with no matching trips.txt shape_id",
            dropped_shapes,
            total_shapes,
        )

    result = dict(gtfs_files)
    result["trips.txt"] = trips
    result["shapes.txt"] = shapes
    return result


def reconcile_route_coverage(
    gtfs_files: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Keep only route_ids present in both trips.txt and routes.txt - a trip referencing a
    route_id with no routes.txt row has no route_short_name to match a GPS vehicle against,
    and a route no trip references is dead weight nobody will ever look up.

    Args:
        gtfs_files: the ingested GTFS files, including trips.txt/routes.txt.

    Returns:
        The same files, with unmatched trips.txt/routes.txt rows dropped.
    """
    trips = gtfs_files["trips.txt"]
    routes = gtfs_files["routes.txt"]

    common_route_ids = set(trips["route_id"]) & set(routes["route_id"])

    total_trips, total_routes = len(trips), len(routes)

    trips = trips[trips["route_id"].isin(common_route_ids)]
    routes = routes[routes["route_id"].isin(common_route_ids)]

    dropped_trips = total_trips - len(trips)
    if dropped_trips:
        logger.warning(
            "trips.txt: dropped %d/%d row(s) with no matching routes.txt route_id",
            dropped_trips,
            total_trips,
        )

    dropped_routes = total_routes - len(routes)
    if dropped_routes:
        logger.warning(
            "routes.txt: dropped %d/%d row(s) with no matching trips.txt route_id",
            dropped_routes,
            total_routes,
        )

    result = dict(gtfs_files)
    result["trips.txt"] = trips
    result["routes.txt"] = routes
    return result


def reconcile_trip_coverage(
    gtfs_files: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Keep only trip_ids present in both trips.txt and stop_times.txt (already linearized,
    indexed by trip_id) - a trip needs both its own trips.txt row (route_id/direction_id/
    shape_id) and its stop_times endpoints to be usable, so one without the other is dropped
    from both. frequencies.txt is then trimmed to that same surviving set too, but isn't part
    of the intersection itself: a trip is still a trip without frequency-based scheduling
    (this library just can't build a time window for it from frequencies.txt alone), so its
    absence from frequencies.txt shouldn't drop it from trips.txt/stop_times.txt.

    Args:
        gtfs_files: the ingested GTFS files, including trips.txt/stop_times.txt
            (already linearized, indexed by trip_id)/frequencies.txt.

    Returns:
        The same files, with unmatched rows dropped from all three.
    """
    trips = gtfs_files["trips.txt"]
    stop_times = gtfs_files["stop_times.txt"]
    frequencies = gtfs_files["frequencies.txt"]

    common_trip_ids = set(trips["trip_id"]) & set(stop_times.index)

    total_trips, total_stop_times, total_frequencies = (
        len(trips),
        len(stop_times),
        len(frequencies),
    )

    trips = trips[trips["trip_id"].isin(common_trip_ids)]
    stop_times = stop_times[stop_times.index.isin(common_trip_ids)]
    frequencies = frequencies[frequencies["trip_id"].isin(common_trip_ids)]

    dropped_trips = total_trips - len(trips)
    if dropped_trips:
        logger.warning(
            "trips.txt: dropped %d/%d row(s) with no matching stop_times.txt trip",
            dropped_trips,
            total_trips,
        )

    dropped_stop_times = total_stop_times - len(stop_times)
    if dropped_stop_times:
        logger.warning(
            "stop_times.txt: dropped %d/%d row(s) with no matching trips.txt trip",
            dropped_stop_times,
            total_stop_times,
        )

    dropped_frequencies = total_frequencies - len(frequencies)
    if dropped_frequencies:
        logger.warning(
            "frequencies.txt: dropped %d/%d row(s) whose trip isn't in both "
            "trips.txt and stop_times.txt",
            dropped_frequencies,
            total_frequencies,
        )

    result = dict(gtfs_files)
    result["trips.txt"] = trips
    result["stop_times.txt"] = stop_times
    result["frequencies.txt"] = frequencies
    return result


def _relation_fingerprint(
    gtfs_files: dict[str, pd.DataFrame],
) -> tuple[frozenset, frozenset, frozenset, frozenset, frozenset]:
    """Snapshot of the id sets reconcile_gtfs_schedule_relations iterates on,
    used to detect when a fixed point has been reached.
    """
    return (
        frozenset(gtfs_files["trips.txt"]["trip_id"]),
        frozenset(gtfs_files["shapes.txt"]["shape_id"]),
        frozenset(gtfs_files["routes.txt"]["route_id"]),
        frozenset(gtfs_files["stop_times.txt"].index),
        frozenset(gtfs_files["frequencies.txt"]["trip_id"]),
    )


def reconcile_gtfs_schedule_relations(
    gtfs_files: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Repeatedly apply reconcile_shape_coverage, reconcile_route_coverage and
    reconcile_trip_coverage until trips.txt/shapes.txt/routes.txt/stop_times.txt/
    frequencies.txt stop changing.

    A single pass of each isn't enough: the three relations (trip<->shape,
    trip<->route, trip<->stop_times) can cascade into each other. E.g.
    reconcile_trip_coverage can drop the last trip referencing some shape_id or
    route_id (because that trip had no usable stop_times), which orphans a row in
    shapes.txt/routes.txt that only reconcile_shape_coverage/reconcile_route_coverage
    would catch - and running those again could, in principle, drop a trip that was
    the last one covering some frequencies.txt entry, and so on. Iterating to a
    fixed point (rather than a fixed sequence of calls) is the only way to
    guarantee the five files stay mutually consistent regardless of which
    combination of rows a given feed happens to be missing.

    Terminates because every pass only removes rows - the fingerprint's cardinality
    is monotonically non-increasing and bounded below by 0, so it must eventually
    stop changing.

    Args:
        gtfs_files: the ingested GTFS files, including trips.txt/shapes.txt/
            routes.txt/stop_times.txt/frequencies.txt.

    Returns:
        The same files, with every mutually-inconsistent row removed from
        all five.
    """
    fingerprint = _relation_fingerprint(gtfs_files)
    while True:
        gtfs_files = reconcile_shape_coverage(gtfs_files)
        gtfs_files = reconcile_route_coverage(gtfs_files)
        gtfs_files = reconcile_trip_coverage(gtfs_files)

        new_fingerprint = _relation_fingerprint(gtfs_files)
        if new_fingerprint == fingerprint:
            return gtfs_files
        fingerprint = new_fingerprint


def _zone_geometry(geometry: gpd.GeoSeries, threshold: GeometryThreshold) -> gpd.GeoSeries:
    """Return `geometry` in whichever form a proximity check against
    `threshold.distance` needs: expanded into a buffer polygon
    (threshold.mode == "buffer") or left exactly as-is (mode == "distance") -
    decided once here instead of by every caller.
    """
    if threshold.mode == "buffer":
        return geometry.buffer(threshold.distance)
    return geometry


def build_stop_zones(stops: gpd.GeoDataFrame, threshold: GeometryThreshold) -> gpd.GeoDataFrame:
    """Precompute each stop's proximity zone (settings.stop_geometry) - the region a vehicle
    counts as "at this stop", used as a trip's endpoint when that stop isn't part of a larger
    terminal.

    Args:
        stops: stops.txt, already converted to point geometry.
        threshold: the proximity check to apply.

    Returns:
        `stops` with a `zone` column added.
    """
    result = stops.copy()
    result["zone"] = _zone_geometry(stops["geometry"], threshold)
    return result


def build_shape_zones(
    shapes: gpd.GeoDataFrame,
    path_threshold: GeometryThreshold,
    endpoint_threshold: GeometryThreshold,
) -> gpd.GeoDataFrame:
    """Precompute each shape's two proximity zones: `zone` (settings.shape_geometry) is the
    region a vehicle counts as "on this route's path"; `start_zone`/`end_zone` are the regions
    around the shape's first/last point, used as a trip's terminal fallback when stop_times.txt
    doesn't have usable endpoint data for it. These are checked with `settings.stop_geometry`
    (not `settings.shape_geometry`, and not `settings.terminal_geometry`) - a shape's endpoint
    is a single coordinate, same shape as an individual stop, never a multi-platform terminal
    area; and shape_geometry's margin is meant for "is this vehicle on the route", which is
    typically much tighter than what counts as "arrived".

    Args:
        shapes: shapes.txt, already converted to LineString geometry.
        path_threshold: the proximity check for `zone`.
        endpoint_threshold: the proximity check for `start_zone`/`end_zone`.

    Returns:
        `shapes` with `zone`/`start_zone`/`end_zone` columns added.
    """
    result = shapes.copy()
    result["zone"] = _zone_geometry(shapes["geometry"], path_threshold)

    start_points = gpd.GeoSeries(
        [Point(line.coords[0]) for line in shapes["geometry"]],
        index=shapes.index,
        crs=shapes.crs,
    )
    end_points = gpd.GeoSeries(
        [Point(line.coords[-1]) for line in shapes["geometry"]],
        index=shapes.index,
        crs=shapes.crs,
    )
    result["start_zone"] = _zone_geometry(start_points, endpoint_threshold)
    result["end_zone"] = _zone_geometry(end_points, endpoint_threshold)
    return result


def build_terminal_zones(
    stops: gpd.GeoDataFrame, threshold: TerminalGeometryThreshold
) -> gpd.GeoDataFrame:
    """Build one row per terminal - every stop_id that some other stop's parent_station points
    to - with its reference geometry (threshold.shape_mode: the terminal stop's own point, or
    the convex hull enclosing every child stop under it) and that geometry's proximity zone
    (threshold.distance/mode, same idea as build_stop_zones/build_shape_zones).

    A parent_station value with no matching stops.txt row is a dangling reference - logged and
    skipped, since there's no geometry to build a terminal out of.

    Args:
        stops: stops.txt, already converted to point geometry.
        threshold: the terminal shape/proximity check to apply.

    Returns:
        One row per terminal stop_id, with `geometry` and `zone` columns.
    """
    parent_ids = set(stops["parent_station"].dropna())
    missing_parents = parent_ids - set(stops.index)
    if missing_parents:
        logger.warning(
            "stops.txt: %d parent_station value(s) have no matching stop_id, "
            "skipping those terminals: %s",
            len(missing_parents),
            sorted(missing_parents),
        )
    terminal_ids = sorted(parent_ids & set(stops.index))

    rows = []
    for terminal_id in terminal_ids:
        if threshold.shape_mode == "terminal_point":
            geometry: BaseGeometry = stops.loc[terminal_id, "geometry"]
        else:
            children = stops[stops["parent_station"] == terminal_id]
            geometry = MultiPoint(list(children["geometry"])).convex_hull
        rows.append({"stop_id": terminal_id, "geometry": geometry})

    result = gpd.GeoDataFrame(
        rows, geometry="geometry", crs=stops.crs, columns=["stop_id", "geometry"]
    ).set_index("stop_id", drop=False)
    result["zone"] = _zone_geometry(result["geometry"], threshold)
    return result


def build_gtfs_schedule_zones(
    gtfs_files: dict[str, pd.DataFrame], settings: Settings
) -> dict[str, pd.DataFrame]:
    """Precompute every proximity zone build_stop_zones/build_shape_zones/build_terminal_zones
    can derive purely from settings + already-reconciled stops.txt/shapes.txt - so the
    buffer-vs-distance and terminal-shape choices in Settings only get evaluated once per GTFS
    refresh, not once per vehicle observation per cycle.

    Adds a "terminals" entry alongside the usual per-file keys - not a real GTFS file (hence no
    ".txt"), but derived from stops.txt's parent_station relationships.

    Args:
        gtfs_files: the ingested GTFS files, already reconciled.
        settings: the geometry thresholds to apply.

    Returns:
        The same files, with `zone` columns added to stops.txt/shapes.txt
        and a new "terminals" entry.
    """
    result = dict(gtfs_files)
    result["stops.txt"] = build_stop_zones(gtfs_files["stops.txt"], settings.stop_geometry)
    result["shapes.txt"] = build_shape_zones(
        gtfs_files["shapes.txt"], settings.shape_geometry, settings.stop_geometry
    )
    result["terminals"] = build_terminal_zones(gtfs_files["stops.txt"], settings.terminal_geometry)
    return result


def _resolve_trip_endpoint_zones(
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    stops: gpd.GeoDataFrame,
    terminals: gpd.GeoDataFrame,
    shapes: gpd.GeoDataFrame,
    source: TripEndpointSource,
    position: Literal["start", "end"],
) -> tuple[pd.Series, pd.Series]:
    """Pick, per trip, which already-precomputed zone represents one of its endpoints -
    and which source (stop/terminal/shape) that zone actually came from.

    No new geometry is computed here - stops.txt/terminals/shapes.txt's zone/
    start_zone/end_zone columns already reflect the buffer-vs-distance choice from
    their matching GeometryThreshold, so this only selects among them. `source`
    names the preferred one; combine_first fills any trip where that source can't
    resolve (e.g. "terminal" but the stop has no parent_station) from the
    remaining two, in order of specificity - stop, then terminal, then shape -
    since shape always resolves (every surviving trip has a valid shape_id, per
    reconcile_shape_coverage). The resolved source is tracked alongside the zone
    itself (not just the zone geometry) because different sources were built
    with different GeometryThreshold configs (see build_gtfs_schedule_zones) -
    checking a resolved zone later against the wrong one (e.g. always
    terminal_geometry, regardless of what actually built it) silently mismatches
    buffer-vs-distance mode and/or distance value.

    Args:
        trips: trips.txt, already reconciled.
        stop_times: stop_times.txt, already linearized.
        stops: stops.txt, with a `zone` column.
        terminals: the "terminals" entry, with a `zone` column.
        shapes: shapes.txt, with `start_zone`/`end_zone` columns.
        source: which zone to prefer for this endpoint.
        position: which end of the trip to resolve.

    Returns:
        A (zone, zone_source) pair of Series, both indexed like `trips` - the
        resolved zone geometry, and which of "stop"/"terminal"/"shape" it came
        from.
    """
    stop_column = "first_stop_id" if position == "start" else "last_stop_id"
    shape_column = "start_zone" if position == "start" else "end_zone"

    stop_ids = trips["trip_id"].map(stop_times[stop_column])
    stop_zone = stop_ids.map(stops["zone"])

    parent_station = stop_ids.map(stops["parent_station"])
    terminal_zone = parent_station.map(terminals["zone"])

    shape_zone = trips["shape_id"].map(shapes[shape_column])

    zones_by_source = {"stop": stop_zone, "terminal": terminal_zone, "shape": shape_zone}
    order = [source] + [
        candidate for candidate in ("stop", "terminal", "shape") if candidate != source
    ]

    resolved = zones_by_source[order[0]]
    resolved_source = pd.Series(order[0], index=resolved.index, dtype=object)
    resolved_source[resolved.isna()] = None
    for candidate in order[1:]:
        still_missing = resolved.isna()
        resolved = resolved.combine_first(zones_by_source[candidate])
        resolved_source[still_missing & zones_by_source[candidate].notna()] = candidate
    return resolved, resolved_source


def build_trip_endpoints(
    gtfs_files: dict[str, pd.DataFrame], settings: Settings
) -> dict[str, pd.DataFrame]:
    """Resolve and embed each trip's start/end zone directly onto trips.txt.

    This only depends on the already-finished GTFS static content (stops/
    terminals/shapes zones, trips/stop_times already reconciled) - not on any
    trip window or vehicle - so it belongs in the GTFS ingest pipeline,
    computed once per feed refresh, instead of being recomputed on every trip
    window rebuild.

    Args:
        gtfs_files: the ingested GTFS files, with zones already precomputed.
        settings: selects which zone source to prefer (see TripEndpointSource).

    Returns:
        The same files, with `start_zone`/`start_zone_source`/`end_zone`/
        `end_zone_source` columns added to trips.txt - the `_source` columns
        record which of "stop"/"terminal"/"shape" actually resolved that
        endpoint (see _resolve_trip_endpoint_zones), so a consumer can apply
        the matching GeometryThreshold later instead of guessing one.
    """
    trips = gtfs_files["trips.txt"]
    stop_times = gtfs_files["stop_times.txt"]
    stops = gtfs_files["stops.txt"]
    terminals = gtfs_files["terminals"]
    shapes = gtfs_files["shapes.txt"]
    source = settings.trip_endpoint_source

    trips = trips.copy()
    trips["start_zone"], trips["start_zone_source"] = _resolve_trip_endpoint_zones(
        trips, stop_times, stops, terminals, shapes, source, "start"
    )
    trips["end_zone"], trips["end_zone_source"] = _resolve_trip_endpoint_zones(
        trips, stop_times, stops, terminals, shapes, source, "end"
    )

    result = dict(gtfs_files)
    result["trips.txt"] = trips
    return result
