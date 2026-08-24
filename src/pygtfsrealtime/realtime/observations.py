from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Protocol

from shapely.geometry.base import BaseGeometry

from pygtfsrealtime.realtime.ingest import VehicleReport
from pygtfsrealtime.settings import (
    GeometryThreshold,
    Settings,
    StationaryThreshold,
    TripEndpointSource,
)


class TransitionReason(Enum):
    """Why a VehicleFSM ended up (or stayed) FREE this cycle."""

    NO_SIGNAL = auto()
    NO_CANDIDATE_TRIP = auto()
    OFF_PATH = auto()
    AT_TERMINAL = auto()
    STATIONARY = auto()
    STALE_TRIP = auto()


class TripCandidate(Protocol):
    """The shape a matched trip row (e.g. a TripsSnapshot.trips row) must have
    for every consumer that reads one - the observation predicates below
    (shape_geometry/start_zone/end_zone/start_zone_source/end_zone_source),
    pygtfsrealtime.pb.protobuf.build_pb (trip_id/start_dt/route_id/direction_id,
    for GTFS-RT TripDescriptor), and pygtfsrealtime.realtime.matching.select_candidate
    (end_dt, alongside start_dt, to compute a candidate's elapsed-time
    fraction for mode="progress_match"). Decouples all of them from pandas/
    geopandas or any concrete trip-window/matching class.

    start_zone_source/end_zone_source record which of "stop"/"terminal"/"shape"
    actually resolved that endpoint (see
    pygtfsrealtime.schedule.compute.build_trip_endpoints) - each source's zone
    was built with a different GeometryThreshold, so build_observation needs to
    know which one to re-apply when checking `at_terminal`, instead of always
    using settings.terminal_geometry regardless of where the zone came from.
    """

    trip_id: str
    start_dt: datetime
    end_dt: datetime
    route_id: str
    direction_id: int
    shape_geometry: BaseGeometry
    start_zone: BaseGeometry
    start_zone_source: TripEndpointSource
    end_zone: BaseGeometry
    end_zone_source: TripEndpointSource


class ObservationWindow:
    """Sliding window of a vehicle's recent projected positions, used to decide
    whether it has been stationary. Positions are stored already projected
    (VehicleReport.point's CRS, e.g. UTM meters) - no lat/lon degree approximation.

    `entries` seeds the window from a prior cycle's state (see
    pygtfsrealtime.realtime.cache) - defaults to empty, so
    ObservationWindow() with no arguments starts with no restored state.
    """

    def __init__(self, entries: Iterable[tuple[datetime, float, float]] = ()) -> None:
        """Args:
        entries: prior (datetime, x, y) positions to seed the window with.
        """
        self._entries: deque[tuple[datetime, float, float]] = deque(entries)

    def entries(self) -> tuple[tuple[datetime, float, float], ...]:
        """The window's current (datetime, x, y) entries, oldest first."""
        return tuple(self._entries)

    def push(self, vehicle: VehicleReport, now: datetime, interval: timedelta) -> None:
        """Add a new position, dropping entries older than `interval`.

        Args:
            vehicle: this cycle's vehicle observation.
            now: the current cycle's timestamp.
            interval: how far back the window looks.
        """
        cutoff = now - interval
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()
        if vehicle.datetime >= cutoff:
            self._entries.append((vehicle.datetime, vehicle.point.x, vehicle.point.y))

    def is_stationary(self, threshold: StationaryThreshold) -> bool:
        """Whether the window's positions all fit within `threshold`'s
        drift distance - an empty window counts as stationary.
        """
        if not self._entries:
            return True

        xs = [x for _, x, _ in self._entries]
        ys = [y for _, _, y in self._entries]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        diagonal_squared = width * width + height * height
        return diagonal_squared <= threshold.distance**2


def is_close_to_path(
    point: BaseGeometry, geometry: BaseGeometry, threshold: GeometryThreshold
) -> bool:
    """Whether `point` counts as close enough to `geometry` under `threshold`.

    threshold.mode == "buffer" expects `geometry` to already be a buffered
    zone (point-in-polygon containment); "distance" compares the raw
    point-to-geometry distance instead - same convention as
    pygtfsrealtime.schedule.compute._zone_geometry.

    Args:
        point: the vehicle's projected position.
        geometry: the reference geometry (a buffer polygon or raw geometry,
            per `threshold.mode`).
        threshold: the proximity check to apply.

    Returns:
        Whether `point` is close enough to `geometry`.
    """
    if threshold.mode == "buffer":
        return point.within(geometry)
    return point.distance(geometry) <= threshold.distance


def threshold_for_zone_source(source: TripEndpointSource, settings: Settings) -> GeometryThreshold:
    """Which GeometryThreshold governs a resolved trip endpoint's proximity check -
    the same one that built that endpoint's zone geometry in the first place (see
    pygtfsrealtime.schedule.compute.build_trip_endpoints/build_gtfs_schedule_zones),
    so build/check always agree on buffer-vs-distance mode and the distance value.

    "terminal" uses settings.terminal_geometry. Both "stop" and "shape" use
    settings.stop_geometry: a "shape" endpoint is a single coordinate (the
    shape's first/last point), same shape as an individual stop, never a
    multi-platform terminal area - and it's never checked with
    settings.shape_geometry, whose margin is meant for "is this vehicle on the
    route" (typically much tighter than what counts as "arrived").
    """
    if source == "terminal":
        return settings.terminal_geometry
    return settings.stop_geometry


def is_in_terminal_zone(
    point: BaseGeometry,
    start_zone: BaseGeometry,
    start_threshold: GeometryThreshold,
    end_zone: BaseGeometry,
    end_threshold: GeometryThreshold,
) -> bool:
    """Whether `point` is close enough to either the trip's start or end zone -
    each checked with its own threshold, since start/end can each have resolved
    through a different source (see threshold_for_zone_source).
    """
    return is_close_to_path(point, start_zone, start_threshold) or is_close_to_path(
        point, end_zone, end_threshold
    )


def trip_duration_exceeded(ongoing_since: datetime, now: datetime, threshold: timedelta) -> bool:
    """Whether a BUSY vehicle has been on its current trip longer than `threshold`."""
    return now - ongoing_since >= threshold


def signal_lost(last_observed_at: datetime, now: datetime, threshold: timedelta) -> bool:
    """Whether a vehicle hasn't reported a new GPS observation within `threshold`."""
    return now - last_observed_at >= threshold


@dataclass(frozen=True)
class VehicleObservation:
    """This cycle's resolved state for one vehicle, ready for
    `VehicleFSM.transition()`.

    Attributes:
        trip: the trip relevant to the FSM's current decision, or None.
        on_path: whether the vehicle is close enough to the trip's route.
        at_terminal: whether the vehicle is close enough to a terminal zone.
        stationary: whether the vehicle hasn't moved meaningfully recently.
        signal_lost: whether the vehicle hasn't reported recently enough to
            be trusted.
    """

    trip: TripCandidate | None
    on_path: bool
    at_terminal: bool
    stationary: bool
    signal_lost: bool


def build_observation(
    vehicle: VehicleReport,
    window: ObservationWindow,
    trip: TripCandidate | None,
    settings: Settings,
    now: datetime,
) -> VehicleObservation:
    """Compute the observation a VehicleFSM.transition() call needs this cycle.

    Args:
        vehicle: this cycle's vehicle observation.
        window: the vehicle's recent-positions window (see
            ObservationWindow.is_stationary).
        trip: whichever trip is relevant to the FSM's current decision - a
            freshly matched candidate while FREE, or the FSM's own
            current_trip while BUSY. The caller decides which; this
            function only evaluates it.
        settings: proximity/staleness thresholds to apply.
        now: the current cycle's timestamp.

    Returns:
        The resolved `VehicleObservation`.
    """
    on_path = (
        is_close_to_path(vehicle.point, trip.shape_geometry, settings.shape_geometry)
        if trip is not None
        else False
    )
    at_terminal = (
        is_in_terminal_zone(
            vehicle.point,
            trip.start_zone,
            threshold_for_zone_source(trip.start_zone_source, settings),
            trip.end_zone,
            threshold_for_zone_source(trip.end_zone_source, settings),
        )
        if trip is not None
        else False
    )
    return VehicleObservation(
        trip=trip,
        on_path=on_path,
        at_terminal=at_terminal,
        stationary=window.is_stationary(settings.stationary_threshold),
        signal_lost=signal_lost(vehicle.datetime, now, settings.signal_loss_threshold),
    )
