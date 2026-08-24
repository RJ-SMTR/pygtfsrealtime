import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal, get_args
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def _coerce_choice(value, valid_values, default, label):
    """Return `value` if it's one of `valid_values`, else warn and return `default`."""
    if value not in valid_values:
        logger.warning(
            "%s=%r is not one of %s - using default %r instead.",
            label,
            value,
            valid_values,
            default,
        )
        return default
    return value


def _coerce_positive(value, default, label):
    """Return `value` if it's positive (> 0, or > timedelta(0)), else warn and
    return `default`.
    """
    is_positive = value > timedelta(0) if isinstance(value, timedelta) else value > 0
    if not is_positive:
        logger.warning("%s=%r must be positive - using default %r instead.", label, value, default)
        return default
    return value


def _coerce_range(value, low, high, default, label):
    """Return `value` if low <= value <= high, else warn and return `default`.

    Same warn-and-fall-back-to-default style as _coerce_choice/_coerce_positive
    - an out-of-range value is replaced wholesale, never clamped.
    """
    if not (low <= value <= high):
        logger.warning(
            "%s=%r must be between %r and %r - using default %r instead.",
            label,
            value,
            low,
            high,
            default,
        )
        return default
    return value


# What to do when two rows in a GTFS schedule file share a primary key:
# "keep_first" keeps the first occurrence and drops the rest; "drop_all"
# distrusts every row sharing that key and drops all of them, since there's no
# way to tell which one is correct.
PrimaryKeyDuplicatePolicy = Literal["keep_first", "drop_all"]

# How a proximity check decides whether a point counts as "close enough" to a
# reference geometry: "buffer" expands the reference geometry into a Polygon once
# (by `GeometryThreshold.distance`) and tests point-in-polygon containment;
# "distance" never materializes a buffer polygon and instead compares the raw
# point-to-geometry distance against `GeometryThreshold.distance` at check time.
# Both give the same yes/no answer for the same distance - the choice is a
# performance/memory trade-off (a precomputed buffer is cheap to re-check many
# times but costs memory/setup up front; a raw distance check costs nothing to
# set up but recomputes geometry math on every check), not a behavioral one.
GeometryCalculationMode = Literal["buffer", "distance"]

# Which geometry represents a terminal's "endpoint" before GeometryThreshold gets
# applied to it: "terminal_point" uses the parent stop's own coordinate;
# "stops_convex_hull" uses the convex hull enclosing every child stop/platform
# under that terminal - useful when a terminal's platforms are spread out enough
# that a single point undershoots it.
TerminalShapeMode = Literal["terminal_point", "stops_convex_hull"]

# Which already-precomputed zone (see build_stop_zones/build_shape_zones/
# build_terminal_zones) represents a trip's start/end: "stop" uses the trip's own
# first/last stop's zone (settings.stop_geometry); "terminal" uses that stop's
# parent station's zone (via parent_station, settings.terminal_geometry); "shape"
# uses the shape's start_zone/end_zone - a single coordinate (the shape's first/
# last point), so it's checked with settings.stop_geometry too, never
# settings.terminal_geometry (a multi-platform terminal area doesn't apply to a
# single point) nor settings.shape_geometry (that's a much tighter margin, meant
# for "is this vehicle on the route", not "has it arrived"). Picking a source
# never computes new geometry - it only selects a column. If the chosen source
# isn't available for a given trip, resolution falls through the other two in
# order of specificity: stop -> terminal -> shape (shape always resolves, since
# every surviving trip has a valid shape_id). Because different trips (or a
# trip's start vs end) can resolve through different sources, the ACTUAL
# resolved source is tracked per endpoint (trips.txt's start_zone_source/
# end_zone_source, see pygtfsrealtime.schedule.compute.build_trip_endpoints) so
# a consumer applies the matching GeometryThreshold, not a guessed one.
TripEndpointSource = Literal["stop", "terminal", "shape"]

# Whether a loop's interval is measured from the START or the END of the previous
# cycle - i.e. whether the cycle's own execution time is absorbed into the
# interval or not. Same idea as Java's ScheduledExecutorService.scheduleAtFixedRate
# vs scheduleWithFixedDelay: "fixed_rate" anchors the next tick to
# previous_start + interval, so a slow cycle eats into the next one's sleep (and
# can miss the deadline entirely, see MissedDeadlinePolicy); "fixed_delay" anchors
# it to previous_end + interval, so the cycle's own duration is never absorbed and
# a deadline can never be missed by construction.
LoopAccountingMode = Literal["fixed_rate", "fixed_delay"]

# What to do when a "fixed_rate" cycle takes longer than its interval and the next
# scheduled tick has already passed. Ignored under LoopAccountingMode="fixed_delay"
# - there, the next tick is always "now + interval" computed after the cycle
# finishes, so a deadline can never be missed. "immediate": run the next cycle
# right away, no sleep - catches up as fast as possible, but back-to-back cycles
# if work is consistently slow. "wait_full_interval": sleep the full interval
# starting now and reset the schedule to "now + interval" - for that one cycle,
# behaves like a fixed_delay cycle, but (unlike fixed_delay) later cycles that
# don't miss stay anchored to the reset grid. "skip_to_next_tick": keep the
# original grid and advance to whichever grid-aligned tick is next in the future,
# instead of resetting to now - coalesces missed ticks (common as "coalesce" in
# APScheduler or a "do nothing" misfire policy in Quartz) instead of replaying
# them, so cycles that do land stay aligned to the original schedule (e.g. every
# 30s on the :00/:30 mark) even after a long stall.
MissedDeadlinePolicy = Literal["immediate", "wait_full_interval", "skip_to_next_tick"]

# Which vehicle/trip field(s) pygtfsrealtime.realtime.matching uses to pair a
# VehicleReport (GPSEntry) with an active trip instance in TripsSnapshot.trips:
# "trip_id"/"route_id"/"route_short_name" match on a single field;
# "route_id,direction_id"/"route_short_name,direction_id" additionally
# require the trip's direction to match. Each key maps 1:1 to a tuple of
# column/attribute names shared by VehicleReport and TripsSnapshot.trips - see
# MATCH_KEY_COLUMNS below. Deliberately no "default" criterion exists
# anywhere (see MatchingStrategy.key) - which fields a GPS feed actually
# populates varies per operator, so picking one silently would invite
# misconfigured matching that goes unnoticed.
MatchKey = Literal[
    "trip_id",
    "route_id",
    "route_short_name",
    "route_id,direction_id",
    "route_short_name,direction_id",
]

# How to pick one trip among the candidates that share a MatchKey's value:
# "strict" only accepts the match when there's exactly one candidate -
# ambiguous means no match (the vehicle stays unmatched this cycle). Only
# behaves well when the feed guarantees no two candidates sharing a key can
# ever be concurrently active (no overlapping [start_dt, end_dt) windows for
# the same key) - otherwise every cycle both trips are live, "strict" sees an
# ambiguity that will never resolve itself and the vehicle never matches.
# "progress_match" (the default) scores each candidate by how well two
# independent progress signals agree: the fraction of the trip's scheduled
# duration elapsed so far (now relative to [start_dt, end_dt]) versus the
# fraction of the trip's shape already traveled, found by projecting the
# vehicle's GPS point onto the shape (shapely LineString.project) - linear
# referencing, not just a proximity check. A candidate is only eligible at
# all if the vehicle is within settings.shape_geometry.distance of that
# candidate's shape (raw point-to-line distance, regardless of
# GeometryThreshold.mode - buffer-vs-distance is a boolean-check evaluation
# strategy, irrelevant to linear referencing); among eligible candidates,
# only those whose |shape_fraction - time_fraction| deviation is within
# MatchingStrategy.acceptance_margin survive, and a trip already claimed by
# another BUSY vehicle this cycle is excluded unless
# MatchingStrategy.allow_shared_trip is True. The surviving candidate with
# the smallest deviation wins; ties are broken first toward the
# not-yet-claimed candidate, then toward the earlier start_dt (same
# determinism as before, since build_trip_match_index's groups are
# pre-sorted ascending by start_dt). Unlike a pure schedule-adherence
# check, this doesn't require the vehicle to literally be on time - it
# requires the vehicle's physical progress along the route to track its
# temporal progress through the trip, which tends to hold from the very
# first cycle instead of only converging after several BUSY cycles.
# Extension point for a future score/ML-based mode: it would be a third
# Literal value plus a new branch in matching.select_candidate, without
# touching the rest of the pipeline.
MatchSelectionMode = Literal["strict", "progress_match"]


@dataclass
class MatchingStrategy:
    """Which field(s) and selection rule to use when matching a GPS vehicle
    to an active trip instance.

    Attributes:
        key: which vehicle/trip field(s) to match on. No default - see
            MatchKey's comment above, the operator must pick one
            deliberately.
        mode: how to pick one trip among candidates sharing `key`'s value.
        acceptance_margin: only used by mode="progress_match" - the maximum
            allowed |shape_fraction - time_fraction| deviation (0 to 1) for a
            candidate to be selectable at all.
        allow_shared_trip: only used by mode="progress_match" - whether a
            trip instance already claimed by another BUSY vehicle this cycle
            can still be selected. False means at most one vehicle can be
            BUSY on a given trip instance at a time.
    """

    key: MatchKey
    mode: MatchSelectionMode = "progress_match"
    acceptance_margin: float = 0.2
    allow_shared_trip: bool = False

    def __post_init__(self) -> None:
        if self.key not in get_args(MatchKey):
            raise TypeError(
                f"MatchingStrategy.key={self.key!r} is not a valid MatchKey - "
                f"choose one of {get_args(MatchKey)}. There's no default here "
                "since which fields a GPS feed populates is operator-specific; "
                "picking one silently would invite misconfigured matching that "
                "goes unnoticed."
            )
        self.mode = _coerce_choice(
            self.mode, get_args(MatchSelectionMode), "progress_match", "MatchingStrategy.mode"
        )
        self.acceptance_margin = _coerce_range(
            self.acceptance_margin, 0.0, 1.0, 0.2, "MatchingStrategy.acceptance_margin"
        )


# Which VehicleReport/TripsSnapshot.trips attribute(s) a given MatchKey reads - the
# same names apply on both sides (VehicleReport carries GPSEntry's field names
# verbatim, TripsSnapshot.trips carries the matching GTFS column names), so
# one mapping serves both pygtfsrealtime.realtime.ingest.GPSIngester.ingest (drops
# GPS observations missing a field the configured strategy needs, at
# ingestion) and pygtfsrealtime.realtime.matching (builds the per-cycle candidate
# index). Lives here rather than in matching.py so ingest.py can import it
# without a circular import (matching.py already needs to import VehicleReport
# from ingest.py).
MATCH_KEY_COLUMNS: dict[MatchKey, tuple[str, ...]] = {
    "trip_id": ("trip_id",),
    "route_id": ("route_id",),
    "route_short_name": ("route_short_name",),
    "route_id,direction_id": ("route_id", "direction_id"),
    "route_short_name,direction_id": ("route_short_name", "direction_id"),
}


@dataclass
class LoopSchedule:
    """Timing configuration for one of the library's background loops.

    Attributes:
        interval: seconds between cycles (or, for the trip-window loop, only
            the seed for its first cycle - see run_conditional).
        accounting_mode: whether `interval` is measured from the previous
            cycle's start or its end.
        on_missed_deadline: what to do when a "fixed_rate" cycle overruns
            its interval. Ignored under "fixed_delay".
    """

    interval: float
    accounting_mode: LoopAccountingMode = "fixed_rate"
    on_missed_deadline: MissedDeadlinePolicy = "immediate"

    def __post_init__(self) -> None:
        self.accounting_mode = _coerce_choice(
            self.accounting_mode,
            get_args(LoopAccountingMode),
            "fixed_rate",
            "LoopSchedule.accounting_mode",
        )
        self.on_missed_deadline = _coerce_choice(
            self.on_missed_deadline,
            get_args(MissedDeadlinePolicy),
            "immediate",
            "LoopSchedule.on_missed_deadline",
        )


@dataclass
class StationaryThreshold:
    """How little movement, sustained for how long, counts as "stationary".

    Attributes:
        distance: max drift (meters) between the oldest and newest position
            in the observation window still considered stationary.
        interval: how far back the observation window looks.
    """

    distance: int = 30
    interval: timedelta = timedelta(seconds=900)

    def __post_init__(self) -> None:
        self.distance = _coerce_positive(self.distance, 30, "StationaryThreshold.distance")
        self.interval = _coerce_positive(
            self.interval, timedelta(seconds=900), "StationaryThreshold.interval"
        )


@dataclass
class GeometryThreshold:
    """A proximity check's distance and evaluation strategy.

    Attributes:
        distance: the threshold distance, in meters.
        mode: whether the check is done via a precomputed buffer polygon or
            a raw point-to-geometry distance comparison (see
            GeometryCalculationMode above).
    """

    distance: int
    mode: GeometryCalculationMode = "buffer"

    def __post_init__(self) -> None:
        self.mode = _coerce_choice(
            self.mode, get_args(GeometryCalculationMode), "buffer", "GeometryThreshold.mode"
        )


@dataclass
class TerminalGeometryThreshold(GeometryThreshold):
    """A `GeometryThreshold` for terminal proximity checks.

    Attributes:
        shape_mode: which geometry represents the terminal's "endpoint"
            before the distance/buffer threshold is applied (see
            TerminalShapeMode above).
    """

    shape_mode: TerminalShapeMode = "stops_convex_hull"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.shape_mode = _coerce_choice(
            self.shape_mode,
            get_args(TerminalShapeMode),
            "stops_convex_hull",
            "TerminalGeometryThreshold.shape_mode",
        )


@dataclass
class Settings:
    """Library-wide configuration: loop timing, geometry/proximity
    thresholds, trip matching, and validation policy. Every field has a
    usable default except `trip_matching` and `projection`, which must be set
    explicitly (see their comments below) - `GTFSRealtimeEngine` raises at
    construction time if either is still unset/invalid. Every other field
    that's given an invalid value (not one of its allowed choices, or not a
    positive number/duration where one is required) is corrected in
    `__post_init__`: logged as a warning and replaced with its documented
    default, rather than failing the whole program over a non-critical
    misconfiguration.

    Attributes:
        gtfs_loop_schedule: timing for the GTFS static feed refresh loop.
        fsm_loop_schedule: timing for the GPS-polling/FSM loop.
        trip_window_loop_schedule: timing/window-length for the trip-window
            rebuild loop.
        trip_window_margin: safety margin added past the trip window's
            computed end before a rebuild is scheduled.
        trip_window_max_lookback_days: cap on how many days back a trip
            window looks for candidate service days.
        trip_endpoint_source: which precomputed zone represents a trip's
            start/end.
        trip_matching: criterion and selection mode for pairing a vehicle to
            an active trip. No default; must be set explicitly.
        timezone: fallback feed timezone, used only when agency.txt doesn't
            give a single unambiguous value.
        projection: UTM CRS used for all meter-based geometry. No default;
            must be set explicitly to a valid CRS identifier.
        shape_geometry: proximity threshold for a vehicle to its trip's
            route path.
        stop_geometry: proximity threshold for a vehicle to an individual
            stop (used as a trip endpoint when that stop isn't part of a
            larger terminal).
        terminal_geometry: proximity threshold for a vehicle to a terminal
            (a stop with children stops/platforms).
        stale_trip_threshold: how long a BUSY vehicle can go without
            matching its trip's path/terminal before the FSM treats the
            match as stale.
        stationary_threshold: how little movement, sustained for how long,
            counts as a vehicle being stationary.
        signal_loss_threshold: how long a vehicle can go without a new GPS
            observation before the FSM stops trusting its last known
            position/trip match and forces it back to FREE.
        primary_key_duplicate_policy: what to do when two rows in a GTFS
            schedule file share a primary key.
    """

    # How often to check whether the GTFS static feed changed. Short/cheap on
    # purpose - the loop only does real parse/geometry work when the raw
    # feed's hash actually differs from last check.
    gtfs_loop_schedule: LoopSchedule = field(default_factory=lambda: LoopSchedule(interval=300))
    # GPS polling + vehicle FSM transition (FSMLoop), bound by the GTFS-RT
    # spec's 30s max publish interval.
    fsm_loop_schedule: LoopSchedule = field(default_factory=lambda: LoopSchedule(interval=30))
    # Trip-window rebuild. Serves two roles safely at once, because neither
    # reader ever writes back to it: (1) window LENGTH (see
    # pygtfsrealtime.trip_window.compute.build_trips_snapshot, which reads
    # .interval directly as the 8h window size) and (2) config/seed for
    # pygtfsrealtime.runner.run_conditional, which drives TripWindowLoop -
    # .accounting_mode/.on_missed_deadline keep their usual meaning there,
    # and .interval only seeds the very first cycle (every cycle after that,
    # the sleep duration comes from TripWindowLoop.next_interval()'s return
    # value, computed from window_end - trip_window_margin, not from this
    # field). This is the only loop that isn't driven by run_periodic - see
    # run_conditional's docstring for why a fixed-interval runner doesn't fit
    # a loop whose next wake time depends on data it just computed.
    trip_window_loop_schedule: LoopSchedule = field(
        default_factory=lambda: LoopSchedule(interval=28800)
    )
    # Safety margin added past window_start + trip_window_loop_schedule.interval
    # when building a trip window (see pygtfsrealtime.trip_window.compute) -
    # both to give the rebuild slack before the window truly runs out, and to
    # derive the self-scheduling wake time as window_end - trip_window_margin.
    trip_window_margin: timedelta = timedelta(minutes=30)
    # Sanity cap on pygtfsrealtime.trip_window.compute.candidate_service_dates'
    # lookback, so a corrupted feed (e.g. a stray "9999:00:00" time value) can't
    # blow up the range of calendar days checked when building a trip window.
    trip_window_max_lookback_days: int = 3
    # Which zone to use as a trip's start/end - see TripEndpointSource above.
    trip_endpoint_source: TripEndpointSource = "stop"
    # Criterion and selection mode pygtfsrealtime.realtime.matching uses to pair a
    # vehicle (GPSEntry/VehicleReport) with an active trip in TripsSnapshot. No
    # default MatchingStrategy - which fields a GPS feed actually populates
    # is operator-specific, so this stays None until explicitly set (e.g.
    # Settings(trip_matching=MatchingStrategy(key="trip_id"))).
    # pygtfsrealtime.realtime.ingest.GPSIngester.ingest raises a clear TypeError if
    # it's still None when the FSM/GPS cycle actually needs it, instead
    # of silently matching on a criterion nobody chose.
    trip_matching: MatchingStrategy | None = None
    # Explicit override/fallback for the GTFS feed's timezone. None means "not
    # set" - the feed's own agency.txt (agency_timezone) is the primary source
    # (see pygtfsrealtime.schedule.snapshot.resolve_gtfs_timezone); this is only
    # consulted when agency.txt doesn't give a single unambiguous answer
    # (missing, or - a spec violation - multiple different values), and if it's
    # still None at that point ingestion raises, since there'd be no way left
    # to know which timezone the feed's un-timezoned times are in.
    timezone: ZoneInfo | None = None
    # UTM CRS used for all meter-based geometry (buffers, distances). No safe
    # default - which UTM zone applies is specific to where the feed's
    # vehicles actually operate, so this stays None until explicitly set
    # (e.g. Settings(projection="EPSG:32723")). GTFSRealtimeEngine raises a
    # clear TypeError at construction time if it's still None, or if it isn't
    # a CRS pyproj can parse, instead of letting a background loop cycle fail
    # obscurely the first time geometry gets reprojected.
    projection: str | None = None
    # Proximity of a vehicle to its trip's route path (shapes.txt geometry).
    shape_geometry: GeometryThreshold = field(
        default_factory=lambda: GeometryThreshold(distance=30, mode="distance")
    )
    # Proximity of a vehicle to an individual stop (used as a trip endpoint when
    # that stop isn't part of a larger terminal).
    stop_geometry: GeometryThreshold = field(
        default_factory=lambda: GeometryThreshold(distance=250, mode="buffer")
    )
    # Proximity of a vehicle to a terminal (a stop with children stops/platforms).
    terminal_geometry: TerminalGeometryThreshold = field(
        default_factory=lambda: TerminalGeometryThreshold(distance=250, mode="buffer")
    )
    stale_trip_threshold: timedelta = timedelta(seconds=10800)
    stationary_threshold: StationaryThreshold = field(default_factory=StationaryThreshold)
    # How long a vehicle can go without a new GPS observation before the FSM
    # (pygtfsrealtime.realtime.fsm) stops trusting its last known position/trip match
    # and forces it back to FREE, regardless of what that stale observation
    # would otherwise imply (on path, stationary, etc).
    signal_loss_threshold: timedelta = timedelta(minutes=5)
    primary_key_duplicate_policy: PrimaryKeyDuplicatePolicy = "keep_first"

    def __post_init__(self) -> None:
        """Warn-and-fall-back-to-default for every field with a sane default,
        given an invalid value. `projection` and `trip_matching` are
        deliberately not handled here - they have no safe default, so
        GTFSRealtimeEngine raises instead of silently substituting one; see
        their comments above. `timezone` is resolved lazily against
        agency.txt (pygtfsrealtime.schedule.snapshot.resolve_gtfs_timezone),
        not here.
        """
        self.gtfs_loop_schedule.interval = _coerce_positive(
            self.gtfs_loop_schedule.interval, 300, "gtfs_loop_schedule.interval"
        )
        self.fsm_loop_schedule.interval = _coerce_positive(
            self.fsm_loop_schedule.interval, 30, "fsm_loop_schedule.interval"
        )
        self.trip_window_loop_schedule.interval = _coerce_positive(
            self.trip_window_loop_schedule.interval, 28800, "trip_window_loop_schedule.interval"
        )
        self.shape_geometry.distance = _coerce_positive(
            self.shape_geometry.distance, 30, "shape_geometry.distance"
        )
        self.stop_geometry.distance = _coerce_positive(
            self.stop_geometry.distance, 250, "stop_geometry.distance"
        )
        self.terminal_geometry.distance = _coerce_positive(
            self.terminal_geometry.distance, 250, "terminal_geometry.distance"
        )
        self.trip_window_margin = _coerce_positive(
            self.trip_window_margin, timedelta(minutes=30), "trip_window_margin"
        )
        self.trip_window_max_lookback_days = _coerce_positive(
            self.trip_window_max_lookback_days, 3, "trip_window_max_lookback_days"
        )
        self.trip_endpoint_source = _coerce_choice(
            self.trip_endpoint_source, get_args(TripEndpointSource), "stop", "trip_endpoint_source"
        )
        self.stale_trip_threshold = _coerce_positive(
            self.stale_trip_threshold, timedelta(seconds=10800), "stale_trip_threshold"
        )
        self.signal_loss_threshold = _coerce_positive(
            self.signal_loss_threshold, timedelta(minutes=5), "signal_loss_threshold"
        )
        self.primary_key_duplicate_policy = _coerce_choice(
            self.primary_key_duplicate_policy,
            get_args(PrimaryKeyDuplicatePolicy),
            "keep_first",
            "primary_key_duplicate_policy",
        )
