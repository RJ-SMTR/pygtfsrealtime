import logging
from collections.abc import Callable
from datetime import UTC, datetime

import geopandas as gpd

from pygtfsrealtime.pb.protobuf import build_pb
from pygtfsrealtime.realtime.cache import load_cache, save_cache
from pygtfsrealtime.realtime.fsm import (
    TransitionCallback,
    TransitionEvent,
    VehicleFSM,
    VehicleState,
)
from pygtfsrealtime.realtime.ingest import GPSIngester
from pygtfsrealtime.realtime.matching import (
    build_trip_match_index,
    filter_active_now,
    match_vehicle,
)
from pygtfsrealtime.realtime.observations import (
    ObservationWindow,
    TripCandidate,
    build_observation,
)
from pygtfsrealtime.runner import SnapshotStore
from pygtfsrealtime.settings import Settings
from pygtfsrealtime.trip_window.compute import TripsSnapshot

logger = logging.getLogger(__name__)


class FSMLoop:
    """Polls GPS, matches each vehicle to an active trip, advances its FSM,
    publishes GTFS-RT protobuf, and persists per-vehicle state across cycles.
    Unconditional/fixed-tick like GTFSScheduleLoop - driven by run_periodic,
    no wake_event, since it already re-reads whatever TripsSnapshot is
    current every ~30s regardless of whether it changed.

    Deliberately takes no gtfs_snapshot_store: TripsSnapshot.trips rows
    already carry shape_geometry/start_zone/end_zone/start_zone_source/
    end_zone_source (joined in by build_trips_snapshot), which is everything
    TripCandidate needs - reading GtfsSnapshot directly here would be
    redundant.
    """

    def __init__(
        self,
        settings: Settings,
        trips_snapshot_store: SnapshotStore[TripsSnapshot],
        ingester: GPSIngester,
        publish_protobuf: Callable[[bytes], None],
        get_cache: Callable[[], bytes | None] | None = None,
        set_cache: Callable[[bytes], None] | None = None,
        on_transition: TransitionCallback | None = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        """Args:
        settings: library configuration.
        trips_snapshot_store: where the current `TripsSnapshot` is read from.
        ingester: fetches/validates/projects this cycle's GPS observations.
        publish_protobuf: called each cycle with the serialized GTFS-RT bytes.
        get_cache: optional; restores FSM/window state on the first cycle a
            `TripsSnapshot` is available.
        set_cache: optional; persists FSM/window state after each cycle.
        on_transition: optional; called once per completed cycle (i.e. not
            when the cycle is skipped for lack of a TripsSnapshot) with the
            list of every vehicle's `TransitionEvent` from that cycle - may
            be empty if no vehicle reported this cycle.
        now_fn: clock injection point, mainly for tests.
        """
        self.settings = settings
        self.trips_snapshot_store = trips_snapshot_store
        self.ingester = ingester
        self.publish_protobuf = publish_protobuf
        self.get_cache = get_cache
        self.set_cache = set_cache
        self.on_transition = on_transition
        self.now_fn = now_fn
        self._fsms: dict[str, VehicleFSM] = {}
        self._windows: dict[str, ObservationWindow] = {}
        # Cache load is deferred to the first run_once() call where a
        # TripsSnapshot actually exists (see _maybe_load_cache) - FSMLoop is
        # constructed before the GTFS/trip-window loops have ever run, so
        # trips_snapshot_store is always empty at __init__ time; loading here
        # unconditionally would permanently resolve every cached BUSY
        # vehicle's trip to None and silently demote it to FREE. Nothing to
        # load when get_cache isn't configured, so there's nothing to wait
        # for either.
        self._cache_loaded = get_cache is None

    def _maybe_load_cache(self) -> None:
        """Load the persisted FSM/window cache once a TripsSnapshot exists.

        No-op once already loaded, and retried every cycle until a
        TripsSnapshot is available to resolve cached trips against.
        """
        if self._cache_loaded:
            return
        active_trips = self._active_trips()
        if active_trips is None:
            return  # no TripsSnapshot published yet - retry next cycle
        self._fsms, self._windows = load_cache(self.get_cache, self._trip_resolver())
        self._cache_loaded = True

    def _active_trips(self) -> gpd.GeoDataFrame | None:
        """The current TripsSnapshot's trips filtered to "active now", or
        None if no TripsSnapshot has been published yet.
        """
        snapshot = self.trips_snapshot_store.get()
        if snapshot is None:
            return None
        return filter_active_now(snapshot.trips, self.now_fn())

    def _trip_resolver(self) -> Callable[[tuple[str, datetime]], TripCandidate | None]:
        """Build a (trip_id, start_dt) -> TripCandidate lookup against the
        currently active trips, for restore_fsm() to resolve cached trips.
        """
        active_trips = self._active_trips()

        def resolve(trip_key: tuple[str, datetime]) -> TripCandidate | None:
            # A trip instance's identifier is the ORDERED PAIR (trip_id,
            # start_dt), not trip_id alone - frequencies.txt expands one
            # trip_id into several instances with different start_dt (same
            # composite key TripsSnapshot.trips itself is indexed by).
            if active_trips is None or trip_key not in active_trips.index:
                return None
            return active_trips.loc[trip_key]

        return resolve

    def run_once(self) -> None:
        """Run one cycle: poll GPS, match, transition every FSM, and publish.

        Skips the cycle entirely (no publish) when no TripsSnapshot has been
        published yet.
        """
        if self.trips_snapshot_store.get() is None:
            logger.info("FSM loop: no TripsSnapshot published yet, skipping this cycle")
            return

        now = self.now_fn()
        self._maybe_load_cache()
        # GPSIngester.ingest raises a clear TypeError if trip_matching isn't
        # set - called before touching settings.trip_matching.key below, so
        # that error (not a bare AttributeError) is what a misconfigured
        # Settings actually surfaces.
        vehicles = self.ingester.ingest()
        trip_matching = self.settings.trip_matching
        assert trip_matching is not None, "GPSIngester.ingest already requires this to be set"

        active_trips = self._active_trips()
        index = (
            build_trip_match_index(active_trips, trip_matching.key)
            if active_trips is not None and not active_trips.empty
            else {}
        )
        # Which trip instance each currently-BUSY vehicle is on, as of the
        # start of this cycle - read before any vehicle in this cycle's loop
        # can go BUSY, so a vehicle matching for the first time this cycle
        # never sees itself (or another vehicle's brand-new match) as a
        # claim. Computed unconditionally: "strict" mode's branch in
        # select_candidate simply ignores it, only mode="progress_match"
        # consults it (see MatchingStrategy.allow_shared_trip).
        claimed_by: dict[tuple[str, datetime], str] = {
            (fsm.current_trip.trip_id, fsm.current_trip.start_dt): vid
            for vid, fsm in self._fsms.items()
            if fsm.state == VehicleState.BUSY and fsm.current_trip is not None
        }

        transitions: list[TransitionEvent] = []
        for vehicle_id, vehicle in vehicles.items():
            fsm = self._fsms.setdefault(vehicle_id, VehicleFSM(vehicle_id))
            window = self._windows.setdefault(vehicle_id, ObservationWindow())
            window.push(vehicle, now, self.settings.stationary_threshold.interval)

            trip = (
                fsm.current_trip
                if fsm.state == VehicleState.BUSY
                else match_vehicle(vehicle, index, trip_matching, now, self.settings, claimed_by)
            )
            observation = build_observation(vehicle, window, trip, self.settings, now)
            old_state = fsm.state
            reason = fsm.transition(observation, now, self.settings.stale_trip_threshold)
            transitions.append(
                TransitionEvent(vehicle_id, old_state, fsm.state, reason, observation)
            )

        if self.on_transition:
            self.on_transition(transitions)

        data = build_pb(self._fsms, vehicles, now)
        self.publish_protobuf(data)
        save_cache(self._fsms, self._windows, self.set_cache)
