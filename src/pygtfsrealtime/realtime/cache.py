import pickle
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from pygtfsrealtime.realtime.fsm import VehicleFSM, VehicleState
from pygtfsrealtime.realtime.observations import ObservationWindow, TransitionReason, TripCandidate

TripKey = tuple[str, datetime]
TripResolver = Callable[[TripKey], "TripCandidate | None"]


@dataclass
class VehicleCheckpoint:
    """Everything needed to restore one vehicle's VehicleFSM + ObservationWindow
    across a process restart - deliberately made of only stable, cheaply
    picklable primitives (str/Enum/datetime/tuples of floats), never a live
    TripCandidate. A TripCandidate in the running process is a TripsSnapshot.trips
    row - it carries shapely geometry and is a pandas/geopandas row type
    generated dynamically by itertuples(), both fragile to pickle across
    process/library-version boundaries.

    Attributes:
        vehicle_id: the vehicle's identifier.
        state: the FSM's state (FREE/BUSY) at checkpoint time.
        current_trip_key: the trip instance's identifier - the ORDERED PAIR
            (trip_id, start_dt), never trip_id alone (frequencies.txt expands
            one trip_id into several instances with different start_dt - the
            same composite key TripsSnapshot.trips itself is indexed by).
            One field, not two independently-nullable ones, so the pair can
            never end up half-set. `restore_fsm()` uses it to re-look up the
            actual TripCandidate from whatever TripsSnapshot is current at
            load time, instead of ever deserializing one.
        ongoing_since: when the vehicle entered its current state.
        last_reason: the FSM transition reason that produced this state.
        window_entries: the vehicle's recent projected positions, from
            ObservationWindow.entries().
    """

    vehicle_id: str
    state: VehicleState
    current_trip_key: TripKey | None
    ongoing_since: datetime | None
    last_reason: TransitionReason | None
    window_entries: tuple[tuple[datetime, float, float], ...]


def to_checkpoint(fsm: VehicleFSM, window: ObservationWindow) -> VehicleCheckpoint:
    """Snapshot one vehicle's FSM + observation window into a checkpoint."""
    trip = fsm.current_trip
    return VehicleCheckpoint(
        vehicle_id=fsm.vehicle_id,
        state=fsm.state,
        current_trip_key=(trip.trip_id, trip.start_dt) if trip is not None else None,
        ongoing_since=fsm.ongoing_since,
        last_reason=fsm.last_reason,
        window_entries=window.entries(),
    )


def restore_fsm(
    checkpoint: VehicleCheckpoint,
    resolve_trip: TripResolver,
) -> tuple[VehicleFSM, ObservationWindow]:
    """Rebuild one vehicle's FSM + window from a checkpoint.

    If the checkpoint says BUSY but resolve_trip(current_trip_key) can't find
    that exact (trip_id, start_dt) instance anymore (GTFS changed, or the
    rolling window moved past it while the process was down), self-heals to
    FREE instead of holding a dangling/stale trip reference - the same
    outcome the FSM would reach on its own next cycle if the trip candidate
    disappeared.

    Args:
        checkpoint: the saved vehicle state.
        resolve_trip: looks up a `TripCandidate` by its (trip_id, start_dt)
            key against whatever `TripsSnapshot` is current now.

    Returns:
        The rebuilt FSM and observation window.
    """
    fsm = VehicleFSM(checkpoint.vehicle_id)
    fsm.last_reason = checkpoint.last_reason

    if checkpoint.state == VehicleState.BUSY and checkpoint.current_trip_key is not None:
        trip = resolve_trip(checkpoint.current_trip_key)
        if trip is not None:
            fsm.state = VehicleState.BUSY
            fsm.current_trip = trip
            fsm.ongoing_since = checkpoint.ongoing_since

    window = ObservationWindow(checkpoint.window_entries)
    return fsm, window


def save_cache(
    fsms: dict[str, VehicleFSM],
    windows: dict[str, ObservationWindow],
    set_cache: Callable[[bytes], None] | None,
) -> None:
    """Checkpoint every vehicle's FSM/window and persist via `set_cache`.

    A no-op when `set_cache` isn't configured.
    """
    if not set_cache:
        return

    checkpoints = {
        vehicle_id: to_checkpoint(fsm, windows.get(vehicle_id, ObservationWindow()))
        for vehicle_id, fsm in fsms.items()
    }
    set_cache(pickle.dumps(checkpoints))


def load_cache(
    get_cache: Callable[[], bytes | None] | None,
    resolve_trip: TripResolver,
) -> tuple[dict[str, VehicleFSM], dict[str, ObservationWindow]]:
    """Load and restore every vehicle's FSM/window via `get_cache`.

    Args:
        get_cache: returns the previously persisted cache bytes, or None.
            A no-op (returns empty dicts) when not configured.
        resolve_trip: looks up a `TripCandidate` by its (trip_id, start_dt)
            key, used to restore each BUSY vehicle's current trip.

    Returns:
        The restored `(fsms, windows)` dicts, keyed by vehicle_id.
    """
    if not get_cache:
        return {}, {}

    cache_data = get_cache()
    if not cache_data:
        return {}, {}

    checkpoints: dict[str, VehicleCheckpoint] = pickle.loads(cache_data)
    fsms: dict[str, VehicleFSM] = {}
    windows: dict[str, ObservationWindow] = {}
    for vehicle_id, checkpoint in checkpoints.items():
        fsm, window = restore_fsm(checkpoint, resolve_trip)
        fsms[vehicle_id] = fsm
        windows[vehicle_id] = window
    return fsms, windows
