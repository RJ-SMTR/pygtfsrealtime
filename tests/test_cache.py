from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pygtfsrealtime.realtime.cache import (
    VehicleCheckpoint,
    load_cache,
    restore_fsm,
    save_cache,
    to_checkpoint,
)
from pygtfsrealtime.realtime.fsm import VehicleFSM, VehicleState
from pygtfsrealtime.realtime.observations import ObservationWindow, TransitionReason

T0 = datetime(2026, 1, 1, 5, 0, 0, tzinfo=UTC)


@dataclass
class _Trip:
    trip_id: str
    start_dt: datetime


class _FakeCacheBackend:
    def __init__(self):
        self.data: bytes | None = None

    def set_cache(self, data: bytes) -> None:
        self.data = data

    def get_cache(self) -> bytes | None:
        return self.data


# --- to_checkpoint -----------------------------------------------------------


def test_to_checkpoint_free_vehicle_has_no_trip_key():
    fsm = VehicleFSM("V1")
    window = ObservationWindow([(T0, 1.0, 2.0)])

    checkpoint = to_checkpoint(fsm, window)

    assert checkpoint.vehicle_id == "V1"
    assert checkpoint.state == VehicleState.FREE
    assert checkpoint.current_trip_key is None
    assert checkpoint.window_entries == ((T0, 1.0, 2.0),)


def test_to_checkpoint_busy_vehicle_captures_the_trip_id_start_dt_pair():
    fsm = VehicleFSM("V1")
    fsm.state = VehicleState.BUSY
    fsm.current_trip = _Trip("T1", T0)
    fsm.ongoing_since = T0

    checkpoint = to_checkpoint(fsm, ObservationWindow())

    assert checkpoint.current_trip_key == ("T1", T0)
    assert checkpoint.ongoing_since == T0


# --- restore_fsm ---------------------------------------------------------------


def test_restore_fsm_free_checkpoint_stays_free_and_keeps_last_reason():
    checkpoint = VehicleCheckpoint(
        vehicle_id="V1",
        state=VehicleState.FREE,
        current_trip_key=None,
        ongoing_since=None,
        last_reason=TransitionReason.NO_CANDIDATE_TRIP,
        window_entries=(),
    )

    fsm, _ = restore_fsm(checkpoint, resolve_trip=lambda trip_key: None)

    assert fsm.state == VehicleState.FREE
    assert fsm.current_trip is None
    assert fsm.last_reason == TransitionReason.NO_CANDIDATE_TRIP


def test_restore_fsm_busy_checkpoint_resolves_the_exact_instance():
    trip = _Trip("T1", T0)
    checkpoint = VehicleCheckpoint(
        vehicle_id="V1",
        state=VehicleState.BUSY,
        current_trip_key=("T1", T0),
        ongoing_since=T0,
        last_reason=None,
        window_entries=(),
    )

    fsm, _ = restore_fsm(
        checkpoint, resolve_trip=lambda trip_key: trip if trip_key == ("T1", T0) else None
    )

    assert fsm.state == VehicleState.BUSY
    assert fsm.current_trip is trip
    assert fsm.ongoing_since == T0


def test_restore_fsm_disambiguates_same_trip_id_different_start_dt():
    """frequencies.txt expands one trip_id into several instances
    with different start_dt - a checkpoint keyed only by trip_id would
    resolve to the wrong instance. The identifier is the ORDERED PAIR
    (trip_id, start_dt), which is exactly what current_trip_key is.
    """
    trip_early = _Trip("T1", T0)
    trip_late = _Trip("T1", T0 + timedelta(hours=1))
    instances = {
        ("T1", T0): trip_early,
        ("T1", T0 + timedelta(hours=1)): trip_late,
    }

    checkpoint_early = VehicleCheckpoint("V1", VehicleState.BUSY, ("T1", T0), T0, None, ())
    checkpoint_late = VehicleCheckpoint(
        "V2", VehicleState.BUSY, ("T1", T0 + timedelta(hours=1)), T0 + timedelta(hours=1), None, ()
    )

    fsm_early, _ = restore_fsm(checkpoint_early, instances.get)
    fsm_late, _ = restore_fsm(checkpoint_late, instances.get)

    assert fsm_early.current_trip is trip_early
    assert fsm_late.current_trip is trip_late
    assert fsm_early.current_trip is not fsm_late.current_trip


def test_restore_fsm_self_heals_to_free_when_trip_no_longer_resolvable():
    checkpoint = VehicleCheckpoint(
        vehicle_id="V1",
        state=VehicleState.BUSY,
        current_trip_key=("T1", T0),
        ongoing_since=T0,
        last_reason=None,
        window_entries=(),
    )

    fsm, _ = restore_fsm(checkpoint, resolve_trip=lambda trip_key: None)

    assert fsm.state == VehicleState.FREE
    assert fsm.current_trip is None
    assert fsm.ongoing_since is None


def test_restore_fsm_restores_window_entries():
    entries = ((T0, 1.0, 2.0), (T0 + timedelta(seconds=30), 1.5, 2.5))
    checkpoint = VehicleCheckpoint("V1", VehicleState.FREE, None, None, None, entries)

    _, window = restore_fsm(checkpoint, resolve_trip=lambda trip_key: None)

    assert window.entries() == entries


# --- save_cache / load_cache ----------------------------------------------------


def test_save_cache_is_a_noop_when_set_cache_is_none():
    save_cache({}, {}, None)  # must not raise


def test_load_cache_returns_empty_when_get_cache_is_none():
    fsms, windows = load_cache(None, resolve_trip=lambda trip_key: None)
    assert fsms == {}
    assert windows == {}


def test_load_cache_returns_empty_when_cache_data_is_empty():
    fsms, windows = load_cache(lambda: None, resolve_trip=lambda trip_key: None)
    assert fsms == {}
    assert windows == {}


def test_save_then_load_cache_roundtrip_restores_busy_vehicle_to_the_right_instance():
    backend = _FakeCacheBackend()
    fsm = VehicleFSM("V1")
    fsm.state = VehicleState.BUSY
    fsm.current_trip = _Trip("T1", T0)
    fsm.ongoing_since = T0
    window = ObservationWindow([(T0, 1.0, 2.0)])

    save_cache({"V1": fsm}, {"V1": window}, backend.set_cache)
    assert backend.data is not None

    trip = _Trip("T1", T0)
    fsms, windows = load_cache(
        backend.get_cache, resolve_trip=lambda trip_key: trip if trip_key == ("T1", T0) else None
    )

    assert fsms["V1"].state == VehicleState.BUSY
    assert fsms["V1"].current_trip is trip
    assert fsms["V1"].ongoing_since == T0
    assert windows["V1"].entries() == ((T0, 1.0, 2.0),)


def test_save_then_load_cache_roundtrip_free_vehicle():
    backend = _FakeCacheBackend()
    fsm = VehicleFSM("V1")

    save_cache({"V1": fsm}, {"V1": ObservationWindow()}, backend.set_cache)
    fsms, _ = load_cache(backend.get_cache, resolve_trip=lambda trip_key: None)

    assert fsms["V1"].state == VehicleState.FREE
    assert fsms["V1"].current_trip is None
