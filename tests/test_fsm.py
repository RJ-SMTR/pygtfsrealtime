from datetime import datetime, timedelta

from pygtfsrealtime.realtime.fsm import VehicleFSM, VehicleState
from pygtfsrealtime.realtime.observations import TransitionReason, VehicleObservation

T0 = datetime(2026, 1, 1, 12, 0, 0)
STALE_TRIP_THRESHOLD = timedelta(hours=3)


class _Trip:
    """Stand-in for a matched trip row - transition() never looks past
    truthiness/identity of `observation.trip`, so no real geometry needed."""


def _observation(
    trip=None, on_path=False, at_terminal=False, stationary=False, signal_lost=False
) -> VehicleObservation:
    return VehicleObservation(
        trip=trip,
        on_path=on_path,
        at_terminal=at_terminal,
        stationary=stationary,
        signal_lost=signal_lost,
    )


def test_new_fsm_starts_free():
    fsm = VehicleFSM("V1")
    assert fsm.state == VehicleState.FREE
    assert fsm.current_trip is None
    assert fsm.ongoing_since is None
    assert fsm.last_reason is None


def test_signal_lost_forces_free_regardless_of_state():
    fsm = VehicleFSM("V1")
    fsm.state = VehicleState.BUSY
    fsm.current_trip = _Trip()
    fsm.ongoing_since = T0

    reason = fsm.transition(_observation(signal_lost=True), T0, STALE_TRIP_THRESHOLD)

    assert reason == TransitionReason.NO_SIGNAL
    assert fsm.state == VehicleState.FREE
    assert fsm.current_trip is None
    assert fsm.ongoing_since is None


def test_signal_lost_takes_priority_over_other_busy_conditions():
    fsm = VehicleFSM("V1")
    fsm.state = VehicleState.BUSY
    fsm.current_trip = _Trip()
    fsm.ongoing_since = T0

    observation = _observation(signal_lost=True, stationary=True, at_terminal=True)
    reason = fsm.transition(observation, T0, STALE_TRIP_THRESHOLD)

    assert reason == TransitionReason.NO_SIGNAL


def test_free_stays_free_with_no_candidate_trip():
    fsm = VehicleFSM("V1")
    reason = fsm.transition(_observation(trip=None), T0, STALE_TRIP_THRESHOLD)
    assert reason == TransitionReason.NO_CANDIDATE_TRIP
    assert fsm.state == VehicleState.FREE


def test_free_stays_free_when_off_path():
    fsm = VehicleFSM("V1")
    observation = _observation(trip=_Trip(), on_path=False)
    reason = fsm.transition(observation, T0, STALE_TRIP_THRESHOLD)
    assert reason == TransitionReason.OFF_PATH
    assert fsm.state == VehicleState.FREE


def test_free_stays_free_when_candidate_at_terminal():
    fsm = VehicleFSM("V1")
    observation = _observation(trip=_Trip(), on_path=True, at_terminal=True)
    reason = fsm.transition(observation, T0, STALE_TRIP_THRESHOLD)
    assert reason == TransitionReason.AT_TERMINAL
    assert fsm.state == VehicleState.FREE


def test_free_to_busy_when_matched_on_path_and_not_at_terminal():
    fsm = VehicleFSM("V1")
    trip = _Trip()
    observation = _observation(trip=trip, on_path=True, at_terminal=False)

    reason = fsm.transition(observation, T0, STALE_TRIP_THRESHOLD)

    assert reason is None
    assert fsm.state == VehicleState.BUSY
    assert fsm.current_trip is trip
    assert fsm.ongoing_since == T0
    assert fsm.last_reason is None


def test_busy_to_free_when_stationary():
    fsm = VehicleFSM("V1")
    fsm.state = VehicleState.BUSY
    fsm.current_trip = _Trip()
    fsm.ongoing_since = T0

    reason = fsm.transition(
        _observation(stationary=True), T0 + timedelta(minutes=1), STALE_TRIP_THRESHOLD
    )

    assert reason == TransitionReason.STATIONARY
    assert fsm.state == VehicleState.FREE


def test_busy_to_free_when_at_terminal():
    fsm = VehicleFSM("V1")
    fsm.state = VehicleState.BUSY
    fsm.current_trip = _Trip()
    fsm.ongoing_since = T0

    reason = fsm.transition(
        _observation(at_terminal=True), T0 + timedelta(minutes=1), STALE_TRIP_THRESHOLD
    )

    assert reason == TransitionReason.AT_TERMINAL
    assert fsm.state == VehicleState.FREE


def test_busy_to_free_when_trip_duration_exceeded():
    fsm = VehicleFSM("V1")
    fsm.state = VehicleState.BUSY
    fsm.current_trip = _Trip()
    fsm.ongoing_since = T0

    now = T0 + STALE_TRIP_THRESHOLD
    reason = fsm.transition(_observation(), now, STALE_TRIP_THRESHOLD)

    assert reason == TransitionReason.STALE_TRIP
    assert fsm.state == VehicleState.FREE


def test_busy_stays_busy_when_nothing_triggers_free():
    fsm = VehicleFSM("V1")
    trip = _Trip()
    fsm.state = VehicleState.BUSY
    fsm.current_trip = trip
    fsm.ongoing_since = T0

    now = T0 + timedelta(minutes=5)
    reason = fsm.transition(_observation(), now, STALE_TRIP_THRESHOLD)

    assert reason is None
    assert fsm.state == VehicleState.BUSY
    assert fsm.current_trip is trip
    assert fsm.ongoing_since == T0
