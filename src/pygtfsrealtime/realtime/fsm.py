from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import NamedTuple

from pygtfsrealtime.realtime.observations import (
    TransitionReason,
    TripCandidate,
    VehicleObservation,
    trip_duration_exceeded,
)


class VehicleState(Enum):
    """A vehicle's confirmation state: not yet matched to a trip, or matched."""

    FREE = auto()
    BUSY = auto()


class TransitionEvent(NamedTuple):
    """One vehicle's outcome from one `VehicleFSM.transition()` call - the
    unit `FSMLoop` accumulates, one per vehicle processed each cycle, into
    the list handed to `TransitionCallback`. A NamedTuple, not a dataclass,
    so a consumer can keep unpacking positionally
    (`for vehicle_id, old_state, new_state, reason, observation in
    transitions:`) or read by attribute (`event.vehicle_id`).
    """

    vehicle_id: str
    old_state: VehicleState
    new_state: VehicleState
    reason: "TransitionReason | None"
    observation: VehicleObservation


TransitionCallback = Callable[[list[TransitionEvent]], None]


class VehicleFSM:
    """Per-vehicle state machine: FREE (not confirmed on any trip) or BUSY (on
    a confirmed trip). Holds only its own state - no geometry, Settings, or
    schedule/matching objects - those are the caller's job to resolve into a
    VehicleObservation (pygtfsrealtime.realtime.observations.build_observation)
    before calling transition().

    Knows nothing about TransitionCallback/TransitionEvent - accumulating
    every vehicle's outcome into one per-cycle batch and invoking a callback
    is the caller's job (see FSMLoop.run_once()), not this class's. This
    class only ever advances one vehicle at a time and has no notion of "a
    cycle" spanning multiple vehicles.
    """

    def __init__(self, vehicle_id: str):
        """Args:
        vehicle_id: the vehicle this FSM tracks.
        """
        self.vehicle_id = vehicle_id
        self.state = VehicleState.FREE
        self.current_trip: TripCandidate | None = None
        self.ongoing_since: datetime | None = None
        self.last_reason: TransitionReason | None = None

    def _go_free(self, reason: TransitionReason) -> TransitionReason:
        """Reset to FREE, clearing any trip/timing state, and return `reason`."""
        self.state = VehicleState.FREE
        self.current_trip = None
        self.ongoing_since = None
        self.last_reason = reason
        return reason

    def transition(
        self,
        observation: VehicleObservation,
        now: datetime,
        stale_trip_threshold: timedelta,
    ) -> TransitionReason | None:
        """Advance the FSM by one cycle.

        Args:
            observation: this cycle's resolved vehicle state (see
                pygtfsrealtime.realtime.observations.build_observation).
            now: the current cycle's timestamp.
            stale_trip_threshold: how long a BUSY vehicle can go without
                confirming its trip before it's forced back to FREE.

        Returns:
            The reason when the FSM ends up (or stays) FREE, None when it
            enters or stays BUSY.
        """
        return self._advance(observation, now, stale_trip_threshold)

    def _advance(
        self,
        observation: VehicleObservation,
        now: datetime,
        stale_trip_threshold: timedelta,
    ) -> TransitionReason | None:
        """signal_lost is checked before anything else, in either state: it's
        a statement about whether `observation` itself is trustworthy, not
        about geometry/trip state, so it pre-empts every other condition
        below.
        """
        if observation.signal_lost:
            return self._go_free(TransitionReason.NO_SIGNAL)

        if self.state == VehicleState.FREE:
            if not observation.trip:
                return self._go_free(TransitionReason.NO_CANDIDATE_TRIP)
            if not observation.on_path:
                return self._go_free(TransitionReason.OFF_PATH)
            if observation.at_terminal:
                return self._go_free(TransitionReason.AT_TERMINAL)

            self.state = VehicleState.BUSY
            self.current_trip = observation.trip
            self.ongoing_since = now
            self.last_reason = None
            return None

        # BUSY - ongoing_since is always set here, since the only way into BUSY
        # (above) sets it, and the only way out of BUSY (_go_free) clears both
        # together.
        assert self.ongoing_since is not None
        if observation.stationary:
            return self._go_free(TransitionReason.STATIONARY)
        if observation.at_terminal:
            return self._go_free(TransitionReason.AT_TERMINAL)
        if trip_duration_exceeded(self.ongoing_since, now, stale_trip_threshold):
            return self._go_free(TransitionReason.STALE_TRIP)

        return None
