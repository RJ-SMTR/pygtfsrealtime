import logging
from collections.abc import Callable
from datetime import UTC, datetime

from pygtfsrealtime.runner import SnapshotStore
from pygtfsrealtime.schedule.snapshot import GtfsSnapshot
from pygtfsrealtime.settings import Settings
from pygtfsrealtime.trip_window.compute import TripsSnapshot, build_trips_snapshot

logger = logging.getLogger(__name__)


class TripWindowLoop:
    """Consumes the published GtfsSnapshot, builds the rolling trip window,
    and publishes a TripsSnapshot. Driven by run_conditional, not
    run_periodic, since its own next-run delay depends on the window it just
    built (window_end - trip_window_margin), not on a fixed tick.
    """

    def __init__(
        self,
        settings: Settings,
        gtfs_snapshot_store: SnapshotStore[GtfsSnapshot],
        trips_snapshot_store: SnapshotStore[TripsSnapshot],
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        """Args:
        settings: window length/margin/lookback configuration.
        gtfs_snapshot_store: where the current `GtfsSnapshot` is read from.
        trips_snapshot_store: where each cycle's `TripsSnapshot` is published.
        now_fn: clock injection point, mainly for tests.
        """
        self.settings = settings
        self.gtfs_snapshot_store = gtfs_snapshot_store
        self.trips_snapshot_store = trips_snapshot_store
        self.now_fn = now_fn

    def run_once(self) -> TripsSnapshot | None:
        """Run one cycle: rebuild and publish the trip window.

        Returns:
            The published `TripsSnapshot`, or None if no `GtfsSnapshot` has
            been published yet.
        """
        gtfs_snapshot = self.gtfs_snapshot_store.get()
        if gtfs_snapshot is None:
            logger.info("Trip window loop: no GtfsSnapshot published yet")
            return None

        trips_snapshot = build_trips_snapshot(gtfs_snapshot, self.settings, self.now_fn())
        self.trips_snapshot_store.set(trips_snapshot)
        return trips_snapshot

    def next_interval(self) -> float:
        """work_fn for run_conditional: runs one cycle and returns seconds
        until the resulting window needs rebuilding - or a short retry
        cadence (settings.gtfs_loop_schedule.interval) if no GtfsSnapshot
        was available yet to build from.

        Returns:
            Seconds until the next cycle should run.
        """
        snapshot = self.run_once()
        if snapshot is None:
            return self.settings.gtfs_loop_schedule.interval

        wake_at = snapshot.window_end - self.settings.trip_window_margin
        return max(0.0, (wake_at - self.now_fn()).total_seconds())
