import hashlib
import threading

from pygtfsrealtime.runner import SnapshotStore
from pygtfsrealtime.schedule.ingest import GTFSScheduleIngester
from pygtfsrealtime.schedule.snapshot import GtfsSnapshot, build_gtfs_snapshot


def _hash_raw_gtfs(raw_schedule: bytes) -> str:
    """Hash the raw feed bytes, used to detect an unchanged feed."""
    return hashlib.sha256(raw_schedule).hexdigest()


class GTFSScheduleLoop:
    """Fetches, hashes, and parses the GTFS static feed, publishing a
    `GtfsSnapshot`. Skips the parse/publish step when the raw feed's hash is
    unchanged since the last cycle.
    """

    def __init__(
        self,
        ingester: GTFSScheduleIngester,
        snapshot_store: SnapshotStore[GtfsSnapshot],
        new_gtfs_event: threading.Event,
    ):
        """Args:
        ingester: fetches/validates/enriches the GTFS static feed.
        snapshot_store: where each cycle's `GtfsSnapshot` is published.
        new_gtfs_event: set() whenever a new snapshot is published, so
            `TripWindowLoop` can wake up and rebuild the trip window early.
        """
        self.ingester = ingester
        self.snapshot_store = snapshot_store
        self.new_gtfs_event = new_gtfs_event
        self._last_hash: str | None = None

    def run_once(self) -> None:
        """Run one cycle: fetch, hash-check, and publish if changed."""
        raw_schedule = self.ingester.fetch()
        gtfs_hash = _hash_raw_gtfs(raw_schedule)
        if gtfs_hash == self._last_hash:
            return

        # ingester already carries Settings (projection, buffer/zone thresholds)
        # from construction, so gtfs_files comes back fully zone-augmented.
        gtfs_files = self.ingester.ingest(raw_schedule)
        snapshot = build_gtfs_snapshot(gtfs_files, gtfs_hash, self.ingester.settings)
        self.snapshot_store.set(snapshot)
        self._last_hash = gtfs_hash
        self.new_gtfs_event.set()
