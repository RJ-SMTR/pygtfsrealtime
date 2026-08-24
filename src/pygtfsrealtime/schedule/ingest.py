import io
import zipfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from pygtfsrealtime.ingester import Ingester
from pygtfsrealtime.schedule.compute import (
    build_gtfs_schedule_geometries,
    build_gtfs_schedule_zones,
    build_trip_endpoints,
    drop_stop_times_missing_endpoints,
    linearize_stop_times,
    reconcile_gtfs_schedule_relations,
    reconcile_shape_coverage,
)
from pygtfsrealtime.schedule.exceptions import GTFSIngestError
from pygtfsrealtime.schedule.gtfs_spec import (
    GTFS_SCHEDULE_OPTIONAL_FILES,
    GTFS_SCHEDULE_REQUIRED_COLUMNS,
    GTFS_SCHEDULE_REQUIRED_FILES,
)
from pygtfsrealtime.schedule.validate import (
    drop_duplicate_rows,
    drop_incomplete_rows,
    drop_primary_key_violations,
    validate_gtfs_schedule_columns,
    validate_gtfs_schedule_required_files,
    validate_gtfs_schedule_types,
)
from pygtfsrealtime.settings import PrimaryKeyDuplicatePolicy, Settings


class GTFSScheduleIngester(Ingester[bytes, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]):
    """Fetches, validates, and geometrically enriches a GTFS static feed.

    Implements the `Ingester` contract: `fetch()` retrieves this cycle's raw
    zip bytes, `validate()` parses and cleans them into one DataFrame per
    GTFS file, and `ingest()` runs both plus the geometry/zone precomputation
    pipeline (see pygtfsrealtime.schedule.compute), producing the
    fully-augmented `gtfs_files` dict a `GtfsSnapshot` wraps.
    """

    def __init__(self, callback: Callable[[], io.BytesIO], settings: Settings | None = None):
        """Args:
        callback: returns the raw GTFS zip file each time it's called.
        settings: library configuration; defaults to `Settings()`.
        """
        self.callback = callback
        self.settings = settings if settings is not None else Settings()

    @staticmethod
    def read_gtfs_schedule_locally(file_path: Path) -> io.BytesIO:
        """Read a local GTFS zip file into an in-memory buffer."""
        return io.BytesIO(Path(file_path).read_bytes())

    @classmethod
    def from_local_file(
        cls, file_path: Path, settings: Settings | None = None
    ) -> "GTFSScheduleIngester":
        """Build an ingester whose callback reads a fixed local zip file."""
        return cls(
            callback=lambda: cls.read_gtfs_schedule_locally(file_path),
            settings=settings,
        )

    def fetch(self) -> bytes:
        """Call the configured callback and return the raw zip bytes.

        Raises:
            GTFSIngestError: if the callback raises.
        """
        try:
            raw_schedule = self.callback()
        except Exception as exc:
            raise GTFSIngestError(
                f"ingest_gtfs_schedule callback raised an exception: {exc}"
            ) from exc

        return raw_schedule.read()

    def validate(self, raw: bytes) -> dict[str, pd.DataFrame]:
        """Unzip, parse, and clean the raw GTFS feed into per-file DataFrames.

        Args:
            raw: the raw zip bytes, as returned by `fetch()`.

        Returns:
            One DataFrame per required GTFS file, with only the columns this
            library needs, duplicate/incomplete/primary-key-conflicting rows
            dropped, and values cast to their declared types.

        Raises:
            GTFSIngestError: if the zip is invalid or a required file is
                missing/unparseable.
        """
        extracted_files = self._unzip(raw)
        self._validate_required_files(extracted_files)
        extracted_files = self._fill_missing_optional_files(extracted_files)
        gtfs_files = self._parse(extracted_files)
        gtfs_files = self._validate_columns(gtfs_files)
        gtfs_files = self._drop_unused_columns(gtfs_files)
        gtfs_files = self._drop_duplicate_rows(gtfs_files)
        gtfs_files = self._drop_incomplete_rows(gtfs_files)
        gtfs_files = self._validate_types(gtfs_files)
        return self._drop_primary_key_violations(
            gtfs_files, self.settings.primary_key_duplicate_policy
        )

    def ingest(self, raw: bytes | None = None) -> dict[str, pd.DataFrame]:
        """Fetch (if needed), validate, and geometrically enrich the feed.

        Args:
            raw: the raw zip bytes; fetched via `fetch()` if not given.

        Returns:
            The fully zone/geometry-augmented `gtfs_files` dict, ready to be
            wrapped in a `GtfsSnapshot`.
        """
        if raw is None:
            raw = self.fetch()
        gtfs_files = self.validate(raw)
        # Run once on the raw (one row per point) shapes.txt, before the geometry
        # step reprojects/builds a LineString per shape_id - no point paying that
        # cost for a shape_id no trip references.
        gtfs_files = self._reconcile_shape_coverage(gtfs_files)
        projection = self.settings.projection
        assert projection is not None, "GTFSRealtimeEngine requires Settings.projection to be set"
        gtfs_files = self._build_geometries(gtfs_files, projection)
        gtfs_files = self._linearize_stop_times(gtfs_files)
        gtfs_files = self._drop_stop_times_missing_endpoints(gtfs_files)
        # From here, trips.txt/shapes.txt/routes.txt/stop_times.txt/frequencies.txt
        # can cascade into each other (e.g. build_shapes_geometry dropping a
        # shape_id for having fewer than 2 points orphans a trip, and reconciling
        # that trip out because its stop_times are gone can in turn orphan a
        # shape_id or route_id) - iterate to a fixed point instead of guessing how
        # many passes today's specific drops need.
        gtfs_files = self._reconcile_gtfs_schedule_relations(gtfs_files)
        # Last, since it depends on stops.txt/shapes.txt being final: any zone
        # precomputed on a shape_id/stop_id the reconciliation above still ends up
        # dropping would just be wasted work.
        gtfs_files = self._build_zones(gtfs_files, self.settings)
        # Depends on stops.txt/terminals/shapes.txt's zone columns above, so it
        # has to run after them - embeds each trip's resolved start/end zone
        # directly onto trips.txt (see build_trip_endpoints).
        return self._build_trip_endpoints(gtfs_files, self.settings)

    @staticmethod
    def _unzip(raw_schedule: bytes) -> dict[str, bytes]:
        """Extract only the required/optional GTFS files from the raw zip bytes."""
        wanted_files = GTFS_SCHEDULE_REQUIRED_FILES | GTFS_SCHEDULE_OPTIONAL_FILES
        extracted_files: dict[str, bytes] = {}
        try:
            with zipfile.ZipFile(io.BytesIO(raw_schedule)) as archive:
                for file_info in archive.infolist():
                    if file_info.is_dir():
                        continue
                    if file_info.filename not in wanted_files:
                        continue
                    extracted_files[file_info.filename] = archive.read(file_info.filename)
        except zipfile.BadZipFile as exc:
            raise GTFSIngestError(f"GTFS schedule data is not a valid zip file: {exc}") from exc

        return extracted_files

    @staticmethod
    def _validate_required_files(extracted_files: dict[str, bytes]) -> None:
        """Raise if any required GTFS file is missing from the archive."""
        validate_gtfs_schedule_required_files(set(extracted_files.keys()))

    @staticmethod
    def _fill_missing_optional_files(
        extracted_files: dict[str, bytes],
    ) -> dict[str, bytes]:
        """Synthesize a header-only frequencies.txt when the feed doesn't ship
        one - schedule/compute.py and trip_window/compute.py both read
        gtfs_files["frequencies.txt"] unconditionally, and an empty DataFrame
        is exactly the right value for "this feed has no frequency-based
        trips".
        """
        if "frequencies.txt" in extracted_files:
            return extracted_files

        result = dict(extracted_files)
        header = ",".join(sorted(GTFS_SCHEDULE_REQUIRED_COLUMNS["frequencies.txt"]))
        result["frequencies.txt"] = f"{header}\n".encode()
        return result

    @staticmethod
    def _parse(extracted_files: dict[str, bytes]) -> dict[str, pd.DataFrame]:
        """Parse each extracted CSV file into a DataFrame of raw strings."""
        gtfs_files = {}
        for filename, content in extracted_files.items():
            try:
                gtfs_files[filename] = pd.read_csv(io.BytesIO(content), dtype=str)
            except Exception as exc:
                raise GTFSIngestError(f"failed to parse {filename}: {exc}") from exc
        return gtfs_files

    @staticmethod
    def _validate_columns(
        gtfs_files: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Delegate to pygtfsrealtime.schedule.validate.validate_gtfs_schedule_columns."""
        return validate_gtfs_schedule_columns(gtfs_files)

    @staticmethod
    def _drop_unused_columns(
        gtfs_files: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Keep only the columns this library actually reads from each file."""
        return {
            filename: df[sorted(GTFS_SCHEDULE_REQUIRED_COLUMNS[filename])]
            for filename, df in gtfs_files.items()
        }

    @staticmethod
    def _drop_incomplete_rows(
        gtfs_files: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Delegate to pygtfsrealtime.schedule.validate.drop_incomplete_rows."""
        return drop_incomplete_rows(gtfs_files)

    @staticmethod
    def _validate_types(
        gtfs_files: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Delegate to pygtfsrealtime.schedule.validate.validate_gtfs_schedule_types."""
        return validate_gtfs_schedule_types(gtfs_files)

    @staticmethod
    def _drop_duplicate_rows(
        gtfs_files: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Delegate to pygtfsrealtime.schedule.validate.drop_duplicate_rows."""
        return drop_duplicate_rows(gtfs_files)

    @staticmethod
    def _drop_primary_key_violations(
        gtfs_files: dict[str, pd.DataFrame],
        policy: PrimaryKeyDuplicatePolicy,
    ) -> dict[str, pd.DataFrame]:
        """Delegate to pygtfsrealtime.schedule.validate.drop_primary_key_violations."""
        return drop_primary_key_violations(gtfs_files, policy)

    @staticmethod
    def _build_geometries(
        gtfs_files: dict[str, pd.DataFrame], projection: str
    ) -> dict[str, pd.DataFrame]:
        """Delegate to pygtfsrealtime.schedule.compute.build_gtfs_schedule_geometries."""
        return build_gtfs_schedule_geometries(gtfs_files, projection)

    @staticmethod
    def _linearize_stop_times(
        gtfs_files: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Delegate to pygtfsrealtime.schedule.compute.linearize_stop_times."""
        result = dict(gtfs_files)
        result["stop_times.txt"] = linearize_stop_times(gtfs_files["stop_times.txt"])
        return result

    @staticmethod
    def _drop_stop_times_missing_endpoints(
        gtfs_files: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Delegate to pygtfsrealtime.schedule.compute.drop_stop_times_missing_endpoints."""
        result = dict(gtfs_files)
        result["stop_times.txt"] = drop_stop_times_missing_endpoints(gtfs_files["stop_times.txt"])
        return result

    @staticmethod
    def _reconcile_shape_coverage(
        gtfs_files: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Delegate to pygtfsrealtime.schedule.compute.reconcile_shape_coverage."""
        return reconcile_shape_coverage(gtfs_files)

    @staticmethod
    def _reconcile_gtfs_schedule_relations(
        gtfs_files: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Delegate to pygtfsrealtime.schedule.compute.reconcile_gtfs_schedule_relations."""
        return reconcile_gtfs_schedule_relations(gtfs_files)

    @staticmethod
    def _build_zones(
        gtfs_files: dict[str, pd.DataFrame], settings: Settings
    ) -> dict[str, pd.DataFrame]:
        """Delegate to pygtfsrealtime.schedule.compute.build_gtfs_schedule_zones."""
        return build_gtfs_schedule_zones(gtfs_files, settings)

    @staticmethod
    def _build_trip_endpoints(
        gtfs_files: dict[str, pd.DataFrame], settings: Settings
    ) -> dict[str, pd.DataFrame]:
        """Delegate to pygtfsrealtime.schedule.compute.build_trip_endpoints."""
        return build_trip_endpoints(gtfs_files, settings)
