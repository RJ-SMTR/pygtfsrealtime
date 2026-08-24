from collections.abc import Callable
from datetime import datetime

import geopandas as gpd
from shapely.geometry import Point

from pygtfsrealtime.ingester import Ingester
from pygtfsrealtime.models import GPSEntry
from pygtfsrealtime.realtime.validate import validate_gps_entries
from pygtfsrealtime.settings import Settings


class VehicleReport:
    """One vehicle's validated, geometry-projected GPS observation for this
    cycle - built by `GPSIngester.ingest()` from a `GPSEntry`, with `point`
    added as a projected shapely Point. Not part of the public contract
    (only `GPSEntry` and the `GTFSRealtimeEngine` constructor are); built via
    `**kwargs` from a GeoDataFrame row.
    """

    vehicle_id: str
    latitude: float
    longitude: float
    datetime: datetime
    point: Point
    route_short_name: str | None
    direction_id: int | None
    route_id: str | None
    trip_id: str | None
    speed: float | None
    bearing: int | None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class GPSIngester(Ingester[list[GPSEntry], list[GPSEntry], dict[str, VehicleReport]]):
    """The GPS-polling ingest step, same fetch/validate/ingest shape as
    GTFSScheduleIngester (pygtfsrealtime.schedule.ingest): carries the user's
    callback and Settings as instance attributes. fetch() calls the callback
    for this cycle's raw GPSEntry list; validate() delegates to
    pygtfsrealtime.realtime.validate.validate_gps_entries, dropping any entry
    missing a field the configured MatchingStrategy needs - same
    orchestrator-doesn't-know-the-rules split as GTFSScheduleIngester.validate()
    delegating to pygtfsrealtime.schedule.validate; ingest() runs both, then
    projects the survivors into a dict[vehicle_id, VehicleReport] ready for
    matching. Unlike GTFSScheduleIngester's ingest(raw), GPS data has no
    cross-cycle change-detection hash to skip on, so ingest() always fetches
    and validates fresh rather than accepting a previously-fetched `raw`.
    """

    def __init__(self, callback: Callable[[], list[GPSEntry]], settings: Settings | None = None):
        """Args:
        callback: returns this cycle's raw GPSEntry list each time it's
            called.
        settings: library configuration; defaults to `Settings()`.
        """
        self.callback = callback
        self.settings = settings if settings is not None else Settings()

    def fetch(self) -> list[GPSEntry]:
        """Call the configured callback and return the raw GPSEntry list."""
        return self.callback()

    def validate(self, raw: list[GPSEntry]) -> list[GPSEntry]:
        """Drop any entry missing a field the configured MatchingStrategy needs.

        Args:
            raw: this cycle's raw GPSEntry list.

        Returns:
            The entries with all fields the configured strategy needs.
        """
        return validate_gps_entries(raw, self.settings.trip_matching)

    def ingest(self, raw: list[GPSEntry] | None = None) -> dict[str, VehicleReport]:
        """Fetch (if needed), validate, and project into VehicleReports.

        Args:
            raw: this cycle's raw GPSEntry list; fetched via `fetch()` if
                not given.

        Returns:
            A dict of `VehicleReport`, keyed by vehicle_id, with positions
            projected into `settings.projection`.
        """
        if raw is None:
            raw = self.fetch()
        validated = self.validate(raw)

        vehicles: dict[str, VehicleReport] = {}
        if not validated:
            # GeoDataFrame.from_records([]) has no columns at all (not even
            # empty ones) - building "point" from gdf.longitude/gdf.latitude
            # below would raise AttributeError. An empty feed is a normal
            # transient state (fleet not reporting yet), not an error.
            return vehicles

        records = [r.model_dump() for r in validated]
        gdf = gpd.GeoDataFrame.from_records(records)
        gdf["point"] = gpd.points_from_xy(
            gdf.longitude,
            gdf.latitude,
        )
        gdf.set_geometry("point", inplace=True)
        gdf.set_crs("EPSG:4326", inplace=True)
        gdf = gdf.to_crs(self.settings.projection)

        for row in gdf.itertuples():
            vehicles[row.vehicle_id] = VehicleReport(**row._asdict())
        return vehicles
