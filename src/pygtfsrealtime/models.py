from datetime import datetime

from pydantic import BaseModel, field_validator


class GPSEntry(BaseModel):
    """One vehicle's raw GPS observation, as returned by an `ingest_gps_data`
    callback. Only `vehicle_id`/`latitude`/`longitude`/`datetime` are always
    required; the rest are optional and only need to be populated if the
    configured `MatchingStrategy` reads them.

    Attributes:
        vehicle_id: unique identifier for the vehicle.
        latitude: WGS84 latitude in degrees.
        longitude: WGS84 longitude in degrees.
        datetime: timestamp of the observation.
        route_short_name: the route's short name, if known.
        direction_id: the trip's direction (0 or 1), if known.
        route_id: the GTFS route_id, if known.
        trip_id: the GTFS trip_id, if known.
        route_long_name: the route's long name, if known.
        speed: vehicle speed, if reported by the source.
        bearing: vehicle heading in degrees, if reported by the source.
    """

    vehicle_id: str
    latitude: float
    longitude: float
    datetime: datetime
    route_short_name: str | None = None
    direction_id: int | None = None
    route_id: str | None = None
    trip_id: str | None = None
    route_long_name: str | None = None
    speed: float | None = None
    bearing: int | None = None

    @field_validator("bearing", mode="before")
    @classmethod
    def validate_bearing(cls, v):
        """Coerce bearing to an int when possible, else None."""
        if v is None:
            return v
        elif isinstance(v, int):
            return v
        elif isinstance(v, float):
            return int(v)
        try:
            return int(v)
        except (ValueError, TypeError):
            return None
