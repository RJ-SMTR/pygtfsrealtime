from datetime import UTC, datetime

import pytest

from pygtfsrealtime.models import GPSEntry
from pygtfsrealtime.realtime.ingest import GPSIngester
from pygtfsrealtime.settings import MatchingStrategy, Settings

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _entry(vehicle_id: str, route_short_name: str | None = "100") -> GPSEntry:
    return GPSEntry(
        vehicle_id=vehicle_id,
        latitude=-22.9,
        longitude=-43.2,
        datetime=NOW,
        route_short_name=route_short_name,
        direction_id=0,
    )


# --- fetch ----------------------------------------------------------------------


def test_fetch_returns_callback_result():
    entries = [_entry("V1")]
    ingester = GPSIngester(callback=lambda: entries)
    assert ingester.fetch() is entries


# --- validate ---------------------------------------------------------------------


def test_validate_drops_entries_missing_a_field_the_match_key_needs(caplog):
    settings = Settings(trip_matching=MatchingStrategy(key="route_short_name,direction_id"))
    ingester = GPSIngester(callback=lambda: [], settings=settings)
    entries = [_entry("V1"), _entry("V2", route_short_name=None)]

    with caplog.at_level("WARNING", logger="pygtfsrealtime.realtime.validate"):
        result = ingester.validate(entries)

    assert [e.vehicle_id for e in result] == ["V1"]
    assert any("V2" in record.message for record in caplog.records)


def test_validate_keeps_entries_with_every_required_field_present():
    settings = Settings(trip_matching=MatchingStrategy(key="route_short_name,direction_id"))
    ingester = GPSIngester(callback=lambda: [], settings=settings)
    entries = [_entry("V1"), _entry("V2")]

    result = ingester.validate(entries)

    assert [e.vehicle_id for e in result] == ["V1", "V2"]


def test_validate_raises_when_trip_matching_is_not_set():
    ingester = GPSIngester(callback=lambda: [], settings=Settings())
    with pytest.raises(TypeError):
        ingester.validate([])


# --- ingest -----------------------------------------------------------------------


def test_gps_ingester_ingest_returns_empty_dict_for_an_empty_gps_feed():
    """Regression: GeoDataFrame.from_records([]) has no columns at all, so
    building geometry from gdf.longitude/gdf.latitude used to raise
    AttributeError on a momentarily-empty GPS feed - a normal transient
    state (fleet not reporting yet), not an error.
    """
    settings = Settings(trip_matching=MatchingStrategy(key="route_short_name,direction_id"))
    ingester = GPSIngester(callback=lambda: [], settings=settings)

    vehicles = ingester.ingest()

    assert vehicles == {}


def test_gps_ingester_ingest_raises_when_trip_matching_is_not_set():
    ingester = GPSIngester(callback=lambda: [], settings=Settings())
    with pytest.raises(TypeError):
        ingester.ingest()


def test_gps_ingester_ingest_drops_entries_missing_a_required_field():
    settings = Settings(
        projection="EPSG:32723",
        trip_matching=MatchingStrategy(key="route_short_name,direction_id"),
    )
    ingester = GPSIngester(
        callback=lambda: [_entry("V1"), _entry("V2", route_short_name=None)],
        settings=settings,
    )

    vehicles = ingester.ingest()

    assert list(vehicles.keys()) == ["V1"]


def test_gps_ingester_uses_default_settings_when_none_given():
    ingester = GPSIngester(callback=lambda: [])
    assert isinstance(ingester.settings, Settings)
    assert ingester.settings.trip_matching is None
