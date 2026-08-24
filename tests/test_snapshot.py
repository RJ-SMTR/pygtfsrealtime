import io
from zoneinfo import ZoneInfo

import pytest

from pygtfsrealtime.exceptions import FatalConfigurationError
from pygtfsrealtime.schedule.ingest import GTFSScheduleIngester
from pygtfsrealtime.schedule.snapshot import (
    GtfsSnapshot,
    build_gtfs_snapshot,
    build_trip_terminal_zones,
    resolve_gtfs_timezone,
)
from pygtfsrealtime.settings import Settings
from tests.gtfs_data import build_gtfs_zip

# TERM1 is a real terminal (something else's parent_station) with 3
# non-collinear children - mirrors tests/test_validation.py's _terminal_stops_gdf.
TERMINAL_STOPS_CSV = (
    "stop_id,stop_lat,stop_lon,parent_station\n"
    "TERM1,-22.900,-43.200,\n"
    "STA,-22.901,-43.200,TERM1\n"
    "STB,-22.900,-43.201,TERM1\n"
    "STC,-22.902,-43.202,TERM1\n"
    "STZ,-22.950,-43.250,\n"
)


def _ingest(overrides: dict[str, str]) -> dict:
    ingester = GTFSScheduleIngester(
        callback=lambda: io.BytesIO(), settings=Settings(projection="EPSG:32723")
    )
    return ingester.ingest(build_gtfs_zip(overrides))


def test_build_trip_terminal_zones_resolves_stop_with_parent_to_terminal_zone():
    gtfs_files = _ingest(
        {
            "stops.txt": TERMINAL_STOPS_CSV,
            "stop_times.txt": (
                "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
                "T1,STA,1,05:00:00,05:01:00\n"
                "T1,STB,2,05:10:00,05:11:00\n"
            ),
        }
    )

    zones = build_trip_terminal_zones(gtfs_files)

    terminal_zone = gtfs_files["terminals"].loc["TERM1", "zone"]
    zone_first, zone_last = zones["T1"]
    assert zone_first is terminal_zone
    assert zone_last is terminal_zone


def test_build_trip_terminal_zones_resolves_stop_without_parent_to_its_own_zone():
    gtfs_files = _ingest(
        {
            "stops.txt": TERMINAL_STOPS_CSV,
            "stop_times.txt": (
                "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
                "T1,STZ,1,05:00:00,05:01:00\n"
                "T1,STZ,2,05:10:00,05:11:00\n"
            ),
        }
    )

    zones = build_trip_terminal_zones(gtfs_files)

    own_zone = gtfs_files["stops.txt"].loc["STZ", "zone"]
    zone_first, zone_last = zones["T1"]
    assert zone_first is own_zone
    assert zone_last is own_zone


def test_build_trip_terminal_zones_falls_back_to_shape_zone_when_stop_is_missing():
    gtfs_files = _ingest(
        {
            "stops.txt": TERMINAL_STOPS_CSV,
            # STMISSING is never declared in stops.txt - stop_times.txt isn't
            # cross-reconciled against stops.txt during ingest, so this survives.
            "stop_times.txt": (
                "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
                "T1,STMISSING,1,05:00:00,05:01:00\n"
                "T1,STMISSING,2,05:10:00,05:11:00\n"
            ),
        }
    )

    zones = build_trip_terminal_zones(gtfs_files)

    shape_row = gtfs_files["shapes.txt"].loc["SH1"]
    zone_first, zone_last = zones["T1"]
    assert zone_first is shape_row["start_zone"]
    assert zone_last is shape_row["end_zone"]


def test_build_gtfs_snapshot_wraps_gtfs_files_hash_and_trip_terminal_zones():
    gtfs_files = _ingest({"stops.txt": TERMINAL_STOPS_CSV})

    snapshot = build_gtfs_snapshot(gtfs_files, gtfs_hash="deadbeef", settings=Settings())

    assert isinstance(snapshot, GtfsSnapshot)
    assert snapshot.gtfs_files is gtfs_files
    assert snapshot.gtfs_hash == "deadbeef"
    assert snapshot.trip_terminal_zones == build_trip_terminal_zones(gtfs_files)


def test_build_gtfs_snapshot_resolves_timezone():
    gtfs_files = _ingest({})

    snapshot = build_gtfs_snapshot(gtfs_files, gtfs_hash="deadbeef", settings=Settings())

    assert snapshot.resolved_timezone == ZoneInfo("UTC")


# --- resolve_gtfs_timezone -------------------------------------------------------


def test_resolve_gtfs_timezone_uses_single_agency_value():
    gtfs_files = _ingest({"agency.txt": "agency_timezone\nAmerica/Sao_Paulo\n"})

    result = resolve_gtfs_timezone(gtfs_files, Settings())

    assert result == ZoneInfo("America/Sao_Paulo")


def test_resolve_gtfs_timezone_ignores_settings_when_agency_is_unambiguous():
    # agency.txt gives an unambiguous answer, so Settings.timezone (even if set
    # to something different) is never consulted.
    gtfs_files = _ingest({"agency.txt": "agency_timezone\nAmerica/Sao_Paulo\n"})

    result = resolve_gtfs_timezone(gtfs_files, Settings(timezone=ZoneInfo("UTC")))

    assert result == ZoneInfo("America/Sao_Paulo")


def test_resolve_gtfs_timezone_falls_back_on_conflict(caplog):
    gtfs_files = _ingest(
        {"agency.txt": ("agency_id,agency_timezone\nA1,America/Sao_Paulo\nA2,America/Manaus\n")}
    )

    with caplog.at_level("WARNING", logger="pygtfsrealtime.schedule.snapshot"):
        result = resolve_gtfs_timezone(gtfs_files, Settings(timezone=ZoneInfo("UTC")))

    assert result == ZoneInfo("UTC")
    messages = [record.message for record in caplog.records]
    assert any("conflicting" in m for m in messages)


def test_resolve_gtfs_timezone_raises_on_conflict_with_no_fallback():
    gtfs_files = _ingest(
        {"agency.txt": ("agency_id,agency_timezone\nA1,America/Sao_Paulo\nA2,America/Manaus\n")}
    )

    with pytest.raises(FatalConfigurationError):
        resolve_gtfs_timezone(gtfs_files, Settings(timezone=None))


def test_resolve_gtfs_timezone_falls_back_when_no_usable_value(caplog):
    # Every agency.txt row is invalid (bad timezone key), so agency_timezone
    # ends up empty after validation - same fallback path as a real conflict.
    gtfs_files = _ingest({"agency.txt": "agency_timezone\nNot/AZone\n"})

    with caplog.at_level("WARNING", logger="pygtfsrealtime.schedule.snapshot"):
        result = resolve_gtfs_timezone(gtfs_files, Settings(timezone=ZoneInfo("UTC")))

    assert result == ZoneInfo("UTC")
    messages = [record.message for record in caplog.records]
    assert any("no usable" in m for m in messages)


def test_resolve_gtfs_timezone_raises_when_no_usable_value_and_no_fallback():
    gtfs_files = _ingest({"agency.txt": "agency_timezone\nNot/AZone\n"})

    with pytest.raises(FatalConfigurationError):
        resolve_gtfs_timezone(gtfs_files, Settings(timezone=None))


def test_resolve_gtfs_timezone_default_settings_timezone_is_none():
    assert Settings().timezone is None


def test_resolve_gtfs_timezone_falls_back_when_agency_has_zero_rows(caplog):
    # Header-only agency.txt (no data rows at all), as opposed to rows that get
    # dropped during validation - same "no usable value" fallback path either way.
    gtfs_files = _ingest({"agency.txt": "agency_timezone\n"})
    assert gtfs_files["agency.txt"].empty

    with caplog.at_level("WARNING", logger="pygtfsrealtime.schedule.snapshot"):
        result = resolve_gtfs_timezone(gtfs_files, Settings(timezone=ZoneInfo("UTC")))

    assert result == ZoneInfo("UTC")
    messages = [record.message for record in caplog.records]
    assert any("no usable" in m for m in messages)


def test_resolve_gtfs_timezone_raises_when_agency_has_zero_rows_and_no_fallback():
    gtfs_files = _ingest({"agency.txt": "agency_timezone\n"})
    assert gtfs_files["agency.txt"].empty

    with pytest.raises(FatalConfigurationError):
        resolve_gtfs_timezone(gtfs_files, Settings(timezone=None))
