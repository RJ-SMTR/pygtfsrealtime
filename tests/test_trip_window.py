import io
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from pygtfsrealtime.schedule.ingest import GTFSScheduleIngester
from pygtfsrealtime.schedule.snapshot import build_gtfs_snapshot
from pygtfsrealtime.settings import Settings
from pygtfsrealtime.trip_window.compute import (
    TripsSnapshot,
    anchor_offsets_to_service_day,
    assemble_trip_window_columns,
    build_service_day_instances,
    build_trips_snapshot,
    build_window_trip_instances,
    candidate_service_dates,
    expand_frequencies_instances,
    resolve_active_service_ids,
    stop_times_only_instances,
)
from tests.gtfs_data import build_gtfs_zip

UTC = ZoneInfo("UTC")


def _ingest(overrides: dict[str, str], settings: Settings | None = None) -> dict:
    settings = settings or Settings(projection="EPSG:32723")
    ingester = GTFSScheduleIngester(callback=lambda: io.BytesIO(), settings=settings)
    return ingester.ingest(build_gtfs_zip(overrides))


# --- resolve_active_service_ids -----------------------------------------------


def _calendar(service_id="S1", weekday=True, start=date(2026, 1, 1), end=date(2026, 12, 31)):
    return pd.DataFrame(
        {
            "service_id": [service_id],
            "monday": [weekday],
            "tuesday": [weekday],
            "wednesday": [weekday],
            "thursday": [weekday],
            "friday": [weekday],
            "saturday": [False],
            "sunday": [False],
            "start_date": [start],
            "end_date": [end],
        }
    )


def _empty_calendar_dates():
    return pd.DataFrame(columns=["service_id", "date", "exception_type"])


def test_resolve_active_service_ids_regular_weekday_match():
    calendar = _calendar()
    result = resolve_active_service_ids(calendar, _empty_calendar_dates(), date(2026, 1, 5))
    assert result == {"S1"}


def test_resolve_active_service_ids_excludes_date_outside_range():
    calendar = _calendar(start=date(2026, 2, 1), end=date(2026, 12, 31))
    result = resolve_active_service_ids(calendar, _empty_calendar_dates(), date(2026, 1, 5))
    assert result == set()


def test_resolve_active_service_ids_exception_adds_service():
    calendar = _calendar()
    calendar_dates = pd.DataFrame(
        {"service_id": ["S2"], "date": [date(2026, 1, 5)], "exception_type": [1]}
    )
    result = resolve_active_service_ids(calendar, calendar_dates, date(2026, 1, 5))
    assert result == {"S1", "S2"}


def test_resolve_active_service_ids_exception_removes_service():
    calendar = _calendar()
    calendar_dates = pd.DataFrame(
        {"service_id": ["S1"], "date": [date(2026, 1, 5)], "exception_type": [2]}
    )
    result = resolve_active_service_ids(calendar, calendar_dates, date(2026, 1, 5))
    assert result == set()


# --- candidate_service_dates ----------------------------------------------------


def test_candidate_service_dates_window_within_one_day():
    window_start = pd.Timestamp(2026, 1, 15, 8, tz=UTC)
    window_end = pd.Timestamp(2026, 1, 15, 16, tz=UTC)
    result = candidate_service_dates(window_start, window_end, pd.Timedelta(0))
    assert result == [date(2026, 1, 14), date(2026, 1, 15)]


def test_candidate_service_dates_crossing_midnight():
    window_start = pd.Timestamp(2026, 1, 15, 22, tz=UTC)
    window_end = pd.Timestamp(2026, 1, 16, 6, tz=UTC)
    result = candidate_service_dates(window_start, window_end, pd.Timedelta(0))
    assert result == [date(2026, 1, 14), date(2026, 1, 15), date(2026, 1, 16)]


def test_candidate_service_dates_large_offset_extends_lookback():
    window_start = pd.Timestamp(2026, 1, 15, 8, tz=UTC)
    window_end = pd.Timestamp(2026, 1, 15, 16, tz=UTC)
    result = candidate_service_dates(window_start, window_end, pd.Timedelta(hours=30))
    assert result[0] == date(2026, 1, 13)


def test_candidate_service_dates_clamps_absurd_offset(caplog):
    window_start = pd.Timestamp(2026, 1, 15, 8, tz=UTC)
    window_end = pd.Timestamp(2026, 1, 15, 16, tz=UTC)
    with caplog.at_level("WARNING", logger="pygtfsrealtime.trip_window.compute"):
        result = candidate_service_dates(window_start, window_end, pd.Timedelta(hours=9999))
    # Clamped to the default max_lookback_days=3, not the ~416 days the raw
    # offset implies.
    assert result[0] == date(2026, 1, 12)
    assert any("clamping" in record.message for record in caplog.records)


def test_settings_default_trip_window_max_lookback_days_is_three():
    assert Settings().trip_window_max_lookback_days == 3


def test_candidate_service_dates_respects_custom_max_lookback_days():
    window_start = pd.Timestamp(2026, 1, 15, 8, tz=UTC)
    window_end = pd.Timestamp(2026, 1, 15, 16, tz=UTC)
    result = candidate_service_dates(
        window_start, window_end, pd.Timedelta(hours=9999), max_lookback_days=1
    )
    assert result[0] == date(2026, 1, 14)


# --- expand_frequencies_instances -----------------------------------------------


def test_expand_frequencies_instances_end_offset_comes_from_stop_times_duration():
    # headway=600s but the trip's own duration (from stop_times) is 1800s - the
    # regression this locks in: end_offset - start_offset must be 1800s, never
    # the 600s headway the legacy code used.
    frequencies = pd.DataFrame(
        {
            "trip_id": ["T1"],
            "start_time": [pd.Timedelta(hours=5)],
            "end_time": [pd.Timedelta(hours=6)],
            "headway_secs": [pd.Timedelta(seconds=600)],
        }
    )
    stop_times = pd.DataFrame(
        {
            "departure_time": [pd.Timedelta(hours=5)],
            "arrival_time": [pd.Timedelta(hours=5, minutes=30)],
        },
        index=pd.Index(["T1"], name="trip_id"),
    )

    result = expand_frequencies_instances(frequencies, stop_times)

    assert len(result) == 7  # (6h-5h)//600s + 1 = 3600//600 + 1
    durations = result["end_offset"] - result["start_offset"]
    assert (durations == pd.Timedelta(minutes=30)).all()
    # k*headway spacing between consecutive instances.
    assert result["start_offset"].iloc[1] - result["start_offset"].iloc[0] == pd.Timedelta(
        seconds=600
    )
    assert result["start_offset"].iloc[0] == pd.Timedelta(hours=5)
    assert result["start_offset"].iloc[-1] == pd.Timedelta(hours=6)


# --- stop_times_only_instances --------------------------------------------------


def test_stop_times_only_instances_excludes_trips_with_frequencies():
    stop_times = pd.DataFrame(
        {
            "trip_id": ["T1", "T2"],
            "departure_time": [pd.Timedelta(hours=5), pd.Timedelta(hours=6)],
            "arrival_time": [pd.Timedelta(hours=5, minutes=30), pd.Timedelta(hours=6, minutes=20)],
        },
        index=pd.Index(["T1", "T2"], name="trip_id"),
    )
    frequencies = pd.DataFrame({"trip_id": ["T1"]})

    result = stop_times_only_instances(stop_times, frequencies)

    assert result["trip_id"].tolist() == ["T2"]
    assert result["start_offset"].iloc[0] == pd.Timedelta(hours=6)
    assert result["end_offset"].iloc[0] == pd.Timedelta(hours=6, minutes=20)


# --- anchor_offsets_to_service_day ----------------------------------------------


def test_anchor_offsets_to_service_day_handles_past_midnight_offset():
    offsets = pd.DataFrame(
        {
            "trip_id": ["T1"],
            "start_offset": [pd.Timedelta(hours=25)],
            "end_offset": [pd.Timedelta(hours=25, minutes=30)],
        }
    )
    result = anchor_offsets_to_service_day(offsets, date(2026, 1, 15), UTC)

    assert result["start_dt"].iloc[0] == pd.Timestamp(2026, 1, 16, 1, tz=UTC)
    assert result["end_dt"].iloc[0] == pd.Timestamp(2026, 1, 16, 1, 30, tz=UTC)


# --- build_service_day_instances -------------------------------------------------


def test_build_service_day_instances_mixes_frequency_and_stop_times_only_trips():
    gtfs_files = {
        "trips.txt": pd.DataFrame(
            {
                "trip_id": ["T1", "T2", "T3"],
                "service_id": ["S1", "S1", "S2"],
            }
        ),
        "frequencies.txt": pd.DataFrame(
            {
                "trip_id": ["T1"],
                "start_time": [pd.Timedelta(hours=5)],
                "end_time": [pd.Timedelta(hours=5, minutes=20)],
                "headway_secs": [pd.Timedelta(seconds=600)],
            }
        ),
        "stop_times.txt": pd.DataFrame(
            {
                "trip_id": ["T1", "T2", "T3"],
                "departure_time": [
                    pd.Timedelta(hours=5),
                    pd.Timedelta(hours=6),
                    pd.Timedelta(hours=7),
                ],
                "arrival_time": [
                    pd.Timedelta(hours=5, minutes=10),
                    pd.Timedelta(hours=6, minutes=15),
                    pd.Timedelta(hours=7, minutes=15),
                ],
            },
            index=pd.Index(["T1", "T2", "T3"], name="trip_id"),
        ),
    }

    result = build_service_day_instances(
        gtfs_files, date(2026, 1, 15), active_service_ids={"S1"}, timezone=UTC
    )

    # T3 is filtered out (service_id S2 not active); T1 expands (3 instances:
    # 5:00, 5:10, 5:20), T2 contributes one stop-times-only instance.
    assert sorted(result["trip_id"].tolist()) == ["T1", "T1", "T1", "T2"]
    assert "T3" not in result["trip_id"].tolist()


# --- assemble_trip_window_columns ------------------------------------------------


def test_assemble_trip_window_columns_joins_route_and_shape_metadata():
    instances = pd.DataFrame(
        {
            "trip_id": ["T1"],
            "start_dt": [pd.Timestamp(2026, 1, 15, 5, tz=UTC)],
            "end_dt": [pd.Timestamp(2026, 1, 15, 5, 30, tz=UTC)],
        }
    )
    start_zone_geom = Point(0, 0)
    end_zone_geom = Point(1, 1)
    shape_geom = LineString([(0, 0), (1, 1)])
    gtfs_files = {
        "trips.txt": pd.DataFrame(
            {
                "trip_id": ["T1"],
                "route_id": ["R1"],
                "direction_id": [0],
                "shape_id": ["SH1"],
                "start_zone": [start_zone_geom],
                "start_zone_source": ["stop"],
                "end_zone": [end_zone_geom],
                "end_zone_source": ["stop"],
            }
        ),
        "routes.txt": pd.DataFrame({"route_id": ["R1"], "route_short_name": ["100"]}),
        "shapes.txt": gpd.GeoDataFrame(
            {"shape_id": ["SH1"], "geometry": [shape_geom]}, crs="EPSG:32723"
        ).set_index("shape_id", drop=False),
    }

    result = assemble_trip_window_columns(instances, gtfs_files)

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == "EPSG:32723"
    assert result.geometry.name == "shape_geometry"
    assert list(result.columns) == [
        "trip_id",
        "start_dt",
        "end_dt",
        "route_id",
        "route_short_name",
        "direction_id",
        "start_zone",
        "start_zone_source",
        "end_zone",
        "end_zone_source",
        "shape_geometry",
    ]
    row = result.iloc[0]
    assert row["route_short_name"] == "100"
    assert row["start_zone"] is start_zone_geom
    assert row["start_zone_source"] == "stop"
    assert row["end_zone"] is end_zone_geom
    assert row["end_zone_source"] == "stop"
    assert row["shape_geometry"] == shape_geom


# --- build_window_trip_instances / build_trips_snapshot (integration) -----------


def test_build_window_trip_instances_crossing_midnight_includes_past_midnight_trip():
    overrides = {
        "trips.txt": ("trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\n"),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,ST1,1,25:00:00,25:00:00\n"
            "T1,ST1,2,25:30:00,25:30:00\n"
        ),
        "frequencies.txt": "trip_id,start_time,end_time,headway_secs\n",
    }
    gtfs_files = _ingest(overrides)

    # Window covers today 00:00-02:00 - a trip from yesterday's service offset
    # 25:00 (=today 01:00) should still show up.
    window_start = pd.Timestamp(2026, 1, 2, 0, tz=UTC)
    window_end = pd.Timestamp(2026, 1, 2, 2, tz=UTC)
    result = build_window_trip_instances(gtfs_files, window_start, window_end, UTC)

    assert result.loc[("T1", pd.Timestamp(2026, 1, 2, 1, tz=UTC))]["trip_id"] == "T1"


def test_build_window_trip_instances_includes_trip_already_in_progress():
    overrides = {
        "trips.txt": ("trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\n"),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,ST1,1,05:00:00,05:00:00\n"
            "T1,ST1,2,06:30:00,06:30:00\n"
        ),
        "frequencies.txt": "trip_id,start_time,end_time,headway_secs\n",
    }
    gtfs_files = _ingest(overrides)

    # Window starts at 06:00, but T1 started at 05:00 and only ends at 06:30 -
    # overlap semantics must still include it.
    window_start = pd.Timestamp(2026, 1, 1, 6, tz=UTC)
    window_end = pd.Timestamp(2026, 1, 1, 8, tz=UTC)
    result = build_window_trip_instances(gtfs_files, window_start, window_end, UTC)

    assert ("T1", pd.Timestamp(2026, 1, 1, 5, tz=UTC)) in result.index


def test_build_window_trip_instances_excludes_trip_entirely_before_window():
    overrides = {
        "trips.txt": ("trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\n"),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,ST1,1,05:00:00,05:00:00\n"
            "T1,ST1,2,05:10:00,05:10:00\n"
        ),
        "frequencies.txt": "trip_id,start_time,end_time,headway_secs\n",
    }
    gtfs_files = _ingest(overrides)

    window_start = pd.Timestamp(2026, 1, 1, 6, tz=UTC)
    window_end = pd.Timestamp(2026, 1, 1, 8, tz=UTC)
    result = build_window_trip_instances(gtfs_files, window_start, window_end, UTC)

    assert result.empty


def test_build_trips_snapshot_window_size_and_localization():
    settings = Settings(projection="EPSG:32723", timezone=UTC)
    gtfs_files = _ingest({}, settings=settings)
    snapshot = build_gtfs_snapshot(gtfs_files, gtfs_hash="abc123", settings=settings)

    now = datetime(2026, 1, 1, 12, 0, 0)  # naive
    result = build_trips_snapshot(snapshot, settings, now)

    assert isinstance(result, TripsSnapshot)
    assert result.window_start == pd.Timestamp(2026, 1, 1, 12, tz=UTC)
    expected_length = (
        timedelta(seconds=settings.trip_window_loop_schedule.interval) + settings.trip_window_margin
    )
    assert result.window_end - result.window_start == expected_length
    assert result.gtfs_hash == "abc123"


def test_build_trips_snapshot_converts_tz_aware_now():
    settings = Settings(projection="EPSG:32723", timezone=UTC)
    gtfs_files = _ingest({}, settings=settings)
    snapshot = build_gtfs_snapshot(gtfs_files, gtfs_hash="abc123", settings=settings)

    now = datetime(2026, 1, 1, 9, 0, 0, tzinfo=ZoneInfo("America/Sao_Paulo"))
    result = build_trips_snapshot(snapshot, settings, now)

    assert result.window_start == pd.Timestamp(now).tz_convert(UTC)


def test_build_trips_snapshot_threads_max_lookback_days():
    # Offset 49h can only be found by anchoring 2 service days back
    # (2026-01-03 + 49h = 2026-01-05 01:00, inside the window below) - the
    # default cap (3) reaches that far back, Settings(trip_window_max_lookback_
    # days=1) doesn't.
    overrides = {
        "calendar.txt": (
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
            "start_date,end_date\nS1,1,1,1,1,1,1,1,20260101,20261231\n"
        ),
        "trips.txt": ("trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\n"),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,ST1,1,49:00:00,49:00:00\n"
            "T1,ST1,2,49:30:00,49:30:00\n"
        ),
        "frequencies.txt": "trip_id,start_time,end_time,headway_secs\n",
    }
    settings_default = Settings(projection="EPSG:32723", timezone=UTC)
    gtfs_files = _ingest(overrides, settings=settings_default)
    snapshot = build_gtfs_snapshot(gtfs_files, gtfs_hash="abc123", settings=settings_default)
    now = datetime(2026, 1, 5, 0, 0, 0)

    result_default = build_trips_snapshot(snapshot, settings_default, now)
    assert len(result_default.trips) == 1

    settings_clamped = Settings(timezone=UTC, trip_window_max_lookback_days=1)
    result_clamped = build_trips_snapshot(snapshot, settings_clamped, now)
    assert result_clamped.trips.empty


def test_build_trips_snapshot_trips_multiindex_has_no_duplicates():
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\nT2,R1,S1,0,SH1\n"
        ),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,ST1,1,05:00:00,05:00:00\nT1,ST1,2,05:30:00,05:30:00\n"
            "T2,ST1,1,06:00:00,06:00:00\nT2,ST1,2,06:30:00,06:30:00\n"
        ),
        "frequencies.txt": "trip_id,start_time,end_time,headway_secs\n",
    }
    settings = Settings(projection="EPSG:32723", timezone=UTC)
    gtfs_files = _ingest(overrides, settings=settings)
    snapshot = build_gtfs_snapshot(gtfs_files, gtfs_hash="abc123", settings=settings)

    result = build_trips_snapshot(snapshot, settings, datetime(2026, 1, 1, 4, 0, 0))

    assert not result.trips.index.duplicated().any()
    assert result.trips.index.names == ["trip_id", "start_dt"]
