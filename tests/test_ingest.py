import io
import zipfile

import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from pygtfsrealtime.schedule.exceptions import GTFSIngestError
from pygtfsrealtime.schedule.ingest import GTFSScheduleIngester
from pygtfsrealtime.settings import GeometryThreshold, Settings, TerminalGeometryThreshold
from tests.gtfs_data import VALID_GTFS_CSV, build_gtfs_zip

# --- read_gtfs_schedule_locally / from_local_file ------------------------------


def test_read_gtfs_schedule_locally_reads_file_bytes(tmp_path):
    file_path = tmp_path / "gtfs.zip"
    file_path.write_bytes(b"raw-bytes")

    result = GTFSScheduleIngester.read_gtfs_schedule_locally(file_path)

    assert isinstance(result, io.BytesIO)
    assert result.read() == b"raw-bytes"


def test_from_local_file_callback_reads_the_given_file(tmp_path):
    file_path = tmp_path / "gtfs.zip"
    file_path.write_bytes(build_gtfs_zip())

    ingester = GTFSScheduleIngester.from_local_file(file_path)

    assert ingester.callback().read() == file_path.read_bytes()


# --- fetch --------------------------------------------------------------------


def test_fetch_returns_callback_bytes():
    ingester = GTFSScheduleIngester(callback=lambda: io.BytesIO(b"payload"))
    assert ingester.fetch() == b"payload"


def test_fetch_wraps_callback_exception():
    def bad_callback():
        raise ValueError("boom")

    ingester = GTFSScheduleIngester(callback=bad_callback)
    with pytest.raises(GTFSIngestError, match="boom"):
        ingester.fetch()


# --- _unzip -----------------------------------------------------------------------


def test_unzip_keeps_only_required_files_and_ignores_directories():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("calendar.txt", "service_id\nS1\n")
        archive.writestr("readme.txt", "not a gtfs file")
        archive.writestr("some_dir/", "")

    extracted = GTFSScheduleIngester._unzip(buffer.getvalue())

    assert set(extracted.keys()) == {"calendar.txt"}


def test_unzip_raises_on_invalid_zip():
    with pytest.raises(GTFSIngestError):
        GTFSScheduleIngester._unzip(b"not a zip file")


# --- _parse -------------------------------------------------------------------


def test_parse_wraps_csv_error_as_gtfs_ingest_error():
    extracted_files = {"calendar.txt": b'a,"b\n1,2\n'}
    with pytest.raises(GTFSIngestError, match="calendar.txt"):
        GTFSScheduleIngester._parse(extracted_files)


# --- ingest: end to end ---------------------------------------------------------


@pytest.fixture
def ingester():
    return GTFSScheduleIngester(
        callback=lambda: io.BytesIO(), settings=Settings(projection="EPSG:32723")
    )


def test_ingest_produces_correctly_typed_dataframes(ingester):
    result = ingester.ingest(build_gtfs_zip())

    assert result["calendar.txt"]["monday"].dtype == bool
    assert result["trips.txt"]["direction_id"].dtype == int

    stops = result["stops.txt"]
    assert stops.index.name == "stop_id"
    assert list(stops.columns) == ["stop_id", "parent_station", "geometry", "zone"]
    assert isinstance(stops["geometry"].iloc[0], Point)
    # stop_geometry defaults to mode="buffer", so zone is geometry expanded, not itself.
    assert isinstance(stops["zone"].iloc[0], Polygon)

    shapes = result["shapes.txt"]
    assert shapes.index.name == "shape_id"
    assert list(shapes.columns) == [
        "shape_id",
        "geometry",
        "zone",
        "start_zone",
        "end_zone",
    ]
    assert isinstance(shapes["geometry"].iloc[0], LineString)
    # shape_geometry defaults to mode="distance", so zone is the raw path geometry.
    assert isinstance(shapes["zone"].iloc[0], LineString)
    # terminal_geometry defaults to mode="buffer", so the endpoint zones are Polygons.
    assert isinstance(shapes["start_zone"].iloc[0], Polygon)
    assert isinstance(shapes["end_zone"].iloc[0], Polygon)

    # VALID_GTFS_CSV's only stop points at a parent_station with no stops.txt row of
    # its own, so it's a dangling reference, not a real terminal.
    assert result["terminals"].empty

    stop_times = result["stop_times.txt"]
    assert stop_times.index.name == "trip_id"
    assert list(stop_times.columns) == [
        "trip_id",
        "first_stop_id",
        "departure_time",
        "last_stop_id",
        "arrival_time",
    ]


def test_ingest_linearizes_stop_times_discarding_intermediate_stops(ingester):
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
            "T1,STC,3,05:10:00,05:11:00\n"
        )
    }
    result = ingester.ingest(build_gtfs_zip(overrides))
    stop_times = result["stop_times.txt"]

    assert stop_times.index.tolist() == ["T1"]
    row = stop_times.loc["T1"]
    assert row["first_stop_id"] == "STA"
    assert row["last_stop_id"] == "STC"
    assert row["departure_time"] == pd.Timedelta(hours=5, minutes=1)
    assert row["arrival_time"] == pd.Timedelta(hours=5, minutes=10)


def test_ingest_drops_trips_missing_a_departure_or_arrival_time(ingester):
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
            "T2,STX,1,,06:00:00\n"
        )
    }
    result = ingester.ingest(build_gtfs_zip(overrides))
    assert result["stop_times.txt"].index.tolist() == ["T1"]


def test_ingest_reconciles_trips_and_shapes_by_shape_id(ingester):
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\n"
            "T1,R1,S1,0,SH1\n"
            "T2,R1,S1,0,SH2\n"  # SH2 has no shapes.txt geometry
        ),
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.90,-43.20\n"
            "SH1,2,-22.91,-43.21\n"
            "SH3,1,-23.00,-43.30\n"  # no trip references SH3
            "SH3,2,-23.01,-43.31\n"
        ),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
            "T2,STC,1,06:00:00,06:01:00\n"
            "T2,STD,2,06:10:00,06:11:00\n"
        ),
    }
    result = ingester.ingest(build_gtfs_zip(overrides))
    assert result["trips.txt"]["trip_id"].tolist() == ["T1"]
    assert result["shapes.txt"].index.tolist() == ["SH1"]
    # T2 cascades out of stop_times.txt too, via the trip/stop_times reconciliation
    # that runs right after this one.
    assert result["stop_times.txt"].index.tolist() == ["T1"]


def test_ingest_reconciles_trips_and_routes_by_route_id(ingester):
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\n"
            "T1,R1,S1,0,SH1\n"
            "T2,R2,S1,0,SH1\n"  # R2 has no routes.txt row
        ),
        "routes.txt": (
            "route_id,route_short_name\nR1,100\nR3,300\n"  # no trip references R3
        ),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
            "T2,STC,1,06:00:00,06:01:00\n"
            "T2,STD,2,06:10:00,06:11:00\n"
        ),
    }
    result = ingester.ingest(build_gtfs_zip(overrides))
    assert result["trips.txt"]["trip_id"].tolist() == ["T1"]
    assert result["routes.txt"]["route_id"].tolist() == ["R1"]
    assert result["stop_times.txt"].index.tolist() == ["T1"]


def test_ingest_reconciles_trip_whose_shape_has_too_few_points(ingester):
    # SH2 has exactly 1 point: the pre-geometry filter can't catch it (it does have
    # a point row and a matching trip) — only build_shapes_geometry's own >= 2
    # points rule drops it, which the second reconcile_shape_coverage pass then
    # cascades into trips.txt/stop_times.txt.
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\nT2,R1,S1,0,SH2\n"
        ),
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.90,-43.20\n"
            "SH1,2,-22.91,-43.21\n"
            "SH2,1,-23.00,-43.30\n"
        ),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
            "T2,STC,1,06:00:00,06:01:00\n"
            "T2,STD,2,06:10:00,06:11:00\n"
        ),
    }
    result = ingester.ingest(build_gtfs_zip(overrides))
    assert result["trips.txt"]["trip_id"].tolist() == ["T1"]
    assert result["shapes.txt"].index.tolist() == ["SH1"]
    assert result["stop_times.txt"].index.tolist() == ["T1"]


def test_ingest_reconciles_trips_stop_times_and_frequencies(ingester):
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\n"
            "T1,R1,S1,0,SH1\n"
            "T2,R1,S1,0,SH1\n"  # no stop_times.txt rows below
        ),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
            "T3,STC,1,06:00:00,06:01:00\n"  # no trips.txt row
            "T3,STD,2,06:10:00,06:11:00\n"
        ),
        "frequencies.txt": (
            "trip_id,start_time,end_time,headway_secs\n"
            "T1,05:00:00,23:00:00,600\n"
            "T4,05:00:00,23:00:00,600\n"  # no trips.txt/stop_times.txt row at all
        ),
    }
    result = ingester.ingest(build_gtfs_zip(overrides))
    assert result["trips.txt"]["trip_id"].tolist() == ["T1"]
    assert result["stop_times.txt"].index.tolist() == ["T1"]
    assert result["frequencies.txt"]["trip_id"].tolist() == ["T1"]


def test_ingest_cascades_shape_orphaned_by_the_final_trip_stop_times_reconciliation(
    ingester,
):
    # T1 has no stop_times.txt rows, so it's only dropped from trips.txt by the
    # trip/stop_times reconciliation - by which point the shape/trip reconciliation
    # has already run and had no reason to touch SH1 (T1 was still there, still
    # pointing at a real shape). Without iterating to a fixed point, SH1 would be
    # left behind in shapes.txt with no trip referencing it anymore.
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\nT2,R1,S1,0,SH2\n"
        ),
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.90,-43.20\n"
            "SH1,2,-22.91,-43.21\n"
            "SH2,1,-23.00,-43.30\n"
            "SH2,2,-23.01,-43.31\n"
        ),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T2,STC,1,06:00:00,06:01:00\n"
            "T2,STD,2,06:10:00,06:11:00\n"
        ),
    }
    result = ingester.ingest(build_gtfs_zip(overrides))
    assert result["trips.txt"]["trip_id"].tolist() == ["T2"]
    assert result["shapes.txt"].index.tolist() == ["SH2"]
    assert result["stop_times.txt"].index.tolist() == ["T2"]


def test_ingest_cascades_route_orphaned_by_the_final_trip_stop_times_reconciliation(
    ingester,
):
    # Same cascade as the shape case above, but for routes.txt: T1 has no
    # stop_times.txt rows, so it's only dropped by the trip/stop_times
    # reconciliation - by which point R1 has already survived the route/trip
    # reconciliation and would be left orphaned in routes.txt without a fixed-point
    # loop.
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\nT2,R2,S1,0,SH1\n"
        ),
        "routes.txt": "route_id,route_short_name\nR1,100\nR2,200\n",
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T2,STC,1,06:00:00,06:01:00\n"
            "T2,STD,2,06:10:00,06:11:00\n"
        ),
    }
    result = ingester.ingest(build_gtfs_zip(overrides))
    assert result["trips.txt"]["trip_id"].tolist() == ["T2"]
    assert result["routes.txt"]["route_id"].tolist() == ["R2"]
    assert result["stop_times.txt"].index.tolist() == ["T2"]


def test_ingest_raises_on_missing_required_file(ingester):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("calendar.txt", "service_id\nS1\n")

    with pytest.raises(TypeError):
        ingester.ingest(buffer.getvalue())


def test_ingest_defaults_to_an_empty_frequencies_when_the_file_is_absent(ingester):
    # frequencies.txt is optional per the GTFS spec - a feed with no
    # frequency-based trips is free to omit it entirely.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, content in VALID_GTFS_CSV.items():
            if filename == "frequencies.txt":
                continue
            archive.writestr(filename, content)

    result = ingester.ingest(buffer.getvalue())

    assert result["frequencies.txt"].empty
    assert list(result["frequencies.txt"].columns) == [
        "end_time",
        "headway_secs",
        "start_time",
        "trip_id",
    ]


def test_ingest_raises_on_missing_required_column(ingester):
    overrides = {"stops.txt": "stop_id,stop_lon,parent_station\nST1,-43.2,TERM1\n"}
    with pytest.raises(TypeError, match="stop_lat"):
        ingester.ingest(build_gtfs_zip(overrides))


def test_ingest_drops_rows_with_out_of_range_value(ingester):
    overrides = {
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.9,-43.2\n"
            "SH1,2,-22.91,-43.21\n"
            "SH2,1,999,-43.2\n"
        )
    }
    result = ingester.ingest(build_gtfs_zip(overrides))
    assert result["shapes.txt"]["shape_id"].tolist() == ["SH1"]
    assert isinstance(result["shapes.txt"]["geometry"].iloc[0], LineString)


def test_ingest_drops_incomplete_rows_before_type_validation(ingester):
    overrides = {
        "stops.txt": (
            "stop_id,stop_lat,stop_lon,parent_station\nST1,-22.9,-43.2,TERM1\nST2,,-43.2,TERM1\n"
        )
    }
    result = ingester.ingest(build_gtfs_zip(overrides))
    assert result["stops.txt"]["stop_id"].tolist() == ["ST1"]


def test_ingest_keeps_stops_with_no_parent_station(ingester):
    # parent_station is optional per the GTFS spec — most stops aren't part of a
    # larger station and legitimately have it blank. That shouldn't be treated as
    # an incomplete row.
    overrides = {
        "stops.txt": (
            "stop_id,stop_lat,stop_lon,parent_station\nST1,-22.9,-43.2,\nST2,-22.8,-43.1,TERM1\n"
        )
    }
    result = ingester.ingest(build_gtfs_zip(overrides))
    assert result["stops.txt"]["stop_id"].tolist() == ["ST1", "ST2"]


def test_ingest_drops_exact_duplicate_rows(ingester):
    overrides = {
        "routes.txt": "route_id,route_short_name\nR1,100\nR1,100\n",
    }
    result = ingester.ingest(build_gtfs_zip(overrides))
    assert result["routes.txt"]["route_id"].tolist() == ["R1"]


def test_ingest_drops_rows_that_repeat_a_primary_key(ingester):
    overrides = {
        "stops.txt": (
            "stop_id,stop_lat,stop_lon,parent_station\n"
            "ST1,-22.9,-43.2,TERM1\n"
            "ST1,-23.0,-43.3,TERM1\n"  # same stop_id, different coordinates
        ),
    }
    result = ingester.ingest(build_gtfs_zip(overrides))
    assert result["stops.txt"]["stop_id"].tolist() == ["ST1"]
    assert isinstance(result["stops.txt"]["geometry"].iloc[0], Point)


def test_ingest_uses_default_settings_when_none_given():
    ingester = GTFSScheduleIngester(callback=lambda: io.BytesIO())
    assert ingester.settings.primary_key_duplicate_policy == "keep_first"


def test_ingest_drop_all_policy_removes_every_conflicting_row():
    overrides = {
        "stops.txt": (
            "stop_id,stop_lat,stop_lon,parent_station\n"
            "ST1,-22.9,-43.2,TERM1\n"
            "ST1,-23.0,-43.3,TERM1\n"
            "ST2,-22.9,-43.2,TERM1\n"
        ),
    }
    ingester = GTFSScheduleIngester(
        callback=lambda: io.BytesIO(),
        settings=Settings(projection="EPSG:32723", primary_key_duplicate_policy="drop_all"),
    )
    result = ingester.ingest(build_gtfs_zip(overrides))
    assert result["stops.txt"]["stop_id"].tolist() == ["ST2"]


def test_ingest_builds_terminal_zone_for_stops_with_children(ingester):
    overrides = {
        "stops.txt": (
            "stop_id,stop_lat,stop_lon,parent_station\n"
            "TERM1,-22.900,-43.200,\n"
            "STA,-22.901,-43.200,TERM1\n"
            "STB,-22.900,-43.201,TERM1\n"
            "STC,-22.902,-43.202,TERM1\n"
        ),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
        ),
    }
    result = ingester.ingest(build_gtfs_zip(overrides))

    terminals = result["terminals"]
    assert terminals.index.tolist() == ["TERM1"]
    # Default terminal_geometry: mode="buffer", shape_mode="stops_convex_hull".
    assert isinstance(terminals.loc["TERM1", "geometry"], Polygon)
    assert isinstance(terminals.loc["TERM1", "zone"], Polygon)
    assert terminals.loc["TERM1", "zone"].contains(terminals.loc["TERM1", "geometry"])


def test_ingest_threads_geometry_settings_into_zone_precomputation():
    settings = Settings(
        projection="EPSG:32723",
        stop_geometry=GeometryThreshold(distance=50, mode="distance"),
        shape_geometry=GeometryThreshold(distance=20, mode="buffer"),
        terminal_geometry=TerminalGeometryThreshold(
            distance=10, mode="distance", shape_mode="terminal_point"
        ),
    )
    ingester = GTFSScheduleIngester(callback=lambda: io.BytesIO(), settings=settings)
    overrides = {
        "stops.txt": (
            "stop_id,stop_lat,stop_lon,parent_station\n"
            "TERM1,-22.900,-43.200,\n"
            "STA,-22.901,-43.200,TERM1\n"
            "STB,-22.900,-43.201,TERM1\n"
        ),
    }
    result = ingester.ingest(build_gtfs_zip(overrides))

    stops = result["stops.txt"]
    assert stops.loc["STA", "zone"] is stops.loc["STA", "geometry"]

    shapes = result["shapes.txt"]
    assert isinstance(shapes.loc["SH1", "zone"], Polygon)

    terminals = result["terminals"]
    assert terminals.loc["TERM1", "geometry"].equals(stops.loc["TERM1", "geometry"])
    assert terminals.loc["TERM1", "zone"] is terminals.loc["TERM1", "geometry"]


# --- ingest: trip endpoint zones (build_trip_endpoints) -------------------------


def _trip_zones(result, trip_id="T1"):
    trips = result["trips.txt"].set_index("trip_id")
    return trips.loc[trip_id, "start_zone"], trips.loc[trip_id, "end_zone"]


def test_ingest_embeds_start_and_end_zone_on_trips_using_stop_source_by_default(
    ingester,
):
    overrides = {
        "stops.txt": (
            "stop_id,stop_lat,stop_lon,parent_station\n"
            "TERM1,-22.900,-43.200,\n"
            "STA,-22.901,-43.200,TERM1\n"
            "STB,-22.900,-43.201,TERM1\n"
        ),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
        ),
    }
    result = ingester.ingest(build_gtfs_zip(overrides))

    start_zone, end_zone = _trip_zones(result)
    assert start_zone is result["stops.txt"].loc["STA", "zone"]
    assert end_zone is result["stops.txt"].loc["STB", "zone"]


def test_ingest_trip_endpoint_source_terminal_uses_parent_zone():
    settings = Settings(projection="EPSG:32723", trip_endpoint_source="terminal")
    ingester = GTFSScheduleIngester(callback=lambda: io.BytesIO(), settings=settings)
    overrides = {
        "stops.txt": (
            "stop_id,stop_lat,stop_lon,parent_station\n"
            "TERM1,-22.900,-43.200,\n"
            "STA,-22.901,-43.200,TERM1\n"
            "STB,-22.900,-43.201,TERM1\n"
        ),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
        ),
    }
    result = ingester.ingest(build_gtfs_zip(overrides))

    start_zone, end_zone = _trip_zones(result)
    terminal_zone = result["terminals"].loc["TERM1", "zone"]
    assert start_zone is terminal_zone
    assert end_zone is terminal_zone


def test_ingest_trip_endpoint_falls_back_to_shape_when_stop_is_missing(ingester):
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STMISSING,1,05:00:00,05:01:00\n"
            "T1,STMISSING,2,05:10:00,05:11:00\n"
        ),
    }
    result = ingester.ingest(build_gtfs_zip(overrides))

    start_zone, end_zone = _trip_zones(result)
    shape_row = result["shapes.txt"].loc["SH1"]
    assert start_zone is shape_row["start_zone"]
    assert end_zone is shape_row["end_zone"]


def test_ingest_default_trip_endpoint_source_is_stop():
    assert Settings().trip_endpoint_source == "stop"
