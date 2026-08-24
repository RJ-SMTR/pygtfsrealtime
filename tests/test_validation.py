from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from pygtfsrealtime.contracts import (
    validate_cache_result,
    validate_gps_data_result,
    validate_schedule_result,
)
from pygtfsrealtime.models import GPSEntry
from pygtfsrealtime.schedule.compute import (
    build_gtfs_schedule_geometries,
    build_gtfs_schedule_zones,
    build_shape_zones,
    build_shapes_geometry,
    build_stop_zones,
    build_stops_geometry,
    build_terminal_zones,
    build_trip_endpoints,
    drop_stop_times_missing_endpoints,
    linearize_stop_times,
    reconcile_gtfs_schedule_relations,
    reconcile_route_coverage,
    reconcile_shape_coverage,
    reconcile_trip_coverage,
)
from pygtfsrealtime.schedule.gtfs_spec import (
    GTFS_SCHEDULE_OPTIONAL_FILES,
    GTFS_SCHEDULE_PRIMARY_KEYS,
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
from pygtfsrealtime.settings import GeometryThreshold, Settings, TerminalGeometryThreshold
from tests.gtfs_data import VALID_GTFS_CSV, build_gtfs_dataframes


def test_gtfs_data_fixture_covers_all_required_files():
    assert set(VALID_GTFS_CSV.keys()) == GTFS_SCHEDULE_REQUIRED_FILES | GTFS_SCHEDULE_OPTIONAL_FILES


# --- validate_gtfs_schedule_required_files ------------------------------------


def test_validate_required_files_passes_with_full_set():
    files = set(VALID_GTFS_CSV.keys())
    assert validate_gtfs_schedule_required_files(files) == files


def test_validate_required_files_raises_when_missing():
    files = set(VALID_GTFS_CSV.keys()) - {"stops.txt"}
    with pytest.raises(TypeError, match="stops.txt"):
        validate_gtfs_schedule_required_files(files)


# --- validate_gtfs_schedule_columns --------------------------------------------


def test_validate_columns_passes_with_required_columns_present():
    gtfs_files = build_gtfs_dataframes()
    assert validate_gtfs_schedule_columns(gtfs_files) is gtfs_files


def test_validate_columns_raises_when_missing_column():
    gtfs_files = build_gtfs_dataframes()
    gtfs_files["stops.txt"] = gtfs_files["stops.txt"].drop(columns=["stop_lat"])
    with pytest.raises(TypeError, match="stop_lat"):
        validate_gtfs_schedule_columns(gtfs_files)


# --- validate_schedule_result / validate_gps_data_result / validate_cache_result -


def test_validate_schedule_result_passes_dict():
    result = {"a.txt": pd.DataFrame()}
    assert validate_schedule_result(result) is result


def test_validate_schedule_result_raises_on_non_dict():
    with pytest.raises(TypeError):
        validate_schedule_result([1, 2, 3])


def test_validate_gps_data_result_passes_list_of_gps_entry():
    entries = [
        GPSEntry(
            vehicle_id="V1",
            route_short_name="100",
            latitude=-22.9,
            longitude=-43.2,
            datetime=datetime(2026, 1, 1),
            direction_id=0,
        )
    ]
    assert validate_gps_data_result(entries) is entries


@pytest.mark.parametrize("bad_result", [[{"vehicle_id": "V1"}], "not-a-list", None])
def test_validate_gps_data_result_raises_on_invalid(bad_result):
    with pytest.raises(TypeError):
        validate_gps_data_result(bad_result)


@pytest.mark.parametrize("value", [b"abc", None])
def test_validate_cache_result_passes(value):
    assert validate_cache_result(value) == value


def test_validate_cache_result_raises_on_non_bytes():
    with pytest.raises(TypeError):
        validate_cache_result("not-bytes")


# --- validate_gtfs_schedule_types: dtype coercion ------------------------------


def test_validate_types_coerces_expected_dtypes():
    result = validate_gtfs_schedule_types(build_gtfs_dataframes())

    assert result["calendar.txt"]["monday"].dtype == bool
    assert result["shapes.txt"]["shape_pt_lat"].dtype == float
    assert result["trips.txt"]["direction_id"].dtype == int
    assert pd.api.types.is_string_dtype(result["stops.txt"]["stop_id"])


def test_validate_types_leaves_columns_outside_schema_untouched():
    gtfs_files = build_gtfs_dataframes()
    gtfs_files["stops.txt"]["extra_column"] = ["unchanged"]
    result = validate_gtfs_schedule_types(gtfs_files)
    assert result["stops.txt"]["extra_column"].tolist() == ["unchanged"]


# --- validate_gtfs_schedule_types: agency_timezone (is_timezone) ---------------


def test_validate_types_converts_valid_iana_timezone_to_zoneinfo():
    overrides = {"agency.txt": "agency_timezone\nAmerica/Sao_Paulo\n"}
    result = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))
    value = result["agency.txt"]["agency_timezone"].iloc[0]
    assert value == ZoneInfo("America/Sao_Paulo")


def test_validate_types_drops_row_with_invalid_timezone_key():
    overrides = {"agency.txt": "agency_timezone\nAmerica/Sao_Paulo\nNot/AZone\n"}
    result = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))
    values = result["agency.txt"]["agency_timezone"].tolist()
    assert values == [ZoneInfo("America/Sao_Paulo")]


def test_validate_types_drops_row_with_empty_timezone():
    overrides = {"agency.txt": "agency_id,agency_timezone\nA1,\nA2,America/Sao_Paulo\n"}
    result = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))
    values = result["agency.txt"]["agency_timezone"].tolist()
    assert values == [ZoneInfo("America/Sao_Paulo")]


def test_validate_types_ignores_files_without_schema():
    gtfs_files = build_gtfs_dataframes()
    gtfs_files["translations.txt"] = pd.DataFrame({"trans_id": ["T1"]})
    result = validate_gtfs_schedule_types(gtfs_files)
    assert result["translations.txt"]["trans_id"].tolist() == ["T1"]


# --- validate_gtfs_schedule_types: nullable columns -----------------------------


def test_validate_types_keeps_row_with_missing_value_in_nullable_column():
    # parent_station is optional — a stop with no parent shouldn't be dropped or
    # have its missing value treated as an invalid one.
    overrides = {
        "stops.txt": (
            "stop_id,stop_lat,stop_lon,parent_station\nST1,-22.9,-43.2,\nST2,-22.8,-43.1,TERM1\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = validate_gtfs_schedule_types(gtfs_files)
    assert result["stops.txt"]["stop_id"].tolist() == ["ST1", "ST2"]
    assert pd.isna(result["stops.txt"]["parent_station"].iloc[0])
    assert result["stops.txt"]["parent_station"].iloc[1] == "TERM1"


def test_validate_types_does_not_stringify_missing_nullable_value():
    # A naive df.astype(str) turns NaN into the literal string "nan" — the real
    # missing value must survive the cast instead.
    overrides = {"stops.txt": "stop_id,stop_lat,stop_lon,parent_station\nST1,-22.9,-43.2,\n"}
    gtfs_files = build_gtfs_dataframes(overrides)
    result = validate_gtfs_schedule_types(gtfs_files)
    value = result["stops.txt"]["parent_station"].iloc[0]
    assert value != "nan"
    assert pd.isna(value)


def test_validate_types_keeps_stop_time_row_missing_only_departure_time():
    # departure_time is only required on a trip's first stop_time — a trip's
    # last stop commonly has no departure (the vehicle doesn't leave again), and
    # dropping that row would also throw away its (present) arrival_time, which
    # trip_instances.py needs for the trip's end-of-window calculation.
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,ST1,1,05:00:00,05:01:00\n"
            "T1,ST2,2,05:10:00,\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = validate_gtfs_schedule_types(gtfs_files)
    stop_times = result["stop_times.txt"]
    assert stop_times["stop_id"].tolist() == ["ST1", "ST2"]
    assert stop_times["arrival_time"].iloc[1] == pd.Timedelta(hours=5, minutes=10)
    assert pd.isna(stop_times["departure_time"].iloc[1])


def test_validate_types_keeps_stop_time_row_missing_only_arrival_time():
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,ST1,1,,05:01:00\n"
            "T1,ST2,2,05:10:00,05:11:00\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = validate_gtfs_schedule_types(gtfs_files)
    stop_times = result["stop_times.txt"]
    assert stop_times["stop_id"].tolist() == ["ST1", "ST2"]
    assert pd.isna(stop_times["arrival_time"].iloc[0])
    assert stop_times["departure_time"].iloc[0] == pd.Timedelta(hours=5, minutes=1)


# --- validate_gtfs_schedule_types: valid_values (row dropped, not raised) ------


@pytest.mark.parametrize(
    "overrides, filename, id_column, kept_id",
    [
        (
            {
                "calendar.txt": (
                    "service_id,monday,tuesday,wednesday,thursday,friday,saturday,"
                    "sunday,start_date,end_date\n"
                    "S1,1,1,1,1,1,0,0,20260101,20261231\n"
                    "S2,2,1,1,1,1,0,0,20260101,20261231\n"
                )
            },
            "calendar.txt",
            "service_id",
            "S1",
        ),
        (
            {
                "calendar_dates.txt": (
                    "service_id,date,exception_type\nS1,20260101,1\nS2,20260101,3\n"
                )
            },
            "calendar_dates.txt",
            "service_id",
            "S1",
        ),
        (
            {
                "trips.txt": (
                    "trip_id,route_id,service_id,direction_id,shape_id\n"
                    "T1,R1,S1,0,SH1\n"
                    "T2,R1,S1,7,SH1\n"
                )
            },
            "trips.txt",
            "trip_id",
            "T1",
        ),
    ],
)
def test_validate_types_drops_rows_with_invalid_valid_values(
    overrides, filename, id_column, kept_id
):
    gtfs_files = build_gtfs_dataframes(overrides)
    result = validate_gtfs_schedule_types(gtfs_files)
    assert result[filename][id_column].tolist() == [kept_id]


# --- validate_gtfs_schedule_types: min/max ranges (row dropped, not raised) ----


@pytest.mark.parametrize(
    "overrides, filename, id_column, kept_id",
    [
        (
            {
                "shapes.txt": (
                    "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
                    "SH1,1,-22.9,-43.2\n"
                    "SH2,1,999,-43.2\n"
                )
            },
            "shapes.txt",
            "shape_id",
            "SH1",
        ),
        (
            {
                "shapes.txt": (
                    "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
                    "SH1,1,-22.9,-43.2\n"
                    "SH2,1,-22.9,-999\n"
                )
            },
            "shapes.txt",
            "shape_id",
            "SH1",
        ),
        (
            {
                "stops.txt": (
                    "stop_id,stop_lat,stop_lon,parent_station\n"
                    "ST1,-22.9,-43.2,TERM1\n"
                    "ST2,-91,-43.2,TERM1\n"
                )
            },
            "stops.txt",
            "stop_id",
            "ST1",
        ),
        (
            {
                "frequencies.txt": (
                    "trip_id,start_time,end_time,headway_secs\n"
                    "T1,05:00:00,23:00:00,600\n"
                    "T2,05:00:00,23:00:00,0\n"
                )
            },
            "frequencies.txt",
            "trip_id",
            "T1",
        ),
        (
            {
                "stop_times.txt": (
                    "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
                    "T1,ST1,1,05:00:00,05:01:00\n"
                    "T2,ST1,-1,05:00:00,05:01:00\n"
                )
            },
            "stop_times.txt",
            "trip_id",
            "T1",
        ),
    ],
)
def test_validate_types_drops_rows_out_of_range(overrides, filename, id_column, kept_id):
    gtfs_files = build_gtfs_dataframes(overrides)
    result = validate_gtfs_schedule_types(gtfs_files)
    assert result[filename][id_column].tolist() == [kept_id]


# --- validate_gtfs_schedule_types: time pattern (row dropped, not raised) ------


@pytest.mark.parametrize("bad_time", ["5am", "25:70:00", "5:0:0", "12:30"])
def test_validate_types_drops_rows_with_malformed_time(bad_time):
    overrides = {
        "frequencies.txt": (
            "trip_id,start_time,end_time,headway_secs\n"
            "T1,05:00:00,23:00:00,600\n"
            f"T2,{bad_time},23:00:00,600\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = validate_gtfs_schedule_types(gtfs_files)
    assert result["frequencies.txt"]["trip_id"].tolist() == ["T1"]


def test_validate_types_accepts_time_past_midnight():
    overrides = {
        "frequencies.txt": ("trip_id,start_time,end_time,headway_secs\nT1,24:30:00,26:00:00,600\n")
    }
    result = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))
    end_time = result["frequencies.txt"]["end_time"].iloc[0]
    assert end_time == pd.Timedelta(hours=26)
    assert end_time.total_seconds() == 26 * 3600


# --- validate_gtfs_schedule_types: dtype coercion failure (row dropped) --------


def test_validate_types_drops_rows_that_cannot_be_cast():
    overrides = {
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.9,-43.2\n"
            "SH2,1,abc,-43.2\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = validate_gtfs_schedule_types(gtfs_files)
    assert result["shapes.txt"]["shape_id"].tolist() == ["SH1"]
    assert result["shapes.txt"]["shape_pt_lat"].dtype == float


# --- validate_gtfs_schedule_types: strict string shape per dtype ---------------


def test_validate_types_drops_fractional_value_in_int_column_instead_of_truncating():
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,ST1,1,05:00:00,05:01:00\n"
            "T2,ST1,2.7,05:02:00,05:03:00\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = validate_gtfs_schedule_types(gtfs_files)
    assert result["stop_times.txt"]["trip_id"].tolist() == ["T1"]


def test_validate_types_drops_whole_number_float_string_in_int_column():
    # "3.0" is numerically a whole number, but it isn't the string shape an int
    # column requires (GTFS int fields are plain digits) — we don't rely on
    # pandas' numeric-parsing leniency to decide that's acceptable.
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,ST1,1,05:00:00,05:01:00\n"
            "T2,ST1,3.0,05:02:00,05:03:00\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = validate_gtfs_schedule_types(gtfs_files)
    assert result["stop_times.txt"]["trip_id"].tolist() == ["T1"]


@pytest.mark.parametrize("bad_value", ["1e2", "nan", "inf", "-inf", "+5", "5_000"])
def test_validate_types_rejects_values_pandas_would_leniently_parse(bad_value):
    overrides = {
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.9,-43.2\n"
            f"SH2,1,{bad_value},-43.2\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = validate_gtfs_schedule_types(gtfs_files)
    assert result["shapes.txt"]["shape_id"].tolist() == ["SH1"]


# --- validate_gtfs_schedule_types: real calendar-date validation ---------------


@pytest.mark.parametrize(
    "bad_date",
    [
        "20265599",  # month 55, day 99 — not a real month/day, just 8 digits
        "20261301",  # month 13
        "20260231",  # Feb never has a 31st
        "20260230",  # Feb 30th never exists
        "20260229",  # 2026 is not a leap year, so Feb 29th doesn't exist
        "00000101",  # year 0000
    ],
)
def test_validate_types_drops_calendar_nonsense_dates(bad_date):
    overrides = {
        "calendar.txt": (
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
            "start_date,end_date\n"
            "S1,1,1,1,1,1,0,0,20260101,20261231\n"
            f"S2,1,1,1,1,1,0,0,{bad_date},20261231\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = validate_gtfs_schedule_types(gtfs_files)
    assert result["calendar.txt"]["service_id"].tolist() == ["S1"]


def test_validate_types_accepts_leap_day_in_a_leap_year():
    overrides = {
        "calendar.txt": (
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
            "start_date,end_date\n"
            "S1,1,1,1,1,1,0,0,20280229,20281231\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = validate_gtfs_schedule_types(gtfs_files)
    assert result["calendar.txt"]["service_id"].tolist() == ["S1"]
    assert result["calendar.txt"]["start_date"].tolist() == [date(2028, 2, 29)]
    assert all(type(value) is date for value in result["calendar.txt"]["start_date"])


def test_validate_types_drops_calendar_dates_txt_nonsense_date():
    overrides = {
        "calendar_dates.txt": ("service_id,date,exception_type\nS1,20260101,1\nS2,20261332,1\n")
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = validate_gtfs_schedule_types(gtfs_files)
    assert result["calendar_dates.txt"]["service_id"].tolist() == ["S1"]


# --- validate_gtfs_schedule_types: logging of dropped rows ---------------------


def test_validate_types_logs_dropped_rows(caplog):
    overrides = {
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.9,-43.2\n"
            "SH2,1,999,-43.2\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)

    with caplog.at_level("WARNING", logger="pygtfsrealtime.schedule.validate"):
        validate_gtfs_schedule_types(gtfs_files)

    messages = [record.message for record in caplog.records]
    assert any("shapes.txt" in m and "shape_pt_lat" in m and "1/2" in m for m in messages)
    assert any("shapes.txt" in m and "dropped 1/2" in m for m in messages)


def test_validate_types_does_not_log_when_all_rows_valid(caplog):
    gtfs_files = build_gtfs_dataframes()

    with caplog.at_level("WARNING", logger="pygtfsrealtime.schedule.validate"):
        validate_gtfs_schedule_types(gtfs_files)

    assert caplog.records == []


# --- drop_incomplete_rows ---------------------------------------------------------


def test_drop_incomplete_rows_keeps_rows_missing_only_a_nullable_column():
    overrides = {
        "stops.txt": (
            "stop_id,stop_lat,stop_lon,parent_station\nST1,-22.9,-43.2,\nST2,-22.8,-43.1,TERM1\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = drop_incomplete_rows(gtfs_files)
    assert result["stops.txt"]["stop_id"].tolist() == ["ST1", "ST2"]


def test_drop_incomplete_rows_still_drops_rows_missing_a_required_column():
    overrides = {
        "stops.txt": (
            "stop_id,stop_lat,stop_lon,parent_station\nST1,-22.9,-43.2,TERM1\nST2,,-43.1,TERM1\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = drop_incomplete_rows(gtfs_files)
    assert result["stops.txt"]["stop_id"].tolist() == ["ST1"]


def test_drop_incomplete_rows_keeps_last_stop_time_missing_departure_time():
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,ST1,1,05:00:00,05:01:00\n"
            "T1,ST2,2,05:10:00,\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = drop_incomplete_rows(gtfs_files)
    assert result["stop_times.txt"]["stop_id"].tolist() == ["ST1", "ST2"]


def test_drop_incomplete_rows_falls_back_to_full_dropna_for_files_without_schema():
    gtfs_files = {
        "translations.txt": pd.DataFrame({"trans_id": ["T1", None], "translation": ["N1", "N2"]})
    }
    result = drop_incomplete_rows(gtfs_files)
    assert result["translations.txt"]["trans_id"].tolist() == ["T1"]


def test_drop_incomplete_rows_logs_dropped_count(caplog):
    overrides = {
        "stops.txt": (
            "stop_id,stop_lat,stop_lon,parent_station\nST1,-22.9,-43.2,TERM1\nST2,,-43.1,TERM1\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)

    with caplog.at_level("WARNING", logger="pygtfsrealtime.schedule.validate"):
        drop_incomplete_rows(gtfs_files)

    messages = [record.message for record in caplog.records]
    assert any("stops.txt" in m and "dropped 1/2" in m for m in messages)


# --- drop_duplicate_rows: exact duplicates --------------------------------------


def test_drop_duplicate_rows_keeps_first_occurrence_of_exact_duplicates():
    overrides = {"routes.txt": ("route_id,route_short_name\nR1,100\nR2,200\nR1,100\n")}
    gtfs_files = build_gtfs_dataframes(overrides)
    result = drop_duplicate_rows(gtfs_files)
    assert result["routes.txt"]["route_id"].tolist() == ["R1", "R2"]


def test_drop_duplicate_rows_keeps_rows_that_only_partially_match():
    # Same route_id, different route_short_name — not an exact duplicate, both
    # rows survive here (a duplicate route_id is a primary-key concern, not an
    # exact-duplicate concern).
    overrides = {"routes.txt": "route_id,route_short_name\nR1,100\nR1,200\n"}
    gtfs_files = build_gtfs_dataframes(overrides)
    result = drop_duplicate_rows(gtfs_files)
    assert result["routes.txt"]["route_short_name"].tolist() == ["100", "200"]


def test_drop_duplicate_rows_logs_dropped_count(caplog):
    overrides = {"routes.txt": "route_id,route_short_name\nR1,100\nR1,100\n"}
    gtfs_files = build_gtfs_dataframes(overrides)

    with caplog.at_level("WARNING", logger="pygtfsrealtime.schedule.validate"):
        drop_duplicate_rows(gtfs_files)

    messages = [record.message for record in caplog.records]
    assert any("routes.txt" in m and "dropped 1/2" in m for m in messages)


# --- drop_primary_key_violations -------------------------------------------------


def test_gtfs_data_fixture_primary_keys_are_all_defined():
    # agency.txt is deliberately excluded: we only require agency_timezone (not
    # agency_id, which the GTFS spec only requires when a feed has more than one
    # agency), so there's no key to enforce uniqueness on.
    assert set(VALID_GTFS_CSV.keys()) - {"agency.txt"} == set(GTFS_SCHEDULE_PRIMARY_KEYS.keys())


def test_drop_primary_key_violations_keeps_first_row_per_key():
    overrides = {
        "stops.txt": (
            "stop_id,stop_lat,stop_lon,parent_station\n"
            "ST1,-22.9,-43.2,TERM1\n"
            "ST1,-23.0,-43.3,TERM1\n"  # same stop_id, different coordinates
            "ST2,-22.9,-43.2,TERM1\n"
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = drop_primary_key_violations(gtfs_files, policy="keep_first")
    assert result["stops.txt"]["stop_id"].tolist() == ["ST1", "ST2"]
    assert result["stops.txt"]["stop_lat"].iloc[0] == "-22.9"


def test_drop_primary_key_violations_handles_composite_keys():
    # frequencies.txt's key is (trip_id, start_time): same trip_id is fine on its
    # own as long as start_time differs, but repeating the same pair isn't.
    overrides = {
        "frequencies.txt": (
            "trip_id,start_time,end_time,headway_secs\n"
            "T1,05:00:00,10:00:00,600\n"
            "T1,10:00:00,15:00:00,600\n"
            "T1,05:00:00,23:00:00,900\n"  # (T1, 05:00:00) repeats the first row's key
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = drop_primary_key_violations(gtfs_files, policy="keep_first")
    assert result["frequencies.txt"]["start_time"].tolist() == ["05:00:00", "10:00:00"]


def test_drop_primary_key_violations_ignores_files_without_a_defined_key():
    gtfs_files = {"agency.txt": pd.DataFrame({"agency_id": ["A1", "A1"]})}
    result = drop_primary_key_violations(gtfs_files, policy="keep_first")
    assert result["agency.txt"]["agency_id"].tolist() == ["A1", "A1"]


def test_drop_primary_key_violations_logs_dropped_count(caplog):
    overrides = {"routes.txt": "route_id,route_short_name\nR1,100\nR1,200\n"}
    gtfs_files = build_gtfs_dataframes(overrides)

    with caplog.at_level("WARNING", logger="pygtfsrealtime.schedule.validate"):
        drop_primary_key_violations(gtfs_files, policy="keep_first")

    messages = [record.message for record in caplog.records]
    assert any("routes.txt" in m and "dropped 1/2" in m for m in messages)


def test_drop_primary_key_violations_default_policy_is_keep_first():
    assert Settings().primary_key_duplicate_policy == "keep_first"


def test_drop_primary_key_violations_drop_all_removes_every_conflicting_row():
    overrides = {
        "stops.txt": (
            "stop_id,stop_lat,stop_lon,parent_station\n"
            "ST1,-22.9,-43.2,TERM1\n"
            "ST1,-23.0,-43.3,TERM1\n"  # conflicts with the row above
            "ST2,-22.9,-43.2,TERM1\n"  # no conflict, untouched
        )
    }
    gtfs_files = build_gtfs_dataframes(overrides)
    result = drop_primary_key_violations(gtfs_files, policy="drop_all")
    # Both ST1 rows are dropped — neither is more trustworthy than the other —
    # while the unrelated ST2 row survives untouched.
    assert result["stops.txt"]["stop_id"].tolist() == ["ST2"]


def test_drop_primary_key_violations_drop_all_logs_dropped_count(caplog):
    overrides = {"routes.txt": "route_id,route_short_name\nR1,100\nR1,200\n"}
    gtfs_files = build_gtfs_dataframes(overrides)

    with caplog.at_level("WARNING", logger="pygtfsrealtime.schedule.validate"):
        drop_primary_key_violations(gtfs_files, policy="drop_all")

    messages = [record.message for record in caplog.records]
    assert any("routes.txt" in m and "dropped 2/2" in m for m in messages)


# --- build_stops_geometry / build_shapes_geometry / build_gtfs_schedule_geometries -----


def test_build_stops_geometry_indexes_by_stop_id_with_expected_columns():
    stops = validate_gtfs_schedule_types(build_gtfs_dataframes())["stops.txt"]
    result = build_stops_geometry(stops, projection="EPSG:32723")

    assert result.index.name == "stop_id"
    assert list(result.columns) == ["stop_id", "parent_station", "geometry"]
    assert isinstance(result["geometry"].iloc[0], Point)
    assert result.loc["ST1", "parent_station"] == "TERM1"


def test_build_stops_geometry_reprojects_coordinates():
    stops = validate_gtfs_schedule_types(build_gtfs_dataframes())["stops.txt"]
    result = build_stops_geometry(stops, projection="EPSG:32723")

    point = result["geometry"].iloc[0]
    # EPSG:32723 (UTM zone 23S) coordinates are meters, not the original lat/lon
    # degrees — a reprojection that silently no-oped would leave x/y in [-180, 180].
    assert abs(point.x) > 1000
    assert abs(point.y) > 1000


def test_build_shapes_geometry_indexes_by_shape_id_with_linestring():
    overrides = {
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.90,-43.20\n"
            "SH1,2,-22.91,-43.21\n"
        )
    }
    shapes = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))["shapes.txt"]
    result = build_shapes_geometry(shapes, projection="EPSG:32723")

    assert result.index.name == "shape_id"
    assert list(result.columns) == ["shape_id", "geometry"]
    assert isinstance(result.loc["SH1", "geometry"], LineString)
    assert len(result.loc["SH1", "geometry"].coords) == 2


def test_build_shapes_geometry_drops_shapes_with_fewer_than_two_points(caplog):
    overrides = {
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.90,-43.20\n"
            "SH1,2,-22.91,-43.21\n"
            "SH2,1,-22.90,-43.20\n"
        )
    }
    shapes = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))["shapes.txt"]

    with caplog.at_level("WARNING", logger="pygtfsrealtime.schedule.compute"):
        result = build_shapes_geometry(shapes, projection="EPSG:32723")

    assert result.index.tolist() == ["SH1"]
    messages = [record.message for record in caplog.records]
    assert any("shapes.txt" in m and "SH2" in m for m in messages)


def test_build_gtfs_schedule_geometries_replaces_both_files():
    gtfs_files = validate_gtfs_schedule_types(build_gtfs_dataframes())
    result = build_gtfs_schedule_geometries(gtfs_files, projection="EPSG:32723")

    assert result["stops.txt"].index.name == "stop_id"
    assert result["shapes.txt"].index.name == "shape_id"
    # Untouched files pass through unchanged.
    assert result["routes.txt"] is gtfs_files["routes.txt"]


# --- build_stop_zones / build_shape_zones / build_terminal_zones / build_gtfs_schedule_zones


def _terminal_stops_gdf():
    # TERM1 is a real terminal (something else's parent_station) with 3
    # non-collinear children, so its convex hull is an actual Polygon rather than
    # degenerating into a Point/LineString.
    overrides = {
        "stops.txt": (
            "stop_id,stop_lat,stop_lon,parent_station\n"
            "TERM1,-22.900,-43.200,\n"
            "STA,-22.901,-43.200,TERM1\n"
            "STB,-22.900,-43.201,TERM1\n"
            "STC,-22.902,-43.202,TERM1\n"
        )
    }
    stops = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))["stops.txt"]
    return build_stops_geometry(stops, projection="EPSG:32723")


def test_build_stop_zones_buffers_when_mode_is_buffer():
    stops = _terminal_stops_gdf()
    result = build_stop_zones(stops, GeometryThreshold(distance=50, mode="buffer"))

    assert isinstance(result.loc["TERM1", "zone"], Polygon)
    assert result.loc["TERM1", "zone"].contains(result.loc["TERM1", "geometry"])


def test_build_stop_zones_keeps_raw_geometry_when_mode_is_distance():
    stops = _terminal_stops_gdf()
    result = build_stop_zones(stops, GeometryThreshold(distance=50, mode="distance"))

    assert result.loc["TERM1", "zone"] is result.loc["TERM1", "geometry"]


def _single_shape_gdf(*points):
    rows = "\n".join(f"SH1,{i + 1},{lat},{lon}" for i, (lat, lon) in enumerate(points))
    overrides = {
        "shapes.txt": ("shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n" + rows + "\n")
    }
    shapes = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))["shapes.txt"]
    return build_shapes_geometry(shapes, projection="EPSG:32723")


def test_build_shape_zones_path_zone_respects_mode():
    shapes = _single_shape_gdf((-22.90, -43.20), (-22.91, -43.21))

    buffered = build_shape_zones(
        shapes,
        path_threshold=GeometryThreshold(distance=30, mode="buffer"),
        endpoint_threshold=GeometryThreshold(distance=100, mode="buffer"),
    )
    assert isinstance(buffered.loc["SH1", "zone"], Polygon)

    raw = build_shape_zones(
        shapes,
        path_threshold=GeometryThreshold(distance=30, mode="distance"),
        endpoint_threshold=GeometryThreshold(distance=100, mode="distance"),
    )
    assert raw.loc["SH1", "zone"] is raw.loc["SH1", "geometry"]


def test_build_shape_zones_endpoint_zones_are_the_first_and_last_points():
    shapes = _single_shape_gdf((-22.90, -43.20), (-22.91, -43.21), (-22.92, -43.22))

    result = build_shape_zones(
        shapes,
        path_threshold=GeometryThreshold(distance=30, mode="distance"),
        endpoint_threshold=GeometryThreshold(distance=100, mode="distance"),
    )

    line = shapes.loc["SH1", "geometry"]
    assert result.loc["SH1", "start_zone"].equals(Point(line.coords[0]))
    assert result.loc["SH1", "end_zone"].equals(Point(line.coords[-1]))


def test_build_shape_zones_endpoint_zones_buffer_when_mode_is_buffer():
    shapes = _single_shape_gdf((-22.90, -43.20), (-22.91, -43.21))

    result = build_shape_zones(
        shapes,
        path_threshold=GeometryThreshold(distance=30, mode="distance"),
        endpoint_threshold=GeometryThreshold(distance=100, mode="buffer"),
    )

    assert isinstance(result.loc["SH1", "start_zone"], Polygon)
    assert isinstance(result.loc["SH1", "end_zone"], Polygon)


def test_build_terminal_zones_uses_convex_hull_of_children_by_default():
    stops = _terminal_stops_gdf()
    result = build_terminal_zones(stops, TerminalGeometryThreshold(distance=10, mode="distance"))

    assert result.index.tolist() == ["TERM1"]
    assert isinstance(result.loc["TERM1", "geometry"], Polygon)
    # mode="distance": zone is the raw hull, not a buffer around it.
    assert result.loc["TERM1", "zone"] is result.loc["TERM1", "geometry"]


def test_build_terminal_zones_uses_terminal_point_when_configured():
    stops = _terminal_stops_gdf()
    result = build_terminal_zones(
        stops,
        TerminalGeometryThreshold(distance=10, mode="distance", shape_mode="terminal_point"),
    )

    assert result.loc["TERM1", "geometry"].equals(stops.loc["TERM1", "geometry"])


def test_build_terminal_zones_buffers_when_mode_is_buffer():
    stops = _terminal_stops_gdf()
    result = build_terminal_zones(stops, TerminalGeometryThreshold(distance=10, mode="buffer"))

    assert isinstance(result.loc["TERM1", "zone"], Polygon)
    assert result.loc["TERM1", "zone"].contains(result.loc["TERM1", "geometry"])


def test_build_terminal_zones_skips_dangling_parent_station(caplog):
    # VALID_GTFS_CSV's only stop points at parent_station=TERM1, which has no
    # stops.txt row of its own.
    stops = validate_gtfs_schedule_types(build_gtfs_dataframes())["stops.txt"]
    stops = build_stops_geometry(stops, projection="EPSG:32723")

    with caplog.at_level("WARNING", logger="pygtfsrealtime.schedule.compute"):
        result = build_terminal_zones(stops, TerminalGeometryThreshold(distance=10, mode="buffer"))

    assert result.empty
    messages = [record.message for record in caplog.records]
    assert any("TERM1" in m for m in messages)


def test_build_gtfs_schedule_zones_adds_terminals_entry_and_zone_columns():
    gtfs_files = build_gtfs_schedule_geometries(
        validate_gtfs_schedule_types(build_gtfs_dataframes()), projection="EPSG:32723"
    )
    result = build_gtfs_schedule_zones(gtfs_files, Settings())

    assert "zone" in result["stops.txt"].columns
    assert {"zone", "start_zone", "end_zone"}.issubset(result["shapes.txt"].columns)
    assert "terminals" in result
    # Untouched files pass through unchanged.
    assert result["routes.txt"] is gtfs_files["routes.txt"]


# --- linearize_stop_times ---------------------------------------------------------


def test_linearize_stop_times_keeps_only_first_and_last_stop():
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
            "T1,STC,3,05:10:00,05:11:00\n"
        )
    }
    stop_times = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))["stop_times.txt"]
    result = linearize_stop_times(stop_times)

    assert result.index.name == "trip_id"
    assert result.index.tolist() == ["T1"]
    assert list(result.columns) == [
        "trip_id",
        "first_stop_id",
        "departure_time",
        "last_stop_id",
        "arrival_time",
    ]
    row = result.loc["T1"]
    assert row["first_stop_id"] == "STA"
    assert row["departure_time"] == pd.Timedelta(hours=5, minutes=1)
    assert row["last_stop_id"] == "STC"
    assert row["arrival_time"] == pd.Timedelta(hours=5, minutes=10)
    # STB (the intermediate stop) doesn't leak into either endpoint.
    assert "STB" not in (row["first_stop_id"], row["last_stop_id"])


def test_linearize_stop_times_single_stop_trip_uses_the_same_row_for_both_ends():
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
        )
    }
    stop_times = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))["stop_times.txt"]
    result = linearize_stop_times(stop_times)

    row = result.loc["T1"]
    assert row["first_stop_id"] == row["last_stop_id"] == "STA"
    assert row["departure_time"] == pd.Timedelta(hours=5, minutes=1)
    assert row["arrival_time"] == pd.Timedelta(hours=5)


def test_linearize_stop_times_is_positional_not_first_non_null():
    # The last stop's departure_time is legitimately missing. A naive
    # groupby(...).agg("first"/"last") skips NaN and would wrongly pull an
    # intermediate stop's time instead — arrival_time here must come from the
    # true last row (STC), even though its departure_time is empty.
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
            "T1,STC,3,05:10:00,\n"
        )
    }
    stop_times = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))["stop_times.txt"]
    result = linearize_stop_times(stop_times)

    row = result.loc["T1"]
    assert row["last_stop_id"] == "STC"
    assert row["arrival_time"] == pd.Timedelta(hours=5, minutes=10)
    assert row["departure_time"] == pd.Timedelta(hours=5, minutes=1)


def test_linearize_stop_times_handles_multiple_trips_independently():
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
            "T2,STX,1,06:00:00,06:01:00\n"
            "T2,STY,2,06:10:00,06:11:00\n"
        )
    }
    stop_times = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))["stop_times.txt"]
    result = linearize_stop_times(stop_times)

    assert sorted(result.index.tolist()) == ["T1", "T2"]
    assert result.loc["T1", "last_stop_id"] == "STB"
    assert result.loc["T2", "last_stop_id"] == "STY"


# --- drop_stop_times_missing_endpoints -------------------------------------------


def test_drop_stop_times_missing_endpoints_drops_trip_missing_arrival_time():
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
            "T2,STX,1,,06:00:00\n"
        )
    }
    stop_times = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))["stop_times.txt"]
    linearized = linearize_stop_times(stop_times)
    result = drop_stop_times_missing_endpoints(linearized)
    assert result.index.tolist() == ["T1"]


def test_drop_stop_times_missing_endpoints_drops_trip_missing_departure_time():
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
            "T2,STX,1,06:00:00,\n"
        )
    }
    stop_times = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))["stop_times.txt"]
    linearized = linearize_stop_times(stop_times)
    result = drop_stop_times_missing_endpoints(linearized)
    assert result.index.tolist() == ["T1"]


def test_drop_stop_times_missing_endpoints_keeps_trip_with_both_present():
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
        )
    }
    stop_times = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))["stop_times.txt"]
    linearized = linearize_stop_times(stop_times)
    result = drop_stop_times_missing_endpoints(linearized)
    assert result.index.tolist() == ["T1"]


def test_drop_stop_times_missing_endpoints_logs_dropped_count(caplog):
    overrides = {
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,STA,1,05:00:00,05:01:00\n"
            "T1,STB,2,05:05:00,05:06:00\n"
            "T2,STX,1,,06:00:00\n"
        )
    }
    stop_times = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))["stop_times.txt"]
    linearized = linearize_stop_times(stop_times)

    with caplog.at_level("WARNING", logger="pygtfsrealtime.schedule.compute"):
        drop_stop_times_missing_endpoints(linearized)

    messages = [record.message for record in caplog.records]
    assert any("stop_times.txt" in m and "dropped 1/2" in m for m in messages)


# --- reconcile_shape_coverage -----------------------------------------------------


def _typed_gtfs_files_with_geometries(overrides):
    gtfs_files = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))
    return build_gtfs_schedule_geometries(gtfs_files, projection="EPSG:32723")


def test_reconcile_shape_coverage_drops_trip_whose_shape_has_no_geometry():
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\nT2,R1,S1,0,SH2\n"
        ),
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.90,-43.20\nSH1,2,-22.91,-43.21\n"
        ),
    }
    gtfs_files = _typed_gtfs_files_with_geometries(overrides)
    result = reconcile_shape_coverage(gtfs_files)
    assert result["trips.txt"]["trip_id"].tolist() == ["T1"]


def test_reconcile_shape_coverage_drops_shape_no_trip_references():
    overrides = {
        "trips.txt": ("trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\n"),
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.90,-43.20\nSH1,2,-22.91,-43.21\n"
            "SH2,1,-23.00,-43.30\nSH2,2,-23.01,-43.31\n"
        ),
    }
    gtfs_files = _typed_gtfs_files_with_geometries(overrides)
    result = reconcile_shape_coverage(gtfs_files)
    assert result["shapes.txt"].index.tolist() == ["SH1"]


def test_reconcile_shape_coverage_keeps_matched_pair():
    overrides = {
        "trips.txt": ("trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\n"),
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.90,-43.20\nSH1,2,-22.91,-43.21\n"
        ),
    }
    gtfs_files = _typed_gtfs_files_with_geometries(overrides)
    result = reconcile_shape_coverage(gtfs_files)
    assert result["trips.txt"]["trip_id"].tolist() == ["T1"]
    assert result["shapes.txt"].index.tolist() == ["SH1"]


def test_reconcile_shape_coverage_logs_dropped_counts(caplog):
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\nT2,R1,S1,0,SH2\n"
        ),
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.90,-43.20\nSH1,2,-22.91,-43.21\n"
        ),
    }
    gtfs_files = _typed_gtfs_files_with_geometries(overrides)

    with caplog.at_level("WARNING", logger="pygtfsrealtime.schedule.compute"):
        reconcile_shape_coverage(gtfs_files)

    messages = [record.message for record in caplog.records]
    assert any("trips.txt" in m and "dropped 1/2" in m for m in messages)


def test_reconcile_shape_coverage_also_works_on_raw_pre_geometry_shapes():
    # Same function, called before build_shapes_geometry: shapes.txt is still one row
    # per point (RangeIndex), shape_id only a column — filtering by the "shape_id"
    # column (not the index) has to work in both shapes.

    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\nT2,R1,S1,0,SH2\n"
        ),
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.90,-43.20\nSH1,2,-22.91,-43.21\n"
            "SH3,1,-23.00,-43.30\nSH3,2,-23.01,-43.31\n"  # no trip references SH3
        ),
    }
    gtfs_files = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))
    result = reconcile_shape_coverage(gtfs_files)

    assert result["trips.txt"]["trip_id"].tolist() == ["T1"]
    # shapes.txt is still point-level here — every SH3 point row is dropped, SH2
    # never had a point row to begin with.
    assert result["shapes.txt"]["shape_id"].tolist() == ["SH1", "SH1"]


def test_reconcile_shape_coverage_second_pass_catches_shape_dropped_for_too_few_points():
    # SH2 has exactly 1 point: it survives the pre-geometry pass (it does have a
    # point row and a matching trip), but build_shapes_geometry itself drops it
    # (a LineString needs >= 2 points) — only a second reconcile pass, run after
    # geometry-building, catches the now-orphaned trip.
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\nT2,R1,S1,0,SH2\n"
        ),
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.90,-43.20\nSH1,2,-22.91,-43.21\n"
            "SH2,1,-23.00,-43.30\n"
        ),
    }
    gtfs_files = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))

    first_pass = reconcile_shape_coverage(gtfs_files)
    assert sorted(first_pass["trips.txt"]["trip_id"].tolist()) == ["T1", "T2"]

    geometries = build_gtfs_schedule_geometries(first_pass, projection="EPSG:32723")
    assert "SH2" not in geometries["shapes.txt"].index

    second_pass = reconcile_shape_coverage(geometries)
    assert second_pass["trips.txt"]["trip_id"].tolist() == ["T1"]


# --- reconcile_route_coverage -------------------------------------------------------


def test_reconcile_route_coverage_drops_trip_whose_route_has_no_routes_row():
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\n"
            "T1,R1,S1,0,SH1\nT2,R2,S1,0,SH1\n"  # R2 has no routes.txt row below
        ),
        "routes.txt": "route_id,route_short_name\nR1,100\n",
    }
    gtfs_files = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))
    result = reconcile_route_coverage(gtfs_files)

    assert result["trips.txt"]["trip_id"].tolist() == ["T1"]
    assert result["routes.txt"]["route_id"].tolist() == ["R1"]


def test_reconcile_route_coverage_drops_route_no_trip_references():
    overrides = {
        "trips.txt": ("trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\n"),
        "routes.txt": (
            "route_id,route_short_name\nR1,100\nR2,200\n"  # no trip references R2
        ),
    }
    gtfs_files = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))
    result = reconcile_route_coverage(gtfs_files)

    assert result["trips.txt"]["trip_id"].tolist() == ["T1"]
    assert result["routes.txt"]["route_id"].tolist() == ["R1"]


def test_reconcile_route_coverage_keeps_matched_pair():
    gtfs_files = validate_gtfs_schedule_types(build_gtfs_dataframes())
    result = reconcile_route_coverage(gtfs_files)

    assert result["trips.txt"]["trip_id"].tolist() == ["T1"]
    assert result["routes.txt"]["route_id"].tolist() == ["R1"]


def test_reconcile_route_coverage_logs_dropped_counts(caplog):
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\nT2,R2,S1,0,SH1\n"
        ),
        "routes.txt": "route_id,route_short_name\nR1,100\n",
    }
    gtfs_files = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))

    with caplog.at_level("WARNING", logger="pygtfsrealtime.schedule.compute"):
        reconcile_route_coverage(gtfs_files)

    messages = [record.message for record in caplog.records]
    assert any("trips.txt" in m and "dropped 1/2" in m for m in messages)


# --- reconcile_trip_coverage ------------------------------------------------------


def _typed_gtfs_files_with_linearized_stop_times(overrides):
    gtfs_files = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))
    gtfs_files["stop_times.txt"] = drop_stop_times_missing_endpoints(
        linearize_stop_times(gtfs_files["stop_times.txt"])
    )
    return gtfs_files


def test_reconcile_trip_coverage_drops_trip_present_only_in_trips():
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\nT2,R1,S1,0,SH1\n"
        ),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,ST1,1,05:00:00,05:01:00\nT1,ST2,2,05:10:00,05:11:00\n"
        ),
    }
    gtfs_files = _typed_gtfs_files_with_linearized_stop_times(overrides)
    result = reconcile_trip_coverage(gtfs_files)
    assert result["trips.txt"]["trip_id"].tolist() == ["T1"]


def test_reconcile_trip_coverage_drops_trip_present_only_in_stop_times():
    overrides = {
        "trips.txt": ("trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\n"),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,ST1,1,05:00:00,05:01:00\nT1,ST2,2,05:10:00,05:11:00\n"
            "T2,ST3,1,06:00:00,06:01:00\nT2,ST4,2,06:10:00,06:11:00\n"
        ),
    }
    gtfs_files = _typed_gtfs_files_with_linearized_stop_times(overrides)
    result = reconcile_trip_coverage(gtfs_files)
    assert result["stop_times.txt"].index.tolist() == ["T1"]


def test_reconcile_trip_coverage_keeps_trip_with_no_frequencies_at_all():
    # A trip is still valid without frequency-based scheduling — only trips.txt +
    # stop_times.txt coverage is required.
    overrides = {
        "trips.txt": ("trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\n"),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,ST1,1,05:00:00,05:01:00\nT1,ST2,2,05:10:00,05:11:00\n"
        ),
        "frequencies.txt": "trip_id,start_time,end_time,headway_secs\n",
    }
    gtfs_files = _typed_gtfs_files_with_linearized_stop_times(overrides)
    result = reconcile_trip_coverage(gtfs_files)
    assert result["trips.txt"]["trip_id"].tolist() == ["T1"]
    assert result["stop_times.txt"].index.tolist() == ["T1"]
    assert result["frequencies.txt"].empty


def test_reconcile_trip_coverage_drops_frequencies_row_for_a_trip_missing_elsewhere():
    overrides = {
        "trips.txt": ("trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\n"),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,ST1,1,05:00:00,05:01:00\nT1,ST2,2,05:10:00,05:11:00\n"
        ),
        "frequencies.txt": (
            "trip_id,start_time,end_time,headway_secs\n"
            "T1,05:00:00,23:00:00,600\n"
            "T2,05:00:00,23:00:00,600\n"  # T2 has no trips.txt/stop_times.txt row
        ),
    }
    gtfs_files = _typed_gtfs_files_with_linearized_stop_times(overrides)
    result = reconcile_trip_coverage(gtfs_files)
    assert result["frequencies.txt"]["trip_id"].tolist() == ["T1"]


def test_reconcile_trip_coverage_logs_dropped_counts(caplog):
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\nT2,R1,S1,0,SH1\n"
        ),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T1,ST1,1,05:00:00,05:01:00\nT1,ST2,2,05:10:00,05:11:00\n"
        ),
    }
    gtfs_files = _typed_gtfs_files_with_linearized_stop_times(overrides)

    with caplog.at_level("WARNING", logger="pygtfsrealtime.schedule.compute"):
        reconcile_trip_coverage(gtfs_files)

    messages = [record.message for record in caplog.records]
    assert any("trips.txt" in m and "dropped 1/2" in m for m in messages)


# --- reconcile_gtfs_schedule_relations ---------------------------------------------


def test_reconcile_gtfs_schedule_relations_cascades_orphaned_shape_after_trip_drop():
    # T1 has no stop_times rows at all, so it survives reconcile_shape_coverage (SH1
    # is still a matched pair) and only gets dropped from trips.txt by
    # reconcile_trip_coverage - which then leaves SH1 an orphan in shapes.txt that
    # nothing references anymore. A single pass of each function, in either order,
    # can't catch that: only iterating until nothing changes does.
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\nT2,R1,S1,0,SH2\n"
        ),
        "shapes.txt": (
            "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
            "SH1,1,-22.90,-43.20\nSH1,2,-22.91,-43.21\n"
            "SH2,1,-23.00,-43.30\nSH2,2,-23.01,-43.31\n"
        ),
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T2,STC,1,06:00:00,06:01:00\nT2,STD,2,06:10:00,06:11:00\n"
        ),
    }
    gtfs_files = _typed_gtfs_files_with_linearized_stop_times(overrides)

    # What a single pass of each, in this order, would have missed:
    single_pass = reconcile_trip_coverage(reconcile_shape_coverage(gtfs_files))
    assert "SH1" in single_pass["shapes.txt"]["shape_id"].tolist()

    result = reconcile_gtfs_schedule_relations(gtfs_files)
    assert result["trips.txt"]["trip_id"].tolist() == ["T2"]
    assert result["shapes.txt"]["shape_id"].tolist() == ["SH2", "SH2"]
    assert result["stop_times.txt"].index.tolist() == ["T2"]


def test_reconcile_gtfs_schedule_relations_cascades_orphaned_route_after_trip_drop():
    # Same shape of problem as the shape-orphan case above, but for routes.txt: T1
    # has no stop_times rows, survives reconcile_route_coverage (R1 is still a
    # matched pair), and only gets dropped by reconcile_trip_coverage - which then
    # leaves R1 an orphan in routes.txt.
    overrides = {
        "trips.txt": (
            "trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\nT2,R2,S1,0,SH1\n"
        ),
        "routes.txt": "route_id,route_short_name\nR1,100\nR2,200\n",
        "stop_times.txt": (
            "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
            "T2,STC,1,06:00:00,06:01:00\nT2,STD,2,06:10:00,06:11:00\n"
        ),
    }
    gtfs_files = _typed_gtfs_files_with_linearized_stop_times(overrides)

    single_pass = reconcile_trip_coverage(reconcile_route_coverage(gtfs_files))
    assert "R1" in single_pass["routes.txt"]["route_id"].tolist()

    result = reconcile_gtfs_schedule_relations(gtfs_files)
    assert result["trips.txt"]["trip_id"].tolist() == ["T2"]
    assert result["routes.txt"]["route_id"].tolist() == ["R2"]
    assert result["stop_times.txt"].index.tolist() == ["T2"]


def test_reconcile_gtfs_schedule_relations_keeps_mutually_consistent_data():
    gtfs_files = _typed_gtfs_files_with_linearized_stop_times({})
    result = reconcile_gtfs_schedule_relations(gtfs_files)
    assert result["trips.txt"]["trip_id"].tolist() == ["T1"]
    assert result["stop_times.txt"].index.tolist() == ["T1"]
    assert result["routes.txt"]["route_id"].tolist() == ["R1"]


# --- build_trip_endpoints ----------------------------------------------------------

# TERM1 is a real terminal (STA's parent_station); STZ has no parent at all.
_TRIP_ENDPOINT_STOPS_CSV = (
    "stop_id,stop_lat,stop_lon,parent_station\n"
    "TERM1,-22.900,-43.200,\n"
    "STA,-22.901,-43.200,TERM1\n"
    "STB,-22.900,-43.201,TERM1\n"
    "STC,-22.902,-43.202,TERM1\n"
    "STZ,-22.950,-43.250,\n"
)


def _gtfs_files_with_zones(overrides, settings=None):
    settings = settings or Settings(projection="EPSG:32723")
    gtfs_files = validate_gtfs_schedule_types(build_gtfs_dataframes(overrides))
    gtfs_files = build_gtfs_schedule_geometries(gtfs_files, projection=settings.projection)
    gtfs_files["stop_times.txt"] = drop_stop_times_missing_endpoints(
        linearize_stop_times(gtfs_files["stop_times.txt"])
    )
    gtfs_files = reconcile_gtfs_schedule_relations(gtfs_files)
    return build_gtfs_schedule_zones(gtfs_files, settings)


def _trip_zones(result):
    trips = result["trips.txt"].set_index("trip_id")
    return trips.loc["T1", "start_zone"], trips.loc["T1", "end_zone"]


def test_build_trip_endpoints_default_source_prefers_the_stop_own_zone():
    # STA has a parent (TERM1), but the default source ("stop") should still pick
    # STA's own zone over the terminal's - "stop" only falls through to "terminal"
    # when the stop itself can't resolve.
    gtfs_files = _gtfs_files_with_zones(
        {
            "stops.txt": _TRIP_ENDPOINT_STOPS_CSV,
            "stop_times.txt": (
                "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
                "T1,STA,1,05:00:00,05:01:00\n"
                "T1,STB,2,05:05:00,05:06:00\n"
            ),
        }
    )
    result = build_trip_endpoints(gtfs_files, Settings())

    start_zone, end_zone = _trip_zones(result)
    assert start_zone is gtfs_files["stops.txt"].loc["STA", "zone"]
    assert end_zone is gtfs_files["stops.txt"].loc["STB", "zone"]


def test_build_trip_endpoints_stop_source_falls_back_to_terminal_then_shape():
    # STMISSING is never declared in stops.txt - "stop" can't resolve it (no row
    # to look up a zone or a parent_station on), so both the "stop" and "terminal"
    # candidates fail and resolution falls all the way to the shape's endpoints.
    gtfs_files = _gtfs_files_with_zones(
        {
            "stops.txt": _TRIP_ENDPOINT_STOPS_CSV,
            "stop_times.txt": (
                "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
                "T1,STMISSING,1,05:00:00,05:01:00\n"
                "T1,STMISSING,2,05:10:00,05:11:00\n"
            ),
        }
    )
    result = build_trip_endpoints(gtfs_files, Settings(trip_endpoint_source="stop"))

    start_zone, end_zone = _trip_zones(result)
    shape_row = gtfs_files["shapes.txt"].loc["SH1"]
    assert start_zone is shape_row["start_zone"]
    assert end_zone is shape_row["end_zone"]


def test_build_trip_endpoints_terminal_source_uses_parent_zone():
    gtfs_files = _gtfs_files_with_zones(
        {
            "stops.txt": _TRIP_ENDPOINT_STOPS_CSV,
            "stop_times.txt": (
                "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
                "T1,STA,1,05:00:00,05:01:00\n"
                "T1,STB,2,05:05:00,05:06:00\n"
            ),
        }
    )
    result = build_trip_endpoints(gtfs_files, Settings(trip_endpoint_source="terminal"))

    start_zone, end_zone = _trip_zones(result)
    terminal_zone = gtfs_files["terminals"].loc["TERM1", "zone"]
    assert start_zone is terminal_zone
    assert end_zone is terminal_zone


def test_build_trip_endpoints_terminal_source_falls_back_to_stop_without_parent():
    gtfs_files = _gtfs_files_with_zones(
        {
            "stops.txt": _TRIP_ENDPOINT_STOPS_CSV,
            "stop_times.txt": (
                "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
                "T1,STZ,1,05:00:00,05:01:00\n"
                "T1,STZ,2,05:10:00,05:11:00\n"
            ),
        }
    )
    result = build_trip_endpoints(gtfs_files, Settings(trip_endpoint_source="terminal"))

    start_zone, end_zone = _trip_zones(result)
    own_zone = gtfs_files["stops.txt"].loc["STZ", "zone"]
    assert start_zone is own_zone
    assert end_zone is own_zone


def test_build_trip_endpoints_shape_source_ignores_the_stop_entirely():
    gtfs_files = _gtfs_files_with_zones(
        {
            "stops.txt": _TRIP_ENDPOINT_STOPS_CSV,
            "stop_times.txt": (
                "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
                "T1,STA,1,05:00:00,05:01:00\n"
                "T1,STB,2,05:05:00,05:06:00\n"
            ),
        }
    )
    result = build_trip_endpoints(gtfs_files, Settings(trip_endpoint_source="shape"))

    start_zone, end_zone = _trip_zones(result)
    shape_row = gtfs_files["shapes.txt"].loc["SH1"]
    assert start_zone is shape_row["start_zone"]
    assert end_zone is shape_row["end_zone"]


def test_build_trip_endpoints_selected_zone_respects_buffer_vs_distance_mode():
    # No new geometry is computed by build_trip_endpoints - it only selects among
    # already-precomputed zone columns, so the buffer-vs-distance choice from
    # Settings.stop_geometry has to already be baked into what gets selected.
    settings = Settings(
        projection="EPSG:32723", stop_geometry=GeometryThreshold(distance=50, mode="distance")
    )
    gtfs_files = _gtfs_files_with_zones(
        {
            "stops.txt": _TRIP_ENDPOINT_STOPS_CSV,
            "stop_times.txt": (
                "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
                "T1,STA,1,05:00:00,05:01:00\n"
                "T1,STB,2,05:05:00,05:06:00\n"
            ),
        },
        settings=settings,
    )
    result = build_trip_endpoints(gtfs_files, settings)

    start_zone, _ = _trip_zones(result)
    assert start_zone is gtfs_files["stops.txt"].loc["STA", "geometry"]


def test_build_trip_endpoints_only_adds_columns_to_trips_txt():
    gtfs_files = _gtfs_files_with_zones({"stops.txt": _TRIP_ENDPOINT_STOPS_CSV})
    result = build_trip_endpoints(gtfs_files, Settings())

    assert {"start_zone", "end_zone"}.issubset(result["trips.txt"].columns)
    # Untouched files pass through unchanged.
    assert result["routes.txt"] is gtfs_files["routes.txt"]
