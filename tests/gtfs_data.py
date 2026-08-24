"""Shared valid GTFS schedule fixture data for ingest/validation tests."""

import io
import zipfile

import pandas as pd

VALID_GTFS_CSV: dict[str, str] = {
    "agency.txt": "agency_timezone\nUTC\n",
    "calendar.txt": (
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
        "start_date,end_date\nS1,1,1,1,1,1,0,0,20260101,20261231\n"
    ),
    "calendar_dates.txt": "service_id,date,exception_type\nS1,20260101,1\n",
    "trips.txt": "trip_id,route_id,service_id,direction_id,shape_id\nT1,R1,S1,0,SH1\n",
    "routes.txt": "route_id,route_short_name\nR1,100\n",
    "frequencies.txt": ("trip_id,start_time,end_time,headway_secs\nT1,05:00:00,23:00:00,600\n"),
    "stop_times.txt": (
        "trip_id,stop_id,stop_sequence,arrival_time,departure_time\nT1,ST1,1,05:00:00,05:01:00\n"
    ),
    "shapes.txt": (
        "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
        "SH1,1,-22.9,-43.2\nSH1,2,-22.91,-43.21\n"
    ),
    "stops.txt": "stop_id,stop_lat,stop_lon,parent_station\nST1,-22.9,-43.2,TERM1\n",
}


def build_gtfs_zip(overrides: dict[str, str] | None = None) -> bytes:
    files = dict(VALID_GTFS_CSV)
    if overrides:
        files.update(overrides)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def build_gtfs_dataframes(
    overrides: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    files = dict(VALID_GTFS_CSV)
    if overrides:
        files.update(overrides)
    return {name: pd.read_csv(io.StringIO(content), dtype=str) for name, content in files.items()}
