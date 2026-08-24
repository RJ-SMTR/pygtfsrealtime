GTFS_SCHEDULE_REQUIRED_FILES = {
    "agency.txt",
    "calendar.txt",
    "calendar_dates.txt",
    "trips.txt",
    "routes.txt",
    "stop_times.txt",
    "shapes.txt",
    "stops.txt",
}

# Present in GTFS_SCHEDULE_REQUIRED_COLUMNS/GTFS_SCHEDULE_PRIMARY_KEYS (so it's
# still parsed/validated/deduplicated like any other file when it IS present),
# but not in GTFS_SCHEDULE_REQUIRED_FILES - a feed with no frequency-based
# trips is free to omit frequencies.txt entirely, per the GTFS spec. See
# GTFSScheduleIngester._fill_missing_optional_files, which synthesizes an
# empty one so every downstream step that reads gtfs_files["frequencies.txt"]
# unconditionally doesn't need its own None-check.
GTFS_SCHEDULE_OPTIONAL_FILES = {"frequencies.txt"}

# Columns read from each file by pygtfsrealtime.schedule.trip_instances and
# pygtfsrealtime.schedule.terminals. Keep in sync with those modules.
GTFS_SCHEDULE_REQUIRED_COLUMNS: dict[str, set[str]] = {
    # agency_id is only conditionally required by the GTFS spec (when a feed
    # has more than one agency) - we don't need it (nothing in this library
    # joins a trip to a specific agency), only agency_timezone, to resolve the
    # feed's timezone (see pygtfsrealtime.schedule.snapshot.resolve_gtfs_timezone).
    "agency.txt": {"agency_timezone"},
    "calendar.txt": {
        "service_id",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "start_date",
        "end_date",
    },
    "calendar_dates.txt": {"service_id", "date", "exception_type"},
    "trips.txt": {"trip_id", "route_id", "service_id", "direction_id", "shape_id"},
    "routes.txt": {"route_id", "route_short_name"},
    "frequencies.txt": {"trip_id", "start_time", "end_time", "headway_secs"},
    "stop_times.txt": {
        "trip_id",
        "stop_id",
        "stop_sequence",
        "arrival_time",
        "departure_time",
    },
    "shapes.txt": {"shape_id", "shape_pt_sequence", "shape_pt_lat", "shape_pt_lon"},
    "stops.txt": {"stop_id", "stop_lat", "stop_lon", "parent_station"},
}

# Primary key per the GTFS Schedule reference — the column(s) that must be
# unique within each file. Two rows sharing one of these keys can't both be
# right, so only the first occurrence is kept.
GTFS_SCHEDULE_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "calendar.txt": ("service_id",),
    "calendar_dates.txt": ("service_id", "date"),
    "trips.txt": ("trip_id",),
    "routes.txt": ("route_id",),
    "frequencies.txt": ("trip_id", "start_time"),
    "stop_times.txt": ("trip_id", "stop_sequence"),
    "shapes.txt": ("shape_id", "shape_pt_sequence"),
    "stops.txt": ("stop_id",),
}
