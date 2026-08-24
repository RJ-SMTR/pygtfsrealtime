import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from pygtfsrealtime.schedule.gtfs_spec import (
    GTFS_SCHEDULE_PRIMARY_KEYS,
    GTFS_SCHEDULE_REQUIRED_COLUMNS,
    GTFS_SCHEDULE_REQUIRED_FILES,
)
from pygtfsrealtime.settings import PrimaryKeyDuplicatePolicy

logger = logging.getLogger(__name__)

_TIME_PATTERN = r"^\d{1,3}:[0-5]\d:[0-5]\d$"

# Strict string shape each numeric dtype must have before we even attempt to parse
# it — we already know the target dtype from the schema, so we decide up front
# what counts as "looks like an int"/"looks like a float" instead of leaning on
# pandas' more permissive numeric-parsing heuristics (which accept things like
# scientific notation, or the literal strings "nan"/"inf" as valid numbers).
_INT_STRING_PATTERN = r"^-?\d+$"
_FLOAT_STRING_PATTERN = r"^-?\d+(\.\d+)?$"


@dataclass(frozen=True)
class GTFSColumnSchema:
    """Validation/casting rules for one required column of one GTFS file.

    Attributes:
        dtype: the raw dtype (str/int/float) a value must parse as.
        valid_values: if set, the only dtype values considered valid.
        min_value: if set, the minimum valid value (inclusive).
        max_value: if set, the maximum valid value (inclusive).
        pattern: if set, a regex the raw string value must match.
        is_calendar_date: whether the (int-encoded YYYYMMDD) value must be a
            real calendar date.
        is_timezone: whether the (string) value must be a real IANA
            timezone key that zoneinfo.ZoneInfo can load.
        nullable: whether a missing value in this column is acceptable (e.g.
            stops.txt's parent_station) rather than making the row
            incomplete.
        to_semantic_type: optional conversion applied only after a row has
            passed every check above, turning the validated raw dtype into
            the semantic type the rest of the program actually works with
            (bool, date, timedelta, ...).
    """

    dtype: type
    valid_values: frozenset[int] | None = None
    min_value: float | None = None
    max_value: float | None = None
    pattern: str | None = None
    is_calendar_date: bool = False
    # Whether the (string) value must be a real IANA timezone key that
    # zoneinfo.ZoneInfo can load - same idea as is_calendar_date, but for
    # agency.txt's agency_timezone.
    is_timezone: bool = False
    # Per the GTFS Schedule reference, most required columns still can't be empty
    # (e.g. stop_lat) — but a few (e.g. stops.txt's parent_station) are legitimately
    # blank for most rows (any stop that isn't part of a larger station). Rows missing
    # a nullable column's value shouldn't be dropped by drop_incomplete_rows just for
    # that.
    nullable: bool = False
    # Applied only after a row has already passed every check above, to turn the
    # validated dtype (int/float/str — the shape we validate against) into the
    # semantic type the rest of the program should actually work with (bool,
    # date, timedelta, ...). Validation and final representation are deliberately
    # separate: what's safe to *check* isn't always what's useful to *use*.
    to_semantic_type: Callable[[pd.Series], pd.Series] | None = None


def _to_bool(series: pd.Series) -> pd.Series:
    """0/1 (already validated against valid_values={0, 1}) -> bool."""
    return series.astype(bool)


def _to_calendar_date(series: pd.Series) -> pd.Series:
    """Int YYYYMMDD (already validated as a real date) -> datetime.date.

    Deliberately .dt.date, not the Timestamp pd.to_datetime returns by default —
    a GTFS service date has no time-of-day component, and we don't want callers
    treating it like one.
    """
    return pd.to_datetime(series.astype(str), format="%Y%m%d").dt.date


def _hms_string_to_timedelta(series: pd.Series) -> pd.Series:
    """ "HH:MM:SS" (already validated, H can exceed 24 for past-midnight trips) ->
    Timedelta. A datetime.time can't represent "25:10:00" — it isn't a time of
    day, it's an offset from the start of the service day — so Timedelta is the
    type that actually preserves that meaning.
    """
    return pd.to_timedelta(series)


def _seconds_to_timedelta(series: pd.Series) -> pd.Series:
    """Plain seconds (already validated as a positive int) -> Timedelta, so it's
    the same duration type as the *_time columns it's used alongside."""
    return pd.to_timedelta(series, unit="s")


def _is_iana_timezone(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


def _to_zoneinfo(series: pd.Series) -> pd.Series:
    """IANA timezone key (already validated) -> zoneinfo.ZoneInfo."""
    # pandas-stubs' Series.apply overloads don't include ZoneInfo in their
    # known scalar-return union.
    return series.apply(ZoneInfo)  # type: ignore[arg-type]


# Forced dtype and valid interval/set for each required column. Keep in sync with
# GTFS_SCHEDULE_REQUIRED_COLUMNS (pygtfsrealtime.schedule.gtfs_spec) and the GTFS
# Schedule reference.
GTFS_SCHEDULE_COLUMN_SCHEMAS: dict[str, dict[str, GTFSColumnSchema]] = {
    "agency.txt": {
        "agency_timezone": GTFSColumnSchema(
            dtype=str, is_timezone=True, to_semantic_type=_to_zoneinfo
        ),
    },
    "calendar.txt": {
        "service_id": GTFSColumnSchema(dtype=str),
        "monday": GTFSColumnSchema(
            dtype=int, valid_values=frozenset({0, 1}), to_semantic_type=_to_bool
        ),
        "tuesday": GTFSColumnSchema(
            dtype=int, valid_values=frozenset({0, 1}), to_semantic_type=_to_bool
        ),
        "wednesday": GTFSColumnSchema(
            dtype=int, valid_values=frozenset({0, 1}), to_semantic_type=_to_bool
        ),
        "thursday": GTFSColumnSchema(
            dtype=int, valid_values=frozenset({0, 1}), to_semantic_type=_to_bool
        ),
        "friday": GTFSColumnSchema(
            dtype=int, valid_values=frozenset({0, 1}), to_semantic_type=_to_bool
        ),
        "saturday": GTFSColumnSchema(
            dtype=int, valid_values=frozenset({0, 1}), to_semantic_type=_to_bool
        ),
        "sunday": GTFSColumnSchema(
            dtype=int, valid_values=frozenset({0, 1}), to_semantic_type=_to_bool
        ),
        "start_date": GTFSColumnSchema(
            dtype=int, is_calendar_date=True, to_semantic_type=_to_calendar_date
        ),
        "end_date": GTFSColumnSchema(
            dtype=int, is_calendar_date=True, to_semantic_type=_to_calendar_date
        ),
    },
    "calendar_dates.txt": {
        "service_id": GTFSColumnSchema(dtype=str),
        "date": GTFSColumnSchema(
            dtype=int, is_calendar_date=True, to_semantic_type=_to_calendar_date
        ),
        "exception_type": GTFSColumnSchema(dtype=int, valid_values=frozenset({1, 2})),
    },
    "trips.txt": {
        "trip_id": GTFSColumnSchema(dtype=str),
        "route_id": GTFSColumnSchema(dtype=str),
        "service_id": GTFSColumnSchema(dtype=str),
        "direction_id": GTFSColumnSchema(dtype=int, valid_values=frozenset({0, 1})),
        "shape_id": GTFSColumnSchema(dtype=str),
    },
    "routes.txt": {
        "route_id": GTFSColumnSchema(dtype=str),
        "route_short_name": GTFSColumnSchema(dtype=str),
    },
    "frequencies.txt": {
        "trip_id": GTFSColumnSchema(dtype=str),
        "start_time": GTFSColumnSchema(
            dtype=str, pattern=_TIME_PATTERN, to_semantic_type=_hms_string_to_timedelta
        ),
        "end_time": GTFSColumnSchema(
            dtype=str, pattern=_TIME_PATTERN, to_semantic_type=_hms_string_to_timedelta
        ),
        "headway_secs": GTFSColumnSchema(
            dtype=int, min_value=1, to_semantic_type=_seconds_to_timedelta
        ),
    },
    "stop_times.txt": {
        "trip_id": GTFSColumnSchema(dtype=str),
        "stop_id": GTFSColumnSchema(dtype=str),
        "stop_sequence": GTFSColumnSchema(dtype=int, min_value=0),
        # Only required for a trip's first and last stop_time — blank on
        # intermediate stops is normal (their times are meant to be interpolated),
        # so a missing value here shouldn't drop the whole row.
        "arrival_time": GTFSColumnSchema(
            dtype=str,
            pattern=_TIME_PATTERN,
            nullable=True,
            to_semantic_type=_hms_string_to_timedelta,
        ),
        "departure_time": GTFSColumnSchema(
            dtype=str,
            pattern=_TIME_PATTERN,
            nullable=True,
            to_semantic_type=_hms_string_to_timedelta,
        ),
    },
    "shapes.txt": {
        "shape_id": GTFSColumnSchema(dtype=str),
        "shape_pt_sequence": GTFSColumnSchema(dtype=int, min_value=0),
        "shape_pt_lat": GTFSColumnSchema(dtype=float, min_value=-90, max_value=90),
        "shape_pt_lon": GTFSColumnSchema(dtype=float, min_value=-180, max_value=180),
    },
    "stops.txt": {
        "stop_id": GTFSColumnSchema(dtype=str),
        "stop_lat": GTFSColumnSchema(dtype=float, min_value=-90, max_value=90),
        "stop_lon": GTFSColumnSchema(dtype=float, min_value=-180, max_value=180),
        "parent_station": GTFSColumnSchema(dtype=str, nullable=True),
    },
}


def drop_incomplete_rows(gtfs_files: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Drop rows missing a value in a column that's actually required to be non-empty.

    A plain df.dropna() would also drop rows just because a nullable column (e.g.
    stops.txt's parent_station) is empty — but an empty parent_station is the normal
    case for any stop that isn't part of a larger station, not incomplete data.

    Args:
        gtfs_files: the parsed GTFS files.

    Returns:
        The same files, with incomplete rows dropped.
    """
    result: dict[str, pd.DataFrame] = {}
    for filename, df in gtfs_files.items():
        schemas = GTFS_SCHEDULE_COLUMN_SCHEMAS.get(filename)
        required_columns = (
            [
                column
                for column, schema in schemas.items()
                if not schema.nullable and column in df.columns
            ]
            if schemas is not None
            else None
        )

        complete = df.dropna(subset=required_columns)
        dropped = len(df) - len(complete)
        if dropped:
            logger.warning(
                "%s: dropped %d/%d row(s) with a missing required value",
                filename,
                dropped,
                len(df),
            )
        result[filename] = complete

    return result


def _soft_cast_column(series: pd.Series, schema: GTFSColumnSchema) -> pd.Series:
    """Cast to schema.dtype without raising: values that don't fit become NaN.

    We already know the required dtype from the schema, so we check the raw
    string against that dtype's exact expected shape first and only parse the
    strings that pass — rather than handing the raw string to pandas' numeric
    parser and hoping its notion of "numeric" (scientific notation, "nan"/"inf"
    literals, "2.7" where an int is required, ...) matches what we actually want.
    """
    if schema.dtype is str:
        return series

    pattern = _INT_STRING_PATTERN if schema.dtype is int else _FLOAT_STRING_PATTERN
    well_formed = series.str.match(pattern, na=False)
    return pd.to_numeric(series.where(well_formed), errors="coerce")


def _is_valid_calendar_date(coerced: pd.Series) -> pd.Series:
    """Whether an int-encoded YYYYMMDD value is a real calendar date.

    A pure range check (e.g. 10000101 <= x <= 99991231) lets nonsense like
    20265599 ("month 55, day 99") through — it has the right number of digits
    but isn't a date. Routing it through pd.to_datetime with a strict format
    catches invalid months/days and non-leap Feb 29th, not just the digit count.
    """
    as_int = coerced.astype("Int64")
    parsed = pd.to_datetime(as_int.astype(str), format="%Y%m%d", errors="coerce")
    return parsed.notna()


def _column_validity_mask(coerced: pd.Series, schema: GTFSColumnSchema) -> pd.Series:
    """Boolean mask of which values in a coerced column satisfy its schema."""
    present = coerced.notna()
    valid = present.copy()

    if schema.valid_values is not None:
        valid &= coerced.isin(schema.valid_values)

    if schema.min_value is not None:
        valid &= coerced >= schema.min_value

    if schema.max_value is not None:
        valid &= coerced <= schema.max_value

    if schema.pattern is not None:
        valid &= coerced.str.match(schema.pattern, na=False)

    if schema.is_calendar_date:
        valid &= _is_valid_calendar_date(coerced)

    if schema.is_timezone:
        # astype(bool): on an empty column, .apply() has no values to infer a
        # return dtype from and falls back to the column's own (str) dtype
        # instead of bool, which then fails the boolean `&=` below.
        valid &= coerced.apply(_is_iana_timezone).astype(bool)

    if schema.nullable:
        # A missing value in a nullable column isn't an invalid value — it's the
        # normal case (e.g. stops.txt's parent_station for a stop with no parent).
        # It only needs to pass the checks above when it's actually present.
        valid |= ~present

    return valid


def _filter_and_cast_file(
    filename: str, df: pd.DataFrame, column_schemas: dict[str, GTFSColumnSchema]
) -> pd.DataFrame:
    """Drop rows failing any column's schema, then cast surviving columns."""
    if not column_schemas:
        return df

    total_rows = len(df)
    coerced_columns: dict[str, pd.Series] = {}
    row_valid = pd.Series(True, index=df.index)

    for column, schema in column_schemas.items():
        if column not in df.columns:
            continue

        coerced = _soft_cast_column(df[column], schema)
        coerced_columns[column] = coerced

        column_valid = _column_validity_mask(coerced, schema)
        invalid_count = int((~column_valid).sum())
        if invalid_count:
            logger.warning(
                "%s: dropping %d/%d row(s) with an invalid %s",
                filename,
                invalid_count,
                total_rows,
                column,
            )

        row_valid &= column_valid

    df = df.loc[row_valid].copy()
    for column, coerced in coerced_columns.items():
        schema = column_schemas[column]
        validated = coerced.loc[row_valid]
        casted = validated.astype(schema.dtype)
        if schema.nullable:
            # astype() turns a NaN into the literal string "nan" for str columns —
            # put the real missing value back wherever it was missing beforehand.
            casted = casted.where(validated.notna(), validated)
        df[column] = (
            schema.to_semantic_type(casted) if schema.to_semantic_type is not None else casted
        )

    dropped_rows = total_rows - len(df)
    if dropped_rows:
        logger.warning(
            "%s: dropped %d/%d row(s) total after type/range validation",
            filename,
            dropped_rows,
            total_rows,
        )

    return df


def validate_gtfs_schedule_types(
    gtfs_files: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Drop rows that violate a required column's type/range/pattern and cast the
    remaining rows to their required dtype. Invalid data is filtered out row by
    row rather than failing the whole file, so one bad row doesn't take down an
    otherwise-usable GTFS feed. Each drop is logged with the offending file/column
    so silently-lost rows stay debuggable.

    Args:
        gtfs_files: the parsed GTFS files.

    Returns:
        The same files, with invalid rows dropped and valid columns cast to
        their schema's dtype (and semantic type, if configured).
    """
    return {
        filename: _filter_and_cast_file(
            filename, df, GTFS_SCHEDULE_COLUMN_SCHEMAS.get(filename, {})
        )
        for filename, df in gtfs_files.items()
    }


def _drop_duplicates(
    filename: str,
    df: pd.DataFrame,
    subset: list[str] | None,
    reason: str,
    keep: Literal["first", False],
) -> pd.DataFrame:
    duplicate_mask = df.duplicated(subset=subset, keep=keep)
    dropped = int(duplicate_mask.sum())
    if dropped:
        logger.warning(
            "%s: dropped %d/%d row(s) that were %s",
            filename,
            dropped,
            len(df),
            reason,
        )
    return df.loc[~duplicate_mask]


def drop_duplicate_rows(
    gtfs_files: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Drop rows that are exact duplicates of an earlier row in the same file
    (identical across every column we kept), keeping the first occurrence.
    Unlike a primary-key conflict, an exact duplicate carries no extra
    information either way, so there's no policy choice here — the extra
    copies are always dropped.

    agency.txt is skipped here: GTFS_SCHEDULE_REQUIRED_COLUMNS narrows it down
    to just agency_timezone, so any feed with more than one agency sharing a
    timezone (the spec-compliant, expected case per
    pygtfsrealtime.schedule.snapshot.resolve_gtfs_timezone) would otherwise trigger
    a spurious "dropped duplicates" warning that reads like a dataset problem
    when there isn't one.

    Args:
        gtfs_files: the parsed GTFS files.

    Returns:
        The same files, with exact-duplicate rows dropped (agency.txt
        unchanged).
    """
    return {
        filename: (
            df
            if filename == "agency.txt"
            else _drop_duplicates(
                filename,
                df,
                None,
                "exact duplicates of an earlier row",
                keep="first",
            )
        )
        for filename, df in gtfs_files.items()
    }


def drop_primary_key_violations(
    gtfs_files: dict[str, pd.DataFrame],
    policy: PrimaryKeyDuplicatePolicy,
) -> dict[str, pd.DataFrame]:
    """Drop rows whose primary key (per the GTFS Schedule reference) repeats an
    earlier row's. Two rows can't both legitimately claim the same
    trip_id/stop_id/(shape_id, shape_pt_sequence)/etc.

    `policy` controls what survives a conflict — there's no sane default to
    fall back on silently, callers must say which they want: "keep_first"
    keeps the first occurrence and drops the rest; "drop_all" trusts neither
    row and drops every row sharing that key, since there's no principled way
    to tell which one is correct.

    Args:
        gtfs_files: the parsed GTFS files.
        policy: what to do when two rows share a primary key.

    Returns:
        The same files, with primary-key-violating rows dropped per policy.
    """
    keep: Literal["first", False] = "first" if policy == "keep_first" else False
    reason = (
        "duplicates of an earlier row's primary key"
        if policy == "keep_first"
        else "sharing a primary key with another row (policy: drop_all)"
    )

    result = {}
    for filename, df in gtfs_files.items():
        primary_key = GTFS_SCHEDULE_PRIMARY_KEYS.get(filename)
        if not primary_key or not set(primary_key).issubset(df.columns):
            result[filename] = df
            continue

        result[filename] = _drop_duplicates(
            filename,
            df,
            list(primary_key),
            f"{reason} {primary_key}",
            keep=keep,
        )

    return result


def validate_gtfs_schedule_required_files(files: set[str]) -> set[str]:
    """Check that every required GTFS file was present in the archive.

    Args:
        files: filenames extracted from the GTFS zip.

    Returns:
        `files`, unchanged, once completeness is confirmed.

    Raises:
        TypeError: if a required file is missing.
    """
    missing_files = GTFS_SCHEDULE_REQUIRED_FILES - files
    if missing_files:
        raise TypeError(f"GTFS schedule is missing required files: {sorted(missing_files)!r}")
    return files


def validate_gtfs_schedule_columns(
    gtfs_files: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Check that every required column is present in its GTFS file.

    Args:
        gtfs_files: the parsed GTFS files.

    Returns:
        `gtfs_files`, unchanged, once completeness is confirmed.

    Raises:
        TypeError: if a required column is missing from a required file.
    """
    for filename, required_columns in GTFS_SCHEDULE_REQUIRED_COLUMNS.items():
        if filename not in gtfs_files:
            continue

        missing_columns = required_columns - set(gtfs_files[filename].columns)
        if missing_columns:
            raise TypeError(f"{filename} is missing required columns: {sorted(missing_columns)!r}")

    return gtfs_files
