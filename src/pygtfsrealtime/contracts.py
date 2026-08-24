import pandas as pd

from pygtfsrealtime.models import GPSEntry


def validate_schedule_result(result) -> dict[str, pd.DataFrame]:
    """Check that a GTFS schedule callback returned the expected shape.

    Args:
        result: the raw return value of an `ingest_gtfs_schedule` callback.

    Returns:
        `result`, unchanged, once its shape is confirmed.

    Raises:
        TypeError: if `result` isn't a `dict[str, DataFrame]`.
    """
    if not isinstance(result, dict):
        raise TypeError(
            f"ingest_gtfs_schedule must return a dict[str, DataFrame], received: {result!r}"
        )

    return result


def validate_gps_data_result(result) -> list[GPSEntry]:
    """Check that an `ingest_gps_data` callback returned the expected shape.

    Args:
        result: the raw return value of an `ingest_gps_data` callback.

    Returns:
        `result`, unchanged, once its shape is confirmed.

    Raises:
        TypeError: if `result` isn't a `list[GPSEntry]`.
    """
    if not isinstance(result, list) or not all(isinstance(item, GPSEntry) for item in result):
        raise TypeError(f"ingest_gps_data must return a list[GPSEntry], received: {result!r}")

    return result


def validate_cache_result(result) -> bytes | None:
    """Check that a `get_cache` callback returned the expected shape.

    Args:
        result: the raw return value of a `get_cache` callback.

    Returns:
        `result`, unchanged, once its shape is confirmed.

    Raises:
        TypeError: if `result` isn't `bytes` or `None`.
    """
    if result is not None and not isinstance(result, bytes):
        raise TypeError(f"get_cache must return bytes or None, received: {result!r}")

    return result
