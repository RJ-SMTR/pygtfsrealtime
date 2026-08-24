import logging

from pygtfsrealtime.models import GPSEntry
from pygtfsrealtime.settings import MATCH_KEY_COLUMNS, MatchingStrategy

logger = logging.getLogger(__name__)


def validate_gps_entries(
    entries: list[GPSEntry], trip_matching: MatchingStrategy | None
) -> list[GPSEntry]:
    """Drop any entry missing a field `trip_matching` needs to build its match
    key - same "log a warning, keep going" shape as pygtfsrealtime.schedule.validate's
    row-level dropping functions, so one incomplete observation doesn't take
    down the whole cycle's feed.

    Args:
        entries: this cycle's raw GPS entries.
        trip_matching: the configured matching strategy.

    Returns:
        The entries with every field `trip_matching.key` needs.

    Raises:
        TypeError: if `trip_matching` is None.
    """
    if trip_matching is None:
        raise TypeError(
            "settings.trip_matching is not set - construct one with "
            "pygtfsrealtime.settings.MatchingStrategy(key=...) and pass it as "
            "Settings(trip_matching=...) before running the GPS/FSM "
            "cycle."
        )

    required_fields = MATCH_KEY_COLUMNS[trip_matching.key]
    validated = []
    dropped_ids = []
    for entry in entries:
        if all(getattr(entry, field) is not None for field in required_fields):
            validated.append(entry)
        else:
            dropped_ids.append(entry.vehicle_id)

    if dropped_ids:
        logger.warning(
            "validate_gps_entries: dropping %d observation(s) missing field(s) %s "
            "required by trip_matching.key=%r: vehicle_id(s) %s",
            len(dropped_ids),
            required_fields,
            trip_matching.key,
            dropped_ids,
        )
    return validated
