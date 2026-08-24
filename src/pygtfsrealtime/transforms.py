from functools import cache

from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import transform


@cache
def _get_transformer(from_crs: str, to_crs: str) -> Transformer:
    """Return a cached pyproj Transformer for the given CRS pair."""
    return Transformer.from_crs(from_crs, to_crs, always_xy=True)


def to_utm(point: Point, projection: str) -> Point:
    """Project a lat/lon (EPSG:4326) point into a UTM projection.

    Args:
        point: a Point in EPSG:4326 (longitude, latitude order).
        projection: the target UTM CRS, e.g. "EPSG:32723".

    Returns:
        The equivalent Point in `projection`, in meters.
    """
    transformer = _get_transformer("EPSG:4326", projection)
    return transform(transformer.transform, point)


def to_latlon(point: Point, projection: str) -> Point:
    """Project a UTM point back into lat/lon (EPSG:4326).

    Args:
        point: a Point in `projection`, in meters.
        projection: the source UTM CRS, e.g. "EPSG:32723".

    Returns:
        The equivalent Point in EPSG:4326 (longitude, latitude order).
    """
    transformer = _get_transformer(projection, "EPSG:4326")
    return transform(transformer.transform, point)
