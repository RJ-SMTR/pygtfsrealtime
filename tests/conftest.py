import pytest

from tests.gtfs_data import build_gtfs_dataframes, build_gtfs_zip


@pytest.fixture
def valid_gtfs_zip() -> bytes:
    return build_gtfs_zip()


@pytest.fixture
def make_gtfs_zip():
    return build_gtfs_zip


@pytest.fixture
def valid_gtfs_dataframes() -> dict:
    return build_gtfs_dataframes()


@pytest.fixture
def make_gtfs_dataframes():
    return build_gtfs_dataframes
