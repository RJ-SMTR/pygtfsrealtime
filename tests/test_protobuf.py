from dataclasses import dataclass
from datetime import UTC, datetime

from shapely.geometry import Point

from pygtfsrealtime.pb.gtfs_realtime_pb2 import FeedMessage  # type: ignore[attr-defined]
from pygtfsrealtime.pb.protobuf import build_pb
from pygtfsrealtime.realtime.fsm import VehicleFSM, VehicleState
from pygtfsrealtime.realtime.ingest import VehicleReport

NOW = datetime(2026, 1, 1, tzinfo=UTC)
T0 = datetime(2026, 1, 1, 12, 0, 0)


@dataclass
class _Trip:
    trip_id: str
    start_dt: datetime
    route_id: str
    direction_id: int


def _vehicle(vehicle_id: str, bearing: int | None = None) -> VehicleReport:
    return VehicleReport(
        vehicle_id=vehicle_id,
        latitude=-22.9,
        longitude=-43.2,
        speed=0.0,
        point=Point(0, 0),
        datetime=NOW,
        bearing=bearing,
    )


def _fsm(vehicle_id: str) -> VehicleFSM:
    fsm = VehicleFSM(vehicle_id)
    fsm.state = VehicleState.BUSY
    fsm.current_trip = _Trip("T1", NOW, "R1", 0)  # type: ignore
    return fsm


def _build_feed(vehicle_id: str, bearing: int | None = None) -> FeedMessage:
    fsm = _fsm(vehicle_id)
    vehicle = _vehicle(vehicle_id, bearing)
    raw = build_pb({vehicle_id: fsm}, {vehicle_id: vehicle}, NOW)

    feed = FeedMessage()
    feed.ParseFromString(raw)  # bytes -> FeedMessage
    return feed


def test_bearing_accepts_zero_as_value():
    """
    bearing=0 (due north) is a valid heading and must be published.
    """

    feed = _build_feed("V1", 0)
    position = feed.entity[0].vehicle.position

    assert position.HasField("bearing")
    assert position.bearing == 0


def test_missing_bearing_doesnt_turn_to_zero():
    feed = _build_feed("V1")
    position = feed.entity[0].vehicle.position

    assert not position.HasField("bearing")
