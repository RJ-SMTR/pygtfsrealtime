from datetime import datetime

# FeedMessage is built dynamically by protoc's generated module, so mypy
# can't see it statically.
from pygtfsrealtime.pb.gtfs_realtime_pb2 import FeedMessage  # type: ignore[attr-defined]
from pygtfsrealtime.realtime.fsm import VehicleFSM, VehicleState
from pygtfsrealtime.realtime.ingest import VehicleReport


def build_pb(
    fsms: dict[str, VehicleFSM], vehicles: dict[str, VehicleReport], now: datetime
) -> bytes:
    """Serialize this cycle's fleet to a GTFS-RT VehiclePositions FeedMessage.

    Args:
        fsms: per-vehicle FSM state (see pygtfsrealtime.realtime.fsm).
        vehicles: this cycle's fresh GPS fetch (see
            pygtfsrealtime.realtime.ingest.GPSIngester.ingest) - VehicleFSM
            itself holds no position, only FSM state, so position always
            comes from here.
        now: the current cycle's timestamp, used for the feed header.

    Returns:
        A serialized `FeedMessage` (protobuf bytes). A vehicle is only
        emitted when it both reported GPS this cycle AND is BUSY (confirmed
        on a trip).
    """
    feed = FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    feed.header.incrementality = feed.header.FULL_DATASET
    feed.header.timestamp = int(now.timestamp())

    for vehicle_id, vehicle in vehicles.items():
        fsm = fsms.get(vehicle_id)
        if fsm is None or fsm.state != VehicleState.BUSY:
            continue
        # BUSY implies current_trip is set - see VehicleFSM._advance.
        assert fsm.current_trip is not None

        entity = feed.entity.add()
        entity.id = vehicle_id

        pb_vehicle = entity.vehicle
        pb_vehicle.trip.trip_id = fsm.current_trip.trip_id
        pb_vehicle.trip.start_time = fsm.current_trip.start_dt.strftime("%H:%M:%S")
        pb_vehicle.trip.start_date = fsm.current_trip.start_dt.strftime("%Y%m%d")
        pb_vehicle.trip.schedule_relationship = 0
        pb_vehicle.trip.route_id = fsm.current_trip.route_id
        pb_vehicle.trip.direction_id = fsm.current_trip.direction_id

        pb_vehicle.position.latitude = vehicle.latitude
        pb_vehicle.position.longitude = vehicle.longitude
        if vehicle.bearing:
            pb_vehicle.position.bearing = vehicle.bearing
        if vehicle.speed is not None:
            pb_vehicle.position.speed = vehicle.speed

        pb_vehicle.timestamp = int(vehicle.datetime.timestamp())
        pb_vehicle.vehicle.id = vehicle_id

    return feed.SerializeToString()
