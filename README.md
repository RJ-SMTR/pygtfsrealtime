# pygtfsrealtime

A Python library that turns raw AVL/GPS feeds into standards-compliant GTFS-Realtime `VehiclePosition` data, resolving the trip ID and start time in case your AVL vendor doesn't provide them.

## Overview

The GTFS-RT `VehiclePosition` spec expects a `TripDescriptor` (`trip_id`, `start_date`, `start_time`, `route_id`, ...) alongside every position report. In practice, a lot of AVL (Automatic Vehicle Location) hardware and legacy fleet-tracking systems don't produce that — they report only a vehicle identifier, a coordinate, and a timestamp, sometimes with a route label but rarely with a trip instance already resolved.

`pygtfsrealtime` closes that gap: given a GTFS static schedule and a stream of raw vehicle positions, it matches each vehicle to a candidate trip instance using a small set of explicit, operator-configured rules and thresholds, and publishes a standard GTFS-RT `FeedMessage`. See [Limitations](#limitations-and-disclaimer).

## How it works

`GTFSRealtimeEngine` runs three background loops:

- **GTFS schedule loop** — periodically re-fetches the static GTFS feed and only reprocesses it when its content actually changed.
- **Trip window loop** — from the current schedule, computes the rolling set of trip *instances* (a `trip_id` plus its scheduled start, with `frequencies.txt` expanded) whose scheduled time range overlaps the near future.
- **FSM loop** — on a fixed cadence (30 seconds by default, in line with the GTFS-RT spec's own publish interval), ingests fresh GPS, matches vehicles against the trip window, advances each vehicle's per-vehicle state machine, and publishes the resulting feed.

Matching is key-based, not path-nearest: you choose which field(s) a vehicle report and a scheduled trip are paired on (`trip_id`, `route_id`, `route_short_name`, optionally combined with `direction_id`), and how ties are broken when more than one trip shares that key. A separate geometric check — proximity to the route's shape, to a stop, or to a terminal, computed in a projected (UTM) coordinate system — decides whether a paired vehicle actually confirms onto that trip.

Each vehicle is tracked by a small state machine with two states, `FREE` (unmatched) and `BUSY` (confirmed on a trip). Only `BUSY` vehicles are published. A vehicle moves to `BUSY` once it has a trip candidate, is close enough to that trip's path, and isn't sitting at a terminal; it reverts to `FREE` on signal loss, prolonged stationarity, terminal arrival, or a trip that's run past its expected duration. An optional callback fires on every transition (whether or not the state actually changed), which is the hook for operational alerting — e.g. flagging vehicles that go off-path.

## Requirements

- Python ≥ 3.11
- A static GTFS feed containing at least `agency.txt`, `calendar.txt`, `calendar_dates.txt`, `routes.txt`, `trips.txt`, `stop_times.txt`, `shapes.txt`, and `stops.txt` — a subset of the full GTFS spec, not the whole feed. `frequencies.txt` is optional, same as in the GTFS spec itself; a feed with no frequency-based trips can omit it.
- A live vehicle-position source (however you fetch it) that yields, per reading, at minimum a vehicle ID, latitude/longitude, and a timestamp — plus whichever of `route_id`/`route_short_name`/`trip_id`/`direction_id` your chosen matching strategy needs.
- A UTM (or otherwise meter-based, projected) CRS identifier covering the feed's operating area, since every proximity/distance check in the engine is done in meters. [epsg.io](https://epsg.io/) lets you search by place name or coordinates to find the right one.
- `Settings.timezone` is optional — by default the feed's timezone is inferred from `agency.txt`'s `agency_timezone`. We still strongly recommend passing it explicitly: if `agency.txt` is missing it or has conflicting values across rows (a spec violation, but it happens), `Settings.timezone` is what it falls back to; with neither, the engine can't tell what timezone the feed's times are in and stops.

## Installation

`pygtfsrealtime` is published on PyPI and installs like any other package:

```bash
pip install pygtfsrealtime
```

## Quickstart

`pygtfsrealtime` needs five inputs to run — everything else (schedule parsing, trip windows, geometry, the per-vehicle state machine) it resolves on its own:

1. a GTFS schedule feed,
2. live GPS/AVL data,
3. a function to publish the resulting GTFS-RT feed,
4. a UTM projection for your operating area, and
5. a trip-matching strategy.

### 1. GTFS schedule feed

`gtfs_schedule` accepts either a local file path, or a callback for anything that has to be fetched.

A local file:

```python
gtfs_schedule = "gtfs.zip"
```

A remote source — any callback returning `io.BytesIO` works; this is an example of reading directly from GCS:

```python
import io

from google.cloud import storage


def gtfs_schedule() -> io.BytesIO:
    blob = storage.Client().bucket("my-bucket").blob("gtfs/gtfs.zip")
    return io.BytesIO(blob.download_as_bytes())
```

### 2. GPS data

`ingest_gps_data` is called once per FSM cycle and must return a `list[GPSEntry]`. Wrap whatever your fleet-tracking API/DB returns:

```python
from pygtfsrealtime import GPSEntry


def ingest_gps_data() -> list[GPSEntry]:
    raw_positions = fetch_positions_from_your_fleet_api()  # however you do this
    entries = []
    for p in raw_positions:
        entries.append(
            GPSEntry(
                vehicle_id=p["vehicle_id"],
                latitude=p["lat"],
                longitude=p["lon"],
                datetime=p["timestamp"],
                route_short_name=p["route"],
                direction_id=p["direction_id"],
            )
        )
    return entries
```

### 3. Publish function

`publish_protobuf` is called once per FSM cycle with the serialized GTFS-RT `FeedMessage` bytes.

A local file:

```python
def publish_protobuf(feed_bytes: bytes) -> None:
    with open("vehicle_positions.pb", "wb") as f:
        f.write(feed_bytes)
```

A remote destination — this is an example of writing directly to GCP:

```python
from google.cloud import storage


def publish_protobuf(feed_bytes: bytes) -> None:
    blob = storage.Client().bucket("my-bucket").blob("gtfs-rt/vehicle_positions.pb")
    blob.upload_from_string(feed_bytes, content_type="application/octet-stream")
```

### 4. UTM projection

Every proximity/distance check in the engine is done in meters, so `Settings.projection` needs a projected CRS covering your operating area. [epsg.io](https://epsg.io/) lets you search by place name or coordinates to find the right one:

```python
from pygtfsrealtime import Settings

settings = Settings(projection="EPSG:32723")
```

### 5. Matching strategy

`Settings.trip_matching` decides which field(s) pair a vehicle to a scheduled trip (`key`: one of `trip_id`, `route_id`, `route_short_name`, `route_id,direction_id`, `route_short_name,direction_id` — whichever your GPS feed actually populates) and how ties between candidates are broken (`mode`: `"strict"` or `"progress_match"`, the default). There's no default `key` — it's operator-specific:

- `"strict"` only matches when exactly one candidate shares that key; if two trips with the same key are ever scheduled concurrently (overlapping time windows), that ambiguity never resolves on its own and the vehicle just stays unmatched for as long as both are active.
- `"progress_match"` scores each candidate by how well two independent progress signals agree: the fraction of the trip's scheduled duration elapsed so far, versus the fraction of the trip's shape already traveled (found by projecting the vehicle's GPS point onto the shape). A candidate is only eligible if the vehicle is within `Settings.shape_geometry`'s distance of that candidate's shape; among eligible candidates, only those whose deviation between the two fractions is within `acceptance_margin` (default `0.2`) survive, and a trip already claimed by another vehicle this cycle is excluded unless `allow_shared_trip` (default `False`) is set. The smallest-deviation survivor wins, ties broken toward the not-yet-claimed candidate and then the earlier scheduled start.

```python
from pygtfsrealtime import MatchingStrategy, Settings

settings = Settings(
    projection="EPSG:32723",
    trip_matching=MatchingStrategy(key="route_short_name,direction_id"),
)
```

### Putting it together

```python
from pygtfsrealtime import GTFSRealtimeEngine, MatchingStrategy, Settings

engine = GTFSRealtimeEngine(
    gtfs_schedule=gtfs_schedule,
    ingest_gps_data=ingest_gps_data,
    publish_protobuf=publish_protobuf,
    settings=Settings(
        projection="EPSG:32723",
        trip_matching=MatchingStrategy(key="route_short_name,direction_id"),
    ),
)

engine.run()  # blocks until Ctrl+C (or a fatal configuration error), then
# stops every loop cleanly
```

## Optional callbacks

Two more constructor arguments aren't part of the five required inputs, but cover common operational needs.

### Persisting state across restarts

Each vehicle is tracked by its own FSM in memory — if the process restarts, that state is gone and every vehicle has to re-earn its match from scratch. `set_cache`/`get_cache` (give both together) let the engine persist and restore that state across restarts, as opaque `bytes`.

```python
from pathlib import Path


def set_cache(state: bytes) -> None:
    Path("engine_cache.pkl").write_bytes(state)


def get_cache() -> bytes | None:
    path = Path("engine_cache.pkl")
    return path.read_bytes() if path.exists() else None


engine = GTFSRealtimeEngine(
    ...,  # the five inputs from above
    set_cache=set_cache,
    get_cache=get_cache,
)
```

### Alerting on state transitions

`on_transition` is called once per completed FSM cycle — not when a cycle is skipped for lack of a `TripsSnapshot` — with the list of every vehicle's transition outcome from that cycle (possibly empty, if no vehicle reported that cycle). Each entry is a `(vehicle_id, old_state, new_state, reason, observation)` `TransitionEvent`, covering every vehicle processed that cycle whether or not its state actually changed. It's independent of the published GTFS-RT feed itself, so it's a hook for operational alerting: e.g. paging when a vehicle goes off its route path, or just logging every cycle's batch of matches gone stale.

```python
from pygtfsrealtime import TransitionReason


def on_transition(transitions) -> None:
    for vehicle_id, old_state, new_state, reason, observation in transitions:
        if reason is TransitionReason.OFF_PATH:
            alert(f"{vehicle_id} went off its route path")


engine = GTFSRealtimeEngine(
    ...,  # the five inputs from above
    on_transition=on_transition,
)
```

## Configuration

Every tunable lives on `Settings` (`pygtfsrealtime.Settings`); the two fields below have no default and must be set explicitly, everything else falls back to a documented default:

| Field | Purpose |
| --- | --- |
| `projection` | UTM CRS for all meter-based geometry. No default — operating-area specific. |
| `trip_matching` | `MatchingStrategy(key=..., mode=...)` — which field(s) pair a vehicle to a trip, and how ties are broken. No default — which fields a feed populates is operator-specific. |
| `shape_geometry` / `stop_geometry` / `terminal_geometry` | Proximity thresholds (distance + buffer-vs-raw-distance mode) for "on path", "at a stop", "at a terminal". |
| `stationary_threshold` | How little movement, sustained for how long, counts as a vehicle standing still. |
| `stale_trip_threshold` | How long a confirmed vehicle can go without reconfirming its trip before the match is dropped. |
| `signal_loss_threshold` | How long without a new GPS reading before a vehicle is forced back to `FREE`. |
| `gtfs_loop_schedule` / `fsm_loop_schedule` / `trip_window_loop_schedule` | Cadence of the three background loops. |

The three geometry thresholds each gate a different proximity check, all computed in `projection`'s meters:

- `shape_geometry` (default 30m, `mode="distance"`) — how close a vehicle needs to be to its trip's route path to count as "on path". This is the main signal that a match is actually plausible, not just a coincidence of sharing a match key: a vehicle only becomes `BUSY` while on path.
- `stop_geometry` (default 250m, `mode="buffer"`) — proximity to a single stop, used as a trip's start/end point when that stop isn't part of a larger terminal.
- `terminal_geometry` (default 250m, `mode="buffer"`) — proximity to a terminal (a stop with child stops/platforms). It has one extra option, `shape_mode`: `"stops_convex_hull"` (default) uses the hull enclosing every platform under the terminal; `"terminal_point"` uses just the parent stop's own coordinate, which undersells a terminal whose platforms are spread out. Being "at a terminal" is what turns a `BUSY` vehicle back to `FREE` on arrival (the trip is done), and what keeps a vehicle from being confirmed while it's still sitting at the terminal it hasn't departed from yet.

`mode` (`"buffer"` vs `"distance"`) on any of the three is a performance/memory trade-off, not a behavioral one — both give the same yes/no answer for the same distance. `"buffer"` precomputes the threshold as a polygon once and checks point-in-polygon containment; `"distance"` skips that precomputation and compares the raw point-to-geometry distance on every check.

See the `Settings` docstring in `pygtfsrealtime/settings.py` for the full field list and every default value.

## Methodology

The three loops from [How it works](#how-it-works) are stages of one pipeline, each handing its output to the next through a snapshot: the GTFS schedule loop produces a `GtfsSnapshot`, the trip window loop consumes it to produce a `TripsSnapshot`, and the FSM loop consumes that every cycle. Static, feed-wide work (parsing, geometry, zones) happens once per GTFS refresh in the first stage; nothing downstream repeats it.

### 1. Schedule ingestion

Runs once per GTFS refresh (only when the fetched feed's content actually changed, checked by hash before any of this runs):

- **Parse and clean.** Extract the required/optional files from the zip, validate columns and required files are present, drop duplicate/incomplete rows and primary-key conflicts, and cast every remaining value to its declared type.
- **Build geometry.** Reproject `stops.txt`/`shapes.txt` into `Settings.projection`'s UTM meters (every proximity check downstream is a metric distance, never lat/lon degrees), turning each shape into a `LineString`.
- **Reconcile to a fixed point.** A shape with fewer than 2 points gets dropped, which can orphan the trips/routes that referenced it, which can in turn orphan a shape or route nothing else uses — so trips/routes/shapes/stop_times are reconciled against each other repeatedly, not in one pass, until a cycle makes no further changes.
- **Precompute proximity zones**, once per feed refresh from `Settings`'s geometry thresholds (see [Configuration](#configuration)): a buffer polygon or raw geometry (per `mode`) for each stop, for each shape's full path plus its two endpoint points, and for each terminal (a stop referenced as another stop's `parent_station`, represented as either its own point or the convex hull of its child stops).
- **Resolve trip endpoints.** For each trip, pick which precomputed zone represents its start/end: prefer `Settings.trip_endpoint_source` (default `"stop"`), falling back through stop → terminal → shape in that order wherever the preferred source can't resolve (e.g. `"terminal"` requested but that stop has no `parent_station`). The winning zone *and* which source it came from are embedded directly onto `trips.txt`, so the realtime stage later reapplies the exact threshold that built that zone instead of guessing one.

### 2. Trip window

Rebuilt on `trip_window_loop_schedule`'s cadence (default every 8h) or right after a schedule change, from the already-built `GtfsSnapshot` — no geometry is recomputed here, only assembled:

- **Resolve active service.** For every calendar date the window could reach — including a lookback derived from the feed's own largest `stop_times`/`frequencies` offset, since a past-midnight trip anchors to the *previous* day — combine `calendar.txt`'s weekly pattern with `calendar_dates.txt`'s day-specific exceptions into that day's active `service_id`s.
- **Expand instances.** A `frequencies.txt` trip becomes one row per headway-spaced departure (each keeping the *original* trip's own scheduled duration, never `headway_secs` itself); every other active trip contributes its one `stop_times.txt`-derived start/end directly.
- **Keep what overlaps.** Only instances whose `[start_dt, end_dt)` intersects `[window_start, window_end)` survive — a trip already running at rebuild time is included, not just ones starting after it — each joined to its route/shape/zone columns from stage 1.

### 3. Realtime matching

Runs every FSM cycle (`fsm_loop_schedule`, default 30s):

- **Ingest GPS**, dropping any reading missing a field the configured `MatchingStrategy.key` needs to look a vehicle up at all.
- **Index candidates.** Filter the trip window to instances active right now, and group them by `key`'s column(s) into a lookup dict — built once per cycle, not once per vehicle.
- **Select a candidate** from each vehicle's key group: `"strict"` accepts only a group of exactly one; `"progress_match"` scores every eligible candidate by how closely its elapsed-time fraction agrees with its shape-progress fraction (found by projecting the vehicle's point onto the shape via linear referencing), discards anything outside `acceptance_margin` or already claimed by another vehicle this cycle (unless `allow_shared_trip`), and keeps the smallest deviation.
- **Confirm geometrically.** A selected candidate isn't final until it passes the same proximity checks as everything else: close enough to the trip's shape (`shape_geometry`) to count as on-path, and *not* still sitting in its resolved start/end zone (whichever threshold that zone was actually built with — see stage 1).
- **Advance the FSM.** `FREE → BUSY` on a fresh match that's on-path and not at a terminal; `BUSY → FREE` on signal loss, sustained stationarity, terminal arrival, or exceeding `stale_trip_threshold`. Signal loss is checked first, in either state, ahead of every other condition.
- **Publish.** Serialize one GTFS-RT `FeedMessage`, with one `VehiclePosition` entity per vehicle that is both currently reporting *and* `BUSY` — a `FREE` vehicle is omitted from the feed entirely, never published with an empty or guessed trip.

## Use cases

- **Legacy AVL hardware.** Fleet trackers that only ever report `vehicle_id` + coordinates + timestamp (no `trip_id`), where the operator still wants to publish a spec-compliant GTFS-RT `VehiclePositions` feed.
- **Bridging any position source into GTFS-RT.** Any system that already has vehicle coordinates on some cadence — regardless of its original format — can be adapted into a GTFS-RT feed by writing an `ingest_gps_data` callback around it.
- **Operational monitoring.** The `on_transition` callback can be used to alert on specific transition reasons (e.g. a vehicle going off its route path, or a trip running long) as an operational signal, independent of whether a GTFS-RT feed is being published at all.

## Limitations and disclaimer

- Matching is **best-effort**, not guaranteed-correct: it depends on the quality/frequency of the underlying AVL data and on the operator choosing thresholds and a matching strategy appropriate to their own network. Sparse GPS, overlapping routes, or a loosely-chosen match key can all produce incorrect or missed matches.
- The project is at an **early stage** (pre-1.0). The public API may change between releases without a long deprecation cycle.
- This project relies heavily on AI assistance for its development.
- The software is provided **as-is, without warranty of any kind**, express or implied, including but not limited to accuracy, reliability, or fitness for a particular purpose. Evaluate it against your own data and operational requirements before relying on it.
- Licensed under **AGPL-3.0-or-later** (see [License](#license)) — a copyleft license with network-use provisions. Review the full license text (or get your own legal advice) before integrating this into a larger system, especially one offered as a network service.

## Contributing

Issues and pull requests are welcome. There isn't yet a formal `CONTRIBUTING.md` or CI pipeline — those are still to be set up. In the meantime:

1. Fork/clone the repository and set up the environment with [uv](https://docs.astral.sh/uv/):

   ```bash
   git clone https://github.com/RJ-SMTR/pygtfsrealtime.git
   cd pygtfsrealtime
   uv sync  # creates .venv and installs locked dependencies
   ```

2. Before opening a PR: run the test suite (`uv run pytest`, adding tests for any behavioral
   change), lint and format with [ruff](https://docs.astral.sh/ruff/) (`uv run ruff check .`,
   `uv run ruff format .`), and type-check with [mypy](https://mypy-lang.org/)
   (`uv run mypy src tests`).
3. Open a pull request describing the change and its motivation.


## License

Copyright (c) 2026 Prefeitura da Cidade do Rio de Janeiro - Secretaria de Transportes do Rio de Janeiro (SMTR)

[GNU Affero General Public License v3.0 (AGPL-3.0-or-later)](LICENSE).
