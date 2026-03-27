"""Pure grouping functions — no I/O.

All functions operate on lists of ImageRecord and return plain dicts or
lists of ImageGroup.  The only stateful object is the dispatcher
get_groups / resolve_group which centralises the GroupBy dispatch.
"""

from __future__ import annotations

import math
from collections import defaultdict

from .models import CompassDirection, GroupBy, ImageGroup, ImageRecord

# ---------------------------------------------------------------------------
# Azimuth bucketing
# ---------------------------------------------------------------------------

_DIRECTIONS = [
    CompassDirection.N,
    CompassDirection.NE,
    CompassDirection.E,
    CompassDirection.SE,
    CompassDirection.S,
    CompassDirection.SW,
    CompassDirection.W,
    CompassDirection.NW,
]


def azimuth_to_compass(azimuth: float) -> CompassDirection:
    """Map a signed azimuth (degrees) to the nearest compass octant.

    Handles negative azimuths (west of north) correctly via modulo normalisation:
      -91.2° → 268.8° → W
      -178.3° → 181.7° → S
       88.5°  →  88.5° → E
    """
    normalised = azimuth % 360  # always in [0, 360)
    index = math.floor((normalised + 22.5) / 45) % 8
    return _DIRECTIONS[index]


# ---------------------------------------------------------------------------
# Individual grouping functions
# ---------------------------------------------------------------------------


def group_by_property(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        groups[r.property_id].append(r)
    return dict(groups)


def group_by_flight(
    records: list[ImageRecord],
    include_no_flight: bool = False,
) -> dict[str, list[ImageRecord]]:
    """Group by flight_id.

    Records with no flight_id are placed in a 'no_flight' bucket only when
    include_no_flight=True; otherwise they are silently omitted.
    """
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        if r.flight_id:
            groups[r.flight_id].append(r)
        elif include_no_flight:
            groups["no_flight"].append(r)
    return dict(groups)


def group_by_azimuth(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        key = azimuth_to_compass(r.azimuth).value
        groups[key].append(r)
    return dict(groups)


def group_by_date(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    """Group by calendar date (UTC) of capture, key format YYYY-MM-DD."""
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        key = r.captured_at.date().isoformat()
        groups[key].append(r)
    return dict(groups)


def group_by_property_azimuth(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    """Composite key '{property_id}::{compass}'.

    Useful for comparing the same building face across different survey dates.
    """
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        compass = azimuth_to_compass(r.azimuth).value
        key = f"{r.property_id}::{compass}"
        groups[key].append(r)
    return dict(groups)


def group_by_property_flight(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    """Composite key '{property_id}::{flight_id}'.

    Only includes records that have a non-null flight_id.
    """
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        if r.flight_id:
            key = f"{r.property_id}::{r.flight_id}"
            groups[key].append(r)
    return dict(groups)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_DISPATCH = {
    GroupBy.PROPERTY: group_by_property,
    GroupBy.FLIGHT: group_by_flight,
    GroupBy.AZIMUTH: group_by_azimuth,
    GroupBy.DATE: group_by_date,
    GroupBy.PROPERTY_AZIMUTH: group_by_property_azimuth,
    GroupBy.PROPERTY_FLIGHT: group_by_property_flight,
}


def get_groups(records: list[ImageRecord], group_by: GroupBy) -> list[ImageGroup]:
    """Return all groups for the given grouping dimension."""
    raw = _DISPATCH[group_by](records)
    return [
        ImageGroup(
            group_key=key,
            group_by=group_by,
            image_count=len(members),
            image_ids=[r.image_id for r in members],
        )
        for key, members in sorted(raw.items())
    ]


def resolve_group(
    records: list[ImageRecord],
    group_by: GroupBy,
    group_key: str,
) -> list[ImageRecord]:
    """Return the ImageRecords that belong to a specific group key."""
    raw = _DISPATCH[group_by](records)
    return raw.get(group_key, [])
