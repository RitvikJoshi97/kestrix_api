"""Tests for grouping.py — pure grouping functions."""

from __future__ import annotations

import pytest

from thermal_palette.grouping import (
    azimuth_to_compass,
    get_groups,
    group_by_azimuth,
    group_by_date,
    group_by_flight,
    group_by_property,
    group_by_property_azimuth,
    group_by_property_flight,
    resolve_group,
)
from thermal_palette.models import CompassDirection, GroupBy


class TestAzimuthToCompass:
    @pytest.mark.parametrize(
        "azimuth, expected",
        [
            (0.0, CompassDirection.N),
            (2.0, CompassDirection.N),
            (22.4, CompassDirection.N),     # just inside N bucket
            (45.0, CompassDirection.NE),    # boundary → NE
            (88.5, CompassDirection.E),
            (90.0, CompassDirection.E),
            (135.0, CompassDirection.SE),
            (178.0, CompassDirection.S),
            (180.0, CompassDirection.S),
            (225.0, CompassDirection.SW),
            (270.0, CompassDirection.W),
            (315.0, CompassDirection.NW),
            (337.4, CompassDirection.NW),   # last degree of NW
            (337.5, CompassDirection.N),    # wraps back to N
            # Negative azimuths (west of north convention used in the data)
            (-91.2, CompassDirection.W),
            (-178.3, CompassDirection.S),
            (-45.0, CompassDirection.NW),
            (-1.0, CompassDirection.N),
        ],
    )
    def test_bucketing(self, azimuth, expected):
        assert azimuth_to_compass(azimuth) == expected


class TestGroupByProperty:
    def test_groups_by_property_id(self, sample_records):
        groups = group_by_property(sample_records)
        assert "prop-111" in groups
        assert "prop-222" in groups
        assert len(groups["prop-111"]) == 2
        assert len(groups["prop-222"]) == 2

    def test_every_record_in_exactly_one_group(self, sample_records):
        groups = group_by_property(sample_records)
        total = sum(len(v) for v in groups.values())
        assert total == len(sample_records)


class TestGroupByFlight:
    def test_excludes_null_flight_by_default(self, sample_records):
        groups = group_by_flight(sample_records)
        assert "no_flight" not in groups
        # img-003 and img-004 have flight_id=None, should be absent
        all_ids = {r.image_id for grp in groups.values() for r in grp}
        assert "img-003" not in all_ids
        assert "img-004" not in all_ids

    def test_includes_no_flight_when_requested(self, sample_records):
        groups = group_by_flight(sample_records, include_no_flight=True)
        assert "no_flight" in groups
        assert len(groups["no_flight"]) == 2

    def test_flight_groups_correct(self, sample_records):
        groups = group_by_flight(sample_records)
        assert "flight-aaa" in groups
        assert len(groups["flight-aaa"]) == 2


class TestGroupByAzimuth:
    def test_correct_compass_keys(self, sample_records):
        groups = group_by_azimuth(sample_records)
        assert CompassDirection.N.value in groups   # img-001
        assert CompassDirection.E.value in groups   # img-002
        assert CompassDirection.W.value in groups   # img-003
        assert CompassDirection.S.value in groups   # img-004
        assert CompassDirection.NE.value in groups  # img-005

    def test_every_record_present(self, sample_records):
        groups = group_by_azimuth(sample_records)
        total = sum(len(v) for v in groups.values())
        assert total == len(sample_records)


class TestGroupByDate:
    def test_groups_by_date_string(self, sample_records):
        groups = group_by_date(sample_records)
        assert "2025-03-25" in groups
        assert len(groups["2025-03-25"]) == len(sample_records)


class TestGroupByPropertyAzimuth:
    def test_composite_key_format(self, sample_records):
        groups = group_by_property_azimuth(sample_records)
        # img-001 (prop-111, N)
        assert "prop-111::N" in groups
        # img-003 (prop-222, W)
        assert "prop-222::W" in groups

    def test_every_record_present(self, sample_records):
        groups = group_by_property_azimuth(sample_records)
        total = sum(len(v) for v in groups.values())
        assert total == len(sample_records)


class TestGroupByPropertyFlight:
    def test_only_records_with_flight_id(self, sample_records):
        groups = group_by_property_flight(sample_records)
        total = sum(len(v) for v in groups.values())
        # 3 records have a flight_id (img-001, img-002, img-005)
        assert total == 3

    def test_composite_key_format(self, sample_records):
        groups = group_by_property_flight(sample_records)
        assert "prop-111::flight-aaa" in groups


class TestGetGroups:
    def test_returns_image_group_objects(self, sample_records):
        from thermal_palette.models import ImageGroup

        groups = get_groups(sample_records, GroupBy.PROPERTY)
        assert all(isinstance(g, ImageGroup) for g in groups)

    def test_image_ids_populated(self, sample_records):
        groups = get_groups(sample_records, GroupBy.AZIMUTH)
        for g in groups:
            assert len(g.image_ids) == g.image_count


class TestResolveGroup:
    def test_returns_matching_records(self, sample_records):
        records = resolve_group(sample_records, GroupBy.PROPERTY, "prop-111")
        assert len(records) == 2
        assert all(r.property_id == "prop-111" for r in records)

    def test_returns_empty_for_unknown_key(self, sample_records):
        records = resolve_group(sample_records, GroupBy.PROPERTY, "nonexistent")
        assert records == []

    def test_azimuth_group_resolve(self, sample_records):
        records = resolve_group(sample_records, GroupBy.AZIMUTH, "N")
        assert any(r.image_id == "img-001" for r in records)
