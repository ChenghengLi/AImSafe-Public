"""
Tests for engine/zone_manager.py — zone loading and point-in-polygon checks.
"""

import pytest
from engine.zone_manager import ZoneManager, Zone
from engine.worker_state import ZoneStatus


class TestPointInPolygon:
    """Test the ray-casting point-in-polygon algorithm."""

    def test_inside_square(self):
        poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert ZoneManager._point_in_polygon(0.5, 0.5, poly) is True

    def test_outside_square(self):
        poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert ZoneManager._point_in_polygon(1.5, 0.5, poly) is False

    def test_on_edge(self):
        # Edge cases are implementation-defined; just ensure no crash
        poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        ZoneManager._point_in_polygon(0.0, 0.5, poly)  # should not raise

    def test_inside_triangle(self):
        poly = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        assert ZoneManager._point_in_polygon(0.5, 0.3, poly) is True

    def test_outside_triangle(self):
        poly = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        assert ZoneManager._point_in_polygon(0.9, 0.9, poly) is False


class TestZoneManager:
    def test_loads_zones_from_file(self, zones_file):
        zm = ZoneManager(zones_path=zones_file)
        assert len(zm.zones) == 2
        assert zm.zones[0].name == "Conveyor Belt"
        assert zm.zones[1].name == "Press Machine"

    def test_missing_file_loads_empty(self, tmp_path):
        zm = ZoneManager(zones_path=str(tmp_path / "nonexistent.json"))
        assert zm.zones == []

    def test_empty_zones(self, empty_zones_file):
        zm = ZoneManager(zones_path=empty_zones_file)
        assert zm.zones == []

    def test_check_in_restricted_zone(self, zones_file):
        zm = ZoneManager(zones_path=zones_file)
        # (0.85, 0.5) is inside the restricted zone polygon [0.7-1.0, 0.0-1.0]
        status, name = zm.check(0.85, 0.5)
        assert status == ZoneStatus.RESTRICTED
        assert name == "Conveyor Belt"

    def test_check_in_machine_proximity(self, zones_file):
        zm = ZoneManager(zones_path=zones_file)
        # (0.07, 0.25) is inside the machine zone [0.0-0.15, 0.0-0.5]
        status, name = zm.check(0.07, 0.25)
        assert status == ZoneStatus.MACHINE_PROXIMITY
        assert name == "Press Machine"

    def test_check_in_safe_area(self, zones_file):
        zm = ZoneManager(zones_path=zones_file)
        # (0.4, 0.5) is in neither zone
        status, name = zm.check(0.4, 0.5)
        assert status == ZoneStatus.SAFE_AREA
        assert name == ""

    def test_zone_types_correct(self, zones_file):
        zm = ZoneManager(zones_path=zones_file)
        assert zm.zones[0].zone_type == ZoneStatus.RESTRICTED
        assert zm.zones[1].zone_type == ZoneStatus.MACHINE_PROXIMITY

    def test_zone_colors_loaded(self, zones_file):
        zm = ZoneManager(zones_path=zones_file)
        assert zm.zones[0].color == (255, 0, 0)
        assert zm.zones[1].color == (255, 165, 0)
