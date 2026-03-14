"""
Zone manager — loads danger zone polygons and checks worker proximity.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from engine.worker_state import ZoneStatus


@dataclass
class Zone:
    id: str
    name: str
    zone_type: ZoneStatus
    color: tuple[int, int, int]
    polygon: list[tuple[float, float]]


class ZoneManager:
    """Manages danger zone definitions and point-in-polygon checks."""

    def __init__(self, zones_path: str = "data/zones.json"):
        self.zones: list[Zone] = []
        self._load(zones_path)

    def _load(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        with open(p) as f:
            data = json.load(f)
        for z in data.get("zones", []):
            self.zones.append(Zone(
                id=z["id"],
                name=z["name"],
                zone_type=ZoneStatus[z["type"]],
                color=tuple(z["color"]),
                polygon=[(pt[0], pt[1]) for pt in z["polygon"]],
            ))

    def check(self, cx: float, cy: float) -> tuple[ZoneStatus, str]:
        """Check which zone the point (cx, cy) is in. Returns (status, zone_name)."""
        for zone in self.zones:
            if self._point_in_polygon(cx, cy, zone.polygon):
                return zone.zone_type, zone.name
        return ZoneStatus.SAFE_AREA, ""

    @staticmethod
    def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
        """Ray-casting algorithm for point-in-polygon."""
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
                inside = not inside
            j = i
        return inside
