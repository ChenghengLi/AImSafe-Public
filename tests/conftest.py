"""
Shared pytest fixtures for AImSafe tests.
"""

import sys
import os
import json
import tempfile
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def zones_file(tmp_path):
    """Create a temporary zones.json for testing."""
    zones_data = {
        "zones": [
            {
                "id": "restricted_1",
                "name": "Conveyor Belt",
                "type": "RESTRICTED",
                "color": [255, 0, 0],
                "polygon": [
                    [0.7, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.7, 1.0],
                ],
            },
            {
                "id": "machine_1",
                "name": "Press Machine",
                "type": "MACHINE_PROXIMITY",
                "color": [255, 165, 0],
                "polygon": [
                    [0.0, 0.0],
                    [0.15, 0.0],
                    [0.15, 0.5],
                    [0.0, 0.5],
                ],
            },
        ]
    }
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(zones_data))
    return str(path)


@pytest.fixture
def empty_zones_file(tmp_path):
    """Create a zones.json with no zones."""
    path = tmp_path / "zones.json"
    path.write_text(json.dumps({"zones": []}))
    return str(path)
