"""
Tests for engine/risk_engine.py — rule evaluation and score fusion.
Uses mock PoseResult/FaceResult to avoid mediapipe model file dependency.
"""

import sys
import types
import enum
import pytest

# Build a comprehensive mediapipe mock that mirrors the new Tasks API
# so that importing vision modules doesn't require model files.

_mp = types.ModuleType("mediapipe")

# mp.ImageFormat
class _ImageFormat:
    SRGB = 1
_mp.ImageFormat = _ImageFormat

# mp.Image
class _Image:
    def __init__(self, **kw): pass
_mp.Image = _Image

# mp.tasks
_tasks = types.ModuleType("mediapipe.tasks")
_mp.tasks = _tasks

# mp.tasks.BaseOptions
class _BaseOptions:
    def __init__(self, **kw): pass
_tasks.BaseOptions = _BaseOptions

# mp.tasks.vision
_vision = types.ModuleType("mediapipe.tasks.vision")
_tasks.vision = _vision

class _RunningMode(enum.Enum):
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    LIVE_STREAM = "LIVE_STREAM"
_vision.RunningMode = _RunningMode

# Stub landmarkers that do nothing (tests don't call analyze())
class _StubLandmarker:
    @classmethod
    def create_from_options(cls, opts): return cls()
    def detect_for_video(self, img, ts): return type("R", (), {"pose_landmarks": [], "face_landmarks": [], "hand_landmarks": []})()
    def close(self): pass

class _PoseLandmarkerOptions:
    def __init__(self, **kw): pass
class _FaceLandmarkerOptions:
    def __init__(self, **kw): pass
_vision.PoseLandmarker = _StubLandmarker
_vision.PoseLandmarkerOptions = _PoseLandmarkerOptions
_vision.FaceLandmarker = _StubLandmarker
_vision.FaceLandmarkerOptions = _FaceLandmarkerOptions

sys.modules["mediapipe"] = _mp
sys.modules["mediapipe.tasks"] = _tasks
sys.modules["mediapipe.tasks.vision"] = _vision

from engine.risk_engine import RiskEngine
from engine.zone_manager import ZoneManager
from engine.worker_state import RiskLevel, ZoneStatus
from vision.pose_analyzer import PoseResult
from vision.face_analyzer import FaceResult


def _make_pose(visible=True, back=10, flags=None, centroid=(0.5, 0.5)):
    """Helper to create a PoseResult for testing."""
    return PoseResult(
        visible=visible,
        angles={"back": back, "left_arm_extension": 90, "right_arm_extension": 90},
        flags=flags or {},
        body_centroid=centroid,
    )


def _make_face(visible=True, blink_rate=18, microsleep=False, head_pitch=0):
    return FaceResult(
        visible=visible,
        blink_rate=blink_rate,
        microsleep=microsleep,
        head_pitch=head_pitch,
    )


class TestRiskEngineScoring:
    def test_safe_state_default(self, empty_zones_file):
        zm = ZoneManager(zones_path=empty_zones_file)
        engine = RiskEngine(zm)
        state = engine.evaluate(_make_pose(), _make_face())
        assert state.overall_safety_score >= 70
        assert state.risk_level == RiskLevel.SAFE

    def test_unsafe_lift_raises_pose_risk(self, empty_zones_file):
        zm = ZoneManager(zones_path=empty_zones_file)
        engine = RiskEngine(zm)
        pose = _make_pose(flags={"unsafe_lift": True}, back=55)
        state = engine.evaluate(pose, _make_face())
        assert state.pose_risk == 80
        assert state.overall_safety_score < 100

    def test_invisible_pose_no_risk(self, empty_zones_file):
        zm = ZoneManager(zones_path=empty_zones_file)
        engine = RiskEngine(zm)
        pose = _make_pose(visible=False)
        state = engine.evaluate(pose, _make_face())
        assert state.pose_risk == 0.0


class TestRiskEngineZones:
    def test_restricted_zone_critical_alert(self, zones_file):
        zm = ZoneManager(zones_path=zones_file)
        engine = RiskEngine(zm)
        pose = _make_pose(centroid=(0.85, 0.5))
        state = engine.evaluate(pose, _make_face())
        assert state.zone_status == ZoneStatus.RESTRICTED
        assert any(a.severity == "INFO" for a in state.active_alerts)

    def test_machine_proximity_warning(self, zones_file):
        zm = ZoneManager(zones_path=zones_file)
        engine = RiskEngine(zm)
        pose = _make_pose(centroid=(0.07, 0.25))
        state = engine.evaluate(pose, _make_face())
        assert state.zone_status == ZoneStatus.MACHINE_PROXIMITY

    def test_safe_area_no_alert(self, zones_file):
        zm = ZoneManager(zones_path=zones_file)
        engine = RiskEngine(zm)
        pose = _make_pose(centroid=(0.4, 0.5))
        state = engine.evaluate(pose, _make_face())
        assert state.zone_status == ZoneStatus.SAFE_AREA
        proximity_alerts = [a for a in state.active_alerts if a.rule_name == "proximity"]
        assert len(proximity_alerts) == 0


class TestRiskEngineFatigue:
    def test_microsleep_critical_alert(self, empty_zones_file):
        zm = ZoneManager(zones_path=empty_zones_file)
        engine = RiskEngine(zm)
        face = _make_face(microsleep=True)
        state = engine.evaluate(_make_pose(), face)
        fatigue_alerts = [a for a in state.active_alerts if a.rule_name == "fatigue"]
        assert len(fatigue_alerts) == 1
        assert fatigue_alerts[0].severity == "CRITICAL"

    def test_no_face_no_fatigue(self, empty_zones_file):
        zm = ZoneManager(zones_path=empty_zones_file)
        engine = RiskEngine(zm)
        face = _make_face(visible=False)
        state = engine.evaluate(_make_pose(), face)
        assert state.fatigue_score == 0.0


class TestRiskEngineWearable:
    def test_heart_rate_passed_through(self, empty_zones_file):
        zm = ZoneManager(zones_path=empty_zones_file)
        engine = RiskEngine(zm)
        state = engine.evaluate(_make_pose(), _make_face(),
                                heart_rate=85.0, hrv=45.0)
        assert state.heart_rate == 85.0
        assert state.hrv == 45.0


class TestRiskLevelMapping:
    def test_safe(self):
        assert RiskEngine._score_to_level(100) == RiskLevel.SAFE
        assert RiskEngine._score_to_level(70) == RiskLevel.SAFE

    def test_caution(self):
        assert RiskEngine._score_to_level(69) == RiskLevel.CAUTION
        assert RiskEngine._score_to_level(40) == RiskLevel.CAUTION

    def test_warning(self):
        assert RiskEngine._score_to_level(39) == RiskLevel.WARNING
        assert RiskEngine._score_to_level(20) == RiskLevel.WARNING

    def test_danger(self):
        assert RiskEngine._score_to_level(19) == RiskLevel.DANGER
        assert RiskEngine._score_to_level(0) == RiskLevel.DANGER
