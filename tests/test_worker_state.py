"""
Tests for engine/worker_state.py — WorkerState dataclass and serialization.
"""

from engine.worker_state import WorkerState, RiskLevel, ZoneStatus, Alert


class TestAlert:
    def test_alert_creation(self):
        a = Alert(rule_name="unsafe_lift", severity="WARNING", message="Bad posture")
        assert a.rule_name == "unsafe_lift"
        assert a.severity == "WARNING"
        assert a.message == "Bad posture"
        assert a.timestamp > 0

    def test_alert_custom_timestamp(self):
        a = Alert(rule_name="test", severity="INFO", message="msg", timestamp=123.0)
        assert a.timestamp == 123.0


class TestWorkerState:
    def test_defaults(self):
        ws = WorkerState()
        assert ws.worker_id == "worker_1"
        assert ws.pose_risk == 0.0
        assert ws.fatigue_score == 0.0
        assert ws.repetition_count == 0
        assert ws.zone_status == ZoneStatus.UNKNOWN
        assert ws.heart_rate is None
        assert ws.overall_safety_score == 100.0
        assert ws.risk_level == RiskLevel.SAFE
        assert ws.active_alerts == []

    def test_to_dict_keys(self):
        ws = WorkerState()
        d = ws.to_dict()
        expected_keys = {
            "worker_id", "timestamp", "pose_risk", "angles", "pose_flags",
            "body_centroid", "fatigue_score", "blink_rate", "microsleep",
            "head_pitch", "repetition_count", "zone_status",
            "current_zone_name", "heart_rate", "hrv",
            "overall_safety_score", "risk_level", "active_alerts",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_rounded(self):
        ws = WorkerState(pose_risk=72.456, fatigue_score=33.789)
        d = ws.to_dict()
        assert d["pose_risk"] == 72.5
        assert d["fatigue_score"] == 33.8

    def test_to_dict_with_alerts(self):
        ws = WorkerState()
        ws.active_alerts.append(
            Alert(rule_name="test_rule", severity="CRITICAL", message="test msg")
        )
        d = ws.to_dict()
        assert len(d["active_alerts"]) == 1
        assert d["active_alerts"][0]["rule"] == "test_rule"
        assert d["active_alerts"][0]["severity"] == "CRITICAL"

    def test_to_dict_enum_serialization(self):
        ws = WorkerState(
            zone_status=ZoneStatus.RESTRICTED,
            risk_level=RiskLevel.DANGER,
        )
        d = ws.to_dict()
        assert d["zone_status"] == "RESTRICTED"
        assert d["risk_level"] == "DANGER"

    def test_to_dict_heart_rate_none(self):
        ws = WorkerState(heart_rate=None, hrv=None)
        d = ws.to_dict()
        assert d["heart_rate"] is None
        assert d["hrv"] is None

    def test_to_dict_heart_rate_present(self):
        ws = WorkerState(heart_rate=75.123, hrv=42.567)
        d = ws.to_dict()
        assert d["heart_rate"] == 75.1
        assert d["hrv"] == 42.6


class TestRiskLevel:
    def test_all_values(self):
        assert RiskLevel.SAFE.value == "SAFE"
        assert RiskLevel.CAUTION.value == "CAUTION"
        assert RiskLevel.WARNING.value == "WARNING"
        assert RiskLevel.DANGER.value == "DANGER"


class TestZoneStatus:
    def test_all_values(self):
        assert ZoneStatus.SAFE_AREA.value == "SAFE_AREA"
        assert ZoneStatus.MACHINE_PROXIMITY.value == "MACHINE_PROXIMITY"
        assert ZoneStatus.RESTRICTED.value == "RESTRICTED"
        assert ZoneStatus.UNKNOWN.value == "UNKNOWN"
