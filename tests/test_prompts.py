"""
Tests for llm/prompts.py — prompt template formatting.
"""

from llm.prompts import format_worker_snapshot, format_shift_summary


class TestFormatWorkerSnapshot:
    def test_basic_formatting(self):
        state = {
            "worker_id": "worker_1",
            "overall_safety_score": 85.0,
            "risk_level": "SAFE",
            "angles": {"back": 12.3},
            "fatigue_score": 10.0,
            "blink_rate": 17.0,
            "microsleep": False,
            "head_pitch": 3.0,
            "repetition_count": 5,
            "zone_status": "SAFE_AREA",
            "current_zone_name": "",
            "heart_rate": 72.0,
            "active_alerts": [],
        }
        result = format_worker_snapshot(state)
        assert "worker_1" in result
        assert "85.0/100" in result
        assert "SAFE" in result
        assert "12.3" in result
        assert "Active Alerts: None" in result

    def test_with_alerts(self):
        state = {
            "worker_id": "w1",
            "overall_safety_score": 30.0,
            "risk_level": "WARNING",
            "angles": {"back": 55},
            "fatigue_score": 60.0,
            "blink_rate": 28.0,
            "microsleep": True,
            "head_pitch": 20.0,
            "repetition_count": 15,
            "zone_status": "RESTRICTED",
            "current_zone_name": "Conveyor",
            "heart_rate": None,
            "active_alerts": [
                {"severity": "CRITICAL", "message": "Entered restricted zone"},
            ],
        }
        result = format_worker_snapshot(state)
        assert "CRITICAL: Entered restricted zone" in result
        assert "N/A" in result  # heart_rate is None

    def test_missing_angle_key(self):
        state = {
            "worker_id": "w1",
            "overall_safety_score": 100,
            "risk_level": "SAFE",
            "angles": {},
            "fatigue_score": 0,
            "blink_rate": 0,
            "microsleep": False,
            "head_pitch": 0,
            "repetition_count": 0,
            "zone_status": "UNKNOWN",
            "current_zone_name": "",
            "heart_rate": None,
            "active_alerts": [],
        }
        result = format_worker_snapshot(state)
        assert "Back Angle: 0" in result  # .get('back', 0)


class TestFormatShiftSummary:
    def test_basic_summary(self):
        result = format_shift_summary(
            alerts=[],
            avg_safety_score=88.0,
            avg_fatigue=12.0,
            total_incidents=3,
            duration_minutes=120,
            peak_blink_rate=22.0,
            total_repetitions=45,
        )
        assert "120 minutes" in result
        assert "88/100" in result
        assert "No alerts recorded" in result

    def test_with_alerts_list(self):
        alerts = [
            {"severity": "WARNING", "time": "14:30:00", "message": "Unsafe lift"},
            {"severity": "CRITICAL", "time": "15:00:00", "message": "Zone breach"},
        ]
        result = format_shift_summary(
            alerts=alerts,
            avg_safety_score=60.0,
            avg_fatigue=35.0,
            total_incidents=5,
            duration_minutes=60,
            peak_blink_rate=30.0,
            total_repetitions=80,
        )
        assert "Unsafe lift" in result
        assert "Zone breach" in result
        assert "[WARNING]" in result
        assert "[CRITICAL]" in result
