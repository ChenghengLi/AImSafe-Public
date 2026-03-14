"""
Tests for alerts/alert_manager.py — alert dispatching and history.
"""

from alerts.alert_manager import AlertManager
from engine.worker_state import Alert


class TestAlertManager:
    def test_empty_on_init(self):
        am = AlertManager()
        assert len(am.history) == 0
        assert am.get_recent() == []

    def test_process_adds_to_history(self):
        am = AlertManager()
        alerts = [
            Alert(rule_name="test", severity="WARNING", message="msg1"),
            Alert(rule_name="test2", severity="INFO", message="msg2"),
        ]
        am.process(alerts)
        assert len(am.history) == 2

    def test_get_recent_returns_newest_first(self):
        am = AlertManager()
        am.process([Alert(rule_name="first", severity="INFO", message="1st")])
        am.process([Alert(rule_name="second", severity="INFO", message="2nd")])
        recent = am.get_recent(2)
        assert recent[0]["rule"] == "second"
        assert recent[1]["rule"] == "first"

    def test_get_recent_limits_count(self):
        am = AlertManager()
        for i in range(10):
            am.process([Alert(rule_name=f"r{i}", severity="INFO", message=f"msg{i}")])
        recent = am.get_recent(3)
        assert len(recent) == 3

    def test_max_history_respected(self):
        am = AlertManager(max_history=5)
        for i in range(10):
            am.process([Alert(rule_name=f"r{i}", severity="INFO", message=f"msg{i}")])
        assert len(am.history) == 5

    def test_listener_called(self):
        am = AlertManager()
        received = []
        am.add_listener(lambda a: received.append(a))
        am.process([Alert(rule_name="test", severity="WARNING", message="msg")])
        assert len(received) == 1
        assert received[0].rule_name == "test"

    def test_listener_exception_doesnt_crash(self):
        am = AlertManager()

        def bad_listener(a):
            raise ValueError("boom")

        am.add_listener(bad_listener)
        # Should not raise
        am.process([Alert(rule_name="test", severity="INFO", message="msg")])
        assert len(am.history) == 1

    def test_get_all(self):
        am = AlertManager()
        am.process([
            Alert(rule_name="a", severity="INFO", message="1"),
            Alert(rule_name="b", severity="WARNING", message="2"),
        ])
        all_alerts = am.get_all()
        assert len(all_alerts) == 2
        assert all_alerts[0].rule_name == "a"

    def test_get_recent_has_time_field(self):
        am = AlertManager()
        am.process([Alert(rule_name="test", severity="INFO", message="msg")])
        recent = am.get_recent()
        assert "time" in recent[0]
        assert ":" in recent[0]["time"]  # formatted as HH:MM:SS
