"""
Tests for integration/mqtt_bridge.py — command generation logic.
"""

from integration.mqtt_bridge import MQTTBridge
from engine.worker_state import WorkerState, RiskLevel


class TestMQTTBridge:
    def test_danger_triggers_stop(self):
        bridge = MQTTBridge()
        state = WorkerState(risk_level=RiskLevel.DANGER)
        commands = bridge._build_commands(state)
        assert commands["factory/conveyor/zone1"]["action"] == "stop"
        assert commands["factory/lighting/zone1"]["color"] == "red"
        assert commands["factory/alert/zone1"]["level"] == "critical"

    def test_warning_triggers_slow(self):
        bridge = MQTTBridge()
        state = WorkerState(risk_level=RiskLevel.WARNING)
        commands = bridge._build_commands(state)
        assert commands["factory/conveyor/zone1"]["action"] == "slow"
        assert commands["factory/conveyor/zone1"]["speed"] == 50
        assert commands["factory/lighting/zone1"]["color"] == "yellow"

    def test_caution_yellow_light(self):
        bridge = MQTTBridge()
        state = WorkerState(risk_level=RiskLevel.CAUTION)
        commands = bridge._build_commands(state)
        assert commands["factory/lighting/zone1"]["color"] == "yellow"
        assert "factory/conveyor/zone1" not in commands

    def test_safe_green_light(self):
        bridge = MQTTBridge()
        state = WorkerState(risk_level=RiskLevel.SAFE)
        commands = bridge._build_commands(state)
        assert commands["factory/conveyor/zone1"]["action"] == "run"
        assert commands["factory/lighting/zone1"]["color"] == "green"

    def test_duplicate_commands_skipped(self):
        bridge = MQTTBridge()
        state = WorkerState(risk_level=RiskLevel.SAFE)
        bridge.publish(state)
        # Second publish with same state — should be a no-op (last_commands match)
        initial_commands = dict(bridge.last_commands)
        bridge.publish(state)
        assert bridge.last_commands == initial_commands
