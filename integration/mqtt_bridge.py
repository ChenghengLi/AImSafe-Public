"""
MQTT bridge for simulated industrial machine control.
Publishes control commands when risk events are detected.
Falls back to logging if no broker is available.
"""

import json
import logging
from engine.worker_state import WorkerState, RiskLevel
import config

logger = logging.getLogger(__name__)

# Only import paho if MQTT is enabled
_mqtt_client = None
if config.MQTT_ENABLED:
    try:
        import paho.mqtt.client as mqtt
        _mqtt_client = mqtt.Client()
        _mqtt_client.connect(config.MQTT_BROKER, config.MQTT_PORT)
        _mqtt_client.loop_start()
        logger.info(f"MQTT connected to {config.MQTT_BROKER}:{config.MQTT_PORT}")
    except Exception as e:
        logger.warning(f"MQTT connection failed: {e} — running in log-only mode")
        _mqtt_client = None


class MQTTBridge:
    """Publishes factory control commands based on worker risk state."""

    def __init__(self):
        self.last_commands: dict[str, dict] = {}

    def publish(self, state: WorkerState):
        """Evaluate state and send appropriate machine control commands."""
        commands = self._build_commands(state)

        for topic, payload in commands.items():
            # Avoid duplicate commands
            if self.last_commands.get(topic) == payload:
                continue
            self.last_commands[topic] = payload

            msg = json.dumps(payload)
            if _mqtt_client:
                _mqtt_client.publish(topic, msg)
            logger.info(f"MQTT → {topic}: {msg}")

    def _build_commands(self, state: WorkerState) -> dict[str, dict]:
        commands = {}

        if state.risk_level == RiskLevel.DANGER:
            commands["factory/conveyor/zone1"] = {"action": "stop"}
            commands["factory/lighting/zone1"] = {"action": "set", "color": "red", "brightness": 100}
            commands["factory/alert/zone1"] = {"action": "alarm", "level": "critical"}

        elif state.risk_level == RiskLevel.WARNING:
            commands["factory/conveyor/zone1"] = {"action": "slow", "speed": 50}
            commands["factory/lighting/zone1"] = {"action": "set", "color": "yellow", "brightness": 80}
            commands["factory/alert/zone1"] = {"action": "alarm", "level": "warning"}

        elif state.risk_level == RiskLevel.CAUTION:
            commands["factory/lighting/zone1"] = {"action": "set", "color": "yellow", "brightness": 60}

        else:
            commands["factory/conveyor/zone1"] = {"action": "run", "speed": 100}
            commands["factory/lighting/zone1"] = {"action": "set", "color": "green", "brightness": 40}

        return commands
