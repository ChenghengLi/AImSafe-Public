"""
Central alert dispatcher.
Collects alerts from the risk engine and routes them to UI, sound, and log.
"""

import time
from collections import deque
from engine.worker_state import Alert


class AlertManager:
    """Manages alert history and dispatching."""

    def __init__(self, max_history: int = 200):
        self.history: deque[Alert] = deque(maxlen=max_history)
        self._listeners: list[callable] = []

    def add_listener(self, callback):
        """Register a callback that fires on every new alert: callback(alert)."""
        self._listeners.append(callback)

    def process(self, alerts: list[Alert]):
        """Process a batch of alerts from one risk engine evaluation."""
        for alert in alerts:
            self.history.append(alert)
            for listener in self._listeners:
                try:
                    listener(alert)
                except Exception:
                    pass

    def get_recent(self, count: int = 20) -> list[dict]:
        """Return the N most recent alerts as dicts."""
        recent = list(self.history)[-count:]
        return [
            {
                "rule": a.rule_name,
                "severity": a.severity,
                "message": a.message,
                "time": time.strftime("%H:%M:%S", time.localtime(a.timestamp)),
            }
            for a in reversed(recent)
        ]

    def get_all(self) -> list[Alert]:
        """Return all stored alerts."""
        return list(self.history)
