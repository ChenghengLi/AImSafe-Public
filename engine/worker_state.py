"""
Unified worker state model.
Fuses all signals (pose, fatigue, proximity, repetition, wearable) into one snapshot.
"""

from dataclasses import dataclass, field
from enum import Enum
import time


class RiskLevel(Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    DANGER = "DANGER"


class ZoneStatus(Enum):
    SAFE_AREA = "SAFE_AREA"
    MACHINE_PROXIMITY = "MACHINE_PROXIMITY"
    RESTRICTED = "RESTRICTED"
    UNKNOWN = "UNKNOWN"


@dataclass
class Alert:
    rule_name: str
    severity: str          # INFO, WARNING, CRITICAL
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkerState:
    """Complete snapshot of a worker's safety status at a point in time."""
    worker_id: str = "worker_1"
    timestamp: float = field(default_factory=time.time)

    # Pose
    pose_risk: float = 0.0
    angles: dict[str, float] = field(default_factory=dict)
    pose_flags: dict[str, bool] = field(default_factory=dict)
    body_centroid: tuple[float, float] = (0.0, 0.0)
    reba_score: int = 0

    # Fatigue
    fatigue_score: float = 0.0
    blink_rate: float = 0.0
    microsleep: bool = False
    head_pitch: float = 0.0
    perclos: float = 0.0
    drowsy: bool = False
    stress_level: float = 0.0
    yawn_detected: bool = False
    gaze_focus: float = 1.0

    # Repetition
    repetition_count: int = 0

    # Zone
    zone_status: ZoneStatus = ZoneStatus.UNKNOWN
    current_zone_name: str = ""

    # Wearable
    heart_rate: float | None = None
    hrv: float | None = None

    # Composite
    overall_safety_score: float = 100.0
    risk_level: RiskLevel = RiskLevel.SAFE
    active_alerts: list[Alert] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for WebSocket / JSON transport."""
        return {
            "worker_id": self.worker_id,
            "timestamp": self.timestamp,
            "pose_risk": round(self.pose_risk, 1),
            "angles": {k: round(v, 1) for k, v in self.angles.items()},
            "pose_flags": self.pose_flags,
            "body_centroid": self.body_centroid,
            "fatigue_score": round(self.fatigue_score, 1),
            "blink_rate": round(self.blink_rate, 1),
            "microsleep": self.microsleep,
            "head_pitch": round(self.head_pitch, 1),
            "repetition_count": self.repetition_count,
            "zone_status": self.zone_status.value,
            "current_zone_name": self.current_zone_name,
            "heart_rate": round(self.heart_rate, 1) if self.heart_rate else None,
            "hrv": round(self.hrv, 1) if self.hrv else None,
            "overall_safety_score": round(self.overall_safety_score, 1),
            "risk_level": self.risk_level.value,
            "active_alerts": [
                {"rule": a.rule_name, "severity": a.severity, "message": a.message}
                for a in self.active_alerts
            ],
        }
