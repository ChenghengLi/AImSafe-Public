"""
Vision-based fall detection using pose landmark analysis.
Detects rapid vertical movement, horizontal body orientation, and post-fall stillness.
"""

from collections import deque
from dataclasses import dataclass
from vision.pose_analyzer import PoseResult


@dataclass
class FallEvent:
    detected: bool = False
    confidence: float = 0.0
    fall_type: str = ""  # "rapid_descent", "horizontal", "collapse"


class FallDetector:
    """Detects falls from pose data without additional ML models."""

    def __init__(self, history_size: int = 30):
        self._centroid_history: deque[tuple[float, float]] = deque(maxlen=history_size)
        self._angle_history: deque[float] = deque(maxlen=history_size)
        self._fall_cooldown: float = -10.0  # prevent repeated alerts
        self._stillness_counter: int = 0

    def update(self, pose: PoseResult, timestamp: float) -> FallEvent:
        """Check for fall indicators in current frame."""
        event = FallEvent()

        if not pose.visible:
            return event

        centroid = pose.body_centroid
        back_angle = pose.angles.get("back", 0)

        self._centroid_history.append(centroid)
        self._angle_history.append(back_angle)

        # Cooldown: don't re-trigger within 10 seconds
        if timestamp - self._fall_cooldown < 10.0:
            return event

        # Need at least 10 frames of history
        if len(self._centroid_history) < 10:
            return event

        # --- Check 1: Rapid vertical descent ---
        # Compare current centroid to 10 frames ago
        old_centroid = self._centroid_history[-10]
        dy = centroid[1] - old_centroid[1]  # positive = moving down
        # If centroid dropped by > 0.25 (normalized) in ~10 frames (~0.33s)
        if dy > 0.25:
            event.detected = True
            event.confidence = min(dy * 2, 1.0)
            event.fall_type = "rapid_descent"
            self._fall_cooldown = timestamp
            return event

        # --- Check 2: Body nearly horizontal ---
        # Back angle > 75° means torso is nearly horizontal
        if back_angle > 75:
            # Check if this is sudden (was upright 15 frames ago)
            if len(self._angle_history) >= 15:
                old_angle = self._angle_history[-15]
                if old_angle < 40:  # was upright, now horizontal
                    event.detected = True
                    event.confidence = min((back_angle - 60) / 30, 1.0)
                    event.fall_type = "horizontal"
                    self._fall_cooldown = timestamp
                    return event

        # --- Check 3: Collapse (centroid drops + goes still) ---
        if len(self._centroid_history) >= 20:
            # Check if position has been very still for last 15 frames
            recent = list(self._centroid_history)[-15:]
            dx_range = max(c[0] for c in recent) - min(c[0] for c in recent)
            dy_range = max(c[1] for c in recent) - min(c[1] for c in recent)
            is_still = dx_range < 0.01 and dy_range < 0.01

            # And centroid is in lower half of frame (on the ground)
            is_low = centroid[1] > 0.7

            if is_still and is_low:
                self._stillness_counter += 1
                if self._stillness_counter > 30:  # ~1 second of stillness at bottom
                    event.detected = True
                    event.confidence = 0.7
                    event.fall_type = "collapse"
                    self._fall_cooldown = timestamp
                    self._stillness_counter = 0
                    return event
            else:
                self._stillness_counter = 0

        return event
