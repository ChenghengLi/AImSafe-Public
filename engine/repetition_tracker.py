"""
Repetition tracker — counts cyclical motions to detect strain risk.
Tracks arm/bend cycles within a rolling time window.
"""

import time
from collections import deque
import config


class RepetitionTracker:
    """Detects repetitive motion patterns from pose angles over time."""

    def __init__(self):
        self._arm_history: deque[tuple[float, float]] = deque(maxlen=1000)
        self._back_history: deque[tuple[float, float]] = deque(maxlen=1000)
        self._arm_cycles: deque[float] = deque(maxlen=200)
        self._back_cycles: deque[float] = deque(maxlen=200)
        self._arm_was_extended = False
        self._back_was_bent = False

    def update(self, arm_angle: float, back_angle: float) -> int:
        """
        Track arm extension and back bending cycles.

        Args:
            arm_angle: current arm extension angle (degrees)
            back_angle: current back angle from vertical (degrees)

        Returns:
            total repetition count in the current time window
        """
        now = time.time()
        cutoff = now - config.REPETITION_WINDOW

        # --- Arm cycle detection ---
        # A "cycle" = arm extends past 120° then returns below 90°
        if arm_angle > 120 and not self._arm_was_extended:
            self._arm_was_extended = True
        elif arm_angle < 90 and self._arm_was_extended:
            self._arm_was_extended = False
            self._arm_cycles.append(now)

        # --- Back cycle detection ---
        # A "cycle" = back bends past 30° then returns below 15°
        if back_angle > 30 and not self._back_was_bent:
            self._back_was_bent = True
        elif back_angle < 15 and self._back_was_bent:
            self._back_was_bent = False
            self._back_cycles.append(now)

        # Trim old cycles
        while self._arm_cycles and self._arm_cycles[0] < cutoff:
            self._arm_cycles.popleft()
        while self._back_cycles and self._back_cycles[0] < cutoff:
            self._back_cycles.popleft()

        return len(self._arm_cycles) + len(self._back_cycles)
