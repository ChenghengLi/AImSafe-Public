"""
Simulated wearable sensor data — generates realistic heart rate and HRV.
Uses a random walk model with mean-reversion.
"""

import random
import time


class WearableSim:
    """Simulates a wrist-worn heart rate monitor."""

    def __init__(self, base_hr: float = 72, base_hrv: float = 50):
        self._base_hr = base_hr
        self._base_hrv = base_hrv
        self._hr = base_hr
        self._hrv = base_hrv
        self._stress_factor = 0.0  # 0.0 = calm, 1.0 = max stress

    def set_stress(self, factor: float):
        """Set stress factor (0-1) to simulate exertion."""
        self._stress_factor = max(0.0, min(1.0, factor))

    def update(self) -> tuple[float, float]:
        """
        Generate next heart rate and HRV sample.
        Returns (heart_rate_bpm, hrv_ms).
        """
        # Target HR increases with stress
        target_hr = self._base_hr + self._stress_factor * 50
        # Mean-revert towards target with noise
        self._hr += (target_hr - self._hr) * 0.1 + random.gauss(0, 1.5)
        self._hr = max(50, min(180, self._hr))

        # HRV decreases under stress
        target_hrv = self._base_hrv * (1 - self._stress_factor * 0.6)
        self._hrv += (target_hrv - self._hrv) * 0.1 + random.gauss(0, 2)
        self._hrv = max(10, min(100, self._hrv))

        return round(self._hr, 1), round(self._hrv, 1)
