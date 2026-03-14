"""
Fatigue tracker — rolling-window analysis of drowsiness signals.
Uses PERCLOS (percentage eye closure) as primary drowsiness indicator.
Yawning and stress contribute as secondary signals.

Calibration (March 2026):
- PERCLOS computed over 30s window (900 frames) in face_analyzer
- Drowsy threshold: 20% closure (up from 15%)
- EAR threshold: 0.21 (up from 0.18)
- Microsleep: 2.0s sustained closure (up from 1.5s)
- Fatigue contribution only starts when PERCLOS > 0.20
"""

import time
from collections import deque
import config


class FatigueTracker:
    """Tracks fatigue signals over a rolling window with smooth transitions.

    Calibration notes:
    - PERCLOS < 0.20 = normal (no contribution)
    - PERCLOS 0.20-0.35 = mild drowsiness (gradual contribution)
    - PERCLOS 0.35-0.50 = significant drowsiness
    - PERCLOS > 0.50 = severe → triggers incident capture
    - Yawning adds a small, sustained contribution
    - Microsleep only fires after 2.0s sustained closure with pitch gating
    """

    def __init__(self, window_seconds: float = 300):
        self._window = window_seconds
        self._velocity_history: deque[tuple[float, float]] = deque(maxlen=500)
        self._prev_centroid: tuple[float, float] | None = None
        self._prev_time: float = 0.0
        self._baseline_velocity: float | None = None
        # Smoothed fatigue score (decays gradually)
        self._smoothed_score: float = 0.0
        # Time-based drowsiness accumulation
        self._drowsy_accumulated: float = 0.0
        self._drowsy_start: float = 0.0
        self._drowsy_active: bool = False
        # Microsleep buildup (separate from PERCLOS)
        self._microsleep_accumulated: float = 0.0
        # Yawn accumulation
        self._yawn_accumulated: float = 0.0

    def update(
        self,
        blink_rate: float,
        microsleep: bool,
        head_pitch: float,
        centroid: tuple[float, float],
        perclos: float = 0.0,
        drowsy: bool = False,
        yawn_detected: bool = False,
        stress_level: float = 0.0,
    ) -> float:
        """
        Compute fatigue score (0-100) with gradual transitions.
        Calibrated to be less aggressive — only real drowsiness triggers high scores.
        """
        now = time.time()
        raw_score = 0.0

        # 1. PERCLOS contribution (0-30 points) — primary signal
        #    Only starts contributing above 0.20 (normal blink closure is ~0.05-0.12)
        #    With 30s window, PERCLOS is stable — 0.20 means genuinely drowsy
        #    Scaled so 0.20→0, 0.30→5, 0.40→10, 0.50→15, 0.70→25, 0.80→30 (capped)
        if perclos > 0.20:
            perclos_excess = perclos - 0.20
            raw_score += min(perclos_excess * 50, 30)

        # 2. Time-based drowsiness accumulation (0-20 points)
        #    Only kicks in when drowsy=True (PERCLOS > 20%)
        #    Gentler growth — takes longer to build up since drowsy threshold is stricter
        if drowsy:
            if not self._drowsy_active:
                self._drowsy_active = True
                self._drowsy_start = now
            duration = now - self._drowsy_start
            # Gentle growth: 0.3 pt/frame base, up to 1.5 pt/frame after 20s
            growth_rate = min(0.3 + duration * 0.06, 1.5)
            self._drowsy_accumulated = min(self._drowsy_accumulated + growth_rate, 20)
        else:
            self._drowsy_active = False
            self._drowsy_start = 0.0
            # Faster decay when not drowsy (0.8 pts/frame)
            self._drowsy_accumulated = max(self._drowsy_accumulated - 0.8, 0)
        raw_score += self._drowsy_accumulated

        # 3. Microsleep contribution — gradual buildup (0-25 max)
        #    Only fires after 2.0s sustained closure with pitch gating + blendshape confirmation
        if microsleep:
            self._microsleep_accumulated = min(self._microsleep_accumulated + 2, 25)
        else:
            self._microsleep_accumulated = max(self._microsleep_accumulated - 2, 0)
        raw_score += self._microsleep_accumulated

        # 4. Yawn contribution (0-10 points) — bostezos / yawning
        #    Each yawn adds 3 points, decays at 0.2/frame
        #    Frequent yawning builds up and signals fatigue
        if yawn_detected:
            self._yawn_accumulated = min(self._yawn_accumulated + 3, 10)
        else:
            self._yawn_accumulated = max(self._yawn_accumulated - 0.2, 0)
        raw_score += self._yawn_accumulated

        # 5. Stress contribution (0-5 points) — minor, but sustained stress = fatigue
        raw_score += stress_level * 5

        # 6. Head nod / droop contribution (0-10 points) — reduced
        if abs(head_pitch) > config.HEAD_NOD_ANGLE:
            raw_score += min(abs(head_pitch) - config.HEAD_NOD_ANGLE, 10)

        # 7. Movement velocity slowdown (0-8 points)
        if self._prev_centroid and (now - self._prev_time) > 0:
            dt = now - self._prev_time
            dx = centroid[0] - self._prev_centroid[0]
            dy = centroid[1] - self._prev_centroid[1]
            velocity = (dx ** 2 + dy ** 2) ** 0.5 / dt
            self._velocity_history.append((now, velocity))

            if self._baseline_velocity is None and len(self._velocity_history) > 30:
                self._baseline_velocity = sum(v for _, v in self._velocity_history) / len(self._velocity_history)

            if self._baseline_velocity and self._baseline_velocity > 0:
                cutoff = now - self._window
                while self._velocity_history and self._velocity_history[0][0] < cutoff:
                    self._velocity_history.popleft()
                if self._velocity_history:
                    avg_recent = sum(v for _, v in self._velocity_history) / len(self._velocity_history)
                    slowdown = max(0, 1 - avg_recent / self._baseline_velocity)
                    raw_score += slowdown * 8

        self._prev_centroid = centroid
        self._prev_time = now

        # Smooth the score — rise moderately, decay faster
        target = min(raw_score, 100)
        if target > self._smoothed_score:
            self._smoothed_score += (target - self._smoothed_score) * 0.15  # moderate rise
        else:
            self._smoothed_score += (target - self._smoothed_score) * 0.08  # faster decay

        return min(self._smoothed_score, 100)
