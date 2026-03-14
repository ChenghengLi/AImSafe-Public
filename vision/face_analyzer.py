"""
Face analysis using MediaPipe FaceLandmarker (Tasks API).
Detects blinks (Eye Aspect Ratio), drowsiness via PERCLOS, and head pose.

Reliability improvements (March 2026):
- Adaptive EAR baseline: learns each person's open-eye EAR and uses 75% of it as threshold
- Soft PERCLOS: uses continuous closure degree (not just binary) for smoother signal
- Noise-tolerant microsleep: allows up to 2 glitch frames without resetting closure
- Blendshape lookup by name instead of fragile index
- Relaxed pitch gating for microsleep (30° instead of 20°) — people nod off head-down
"""

from dataclasses import dataclass
import math
import os
import time
from collections import deque
import numpy as np
import mediapipe as mp
import config

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "face_landmarker.task")


def _load_model_buffer(path: str) -> bytes:
    """Load model file as bytes to bypass MediaPipe's broken Windows path resolution."""
    with open(path, "rb") as f:
        return f.read()

# PERCLOS: percentage of eye closure over a sliding window
_PERCLOS_WINDOW = 900       # frames (~30s at 30fps) — stable window
_PERCLOS_DROWSY = 0.20      # 20% closure = drowsy threshold
_PERCLOS_MIN_FRAMES = 90    # need ~3s of data before computing
_EAR_EMA_ALPHA = 0.25       # smoothing factor (lower = smoother, less flicker)
_PITCH_GATE_ANGLE = 20.0    # suppress normal closure detection when looking down
_PITCH_GATE_MICROSLEEP = 35.0  # relaxed gate for microsleep (people nod off head-down)

# Adaptive baseline
_BASELINE_WARMUP = 60       # frames to collect before establishing baseline
_BASELINE_THRESHOLD_RATIO = 0.72  # closed = EAR < 72% of personal baseline


@dataclass
class FaceResult:
    """Output of a single frame's face analysis."""
    visible: bool = False
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_avg: float = 0.0
    ear_smoothed: float = 0.0
    blink_detected: bool = False
    blink_rate: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    microsleep: bool = False
    perclos: float = 0.0
    drowsy: bool = False
    stress_level: float = 0.0
    yawn_detected: bool = False
    gaze_x: float = 0.5
    gaze_focus: float = 1.0


class FaceAnalyzer:
    """Wraps MediaPipe FaceLandmarker for fatigue-relevant metrics."""

    # Eye landmark indices for EAR (FaceLandmarker 478 landmarks)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_CORNER = 263
    RIGHT_EYE_CORNER = 33
    FOREHEAD = 10

    def __init__(self):
        model_buffer = _load_model_buffer(MODEL_PATH)
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_buffer),
            running_mode=VisionRunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._frame_ts = 0

        # EAR EMA state
        self._ear_ema: float = 0.3  # start with a normal open-eye value

        # Adaptive EAR baseline — learns the person's open-eye EAR
        self._ear_baseline_samples: list[float] = []
        self._ear_baseline: float | None = None  # None until calibrated
        self._ear_threshold: float = config.EAR_THRESHOLD  # fallback until calibrated

        # Blink tracking state
        self._eye_closed = False
        self._close_start: float = 0.0
        self._close_first_frame: float = 0.0  # when closure actually began (before confirmation)
        self._blink_times: deque[float] = deque(maxlen=100)

        # Noise-tolerant closure tracking
        # Instead of strict consecutive counting, use a sliding window of recent frames
        self._recent_closure: deque[bool] = deque(maxlen=8)  # last 8 frames (~260ms)
        self._CLOSED_RATIO_ENTER = 0.75  # 6/8 frames closed = eyes are closed
        self._CLOSED_RATIO_EXIT = 0.25   # 2/8 frames closed = eyes opened
        self._GLITCH_TOLERANCE = 2  # allow up to 2 open frames during a sustained closure

        # PERCLOS reset: track consecutive open-eye frames
        self._consecutive_open: int = 0
        self._OPEN_RESET_FRAMES = 150  # ~5s of eyes open → reset PERCLOS window

        # PERCLOS sliding window — stores continuous closure degree, not just binary
        self._closure_window: deque[float] = deque(maxlen=_PERCLOS_WINDOW)

        # Stress detection EMA state
        self._stress_ema: float = 0.0

        # Gaze tracking history
        self._gaze_history: deque[float] = deque(maxlen=90)

    def analyze(self, frame_rgb: np.ndarray) -> FaceResult:
        """Process one RGB frame and return face analysis."""
        result = FaceResult()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_ts += 33
        detection = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        if not detection.face_landmarks or len(detection.face_landmarks) == 0:
            return result

        lm = detection.face_landmarks[0]
        result.visible = True

        # --- Head pose (compute early — needed for pitch gating) ---
        result.head_pitch = self._estimate_pitch(lm)
        result.head_yaw = self._estimate_yaw(lm)

        # --- Eye Aspect Ratio ---
        result.ear_left = self._compute_ear(lm, self.LEFT_EYE)
        result.ear_right = self._compute_ear(lm, self.RIGHT_EYE)
        result.ear_avg = (result.ear_left + result.ear_right) / 2

        # --- EAR EMA smoothing ---
        self._ear_ema = _EAR_EMA_ALPHA * result.ear_avg + (1 - _EAR_EMA_ALPHA) * self._ear_ema
        result.ear_smoothed = self._ear_ema

        # --- Adaptive baseline calibration ---
        # Collect open-eye samples during the first few seconds (when not pitch-gated)
        if self._ear_baseline is None and abs(result.head_pitch) < 15:
            self._ear_baseline_samples.append(result.ear_avg)
            if len(self._ear_baseline_samples) >= _BASELINE_WARMUP:
                # Use the 75th percentile as baseline (robust to blinks during warmup)
                sorted_samples = sorted(self._ear_baseline_samples)
                p75_idx = int(len(sorted_samples) * 0.75)
                self._ear_baseline = sorted_samples[p75_idx]
                self._ear_threshold = self._ear_baseline * _BASELINE_THRESHOLD_RATIO
                # Clamp to reasonable range
                self._ear_threshold = max(0.15, min(0.28, self._ear_threshold))

        # --- Blendshape confirmation (by name, not index) ---
        blink_bs_score = 0.0
        has_blendshapes = detection.face_blendshapes and len(detection.face_blendshapes) > 0
        if has_blendshapes:
            bs_dict = {bs.category_name: bs.score for bs in detection.face_blendshapes[0]}
            blink_left = bs_dict.get('eyeBlinkLeft', 0.0)
            blink_right = bs_dict.get('eyeBlinkRight', 0.0)
            blink_bs_score = (blink_left + blink_right) / 2
            result.stress_level = self._compute_stress_from_dict(bs_dict)
            result.yawn_detected = bs_dict.get('jawOpen', 0.0) > 0.6

        # Gaze tracking (if iris landmarks available)
        if len(lm) > 473:
            result.gaze_x, result.gaze_focus = self._estimate_gaze(lm)

        # --- Determine eye closure ---
        # Two independent signals: geometric EAR and neural blendshape
        ear_says_closed = self._ear_ema < self._ear_threshold
        bs_says_closed = blink_bs_score > 0.4  # blendshape thinks eyes closed

        # Fusion: either signal can detect closure, but for highest confidence need both
        if has_blendshapes:
            # With blendshapes: closed if BOTH agree, OR if blendshape is very confident
            eyes_closed_now = (ear_says_closed and bs_says_closed) or blink_bs_score > 0.7
        else:
            # Without blendshapes: rely on EAR alone
            eyes_closed_now = ear_says_closed

        # --- Pitch gating for normal closure detection ---
        pitch_gated = abs(result.head_pitch) > _PITCH_GATE_ANGLE
        if pitch_gated:
            eyes_closed_now = False

        # --- Soft PERCLOS: continuous closure degree ---
        # Instead of binary, compute how closed (0.0 = open, 1.0 = fully shut)
        if eyes_closed_now:
            # How far below threshold — deeper closure = higher weight
            if self._ear_ema > 0:
                closure_depth = max(0.0, 1.0 - self._ear_ema / self._ear_threshold)
                closure_degree = min(1.0, 0.5 + closure_depth)  # at least 0.5 if classified closed
            else:
                closure_degree = 1.0
        else:
            closure_degree = 0.0

        self._closure_window.append(closure_degree)

        # PERCLOS reset: if eyes stay open for ~5s, flush old closure data
        if not eyes_closed_now:
            self._consecutive_open += 1
            if self._consecutive_open >= self._OPEN_RESET_FRAMES:
                # Eyes have been solidly open — clear old closure history
                self._closure_window.clear()
                self._consecutive_open = 0
        else:
            self._consecutive_open = 0

        if len(self._closure_window) >= _PERCLOS_MIN_FRAMES:
            # PERCLOS = average closure degree over window (not just binary ratio)
            result.perclos = sum(self._closure_window) / len(self._closure_window)
            result.drowsy = result.perclos > _PERCLOS_DROWSY

        # --- Noise-tolerant blink/closure tracking ---
        self._recent_closure.append(eyes_closed_now)
        now = time.time()

        if len(self._recent_closure) >= 4:
            closed_ratio = sum(self._recent_closure) / len(self._recent_closure)

            if not self._eye_closed and closed_ratio >= self._CLOSED_RATIO_ENTER:
                # Eyes just closed — backdate the start to when closure actually began
                self._eye_closed = True
                # Estimate when closure started: ~N frames ago where N = consecutive closed
                frames_back = 0
                for v in reversed(self._recent_closure):
                    if v:
                        frames_back += 1
                    else:
                        break
                self._close_start = now - (frames_back * 0.033)  # ~33ms per frame
                self._close_first_frame = self._close_start

            elif self._eye_closed and closed_ratio <= self._CLOSED_RATIO_EXIT:
                # Eyes opened — end the closure event
                self._eye_closed = False
                close_duration = now - self._close_start
                if 0.05 < close_duration < 0.5:
                    result.blink_detected = True
                    self._blink_times.append(now)
                self._close_start = 0.0
                self._close_first_frame = 0.0

        # --- Microsleep detection ---
        # Uses relaxed pitch gating (people nod off with head down)
        microsleep_pitch_ok = abs(result.head_pitch) < _PITCH_GATE_MICROSLEEP
        if self._eye_closed and self._close_start > 0 and microsleep_pitch_ok:
            closure_duration = now - self._close_start
            if closure_duration > config.MICROSLEEP_DURATION:
                result.microsleep = True

        # --- Blink rate (blinks/minute over last 60s) ---
        cutoff = now - 60
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        result.blink_rate = len(self._blink_times)

        return result

    @staticmethod
    def _compute_ear(lm, indices: list[int]) -> float:
        """Eye Aspect Ratio — ratio of vertical to horizontal eye opening."""
        def dist(a, b):
            return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

        p = [lm[i] for i in indices]
        vertical_1 = dist(p[1], p[5])
        vertical_2 = dist(p[2], p[4])
        horizontal = dist(p[0], p[3])
        return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-8)

    def _estimate_pitch(self, lm) -> float:
        nose = lm[self.NOSE_TIP]
        forehead = lm[self.FOREHEAD]
        chin = lm[self.CHIN]
        face_height = abs(forehead.y - chin.y) + 1e-8
        nose_ratio = (nose.y - forehead.y) / face_height
        return (nose_ratio - 0.45) * 90

    def _estimate_yaw(self, lm) -> float:
        left = lm[self.LEFT_EYE_CORNER]
        right = lm[self.RIGHT_EYE_CORNER]
        nose = lm[self.NOSE_TIP]
        face_width = abs(right.x - left.x) + 1e-8
        nose_ratio = (nose.x - left.x) / face_width
        return (nose_ratio - 0.5) * 90

    def _compute_stress_from_dict(self, bs_dict: dict[str, float]) -> float:
        """Compute stress level from blendshape scores dict (0.0-1.0)."""
        brow_tension = (bs_dict.get('browDownLeft', 0) + bs_dict.get('browDownRight', 0)) / 2
        worry = bs_dict.get('browInnerUp', 0)
        frown = (bs_dict.get('mouthFrownLeft', 0) + bs_dict.get('mouthFrownRight', 0)) / 2
        squint = (bs_dict.get('eyeSquintLeft', 0) + bs_dict.get('eyeSquintRight', 0)) / 2

        raw = brow_tension * 0.3 + worry * 0.2 + frown * 0.25 + squint * 0.25
        self._stress_ema = 0.3 * raw + 0.7 * self._stress_ema
        return min(self._stress_ema, 1.0)

    def _compute_stress(self, blendshapes) -> float:
        """Legacy: compute stress from blendshape list."""
        scores = {bs.category_name: bs.score for bs in blendshapes}
        return self._compute_stress_from_dict(scores)

    def _check_yawn(self, blendshapes) -> bool:
        """Detect yawning from jawOpen blendshape."""
        for bs in blendshapes:
            if bs.category_name == 'jawOpen' and bs.score > 0.6:
                return True
        return False

    def _estimate_gaze(self, lm) -> tuple[float, float]:
        """Estimate horizontal gaze direction and focus score."""
        left_iris = lm[468]
        left_inner = lm[263]
        left_outer = lm[362]
        eye_width = abs(left_outer.x - left_inner.x) + 1e-8
        gaze_ratio = (left_iris.x - left_inner.x) / eye_width

        self._gaze_history.append(gaze_ratio)
        if len(self._gaze_history) >= 10:
            variance = np.var(list(self._gaze_history))
            focus = max(0.0, 1.0 - variance * 50)
        else:
            focus = 1.0

        return gaze_ratio, focus

    def close(self):
        self._landmarker.close()
