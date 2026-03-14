"""
Hand analysis using MediaPipe HandLandmarker (Tasks API).
Detects hand positions relevant to factory safety.
"""

from dataclasses import dataclass, field
import os
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "hand_landmarker.task")


def _load_model_buffer(path: str) -> bytes:
    """Load model file as bytes to bypass MediaPipe's broken Windows path resolution."""
    with open(path, "rb") as f:
        return f.read()


@dataclass
class HandResult:
    """Output of a single frame's hand analysis."""
    visible: bool = False
    num_hands: int = 0
    hand_positions: list[tuple[float, float]] = field(default_factory=list)
    landmarks: list = field(default_factory=list)


class HandAnalyzer:
    """Wraps MediaPipe HandLandmarker for hand position tracking."""

    WRIST = 0
    MIDDLE_FINGER_MCP = 9

    def __init__(self, max_hands: int = 2):
        model_buffer = _load_model_buffer(MODEL_PATH)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_buffer),
            running_mode=VisionRunningMode.VIDEO,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            num_hands=max_hands,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_ts = 0

    def analyze(self, frame_rgb: np.ndarray) -> HandResult:
        """Process one RGB frame and return hand analysis."""
        result = HandResult()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_ts += 33
        detection = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        if not detection.hand_landmarks or len(detection.hand_landmarks) == 0:
            return result

        result.visible = True
        result.num_hands = len(detection.hand_landmarks)
        result.landmarks = detection.hand_landmarks

        for hand_lm in detection.hand_landmarks:
            wrist = hand_lm[self.WRIST]
            center = hand_lm[self.MIDDLE_FINGER_MCP]
            cx = (wrist.x + center.x) / 2
            cy = (wrist.y + center.y) / 2
            result.hand_positions.append((cx, cy))

        return result

    def close(self):
        self._landmarker.close()
