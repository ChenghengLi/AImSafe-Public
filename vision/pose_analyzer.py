"""
Pose analysis using MediaPipe PoseLandmarker (Tasks API).
Computes safety-relevant angles: back angle, shoulder tilt, knee angle, arm extension.
"""

from dataclasses import dataclass, field
import math
import os
import numpy as np
import mediapipe as mp
import config

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "pose_landmarker_lite.task")


def _load_model_buffer(path: str) -> bytes:
    """Load model file as bytes to bypass MediaPipe's broken Windows path resolution."""
    with open(path, "rb") as f:
        return f.read()


@dataclass
class PoseResult:
    """Output of a single frame's pose analysis."""
    landmarks: list | None = None
    angles: dict[str, float] = field(default_factory=dict)
    body_centroid: tuple[float, float] = (0.0, 0.0)
    flags: dict[str, bool] = field(default_factory=dict)
    visible: bool = False


class PoseAnalyzer:
    """Wraps MediaPipe PoseLandmarker and computes safety-relevant body angles."""

    # Landmark indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 max_persons: int = 5):
        model_buffer = _load_model_buffer(MODEL_PATH)
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_buffer),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_poses=max_persons,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._frame_ts = 0

        # Temporal filter for unsafe lift — require N consecutive bad frames
        self._unsafe_lift_count: dict[int, int] = {}  # person_idx → consecutive bad frames
        self._UNSAFE_LIFT_MIN_FRAMES = 8  # ~260ms at 30fps — filters out brief movements

    def analyze(self, frame_rgb: np.ndarray) -> PoseResult:
        """Process one RGB frame and return first person's pose (backward compat)."""
        results = self.analyze_multi(frame_rgb)
        return results[0] if results else PoseResult()

    def analyze_multi(self, frame_rgb: np.ndarray) -> list[PoseResult]:
        """Process one RGB frame and return pose analysis for ALL detected people."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_ts += 33
        detection = self._landmarker.detect_for_video(mp_image, self._frame_ts)

        if not detection.pose_landmarks or len(detection.pose_landmarks) == 0:
            return []

        results = []
        for idx, lm in enumerate(detection.pose_landmarks):
            result = self._analyze_landmarks(lm, person_idx=idx)
            results.append(result)
        return results

    def _analyze_landmarks(self, lm, person_idx: int = 0) -> PoseResult:
        """Compute angles and flags for a single person's landmarks."""
        result = PoseResult()
        result.landmarks = lm
        result.visible = True

        # --- Compute key angles ---
        result.angles["back"] = self._back_angle(lm)
        result.angles["shoulder_tilt"] = self._shoulder_tilt(lm)
        result.angles["left_knee"] = self._knee_angle(lm, side="left")
        result.angles["right_knee"] = self._knee_angle(lm, side="right")
        result.angles["left_arm_extension"] = self._arm_extension(lm, side="left")
        result.angles["right_arm_extension"] = self._arm_extension(lm, side="right")

        # --- Body centroid (average of hips) ---
        cx = (lm[self.LEFT_HIP].x + lm[self.RIGHT_HIP].x) / 2
        cy = (lm[self.LEFT_HIP].y + lm[self.RIGHT_HIP].y) / 2
        result.body_centroid = (cx, cy)

        # --- Safety flags with temporal filtering ---
        hands_below_hips = (
            lm[self.LEFT_WRIST].y > lm[self.LEFT_HIP].y
            or lm[self.RIGHT_WRIST].y > lm[self.RIGHT_HIP].y
        )
        raw_unsafe = result.angles["back"] > config.UNSAFE_LIFT_BACK_ANGLE and hands_below_hips

        # Require consecutive bad frames to filter glitches
        if raw_unsafe:
            self._unsafe_lift_count[person_idx] = self._unsafe_lift_count.get(person_idx, 0) + 1
        else:
            self._unsafe_lift_count[person_idx] = 0

        result.flags["unsafe_lift"] = self._unsafe_lift_count.get(person_idx, 0) >= self._UNSAFE_LIFT_MIN_FRAMES

        return result

    # ──── angle helpers ────

    @staticmethod
    def _angle_3pts(a, b, c) -> float:
        """Angle at point b formed by points a-b-c, in degrees."""
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return math.degrees(math.acos(np.clip(cos, -1, 1)))

    def _back_angle(self, lm) -> float:
        """Angle of the torso from vertical (0° = upright, 90° = horizontal)."""
        mid_shoulder = type("P", (), {
            "x": (lm[self.LEFT_SHOULDER].x + lm[self.RIGHT_SHOULDER].x) / 2,
            "y": (lm[self.LEFT_SHOULDER].y + lm[self.RIGHT_SHOULDER].y) / 2,
        })()
        mid_hip = type("P", (), {
            "x": (lm[self.LEFT_HIP].x + lm[self.RIGHT_HIP].x) / 2,
            "y": (lm[self.LEFT_HIP].y + lm[self.RIGHT_HIP].y) / 2,
        })()
        vertical_ref = type("P", (), {"x": mid_hip.x, "y": mid_hip.y - 1.0})()
        return self._angle_3pts(mid_shoulder, mid_hip, vertical_ref)

    def _shoulder_tilt(self, lm) -> float:
        return abs(lm[self.LEFT_SHOULDER].y - lm[self.RIGHT_SHOULDER].y) * 100

    def _knee_angle(self, lm, side: str = "left") -> float:
        if side == "left":
            return self._angle_3pts(lm[self.LEFT_HIP], lm[self.LEFT_KNEE], lm[self.LEFT_ANKLE])
        return self._angle_3pts(lm[self.RIGHT_HIP], lm[self.RIGHT_KNEE], lm[self.RIGHT_ANKLE])

    def _arm_extension(self, lm, side: str = "left") -> float:
        if side == "left":
            return self._angle_3pts(lm[self.LEFT_SHOULDER], lm[self.LEFT_ELBOW], lm[self.LEFT_WRIST])
        return self._angle_3pts(lm[self.RIGHT_SHOULDER], lm[self.RIGHT_ELBOW], lm[self.RIGHT_WRIST])

    def close(self):
        self._landmarker.close()
