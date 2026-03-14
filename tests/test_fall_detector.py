"""Tests for engine/fall_detector.py"""
from engine.fall_detector import FallDetector
from vision.pose_analyzer import PoseResult


class TestFallDetector:
    def test_no_fall_when_upright(self):
        fd = FallDetector()
        pose = PoseResult(visible=True, angles={"back": 10}, body_centroid=(0.5, 0.4))
        for i in range(20):
            event = fd.update(pose, timestamp=float(i))
        assert not event.detected

    def test_rapid_descent_detected(self):
        fd = FallDetector()
        # Normal position for 10 frames
        for i in range(10):
            pose = PoseResult(visible=True, angles={"back": 10}, body_centroid=(0.5, 0.3))
            fd.update(pose, timestamp=float(i) * 0.033)
        # Sudden drop
        detected_event = None
        for i in range(10):
            y = 0.3 + (i + 1) * 0.06  # drops from 0.3 to 0.9
            pose = PoseResult(visible=True, angles={"back": 10}, body_centroid=(0.5, y))
            event = fd.update(pose, timestamp=(10 + i) * 0.033)
            if event.detected:
                detected_event = event
        assert detected_event is not None
        assert detected_event.fall_type == "rapid_descent"

    def test_horizontal_body_detected(self):
        fd = FallDetector()
        # Upright for 15 frames
        for i in range(15):
            pose = PoseResult(visible=True, angles={"back": 15}, body_centroid=(0.5, 0.4))
            fd.update(pose, timestamp=float(i) * 0.033)
        # Suddenly horizontal
        detected_event = None
        for i in range(5):
            pose = PoseResult(visible=True, angles={"back": 80}, body_centroid=(0.5, 0.5))
            event = fd.update(pose, timestamp=(15 + i) * 0.033)
            if event.detected:
                detected_event = event
        assert detected_event is not None
        assert detected_event.fall_type == "horizontal"

    def test_invisible_pose_no_detection(self):
        fd = FallDetector()
        pose = PoseResult(visible=False)
        for i in range(30):
            event = fd.update(pose, timestamp=float(i))
        assert not event.detected

    def test_cooldown_prevents_repeated_alerts(self):
        fd = FallDetector()
        # Trigger a fall
        for i in range(10):
            pose = PoseResult(visible=True, angles={"back": 10}, body_centroid=(0.5, 0.3))
            fd.update(pose, timestamp=float(i) * 0.033)
        first_detected = False
        for i in range(10):
            y = 0.3 + (i + 1) * 0.06
            pose = PoseResult(visible=True, angles={"back": 10}, body_centroid=(0.5, y))
            event = fd.update(pose, timestamp=(10 + i) * 0.033)
            if event.detected:
                first_detected = True
        assert first_detected

        # Try to trigger again immediately — should be suppressed by cooldown
        for i in range(10):
            pose = PoseResult(visible=True, angles={"back": 10}, body_centroid=(0.5, 0.3))
            fd.update(pose, timestamp=1.0 + float(i) * 0.033)
        second_detected = False
        for i in range(10):
            y = 0.3 + (i + 1) * 0.06
            pose = PoseResult(visible=True, angles={"back": 10}, body_centroid=(0.5, y))
            event = fd.update(pose, timestamp=1.3 + float(i) * 0.033)
            if event.detected:
                second_detected = True
        assert not second_detected  # within 10s cooldown
