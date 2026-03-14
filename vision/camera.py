"""
Camera capture manager.
Wraps OpenCV VideoCapture in a threaded reader for non-blocking frame access.
Falls back to a synthetic color-cycling frame if no camera is available.
"""

import threading
import time
import cv2
import numpy as np
import config


class Camera:
    """Threaded camera capture — grabs frames in background so the main loop never blocks."""

    def __init__(self, source: int = config.CAMERA_INDEX):
        self._source = source
        self._cap = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._fps = 0.0
        self._demo_mode = False

    @property
    def is_opened(self) -> bool:
        if self._demo_mode:
            return True
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def demo_mode(self) -> bool:
        return self._demo_mode

    def start(self) -> "Camera":
        """Start the background capture thread."""
        if self._running:
            return self

        # Try to open camera with DirectShow (fast fail on Windows)
        self._cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
        time.sleep(0.5)  # brief wait for camera init

        if not self._cap.isOpened():
            # Try default backend
            self._cap.release()
            self._cap = cv2.VideoCapture(self._source)
            time.sleep(0.5)

        if not self._cap.isOpened():
            print("  [!] No camera detected — running in DEMO MODE (synthetic feed)")
            self._cap.release()
            self._cap = None
            self._demo_mode = True
        else:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop capture and release the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()

    def read(self) -> np.ndarray | None:
        """Return the latest frame (thread-safe)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def read_jpeg(self, quality: int = 80) -> bytes | None:
        """Return the latest frame as JPEG bytes for streaming."""
        frame = self.read()
        if frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def _capture_loop(self):
        """Background loop — continuously grabs frames."""
        interval = 1.0 / config.TARGET_FPS
        frame_count = 0
        while self._running:
            t0 = time.perf_counter()

            if self._demo_mode:
                frame = self._generate_demo_frame(frame_count)
                frame_count += 1
            else:
                ok, frame = self._cap.read()
                if not ok:
                    frame = self._generate_demo_frame(frame_count)
                    frame_count += 1

            with self._lock:
                self._frame = frame

            elapsed = time.perf_counter() - t0
            self._fps = 1.0 / max(elapsed, 1e-6)
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    _bg_cache: np.ndarray | None = None

    @classmethod
    def _generate_demo_frame(cls, count: int) -> np.ndarray:
        """Generate a bright, visible synthetic frame for demo mode."""
        w, h = config.CAMERA_WIDTH, config.CAMERA_HEIGHT

        # Cache gradient background
        if cls._bg_cache is None or cls._bg_cache.shape[:2] != (h, w):
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            gradient = np.linspace(0, 1, h).reshape(h, 1)
            # Bright industrial blue-gray palette
            bg[:, :, 0] = (80 + 40 * gradient).astype(np.uint8)   # B
            bg[:, :, 1] = (75 + 35 * gradient).astype(np.uint8)   # G
            bg[:, :, 2] = (70 + 30 * gradient).astype(np.uint8)   # R
            cls._bg_cache = bg

        frame = cls._bg_cache.copy()

        # Grid overlay (factory floor)
        grid_color = (100, 110, 120)
        for gx in range(0, w, 60):
            cv2.line(frame, (gx, 0), (gx, h), grid_color, 1)
        for gy in range(0, h, 60):
            cv2.line(frame, (0, gy), (w, gy), grid_color, 1)

        # Animated worker stick figure
        t = count * 0.05
        cx = int(w * (0.35 + 0.15 * np.sin(t * 0.5)))
        cy = int(h * 0.45)

        # Worker body colors
        body_color = (220, 220, 240)
        joint_color = (100, 255, 100)
        bone_color = (180, 255, 180)

        # Head
        cv2.circle(frame, (cx, cy - 80), 28, body_color, -1)
        cv2.circle(frame, (cx, cy - 80), 28, joint_color, 2)
        # Eyes
        cv2.circle(frame, (cx - 8, cy - 85), 3, (50, 50, 50), -1)
        cv2.circle(frame, (cx + 8, cy - 85), 3, (50, 50, 50), -1)

        # Body
        cv2.line(frame, (cx, cy - 52), (cx, cy + 40), bone_color, 4)
        # Shoulders
        cv2.line(frame, (cx - 40, cy - 30), (cx + 40, cy - 30), bone_color, 3)

        # Arms (animated)
        arm_dx = int(45 * np.sin(t * 2))
        arm_dy = int(20 * np.cos(t * 2))
        left_hand = (cx - 55 + arm_dx, cy + 15 + arm_dy)
        right_hand = (cx + 55 - arm_dx, cy + 15 - arm_dy)
        cv2.line(frame, (cx - 40, cy - 30), left_hand, bone_color, 3)
        cv2.line(frame, (cx + 40, cy - 30), right_hand, bone_color, 3)
        # Hand circles
        cv2.circle(frame, left_hand, 6, joint_color, -1)
        cv2.circle(frame, right_hand, 6, joint_color, -1)

        # Hips
        cv2.line(frame, (cx - 20, cy + 40), (cx + 20, cy + 40), bone_color, 3)

        # Legs (animated)
        leg_dx = int(10 * np.sin(t * 3))
        left_foot = (cx - 25 + leg_dx, cy + 110)
        right_foot = (cx + 25 - leg_dx, cy + 110)
        cv2.line(frame, (cx - 20, cy + 40), left_foot, bone_color, 3)
        cv2.line(frame, (cx + 20, cy + 40), right_foot, bone_color, 3)
        # Foot circles
        cv2.circle(frame, left_foot, 5, joint_color, -1)
        cv2.circle(frame, right_foot, 5, joint_color, -1)

        # Joint dots on body
        for jx, jy in [(cx, cy - 52), (cx - 40, cy - 30), (cx + 40, cy - 30),
                        (cx, cy + 40), (cx - 20, cy + 40), (cx + 20, cy + 40)]:
            cv2.circle(frame, (jx, jy), 5, joint_color, -1)

        # "Danger zone" visualization on right side
        zone_pts = np.array([(int(w * 0.75), 0), (w, 0), (w, h), (int(w * 0.75), h)], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_pts], (50, 50, 200))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [zone_pts], True, (80, 80, 255), 2)
        cv2.putText(frame, "DANGER ZONE", (int(w * 0.76), 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 2)

        # "Machine" box on left
        mx, my = int(w * 0.02), int(h * 0.15)
        mw, mh = int(w * 0.12), int(h * 0.3)
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (180, 180, 60), 2)
        cv2.putText(frame, "MACHINE", (mx + 5, my + mh + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 60), 1)
        # Animated machine indicator
        indicator_color = (0, 255, 0) if int(t * 2) % 2 == 0 else (0, 180, 0)
        cv2.circle(frame, (mx + mw // 2, my + 15), 8, indicator_color, -1)

        # Top status bar
        cv2.rectangle(frame, (0, h - 50), (w, h), (40, 45, 55), -1)
        cv2.putText(frame, "DEMO MODE - No Camera Connected", (w // 2 - 180, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 160, 255), 2)
        cv2.putText(frame, "Simulated worker tracking active", (w // 2 - 150, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 170, 200), 1)

        return frame
