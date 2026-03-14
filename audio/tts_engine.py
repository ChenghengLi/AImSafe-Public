"""
Text-to-speech engine for safety alerts.
Uses pyttsx3 for offline TTS (no API key needed).
Falls back gracefully if pyttsx3 is not installed.
"""

import threading
import queue
import logging

logger = logging.getLogger(__name__)


class TTSEngine:
    """Thread-safe text-to-speech for safety alerts."""

    def __init__(self, enabled: bool = False, rate: int = 160, volume: float = 0.9):
        self._enabled = enabled
        self._queue: queue.Queue = queue.Queue()
        self._engine = None

        if not enabled:
            return

        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', rate)
            self._engine.setProperty('volume', volume)
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            logger.info("TTS engine initialized")
        except ImportError:
            logger.warning("pyttsx3 not installed — TTS disabled. Install with: pip install pyttsx3")
            self._enabled = False
        except Exception as e:
            logger.warning(f"TTS init failed: {e}")
            self._enabled = False

    @property
    def is_available(self) -> bool:
        return self._enabled and self._engine is not None

    def speak(self, text: str, priority: str = "normal"):
        """Queue a message to be spoken. 'critical' clears queue first."""
        if not self.is_available:
            return

        if priority == "critical":
            # Clear pending messages for urgent alerts
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        self._queue.put(text)

    def _worker(self):
        """Background thread that processes TTS queue."""
        while True:
            try:
                text = self._queue.get(timeout=1.0)
                if self._engine:
                    self._engine.say(text)
                    self._engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS error: {e}")
