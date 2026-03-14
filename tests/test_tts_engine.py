"""Tests for audio/tts_engine.py"""
from audio.tts_engine import TTSEngine


class TestTTSEngine:
    def test_disabled_by_default(self):
        tts = TTSEngine(enabled=False)
        assert not tts.is_available
        tts.speak("test")  # should not crash

    def test_graceful_without_pyttsx3(self):
        # Even if enabled=True, should handle missing pyttsx3 gracefully
        tts = TTSEngine(enabled=True)
        tts.speak("test")  # should not crash regardless
