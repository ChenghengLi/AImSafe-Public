"""
Tests for llm/shift_reporter.py — fallback report generation (no API needed).
"""

import pytest
from llm.shift_reporter import ShiftReporter
from llm.openrouter_client import OpenRouterClient


class TestShiftReporterFallback:
    """Test the fallback report when LLM is unavailable."""

    def test_fallback_excellent(self):
        report = ShiftReporter._fallback_report(95.0, 0, 120)
        assert "Excellent" in report
        assert "120 minutes" in report

    def test_fallback_good(self):
        report = ShiftReporter._fallback_report(75.0, 2, 60)
        assert "Good" in report

    def test_fallback_moderate(self):
        report = ShiftReporter._fallback_report(55.0, 5, 90)
        assert "Moderate" in report

    def test_fallback_poor(self):
        report = ShiftReporter._fallback_report(35.0, 10, 120)
        assert "Poor" in report

    def test_fallback_critical(self):
        report = ShiftReporter._fallback_report(15.0, 20, 60)
        assert "Critical" in report

    def test_fallback_contains_stats(self):
        report = ShiftReporter._fallback_report(50.0, 7, 90)
        assert "50/100" in report
        assert "7" in report
        assert "90 minutes" in report

    @pytest.mark.asyncio
    async def test_generate_uses_fallback_when_no_key(self):
        """When OPENROUTER_API_KEY is empty, generate() should return fallback."""
        client = OpenRouterClient()
        client._enabled = False
        reporter = ShiftReporter(client)
        report = await reporter.generate(
            alerts=[], avg_safety_score=80.0, avg_fatigue=10.0,
            total_incidents=1, duration_minutes=30,
            peak_blink_rate=18.0, total_repetitions=10,
        )
        assert "Good" in report
        assert "LLM analysis unavailable" in report
