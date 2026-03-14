"""
Shift report generator — compiles session data and asks the LLM
for a natural-language safety report.
"""

import logging
from llm.openrouter_client import OpenRouterClient
from llm.prompts import SHIFT_REPORT_SYSTEM, format_shift_summary
from engine.worker_state import Alert

logger = logging.getLogger(__name__)


class ShiftReporter:
    """Generates end-of-session AI safety reports."""

    def __init__(self, client: OpenRouterClient):
        self.client = client

    async def generate(
        self,
        alerts: list[dict],
        avg_safety_score: float,
        avg_fatigue: float,
        total_incidents: int,
        duration_minutes: float,
        peak_blink_rate: float,
        total_repetitions: int,
    ) -> str:
        """
        Generate a shift safety report from aggregated data.
        Returns the report text, or a fallback summary if LLM is unavailable.
        """
        summary = format_shift_summary(
            alerts=alerts,
            avg_safety_score=avg_safety_score,
            avg_fatigue=avg_fatigue,
            total_incidents=total_incidents,
            duration_minutes=duration_minutes,
            peak_blink_rate=peak_blink_rate,
            total_repetitions=total_repetitions,
        )

        if not self.client.is_available:
            return self._fallback_report(
                avg_safety_score, total_incidents, duration_minutes
            )

        report = await self.client.query(SHIFT_REPORT_SYSTEM, summary)
        if report:
            logger.info("Shift report generated via LLM")
            return report

        return self._fallback_report(
            avg_safety_score, total_incidents, duration_minutes
        )

    @staticmethod
    def _fallback_report(
        avg_safety: float, incidents: int, duration: float
    ) -> str:
        """Plain text fallback when LLM is unavailable."""
        rating = (
            "Excellent" if avg_safety >= 90
            else "Good" if avg_safety >= 70
            else "Moderate" if avg_safety >= 50
            else "Poor" if avg_safety >= 30
            else "Critical"
        )
        return (
            f"## Shift Report (auto-generated)\n\n"
            f"**Overall Assessment:** {rating} ({avg_safety:.0f}/100)\n\n"
            f"**Duration:** {duration:.0f} minutes\n\n"
            f"**Total Incidents:** {incidents}\n\n"
            f"*LLM analysis unavailable — set OPENROUTER_API_KEY for AI-powered reports.*"
        )
