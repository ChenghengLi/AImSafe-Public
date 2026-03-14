"""
LLM-powered safety coach — generates real-time natural language tips
and incident analyses from worker state data.
"""

import asyncio
import time
import logging
from llm.openrouter_client import OpenRouterClient
from llm.prompts import (
    SAFETY_COACH_SYSTEM,
    INCIDENT_ANALYSIS_SYSTEM,
    format_worker_snapshot,
)
from engine.worker_state import WorkerState, Alert
import config

logger = logging.getLogger(__name__)


class SafetyCoach:
    """Generates human-readable safety coaching and incident analysis via LLM."""

    def __init__(self, client: OpenRouterClient):
        self.client = client
        self._last_tip_time: float = 0.0
        self._latest_tip: str = ""
        self._latest_incident_analysis: str = ""

    @property
    def latest_tip(self) -> str:
        return self._latest_tip

    @property
    def latest_incident_analysis(self) -> str:
        return self._latest_incident_analysis

    async def maybe_coach(self, state: WorkerState) -> str | None:
        """
        Generate a coaching tip if:
        1. There are active alerts
        2. Cooldown has elapsed
        3. LLM is available

        Returns the tip text or None.
        """
        if not self.client.is_available:
            return None

        if not state.active_alerts:
            return None

        now = time.time()
        if now - self._last_tip_time < config.LLM_COACHING_COOLDOWN:
            return None

        snapshot = format_worker_snapshot(state.to_dict())
        tip = await self.client.query(SAFETY_COACH_SYSTEM, snapshot)

        if tip:
            self._latest_tip = tip
            self._last_tip_time = now
            logger.info(f"Safety tip generated: {tip}")

        return tip

    async def analyze_incident(self, alerts: list[Alert]) -> str | None:
        """
        Generate an incident analysis for a batch of critical alerts.
        Called when a CRITICAL alert fires.
        """
        if not self.client.is_available:
            return None

        events = "\n".join(
            f"- [{a.severity}] {a.rule_name}: {a.message}" for a in alerts
        )
        prompt = f"Recent safety events (last 2 minutes):\n{events}"

        analysis = await self.client.query(INCIDENT_ANALYSIS_SYSTEM, prompt)

        if analysis:
            self._latest_incident_analysis = analysis
            logger.info(f"Incident analysis generated: {analysis}")

        return analysis
