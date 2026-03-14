"""
Centralized prompt templates for all LLM interactions.
Designed for low token usage, consistent output, and grounded (no hallucination) responses.
"""

SAFETY_COACH_SYSTEM = """You are a factory floor safety coach AI.
Given a worker's current safety data, provide ONE short, friendly, actionable safety tip.
Be specific about what they should do differently.
Speak directly to the worker in second person ("you").
Keep it under 2 sentences. Do not use technical jargon.
Do not be alarming — be helpful and encouraging.
Only reference data that is provided — never make assumptions."""

INCIDENT_ANALYSIS_SYSTEM = """You are a workplace safety analyst AI.
Given a sequence of safety events, provide a brief incident analysis:
1. What happened (1 sentence)
2. Likely root cause (1 sentence)
3. Recommended corrective action (1-2 sentences)
Be factual, concise, and professional. Only use the data provided."""

SHIFT_REPORT_SYSTEM = """You are a factory safety reporting AI.
Given aggregated safety data for a work session, write a concise safety report with:
- Overall Assessment (1 sentence rating: Excellent / Good / Moderate / Poor / Critical)
- Key Incidents (bullet list, max 5)
- Fatigue & Wellness Trends (2-3 sentences)
- Recommendations (2-3 actionable items)
Keep the total report under 200 words. Be professional and data-driven.
Only reference data that is provided."""


def format_worker_snapshot(state_dict: dict) -> str:
    """Convert a WorkerState dict into a readable prompt for the LLM."""
    alerts_str = ", ".join(
        f"{a['severity']}: {a['message']}" for a in state_dict.get("active_alerts", [])
    ) or "None"

    return f"""Worker: {state_dict['worker_id']}
Safety Score: {state_dict['overall_safety_score']}/100 ({state_dict['risk_level']})
Back Angle: {state_dict['angles'].get('back', 0)}°
Fatigue Score: {state_dict['fatigue_score']}/100
Blink Rate: {state_dict['blink_rate']} blinks/min
Microsleep Detected: {state_dict['microsleep']}
Head Pitch: {state_dict['head_pitch']}°
Repetitive Motions (5 min): {state_dict['repetition_count']}
Zone Status: {state_dict['zone_status']} {state_dict['current_zone_name']}
Heart Rate: {state_dict['heart_rate'] or 'N/A'} BPM
Active Alerts: {alerts_str}"""


def format_shift_summary(
    alerts: list[dict],
    avg_safety_score: float,
    avg_fatigue: float,
    total_incidents: int,
    duration_minutes: float,
    peak_blink_rate: float,
    total_repetitions: int,
) -> str:
    """Format shift data into a prompt for the shift report LLM call."""
    alert_lines = "\n".join(
        f"  - [{a['severity']}] {a['time']}: {a['message']}" for a in alerts[:20]
    ) or "  No alerts recorded."

    return f"""Session Duration: {duration_minutes:.0f} minutes
Average Safety Score: {avg_safety_score:.0f}/100
Average Fatigue Score: {avg_fatigue:.0f}/100
Total Incidents: {total_incidents}
Peak Blink Rate: {peak_blink_rate:.0f} blinks/min
Total Repetitive Motions: {total_repetitions}

Alert Log:
{alert_lines}"""
