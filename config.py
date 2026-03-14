"""
AImSafe — Configuration
Loads settings from .env file, falls back to defaults.
"""

import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()


def _bool(val: str) -> bool:
    return val.lower() in ("true", "1", "yes")


# ─────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30

# ─────────────────────────────────────────────
# Pose thresholds
# ─────────────────────────────────────────────
UNSAFE_LIFT_BACK_ANGLE = 40          # degrees from vertical — bending to grab a box (with temporal filter)
PROXIMITY_WARN_DISTANCE = 0.15       # normalized units

# ─────────────────────────────────────────────
# Fatigue thresholds
# ─────────────────────────────────────────────
NORMAL_BLINK_RATE = (15, 20)         # blinks per minute
FATIGUE_BLINK_RATE = 30              # blinks/min threshold (high to avoid false positives)
MICROSLEEP_DURATION = 2.0            # seconds — conservative with pitch gating and blendshape confirmation
HEAD_NOD_ANGLE = 25                  # degrees pitch (high to ignore normal head movements)
EAR_THRESHOLD = 0.21                 # eye aspect ratio — midpoint between open (~0.30) and closed (~0.10)

# ─────────────────────────────────────────────
# Repetition
# ─────────────────────────────────────────────
REPETITION_WINDOW = 300              # seconds (5 min)
REPETITION_LIMIT = 50               # motions before warning

# ─────────────────────────────────────────────
# Alert cooldowns (seconds)
# ─────────────────────────────────────────────
ALERT_COOLDOWN = {
    "unsafe_lift": 15,
    "proximity": 20,
    "fatigue": 30,
    "repetition": 180,
    "ergonomic": 60,
    "collision": 8,
}

# ─────────────────────────────────────────────
# Risk weights (must sum to 1.0)
# ─────────────────────────────────────────────
RISK_WEIGHTS = {
    "pose": 0.35,
    "fatigue": 0.30,
    "proximity": 0.25,
    "repetition": 0.10,
}

# ─────────────────────────────────────────────
# MQTT (simulated machine control)
# ─────────────────────────────────────────────
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_ENABLED = _bool(os.getenv("MQTT_ENABLED", "false"))

# ─────────────────────────────────────────────
# LLM — OpenRouter
# ─────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "mistralai/mixtral-8x7b-instruct")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "300"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_RATE_LIMIT = int(os.getenv("LLM_RATE_LIMIT", "10"))
LLM_COACHING_COOLDOWN = int(os.getenv("LLM_COACHING_COOLDOWN", "30"))

# ─────────────────────────────────────────────
# Audio / TTS
# ─────────────────────────────────────────────
TTS_ENABLED = _bool(os.getenv("TTS_ENABLED", "false"))
TTS_RATE = int(os.getenv("TTS_RATE", "160"))

# ─────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))
DARK_THEME = _bool(os.getenv("DARK_THEME", "true"))
