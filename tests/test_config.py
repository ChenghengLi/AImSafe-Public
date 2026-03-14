"""
Tests for config.py — .env loading and default values.
"""

import config


class TestConfig:
    def test_camera_defaults(self):
        assert config.CAMERA_WIDTH == 640
        assert config.CAMERA_HEIGHT == 480
        assert config.TARGET_FPS == 30

    def test_pose_thresholds(self):
        assert config.UNSAFE_LIFT_BACK_ANGLE == 50
        assert config.OVERREACH_MULTIPLIER == 1.5

    def test_fatigue_thresholds(self):
        assert config.FATIGUE_BLINK_RATE == 30
        assert config.MICROSLEEP_DURATION == 1.5
        assert config.EAR_THRESHOLD == 0.18

    def test_risk_weights_sum_to_one(self):
        total = sum(config.RISK_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_alert_cooldowns_all_positive(self):
        for rule, cd in config.ALERT_COOLDOWN.items():
            assert cd > 0, f"Cooldown for {rule} should be positive"

    def test_llm_defaults(self):
        assert config.LLM_MAX_TOKENS == 300
        assert config.LLM_TEMPERATURE == 0.3
        assert config.LLM_RATE_LIMIT == 10

    def test_dashboard_defaults(self):
        assert config.DASHBOARD_PORT in (8080, 8090)  # 8090 when .env overrides
        assert config.DARK_THEME is True

    def test_openrouter_base_url(self):
        assert "openrouter.ai" in config.OPENROUTER_BASE_URL
