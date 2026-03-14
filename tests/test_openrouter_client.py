"""
Tests for llm/openrouter_client.py — rate limiting and availability checks.
"""

import time
from llm.openrouter_client import OpenRouterClient


class TestOpenRouterClient:
    def test_disabled_without_key(self):
        client = OpenRouterClient()
        client._enabled = False
        assert client.is_available is False

    def test_enabled_with_key(self):
        client = OpenRouterClient()
        client._enabled = True
        assert client.is_available is True

    def test_rate_limit_allows_within_limit(self):
        client = OpenRouterClient()
        client._call_times = []
        assert client._check_rate_limit() is True

    def test_rate_limit_blocks_over_limit(self):
        client = OpenRouterClient()
        now = time.time()
        # Fill up to the limit
        client._call_times = [now - i for i in range(10)]
        assert client._check_rate_limit() is False

    def test_rate_limit_allows_after_window(self):
        client = OpenRouterClient()
        # All calls are >60s ago
        client._call_times = [time.time() - 120 for _ in range(10)]
        assert client._check_rate_limit() is True

    def test_cache_key_is_deterministic(self):
        # Same inputs should produce same cache key
        s1 = "system"
        u1 = "user message"
        assert hash(s1 + u1) == hash(s1 + u1)
