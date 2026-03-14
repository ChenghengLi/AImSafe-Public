"""
OpenRouter API client — async wrapper using the OpenAI-compatible SDK.
Includes rate limiting, caching, and graceful fallback.
"""

import asyncio
import time
import logging
from openai import AsyncOpenAI
import config

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Async LLM client that talks to OpenRouter (Llama / Mixtral)."""

    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )
        self.model = config.LLM_MODEL
        self.fallback_model = config.LLM_FALLBACK_MODEL
        self._cache: dict[int, str] = {}
        self._call_times: list[float] = []
        self._enabled = bool(config.OPENROUTER_API_KEY)

    @property
    def is_available(self) -> bool:
        return self._enabled

    async def query(self, system_prompt: str, user_message: str) -> str | None:
        """
        Send a chat completion request to OpenRouter.
        Returns the response text, or None if unavailable/rate-limited.
        """
        if not self._enabled:
            logger.debug("LLM disabled — no OPENROUTER_API_KEY set")
            return None

        # Cache check
        cache_key = hash(system_prompt + user_message)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Rate limiting
        if not self._check_rate_limit():
            logger.debug("LLM rate-limited — skipping call")
            return None

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            )
            result = response.choices[0].message.content
            self._cache[cache_key] = result
            self._call_times.append(time.time())
            return result

        except Exception as e:
            logger.warning(f"LLM primary model failed: {e}, trying fallback")
            try:
                response = await self.client.chat.completions.create(
                    model=self.fallback_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=config.LLM_MAX_TOKENS,
                    temperature=config.LLM_TEMPERATURE,
                )
                result = response.choices[0].message.content
                self._cache[cache_key] = result
                return result
            except Exception as e2:
                logger.error(f"LLM fallback also failed: {e2}")
                return None

    def _check_rate_limit(self) -> bool:
        """Return True if we're within the rate limit."""
        now = time.time()
        cutoff = now - 60
        self._call_times = [t for t in self._call_times if t > cutoff]
        return len(self._call_times) < config.LLM_RATE_LIMIT
