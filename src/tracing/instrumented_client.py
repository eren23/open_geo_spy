"""Wraps AsyncOpenAI to record all LLM calls (tokens, cost, latency)."""

from __future__ import annotations

import time
from typing import Any, Optional

from openai import AsyncOpenAI

from src.tracing.recorder import TraceRecorder

# Approximate cost per 1M tokens (input/output) by model prefix.
# These are rough estimates; actual costs come from OpenRouter.
_COST_PER_M: dict[str, tuple[float, float]] = {
    "google/gemini-2.5-flash": (0.15, 0.60),
    "google/gemini-2.5-pro": (1.25, 10.0),
    "google/gemini-3": (0.50, 2.0),
    "qwen/qwen3-vl-235b": (1.0, 4.0),
    "qwen/qwen3-vl-8b": (0.10, 0.40),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD from token counts."""
    for prefix, (inp_rate, out_rate) in _COST_PER_M.items():
        if model.startswith(prefix):
            return (input_tokens * inp_rate + output_tokens * out_rate) / 1_000_000
    # Default estimate
    return (input_tokens * 0.50 + output_tokens * 2.0) / 1_000_000


class InstrumentedOpenAI:
    """Drop-in replacement for AsyncOpenAI that records all calls to a TraceRecorder.

    Usage:
        client = InstrumentedOpenAI(base_client, recorder, default_purpose="reasoning")
        resp = await client.chat.completions.create(model=..., messages=...)
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        recorder: Optional[TraceRecorder] = None,
        default_purpose: str = "general",
    ):
        self._client = client
        self._recorder = recorder
        self._default_purpose = default_purpose
        self.chat = _InstrumentedChat(self)

    @property
    def base_client(self) -> AsyncOpenAI:
        return self._client


class _InstrumentedChat:
    def __init__(self, parent: InstrumentedOpenAI):
        self._parent = parent
        self.completions = _InstrumentedCompletions(parent)


class _InstrumentedCompletions:
    def __init__(self, parent: InstrumentedOpenAI):
        self._parent = parent

    async def create(self, **kwargs: Any) -> Any:
        """Proxy to AsyncOpenAI.chat.completions.create with instrumentation."""
        recorder = self._parent._recorder
        model = kwargs.get("model", "unknown")
        purpose = kwargs.pop("_purpose", self._parent._default_purpose)
        temperature = kwargs.get("temperature", 0.0)

        start = time.monotonic()
        resp = await self._parent._client.chat.completions.create(**kwargs)
        latency_ms = round((time.monotonic() - start) * 1000, 1)

        # Extract usage
        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = _estimate_cost(model, input_tokens, output_tokens)

        if recorder:
            recorder.record_llm_call(
                model=model,
                purpose=purpose,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                temperature=temperature,
            )

        return resp
