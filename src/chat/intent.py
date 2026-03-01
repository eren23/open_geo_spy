"""Chat intent classification via LLM."""

from __future__ import annotations

import json
import re
from enum import Enum

from loguru import logger
from openai import AsyncOpenAI


class ChatIntent(str, Enum):
    ASK_WHY_NOT = "ask_why_not"       # "Why not Turkey?"
    ZOOM_FEATURE = "zoom_feature"     # "What does the sign say?"
    TRY_SEARCH = "try_search"         # "Try searching for X"
    COMPARE = "compare"               # "Compare candidates 1 and 2"
    EXPLAIN = "explain"               # "Explain the evidence for Paris"
    REFINE_HINT = "refine_hint"       # "I think it's in Southeast Asia"
    GENERAL = "general"               # General question


CLASSIFY_PROMPT = """Classify the user's follow-up message into one of these intents:
- ask_why_not: User asks why a particular location was not chosen (e.g., "why not Turkey?")
- zoom_feature: User asks about a specific visual feature (e.g., "what does the sign say?")
- try_search: User wants to trigger a new search query (e.g., "try searching for X")
- compare: User wants to compare two candidates (e.g., "compare #1 and #2")
- explain: User wants explanation of evidence (e.g., "explain why Paris")
- refine_hint: User provides a location hint (e.g., "I think it's in Southeast Asia")
- general: General question or comment

User message: {message}

Respond with only the intent name (one of the above), nothing else."""


async def classify_intent(
    message: str,
    client: AsyncOpenAI,
    model: str = "google/gemini-2.5-flash",
) -> ChatIntent:
    """Classify user message into a ChatIntent."""
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(message=message)}],
            temperature=0.0,
            max_tokens=50,
        )
        raw = resp.choices[0].message.content.strip().lower()

        # Try to match to enum
        for intent in ChatIntent:
            if intent.value in raw:
                return intent

        return ChatIntent.GENERAL

    except Exception as e:
        logger.warning("Intent classification failed: {}", e)
        # Fallback heuristics
        msg = message.lower()
        if "why not" in msg or "why isn't" in msg:
            return ChatIntent.ASK_WHY_NOT
        if "try search" in msg or "search for" in msg:
            return ChatIntent.TRY_SEARCH
        if "compare" in msg:
            return ChatIntent.COMPARE
        if "explain" in msg or "why" in msg:
            return ChatIntent.EXPLAIN
        if "i think" in msg or "it's in" in msg or "probably" in msg:
            return ChatIntent.REFINE_HINT
        return ChatIntent.GENERAL
