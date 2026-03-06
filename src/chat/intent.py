"""Chat intent classification via LLM with configurable pattern fallback."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from loguru import logger
from openai import AsyncOpenAI

from src.patterns import PatternRegistry, classify_intent as pattern_classify_intent
from src.config.llm import LLMCallType, get_llm_params


class ChatIntent(str, Enum):
    ASK_WHY_NOT = "ask_why_not"       # "Why not Turkey?"
    ZOOM_FEATURE = "zoom_feature"     # "What does the sign say?"
    TRY_SEARCH = "try_search"         # "Try searching for X"
    COMPARE = "compare"               # "Compare candidates 1 and 2"
    EXPLAIN = "explain"               # "Explain the evidence for Paris"
    REFINE_HINT = "refine_hint"       # "I think it's in Southeast Asia"
    GENERAL = "general"               # General question


# Map string intent names to enum values
INTENT_MAP = {intent.value: intent for intent in ChatIntent}


CLASSIFY_PROMPT = """Classify the user's follow-up message into one of these intents:
- ask_why_not: User asks why a particular location was not chosen (e.g., "why not Turkey?", "why isn't it Greece?")
- zoom_feature: User asks about a specific visual feature (e.g., "what does the sign say?", "zoom into the building")
- try_search: User wants to trigger a new search with specific terms (e.g., "try searching for X", "google the restaurant name")
- compare: User wants to compare two candidates (e.g., "compare #1 and #2", "difference between top candidates")
- explain: User wants explanation of evidence (e.g., "explain why Paris", "what evidence supports this?")
- refine_hint: User provides a location hint to narrow down the search area (e.g., "I think it's in Southeast Asia", "narrow it down to Turkey", "around Istanbul", "the image is from Germany")
- general: General question or comment not matching other intents

IMPORTANT DISTINCTIONS:
- "try searching for [business name]" → try_search (user wants to search for a specific thing)
- "I think it's in [place]" or "narrow it down to [place]" → refine_hint (user provides location context for fresh discovery)
- "why not [place]?" → ask_why_not (user questions why a location wasn't chosen)

User message: {message}

Respond with only the intent name (one of the above), nothing else."""


async def classify_intent(
    message: str,
    client: Optional[AsyncOpenAI] = None,
    settings: Optional[Any] = None,
    use_llm: bool = True,
) -> ChatIntent:
    """Classify user message into a ChatIntent.
    
    Uses LLM for classification when available, with configurable pattern
    matching as fallback (no hardcoded patterns in code).
    
    Args:
        message: User's message to classify
        client: OpenAI client (optional, if None uses pattern matching only)
        settings: Settings object with LLM configuration
        use_llm: Whether to try LLM first (default True)
        
    Returns:
        ChatIntent enum value
    """
    # Try LLM classification first if client provided and use_llm is True
    if use_llm and client is not None and settings is not None:
        try:
            params = get_llm_params(LLMCallType.INTENT_CLASSIFY, settings)
            resp = await client.chat.completions.create(
                **params,
                messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(message=message)}],
            )
            raw = resp.choices[0].message.content.strip().lower()

            # Try to match to enum
            for intent in ChatIntent:
                if intent.value in raw:
                    return intent

            # If LLM returned something we don't recognize, fall through to pattern matching
            logger.debug("LLM returned unknown intent '{}', using pattern fallback", raw)
            
        except Exception as e:
            logger.warning("Intent classification failed: {}", e)
            # Fall through to pattern matching
    
    # Use configurable pattern matching as fallback (or primary if no client)
    intent_name, confidence = pattern_classify_intent(message)
    
    # Map string to enum
    intent = INTENT_MAP.get(intent_name)
    if intent:
        return intent
    
    logger.debug("Pattern matching returned unknown intent '{}', defaulting to GENERAL", intent_name)
    return ChatIntent.GENERAL


def classify_intent_sync(message: str) -> ChatIntent:
    """Synchronous intent classification using pattern matching only.
    
    Useful when you don't have async context or LLM client.
    """
    intent_name, confidence = pattern_classify_intent(message)
    return INTENT_MAP.get(intent_name, ChatIntent.GENERAL)
