"""Configurable pattern matching for geolocation.

This module replaces hardcoded string patterns with a configurable
pattern system that can be extended without code changes.
"""

from src.patterns.config import (
    PatternRegistry,
    PatternCategory,
    IntentPattern,
    classify_intent,
    classify_text,
    extract_location_hint,
    extract_search_query,
)

__all__ = [
    "PatternRegistry",
    "PatternCategory",
    "IntentPattern",
    "classify_intent",
    "classify_text",
    "extract_location_hint",
    "extract_search_query",
]
