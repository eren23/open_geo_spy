"""Configurable pattern matching system.

Replaces hardcoded string patterns with configurable, language-aware patterns
loaded from YAML/JSON configuration files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loguru import logger


@dataclass
class PatternCategory:
    """A category with associated patterns for classification."""
    
    name: str
    patterns: list[str]  # Regex patterns (case-insensitive)
    keywords: list[str]  # Simple substring matches
    languages: list[str] = field(default_factory=lambda: ["en"])
    compiled_patterns: list[re.Pattern] = field(default_factory=list, repr=False)
    
    def __post_init__(self):
        # Pre-compile regex patterns
        for p in self.patterns:
            try:
                self.compiled_patterns.append(re.compile(p, re.IGNORECASE))
            except re.error as e:
                logger.warning("Invalid regex pattern '{}': {}", p, e)
    
    def matches(self, text: str) -> bool:
        """Check if text matches any pattern or keyword."""
        text_lower = text.lower()
        
        # Check keywords (substring match)
        for kw in self.keywords:
            if kw.lower() in text_lower:
                return True
        
        # Check regex patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return True
        
        return False
    
    def extract(self, text: str) -> list[str]:
        """Extract all matches from text."""
        matches = []
        for pattern in self.compiled_patterns:
            for m in pattern.finditer(text):
                if m.groups():
                    matches.append(m.group(1))
                else:
                    matches.append(m.group(0))
        return matches


@dataclass
class IntentPattern:
    """Pattern configuration for a chat intent."""
    
    intent: str
    description: str
    examples: list[str]
    patterns: list[str]  # Regex patterns
    keywords: list[str]  # Simple keyword matches
    priority: int = 0  # Higher = checked first
    compiled_patterns: list[re.Pattern] = field(default_factory=list, repr=False)
    
    def __post_init__(self):
        for p in self.patterns:
            try:
                self.compiled_patterns.append(re.compile(p, re.IGNORECASE))
            except re.error as e:
                logger.warning("Invalid intent pattern '{}': {}", p, e)
    
    def matches(self, message: str) -> tuple[bool, float]:
        """Check if message matches this intent.
        
        Returns:
            Tuple of (matched, confidence)
        """
        msg_lower = message.lower().strip()
        
        # Check keywords first (lower confidence)
        for kw in self.keywords:
            if kw in msg_lower:
                return True, 0.7
        
        # Check regex patterns (higher confidence)
        for pattern in self.compiled_patterns:
            if pattern.search(message):
                return True, 0.9
        
        return False, 0.0


class PatternRegistry:
    """Registry for configurable patterns.
    
    Loads patterns from configuration files and provides fast lookup.
    """
    
    _instance: "PatternRegistry | None" = None
    
    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path(__file__).parent / "configs"
        self._intent_patterns: list[IntentPattern] = []
        self._text_categories: dict[str, PatternCategory] = {}
        self._street_patterns: dict[str, list[str]] = {}  # lang -> suffixes
        self._prefix_patterns: dict[str, list[str]] = {}  # type -> prefixes
        self._loaded = False
    
    @classmethod
    def get(cls) -> "PatternRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
    
    def load(self, force: bool = False) -> None:
        """Load patterns from configuration files."""
        if self._loaded and not force:
            return
        
        # Load intent patterns
        self._load_intent_patterns()
        
        # Load text categories
        self._load_text_categories()
        
        # Load street patterns
        self._load_street_patterns()
        
        # Load prefix patterns
        self._load_prefix_patterns()
        
        self._loaded = True
        logger.info(
            "PatternRegistry loaded: {} intents, {} categories, {} street patterns",
            len(self._intent_patterns),
            len(self._text_categories),
            len(self._street_patterns),
        )
    
    def _load_intent_patterns(self) -> None:
        """Load chat intent patterns."""
        config_path = self.config_dir / "intents.json"
        
        # Default patterns (embedded for zero-config)
        default_intents = {
            "intents": [
                {
                    "intent": "ask_why_not",
                    "description": "User asks why a location wasn't chosen",
                    "examples": ["why not Turkey?", "why isn't it Greece?"],
                    "patterns": [
                        r"why (?:not|isn't|didn't|won't|wouldn't)\s+(?:it\s+)?(?:be\s+)?(?:in\s+)?(.+\?)",
                        r"why (?:not|isn't|didn't)\s+(.+)",
                    ],
                    "keywords": ["why not", "why isn't", "why didn't", "why won't"],
                    "priority": 10,
                },
                {
                    "intent": "try_search",
                    "description": "User wants to search for specific terms",
                    "examples": ["try searching for Starbucks", "google the restaurant name"],
                    "patterns": [
                        r"(?:try\s+)?(?:search|google|look\s+up|find)\s+(?:for\s+)?(.+)",
                        r"can you (?:search|google|find)\s+(.+)",
                    ],
                    "keywords": ["try search", "search for", "google", "look up", "find information"],
                    "priority": 8,
                },
                {
                    "intent": "refine_hint",
                    "description": "User provides location hint for fresh discovery",
                    "examples": ["I think it's in Southeast Asia", "narrow it down to Turkey"],
                    "patterns": [
                        r"(?:i think|probably|likely|maybe)\s+(?:it'?s?\s+)?(?:in|from)\s+(.+)",
                        r"(?:narrow|focus|concentrate)\s+(?:it\s+)?down\s+(?:to\s+)?(.+)",
                        r"(?:the\s+)?(?:image|photo|picture)\s+(?:is\s+)?from\s+(.+)",
                        r"(?:this|it)\s+(?:is|'s)\s+(?:in|from)\s+(.+)",
                        r"location:\s*(.+)",
                        r"(?:around|near|somewhere\s+in)\s+(.+)",
                    ],
                    "keywords": [
                        "i think it", "it's in", "it is in", "probably in", "likely in",
                        "narrow it down", "the image is from", "this is from", "this is in",
                        "focus on", "concentrate on", "look in", "somewhere in",
                    ],
                    "priority": 5,
                },
                {
                    "intent": "compare",
                    "description": "User wants to compare candidates",
                    "examples": ["compare #1 and #2", "difference between top candidates"],
                    "patterns": [
                        r"compare\s+(.+)",
                        r"(?:what'?s?\s+)?(?:the\s+)?difference\s+between\s+(.+)",
                    ],
                    "keywords": ["compare", "difference between"],
                    "priority": 7,
                },
                {
                    "intent": "explain",
                    "description": "User wants explanation of evidence",
                    "examples": ["explain why Paris", "what evidence supports this?"],
                    "patterns": [
                        r"explain\s+(?:why\s+)?(.+)",
                        r"(?:what|which)\s+evidence\s+(?:supports|for)\s+(.+)",
                        r"why\s+(?:do\s+you\s+)?think\s+(.+)",
                    ],
                    "keywords": ["explain", "what evidence", "why do you think"],
                    "priority": 6,
                },
                {
                    "intent": "zoom_feature",
                    "description": "User asks about a visual feature",
                    "examples": ["what does the sign say?", "zoom into the building"],
                    "patterns": [
                        r"(?:what|which)\s+does\s+(?:the\s+)?(.+)\s+say",
                        r"zoom\s+(?:into|in\s+on)\s+(.+)",
                        r"look\s+(?:at|closely\s+at)\s+(?:the\s+)?(.+)",
                    ],
                    "keywords": ["what does", "say", "zoom", "look at the", "look closely"],
                    "priority": 4,
                },
            ]
        }
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                logger.info("Loaded intent patterns from {}", config_path)
            except Exception as e:
                logger.warning("Failed to load intent patterns: {}, using defaults", e)
                config = default_intents
        else:
            config = default_intents
        
        self._intent_patterns = [
            IntentPattern(**intent_data) 
            for intent_data in config.get("intents", [])
        ]
        # Sort by priority (highest first)
        self._intent_patterns.sort(key=lambda x: x.priority, reverse=True)
    
    def _load_text_categories(self) -> None:
        """Load text categorization patterns."""
        config_path = self.config_dir / "text_categories.json"
        
        default_categories = {
            "categories": [
                {
                    "name": "street_signs",
                    "patterns": [
                        r"\b(street|st\.?|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|lane|ln\.?|drive|dr\.?|way|place|pl\.?)\b",
                        r"\b(straße|strasse|straße|rue|straße|via|road)\b",
                    ],
                    "keywords": ["street", "st.", "avenue", "ave", "road", "rd", "boulevard", "lane", "drive"],
                    "languages": ["en", "de", "fr", "it"],
                },
                {
                    "name": "building_info",
                    "patterns": [r"\b(building|tower|floor|suite|unit|#\d+)\b"],
                    "keywords": ["building", "#", "floor", "suite", "unit", "tower"],
                    "languages": ["en"],
                },
                {
                    "name": "business_names",
                    "patterns": [
                        r"\b(Inc\.|LLC|Ltd\.?|Co\.|Corp\.?|GmbH|AG|SA|Sarl)\b",
                        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Store|Shop|Restaurant|Hotel|Bank|Cafe|Pharmacy)",
                    ],
                    "keywords": ["Inc.", "LLC", "Ltd.", "Co.", "Corp.", "GmbH", "AG"],
                    "languages": ["en", "de", "fr"],
                },
                {
                    "name": "license_plates",
                    "patterns": [
                        r"\b[A-Z]{1,3}[-\s]?\d{1,4}[A-Z]{0,2}\b",  # EU style
                        r"\b[A-Z]{2}\s?\d{2}\s?[A-Z]{3}\b",  # UK style
                    ],
                    "keywords": [],
                    "languages": ["*"],
                },
            ]
        }
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning("Failed to load text categories: {}, using defaults", e)
                config = default_categories
        else:
            config = default_categories
        
        self._text_categories = {
            cat["name"]: PatternCategory(**cat)
            for cat in config.get("categories", [])
        }
    
    def _load_street_patterns(self) -> None:
        """Load street name patterns by language."""
        config_path = self.config_dir / "streets.json"
        
        default_streets = {
            "suffixes": {
                "en": ["street", "st", "avenue", "ave", "road", "rd", "boulevard", "blvd", "lane", "ln", "drive", "dr", "way", "place", "pl", "court", "ct"],
                "de": ["straße", "strasse", "str.", "weg", "platz", "gasse", "allee"],
                "fr": ["rue", "avenue", "av", "boulevard", "bd", "place", "pl", "route"],
                "es": ["calle", "avenida", "av", "plaza", "paseo", "camino"],
                "it": ["via", "viale", "piazza", "corso", "strada"],
                "pt": ["rua", "avenida", "av", "praça", "travessa"],
                "tr": ["sokak", "sk", "caddesi", "cad", "bulvarı", "meidan"],
                "ru": ["улица", "ул", "проспект", "пр", "площадь", "пл"],
                "ja": ["通り", "dōri", "町", "chō"],
            },
            "patterns": {
                "en": r"([A-Z][a-zA-Z\s]+(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Lane|Ln\.?|Drive|Dr\.?|Way|Place|Pl\.?))",
                "de": r"([A-Zäöüß][a-zäöüß]+(?:straße|strasse|str\.|weg|platz|gasse|allee))",
            }
        }
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
            except Exception as e:
                config = default_streets
        else:
            config = default_streets
        
        self._street_patterns = config.get("suffixes", {})
    
    def _load_prefix_patterns(self) -> None:
        """Load prefix patterns for hint extraction."""
        config_path = self.config_dir / "prefixes.json"
        
        default_prefixes = {
            "location_hint": [
                "the image is from ", "the photo is from ", "the picture is from ",
                "this is in ", "this is from ", "it's in ", "it is in ",
                "location: ", "narrow it down to ", "narrow down to ",
                "around ", "near ", "somewhere in ", "focus on ", "concentrate on ",
                "look in ", "i think it's in ", "probably in ", "likely in ",
            ],
        }
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
            except Exception as e:
                config = default_prefixes
        else:
            config = default_prefixes
        
        self._prefix_patterns = config
    
    # --- Public API ---
    
    def classify_intent(self, message: str) -> tuple[str, float]:
        """Classify a chat message into an intent.
        
        Args:
            message: User's message
            
        Returns:
            Tuple of (intent_name, confidence)
        """
        self.load()
        
        for pattern in self._intent_patterns:
            matched, confidence = pattern.matches(message)
            if matched:
                return pattern.intent, confidence
        
        return "general", 0.5
    
    def classify_text(self, text: str) -> str:
        """Classify text into a category.
        
        Args:
            text: Text to classify
            
        Returns:
            Category name or "other"
        """
        self.load()
        
        for name, category in self._text_categories.items():
            if category.matches(text):
                return name
        
        return "other"
    
    def extract_search_query(self, message: str) -> str:
        """Extract search query from a 'try search' message.
        
        Args:
            message: User's search request message
            
        Returns:
            Extracted search query
        """
        self.load()
        
        text = message.strip()
        
        # Patterns for extracting search queries (order matters - more specific first)
        patterns = [
            r"^try\s+(?:searching|search)\s+(?:for\s+)?(.+)$",
            r"^(?:search|google|look\s+up|find)\s+(?:for\s+)?(.+)$",
            r"^can you (?:search|google|find)\s+(?:for\s+)?(.+)$",
            r"(?:search|google|look\s+up|find)\s+(?:for\s+)?(.+)",
        ]
        
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m and m.groups():
                query = m.group(1).strip()
                if query:
                    return query
        
        # Fallback: just return the message
        return text
    
    def extract_location_from_hint(self, message: str) -> str:
        """Extract location from a hint message.
        
        Args:
            message: User's hint message
            
        Returns:
            Extracted location text
        """
        self.load()
        
        text = message.strip()
        
        # Try prefix patterns
        for prefix in self._prefix_patterns.get("location_hint", []):
            if text.lower().startswith(prefix.lower()):
                extracted = text[len(prefix):].strip()
                if extracted:
                    return extracted.rstrip('?.!')
        
        # Try regex extraction from intent patterns
        for pattern in self._intent_patterns:
            if pattern.intent == "refine_hint":
                for regex in pattern.compiled_patterns:
                    m = regex.search(text)
                    if m and m.groups():
                        return m.group(1).strip().rstrip('?.!')
        
        return text
    
    def is_street_name(self, text: str, language: str = "en") -> bool:
        """Check if text looks like a street name."""
        self.load()
        
        suffixes = self._street_patterns.get(language, [])
        text_lower = text.lower()
        
        return any(text_lower.endswith(s.lower()) for s in suffixes)
    
    def get_street_suffixes(self, language: str = "en") -> list[str]:
        """Get street suffixes for a language."""
        self.load()
        return self._street_patterns.get(language, [])
    
    def get_all_languages(self) -> list[str]:
        """Get all supported languages."""
        self.load()
        return list(self._street_patterns.keys())


# Convenience functions
def classify_intent(message: str) -> tuple[str, float]:
    """Classify a chat message into an intent."""
    return PatternRegistry.get().classify_intent(message)


def classify_text(text: str) -> str:
    """Classify text into a category."""
    return PatternRegistry.get().classify_text(text)


def extract_location_hint(message: str) -> str:
    """Extract location from a hint message."""
    return PatternRegistry.get().extract_location_from_hint(message)


def extract_search_query(message: str) -> str:
    """Extract search query from a search request message."""
    return PatternRegistry.get().extract_search_query(message)
