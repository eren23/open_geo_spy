"""Centralized LLM configuration for all model calls.

Eliminates hardcoded model names, temperatures, and token limits
by providing a single source of truth for all LLM call types.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class LLMCallType(str, Enum):
    """Types of LLM calls in the system."""
    
    # Classification & Extraction
    INTENT_CLASSIFY = "intent_classify"
    FEATURE_EXTRACTION = "feature_extraction"
    OCR = "ocr"
    
    # Reasoning & Analysis
    REASONING = "reasoning"
    REASONING_MULTI_CANDIDATE = "reasoning_multi"
    VERIFICATION_QUICK = "verification_quick"
    VERIFICATION_CLAIMS = "verification_claims"
    
    # Geo Resolution
    GEO_COUNTRY_RESOLVE = "geo_country_resolve"
    GEO_LOCATION_RESOLVE = "geo_location_resolve"
    GEO_STREET_RESOLVE = "geo_street_resolve"
    
    # Search & Query
    QUERY_EXPANSION = "query_expansion"
    SMART_QUERY_SUGGESTION = "smart_query_suggestion"
    
    # Evaluation
    JUDGE = "judge"
    
    # VLM (Vision-Language Models)
    VLM_GEO = "vlm_geo"
    VLM_ANALYSIS = "vlm_analysis"
    
    # Chat Handlers
    CHAT_WHY_NOT = "chat_why_not"
    CHAT_EXPLAIN = "chat_explain"
    CHAT_ZOOM_FEATURE = "chat_zoom_feature"
    CHAT_GENERAL = "chat_general"


class ModelTier(str, Enum):
    """Model tier levels for automatic selection."""
    FAST = "fast"          # Cheap, fast responses
    REASONING = "reasoning"  # Better reasoning, more expensive
    HEAVY = "heavy"        # Most capable, most expensive


@dataclass
class LLMCallConfig:
    """Configuration for a specific LLM call type."""
    
    call_type: LLMCallType
    model_tier: ModelTier = ModelTier.FAST
    temperature: float = 0.0
    max_tokens: int = 1000
    timeout_ms: int = 30000
    retry_count: int = 2
    description: str = ""
    
    # Override model name directly (bypasses tier)
    model_override: Optional[str] = None
    
    def get_model(self, settings: Any) -> str:
        """Resolve the actual model name from settings.
        
        Args:
            settings: Settings object with llm.fast_model, llm.reasoning_model, llm.heavy_model
            
        Returns:
            Model name string
        """
        if self.model_override:
            return self.model_override
        
        if self.model_tier == ModelTier.FAST:
            return settings.llm.fast_model
        elif self.model_tier == ModelTier.REASONING:
            return settings.llm.reasoning_model
        else:
            return settings.llm.heavy_model
    
    @classmethod
    def from_dict(cls, data: dict) -> "LLMCallConfig":
        """Create from dictionary."""
        return cls(
            call_type=LLMCallType(data["call_type"]),
            model_tier=ModelTier(data.get("model_tier", "fast")),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens", 1000),
            timeout_ms=data.get("timeout_ms", 30000),
            retry_count=data.get("retry_count", 2),
            description=data.get("description", ""),
            model_override=data.get("model_override"),
        )


# Default configurations embedded for zero-config
DEFAULT_CALL_CONFIGS: dict[LLMCallType, LLMCallConfig] = {
    # Classification - fast, low tokens
    LLMCallType.INTENT_CLASSIFY: LLMCallConfig(
        call_type=LLMCallType.INTENT_CLASSIFY,
        model_tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=50,
        description="Classify user chat intent",
    ),
    
    # Feature extraction - fast, high tokens for detailed output
    LLMCallType.FEATURE_EXTRACTION: LLMCallConfig(
        call_type=LLMCallType.FEATURE_EXTRACTION,
        model_tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=2000,
        description="Extract visual features from image",
    ),
    
    # OCR - fast, high tokens
    LLMCallType.OCR: LLMCallConfig(
        call_type=LLMCallType.OCR,
        model_tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=2000,
        description="Extract text from image",
    ),
    
    # Main reasoning - needs good model, high tokens
    LLMCallType.REASONING: LLMCallConfig(
        call_type=LLMCallType.REASONING,
        model_tier=ModelTier.REASONING,
        temperature=0.1,
        max_tokens=2000,
        description="Main geolocation reasoning",
    ),
    
    LLMCallType.REASONING_MULTI_CANDIDATE: LLMCallConfig(
        call_type=LLMCallType.REASONING_MULTI_CANDIDATE,
        model_tier=ModelTier.REASONING,
        temperature=0.1,
        max_tokens=2000,
        description="Multi-candidate reasoning and ranking",
    ),
    
    # Verification - fast, low tokens
    LLMCallType.VERIFICATION_QUICK: LLMCallConfig(
        call_type=LLMCallType.VERIFICATION_QUICK,
        model_tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=200,
        description="Quick verification of prediction",
    ),
    
    LLMCallType.VERIFICATION_CLAIMS: LLMCallConfig(
        call_type=LLMCallType.VERIFICATION_CLAIMS,
        model_tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=1500,
        description="Detailed claim verification",
    ),
    
    # Geo resolution - fast, very low tokens
    LLMCallType.GEO_COUNTRY_RESOLVE: LLMCallConfig(
        call_type=LLMCallType.GEO_COUNTRY_RESOLVE,
        model_tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=10,
        description="Resolve location hint to country code",
    ),
    
    LLMCallType.GEO_LOCATION_RESOLVE: LLMCallConfig(
        call_type=LLMCallType.GEO_LOCATION_RESOLVE,
        model_tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=500,
        description="Resolve location from description",
    ),
    
    LLMCallType.GEO_STREET_RESOLVE: LLMCallConfig(
        call_type=LLMCallType.GEO_STREET_RESOLVE,
        model_tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=500,
        description="Resolve street information",
    ),
    
    # Query expansion - fast, moderate tokens
    LLMCallType.QUERY_EXPANSION: LLMCallConfig(
        call_type=LLMCallType.QUERY_EXPANSION,
        model_tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=500,
        description="Expand search queries",
    ),
    
    LLMCallType.SMART_QUERY_SUGGESTION: LLMCallConfig(
        call_type=LLMCallType.SMART_QUERY_SUGGESTION,
        model_tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=800,
        description="Smart query suggestion",
    ),
    
    # Judge - fast, moderate tokens
    LLMCallType.JUDGE: LLMCallConfig(
        call_type=LLMCallType.JUDGE,
        model_tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=500,
        description="Evaluate prediction quality",
    ),
    
    # VLM - varies by use case
    LLMCallType.VLM_GEO: LLMCallConfig(
        call_type=LLMCallType.VLM_GEO,
        model_tier=ModelTier.HEAVY,
        temperature=0.1,
        max_tokens=1500,
        description="VLM geolocation analysis",
    ),
    
    LLMCallType.VLM_ANALYSIS: LLMCallConfig(
        call_type=LLMCallType.VLM_ANALYSIS,
        model_tier=ModelTier.FAST,
        temperature=0.0,
        max_tokens=2000,
        description="VLM image analysis",
    ),
    
    # Chat handlers - fast model, varying tokens
    LLMCallType.CHAT_WHY_NOT: LLMCallConfig(
        call_type=LLMCallType.CHAT_WHY_NOT,
        model_tier=ModelTier.REASONING,
        temperature=0.2,
        max_tokens=1000,
        description="Explain why location not chosen",
    ),
    
    LLMCallType.CHAT_EXPLAIN: LLMCallConfig(
        call_type=LLMCallType.CHAT_EXPLAIN,
        model_tier=ModelTier.FAST,
        temperature=0.2,
        max_tokens=800,
        description="Explain evidence for prediction",
    ),
    
    LLMCallType.CHAT_ZOOM_FEATURE: LLMCallConfig(
        call_type=LLMCallType.CHAT_ZOOM_FEATURE,
        model_tier=ModelTier.FAST,
        temperature=0.2,
        max_tokens=600,
        description="Analyze specific visual feature",
    ),
    
    LLMCallType.CHAT_GENERAL: LLMCallConfig(
        call_type=LLMCallType.CHAT_GENERAL,
        model_tier=ModelTier.FAST,
        temperature=0.3,
        max_tokens=600,
        description="General chat response",
    ),
}


class LLMConfig:
    """Centralized LLM configuration manager.
    
    Provides call-specific configurations that can be overridden
    via JSON config files for zero-code customization.
    """
    
    _instance: "LLMConfig | None" = None
    
    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path(__file__).parent
        self._configs: dict[LLMCallType, LLMCallConfig] = {}
        self._loaded = False
    
    @classmethod
    def get(cls) -> "LLMConfig":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
    
    def load(self, force: bool = False) -> None:
        """Load configurations from defaults + config file."""
        if self._loaded and not force:
            return
        
        # Start with defaults
        self._configs = DEFAULT_CALL_CONFIGS.copy()
        
        # Load overrides from config file
        config_path = self.config_dir / "llm_presets.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)
                
                # Apply overrides
                for call_type_str, override in data.get("overrides", {}).items():
                    try:
                        call_type = LLMCallType(call_type_str)
                        base = self._configs.get(call_type)
                        if base:
                            # Merge override with base
                            self._configs[call_type] = LLMCallConfig(
                                call_type=call_type,
                                model_tier=ModelTier(override.get("model_tier", base.model_tier.value)),
                                temperature=override.get("temperature", base.temperature),
                                max_tokens=override.get("max_tokens", base.max_tokens),
                                timeout_ms=override.get("timeout_ms", base.timeout_ms),
                                retry_count=override.get("retry_count", base.retry_count),
                                description=override.get("description", base.description),
                                model_override=override.get("model_override"),
                            )
                            logger.debug("Applied LLM config override for {}", call_type.value)
                    except (ValueError, KeyError) as e:
                        logger.warning("Invalid LLM config override '{}': {}", call_type_str, e)
                
                logger.info("Loaded LLM config overrides from {}", config_path)
            except Exception as e:
                logger.warning("Failed to load LLM config: {}", e)
        
        self._loaded = True
        logger.info("LLMConfig loaded: {} call types configured", len(self._configs))
    
    def get_config(self, call_type: LLMCallType) -> LLMCallConfig:
        """Get configuration for a call type."""
        self.load()
        return self._configs.get(call_type, DEFAULT_CALL_CONFIGS[call_type])
    
    def get_model(self, call_type: LLMCallType, settings: Any) -> str:
        """Get model name for a call type."""
        config = self.get_config(call_type)
        return config.get_model(settings)
    
    def get_params(self, call_type: LLMCallType, settings: Any) -> dict[str, Any]:
        """Get all parameters for an LLM call.
        
        Returns dict with: model, temperature, max_tokens
        """
        config = self.get_config(call_type)
        return {
            "model": config.get_model(settings),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }


# Convenience function for quick access
def get_llm_params(call_type: LLMCallType, settings: Any) -> dict[str, Any]:
    """Get LLM call parameters.
    
    Usage:
        params = get_llm_params(LLMCallType.REASONING, settings)
        response = await client.chat.completions.create(
            **params,
            messages=[{"role": "user", "content": prompt}],
        )
    """
    return LLMConfig.get().get_params(call_type, settings)


def get_llm_model(call_type: LLMCallType, settings: Any) -> str:
    """Get model name for a call type."""
    return LLMConfig.get().get_model(call_type, settings)
