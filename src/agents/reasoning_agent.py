"""Reasoning Agent - final LLM synthesis with CoVe verification.

Receives all evidence from prior agents, applies environment-aware weighting
(kept from location_resolver.py), and produces the final location prediction
with evidence-based confidence (no random scores).
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from src.config.settings import Settings
from src.evidence.chain import Evidence, EvidenceChain, EvidenceSource
from src.evidence.verifier import LocationVerifier

# Environment-aware evidence weights (from original location_resolver.py)
EVIDENCE_WEIGHTS = {
    "URBAN": {"license_plates": 0.9, "business_names": 0.8, "street_signs": 0.8, "building_info": 0.7, "landmarks": 0.6, "visual_match": 0.95},
    "SUBURBAN": {"street_signs": 0.9, "building_info": 0.8, "business_names": 0.7, "license_plates": 0.6, "landmarks": 0.7, "visual_match": 0.95},
    "RURAL": {"landmarks": 0.9, "business_names": 0.8, "geographic_features": 0.8, "license_plates": 0.5, "street_signs": 0.6, "visual_match": 0.95},
    "INDUSTRIAL": {"business_names": 0.9, "building_info": 0.8, "street_signs": 0.7, "license_plates": 0.6, "landmarks": 0.5, "visual_match": 0.95},
    "AIRPORT": {"building_info": 0.9, "business_names": 0.8, "landmarks": 0.7, "license_plates": 0.4, "street_signs": 0.5, "visual_match": 0.95},
    "COASTAL": {"landmarks": 0.9, "business_names": 0.8, "geographic_features": 0.8, "street_signs": 0.6, "license_plates": 0.5, "visual_match": 0.95},
    "FOREST": {"landmarks": 0.9, "geographic_features": 0.9, "business_names": 0.6, "license_plates": 0.3, "street_signs": 0.4, "visual_match": 0.95},
    "MOUNTAIN": {"landmarks": 0.9, "geographic_features": 0.9, "business_names": 0.6, "license_plates": 0.3, "street_signs": 0.4, "visual_match": 0.95},
    "DESERT": {"landmarks": 0.9, "geographic_features": 0.9, "business_names": 0.7, "license_plates": 0.4, "street_signs": 0.5, "visual_match": 0.95},
    "PARK": {"landmarks": 0.9, "business_names": 0.7, "street_signs": 0.6, "license_plates": 0.4, "building_info": 0.5, "visual_match": 0.95},
    "HIGHWAY": {"street_signs": 0.9, "landmarks": 0.8, "business_names": 0.7, "license_plates": 0.6, "building_info": 0.5, "visual_match": 0.95},
}

REASONING_PROMPT = """You are an expert geolocation analyst. Given ALL the evidence below, determine the most precise location possible.

## Evidence Chain
{evidence}

## Evidence Summary
- Total evidences: {total}
- Sources: {sources}
- Countries mentioned: {countries}
- Evidence agreement: {agreement:.2f}
- Centroid (if available): {centroid}

## Environment Type
{env_type}

## Instructions
1. Weigh evidence by source reliability and mutual corroboration
2. Prioritize evidence that multiple independent sources agree on
3. If coordinates from multiple models cluster tightly (<50km), that's very strong evidence
4. Country-level agreement across models is highly diagnostic
5. OCR text (signs, plates, businesses) provides regional/local specificity
6. Be SPECIFIC - neighborhood > city > region > country
7. Confidence MUST reflect actual evidence strength, NOT be inflated
8. Visual match evidence (source=visual_match) compares the query image against reference photos
   of candidate locations. HIGH similarity is strong evidence for that specific location.
   This is the BEST evidence for distinguishing between same-category candidates (e.g., two hotels in the same city).

Return your answer as JSON:
{{
  "name": "Most specific location name (city, district if possible)",
  "country": "Country name",
  "region": "State/province",
  "city": "City name",
  "latitude": float,
  "longitude": float,
  "confidence": 0.0-1.0 (based on evidence strength, NOT random),
  "reasoning": "Detailed explanation citing specific evidence",
  "evidence_used": ["list of key evidence pieces that support this conclusion"]
}}
"""


class ReasoningAgent:
    """Final synthesis and verification of geolocation prediction."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
        )
        self.primary_model = settings.llm.reasoning_model
        self.verification_model = settings.llm.verification_model
        self.verifier = LocationVerifier(self.client, settings.llm.fast_model)

    async def reason(
        self,
        evidence_chain: EvidenceChain,
        features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synthesize all evidence into a final location prediction.

        Args:
            evidence_chain: Combined evidence from all agents
            features: Raw visual features (for environment type)

        Returns:
            Final prediction dict with location, confidence, reasoning, evidence trail.
        """
        logger.info("Starting reasoning with {} evidences", len(evidence_chain.evidences))

        # Get environment type for weight adjustment
        env_type = "UNKNOWN"
        if features:
            env_type = features.get("environment_type", "UNKNOWN")

        # Build context
        summary = evidence_chain.summary()
        centroid = summary.get("centroid")
        centroid_str = f"({centroid['lat']:.4f}, {centroid['lon']:.4f})" if centroid else "No centroid available"

        prompt = REASONING_PROMPT.format(
            evidence=evidence_chain.to_prompt_context(),
            total=summary["total_evidences"],
            sources=", ".join(summary["sources"]),
            countries=", ".join(summary["countries_mentioned"]) or "None",
            agreement=summary["agreement_score"],
            centroid=centroid_str,
            env_type=env_type,
        )

        # Primary reasoning
        try:
            resp = await self.client.chat.completions.create(
                model=self.primary_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
            )
            prediction = _parse_prediction(resp.choices[0].message.content)
        except Exception as e:
            logger.error("Primary reasoning failed: {}", e)
            prediction = _fallback_prediction(evidence_chain)

        # Apply environment-specific confidence adjustment
        prediction = self._adjust_for_environment(prediction, evidence_chain, env_type)

        # Verification step (using fast model for speed)
        try:
            is_plausible, adjusted_conf, reason = await self.verifier.quick_verify(
                prediction, evidence_chain
            )
            if not is_plausible:
                logger.warning("Verification flagged prediction: {}", reason)
                prediction["confidence"] = adjusted_conf
                prediction["verification_warning"] = reason
            else:
                prediction["confidence"] = adjusted_conf
                prediction["verified"] = True
        except Exception as e:
            logger.warning("Verification skipped: {}", e)

        # Add evidence trail to result
        prediction["evidence_trail"] = [e.to_dict() for e in evidence_chain.top_evidence(10)]
        prediction["evidence_summary"] = summary

        logger.info(
            "Reasoning complete: {} (conf={:.2f})",
            prediction.get("name", "Unknown"),
            prediction.get("confidence", 0),
        )

        return prediction

    def _adjust_for_environment(
        self,
        prediction: dict,
        evidence_chain: EvidenceChain,
        env_type: str,
    ) -> dict:
        """Apply environment-specific evidence weighting.

        Adapted from location_resolver.py's environment refinement.
        """
        weights = EVIDENCE_WEIGHTS.get(env_type, {})
        if not weights:
            return prediction

        # Count evidence types and apply weights
        type_scores = []
        for e in evidence_chain.evidences:
            e_type = e.metadata.get("type", "")
            for weight_key, weight_value in weights.items():
                if weight_key in e_type or weight_key in e.content.lower():
                    type_scores.append(e.confidence * weight_value)
                    break

        if type_scores:
            env_confidence = sum(type_scores) / len(type_scores)
            # Blend with LLM confidence (60% LLM, 40% environment-weighted)
            original_conf = prediction.get("confidence", 0.5)
            blended = 0.6 * original_conf + 0.4 * env_confidence
            prediction["confidence"] = round(max(0.0, min(1.0, blended)), 3)

        return prediction


def _parse_prediction(raw: str) -> dict[str, Any]:
    """Parse LLM reasoning response."""
    try:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            data = json.loads(match.group())
            return {
                "name": data.get("name", "Unknown"),
                "country": data.get("country"),
                "region": data.get("region"),
                "city": data.get("city"),
                "lat": _safe_float(data.get("latitude")),
                "lon": _safe_float(data.get("longitude")),
                "confidence": max(0.0, min(1.0, _safe_float(data.get("confidence"), 0.0))),
                "reasoning": data.get("reasoning", ""),
                "evidence_used": data.get("evidence_used", []),
            }
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Failed to parse reasoning response: {}", e)

    return {"name": "Unknown", "lat": 0.0, "lon": 0.0, "confidence": 0.0, "reasoning": "Parse failed"}


def _fallback_prediction(chain: EvidenceChain) -> dict[str, Any]:
    """Generate a prediction from evidence centroid when LLM fails."""
    cluster = chain.location_cluster()
    countries = chain.country_predictions
    top_country = max(set(countries), key=countries.count) if countries else None

    if cluster:
        return {
            "name": top_country or "Unknown",
            "country": top_country,
            "lat": cluster[0],
            "lon": cluster[1],
            "confidence": chain.agreement_score() * 0.7,
            "reasoning": "Fallback: centroid of evidence coordinates",
        }
    return {"name": "Unknown", "lat": 0.0, "lon": 0.0, "confidence": 0.0, "reasoning": "No evidence available"}


def _safe_float(val: Any, default: float | None = None) -> float | None:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default
