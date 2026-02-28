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
        skip_verification: bool = False,
    ) -> dict[str, Any]:
        """Synthesize all evidence into a final location prediction.

        Args:
            evidence_chain: Combined evidence from all agents
            features: Raw visual features (for environment type)
            skip_verification: When True, skip CoVe verification (used by chat handlers
                for faster re-reasoning)

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

        # Full CoVe verification when visual verification is enabled;
        # fall back to quick_verify on error or when disabled.
        # Skip entirely when called from chat handlers for faster response.
        if skip_verification:
            prediction["verified"] = False
        else:
            try:
                if self.settings.ml.enable_visual_verification:
                    result = await self.verifier.verify(prediction, evidence_chain)
                    prediction["confidence"] = result.confidence
                    prediction["verified"] = result.verified
                    prediction["claims"] = [
                        {
                            "text": c.text,
                            "status": c.status.value,
                            "supporting": c.supporting_evidence,
                            "contradicting": c.contradicting_evidence,
                            "confidence": c.confidence,
                        }
                        for c in result.claims
                    ]
                    prediction["contradictions"] = result.contradictions
                    if not result.verified:
                        logger.warning("Full verification flagged prediction: {}", result.reason)
                        prediction["verification_warning"] = result.reason
                else:
                    # Fast-path verification
                    is_plausible, adjusted_conf, reason = await self.verifier.quick_verify(
                        prediction, evidence_chain
                    )
                    prediction["confidence"] = adjusted_conf
                    prediction["verified"] = is_plausible
                    if not is_plausible:
                        prediction["verification_warning"] = reason
            except Exception as e:
                logger.warning("Full verification failed, falling back to quick_verify: {}", e)
                try:
                    is_plausible, adjusted_conf, reason = await self.verifier.quick_verify(
                        prediction, evidence_chain
                    )
                    prediction["confidence"] = adjusted_conf
                    prediction["verified"] = is_plausible
                    if not is_plausible:
                        prediction["verification_warning"] = reason
                except Exception as e2:
                    logger.warning("Quick verification also failed: {}", e2)

        # Add evidence trail to result
        prediction["evidence_trail"] = [e.to_dict() for e in evidence_chain.top_evidence(10)]
        prediction["evidence_summary"] = summary

        logger.info(
            "Reasoning complete: {} (conf={:.2f})",
            prediction.get("name", "Unknown"),
            prediction.get("confidence", 0),
        )

        return prediction

    async def reason_multi_candidate(
        self,
        evidence_chain: EvidenceChain,
        features: dict[str, Any] | None = None,
        max_candidates: int = 5,
        skip_verification: bool = False,
    ) -> list[dict[str, Any]]:
        """Produce top-N ranked candidate locations.

        1. Cluster evidence by geographic proximity (haversine, eps=50km)
        2. For each cluster, synthesize a candidate via LLM
        3. Rank by composite score
        """
        from src.utils.geo_math import haversine_distance

        # Get single prediction as primary
        primary = await self.reason(evidence_chain, features, skip_verification=skip_verification)
        candidates = [primary]

        # Cluster geo evidences
        geo = evidence_chain.geo_evidences
        if len(geo) < 2:
            return self._rank_candidates(candidates)

        clusters = self._cluster_by_proximity(geo, eps_km=50)

        # For each cluster (beyond the primary), create a candidate
        for cluster_evidences in clusters[1:max_candidates]:
            if not cluster_evidences:
                continue

            cluster_chain = EvidenceChain()
            cluster_chain.add_many(cluster_evidences)
            centroid = cluster_chain.location_cluster()

            # Build candidate from cluster centroid + evidence
            countries = cluster_chain.country_predictions
            top_country = max(set(countries), key=countries.count) if countries else None

            candidate = {
                "name": top_country or "Unknown location",
                "country": top_country,
                "region": None,
                "city": None,
                "lat": centroid[0] if centroid else None,
                "lon": centroid[1] if centroid else None,
                "confidence": cluster_chain.agreement_score() * 0.8,
                "reasoning": f"Evidence cluster with {len(cluster_evidences)} data points",
                "evidence_used": [e.content[:80] for e in cluster_evidences[:5]],
                "evidence_trail": [e.to_dict() for e in cluster_evidences[:5]],
                "evidence_summary": cluster_chain.summary(),
            }
            candidates.append(candidate)

        return self._rank_candidates(candidates)

    def _cluster_by_proximity(
        self,
        evidences: list[Evidence],
        eps_km: float = 50,
    ) -> list[list[Evidence]]:
        """Simple iterative clustering by haversine distance."""
        from src.utils.geo_math import haversine_distance

        clusters: list[list[Evidence]] = []
        assigned = [False] * len(evidences)

        for i, e in enumerate(evidences):
            if assigned[i]:
                continue
            cluster = [e]
            assigned[i] = True

            for j in range(i + 1, len(evidences)):
                if assigned[j]:
                    continue
                dist = haversine_distance(
                    e.latitude, e.longitude,
                    evidences[j].latitude, evidences[j].longitude,
                )
                if dist < eps_km:
                    cluster.append(evidences[j])
                    assigned[j] = True

            clusters.append(cluster)

        # Sort clusters by size descending
        clusters.sort(key=len, reverse=True)
        return clusters

    def _rank_candidates(self, candidates: list[dict]) -> list[dict]:
        """Rank candidates by composite score."""
        for i, c in enumerate(candidates):
            evidence_count = len(c.get("evidence_trail", []))
            source_set = set()
            for e in c.get("evidence_trail", []):
                if isinstance(e, dict):
                    source_set.add(e.get("source", ""))

            evidence_agreement = c.get("confidence", 0)
            visual_match = c.get("visual_match_score", 0) or 0
            source_diversity = len(source_set) / 5  # normalize to ~1
            raw_conf = c.get("confidence", 0)

            score = (
                0.4 * evidence_agreement
                + 0.3 * visual_match
                + 0.2 * min(source_diversity, 1.0)
                + 0.1 * raw_conf
            )
            c["_score"] = score
            c["source_diversity"] = len(source_set)

        candidates.sort(key=lambda x: x.get("_score", 0), reverse=True)

        for i, c in enumerate(candidates):
            c["rank"] = i + 1
            c.pop("_score", None)

        return candidates

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
