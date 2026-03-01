"""Evidence-based grounding verdicts.

Replaces opaque ``confidence=0.72`` with explainable verdicts like
``GROUNDED (3 supporting from 2 sources, 0 contradicting)``.

Inspired by visionbot's GroundingResult pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.evidence.chain import Evidence, EvidenceChain
from src.utils.geo_math import haversine_distance


class GroundingVerdict(str, Enum):
    """How well a claim is supported by independent evidence."""

    GROUNDED = "grounded"          # Multi-source support, no contradictions
    SUPPORTED = "supported"        # Some support, minor gaps
    UNCERTAIN = "uncertain"        # Mixed or insufficient evidence
    WEAKENED = "weakened"          # Some contradicting evidence
    CONTRADICTED = "contradicted"  # Strong contradicting evidence


@dataclass
class GroundingResult:
    """Result of grounding a geographic claim against evidence."""

    verdict: GroundingVerdict
    confidence: float  # 0-1 derived from evidence, not from LLM
    supporting: list[Evidence] = field(default_factory=list)
    contradicting: list[Evidence] = field(default_factory=list)
    source_count: int = 0
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 3),
            "supporting_count": len(self.supporting),
            "contradicting_count": len(self.contradicting),
            "source_count": self.source_count,
            "explanation": self.explanation,
        }


class GroundingEngine:
    """Grounds geographic claims against an evidence chain.

    Instead of trusting LLM confidence, checks claims against independent
    evidence and produces explainable verdicts.
    """

    def __init__(
        self,
        proximity_km: float = 200.0,
        tight_proximity_km: float = 50.0,
        min_sources_for_grounded: int = 2,
    ):
        self.proximity_km = proximity_km
        self.tight_proximity_km = tight_proximity_km
        self.min_sources_for_grounded = min_sources_for_grounded

    def ground_country(
        self,
        country: str,
        chain: EvidenceChain,
    ) -> GroundingResult:
        """Ground a country prediction against the evidence chain."""
        if not country:
            return GroundingResult(
                verdict=GroundingVerdict.UNCERTAIN,
                confidence=0.0,
                explanation="No country claim to ground",
            )

        country_lower = country.lower()
        supporting = []
        contradicting = []
        sources = set()

        for e in chain.evidences:
            if not e.country:
                continue
            if e.country.lower() == country_lower:
                supporting.append(e)
                sources.add(e.source.value)
            else:
                contradicting.append(e)

        return self._compute_verdict(supporting, contradicting, sources,
                                      f"country={country}")

    def ground_coordinates(
        self,
        lat: float,
        lon: float,
        chain: EvidenceChain,
    ) -> GroundingResult:
        """Ground coordinates against geo evidence via proximity."""
        supporting = []
        contradicting = []
        sources = set()

        for e in chain.geo_evidences:
            dist = haversine_distance(lat, lon, e.latitude, e.longitude)
            if dist < self.proximity_km:
                supporting.append(e)
                sources.add(e.source.value)
            elif dist > self.proximity_km * 5:
                contradicting.append(e)

        return self._compute_verdict(supporting, contradicting, sources,
                                      f"coords=({lat:.4f}, {lon:.4f})")

    def ground_region(
        self,
        region: str,
        country: str | None,
        chain: EvidenceChain,
    ) -> GroundingResult:
        """Ground a region prediction."""
        if not region:
            return GroundingResult(
                verdict=GroundingVerdict.UNCERTAIN,
                confidence=0.0,
                explanation="No region claim to ground",
            )

        region_lower = region.lower()
        supporting = []
        contradicting = []
        sources = set()

        country_lower = country.lower() if country else None
        for e in chain.evidences:
            if not e.region:
                continue
            if e.region.lower() == region_lower:
                supporting.append(e)
                sources.add(e.source.value)
            elif country_lower and e.country and e.country.lower() != country_lower:
                # Different country entirely — contradicts region claim
                contradicting.append(e)
            elif country_lower and e.country and e.country.lower() == country_lower:
                # Same country but different region
                contradicting.append(e)

        return self._compute_verdict(supporting, contradicting, sources,
                                      f"region={region}")

    def ground_city(
        self,
        city: str,
        chain: EvidenceChain,
    ) -> GroundingResult:
        """Ground a city prediction."""
        if not city:
            return GroundingResult(
                verdict=GroundingVerdict.UNCERTAIN,
                confidence=0.0,
                explanation="No city claim to ground",
            )

        city_lower = city.lower()
        supporting = []
        contradicting = []
        sources = set()

        for e in chain.evidences:
            if not e.city:
                continue
            if e.city.lower() == city_lower:
                supporting.append(e)
                sources.add(e.source.value)
            else:
                contradicting.append(e)

        return self._compute_verdict(supporting, contradicting, sources,
                                      f"city={city}")

    def _compute_verdict(
        self,
        supporting: list[Evidence],
        contradicting: list[Evidence],
        sources: set[str],
        claim_desc: str,
    ) -> GroundingResult:
        """Compute verdict from supporting/contradicting evidence counts."""
        n_sup = len(supporting)
        n_con = len(contradicting)
        n_sources = len(sources)

        if n_sup == 0 and n_con == 0:
            return GroundingResult(
                verdict=GroundingVerdict.UNCERTAIN,
                confidence=0.3,
                supporting=supporting,
                contradicting=contradicting,
                source_count=n_sources,
                explanation=f"{claim_desc}: no relevant evidence",
            )

        if n_con > n_sup:
            verdict = GroundingVerdict.CONTRADICTED
            confidence = max(0.05, 0.2 * (n_sup / (n_sup + n_con)))
        elif n_con > 0 and n_con >= n_sup / 2:
            verdict = GroundingVerdict.WEAKENED
            confidence = 0.3 + 0.2 * (n_sup / (n_sup + n_con))
        elif n_sources >= self.min_sources_for_grounded and n_sup >= 3:
            verdict = GroundingVerdict.GROUNDED
            confidence = min(0.95, 0.6 + 0.1 * min(n_sources, 4))
        elif n_sup >= 1:
            verdict = GroundingVerdict.SUPPORTED
            confidence = min(0.8, 0.4 + 0.1 * n_sup + 0.05 * n_sources)
        else:
            verdict = GroundingVerdict.UNCERTAIN
            confidence = 0.3

        explanation = (
            f"{claim_desc}: {n_sup} supporting from {n_sources} sources, "
            f"{n_con} contradicting"
        )

        return GroundingResult(
            verdict=verdict,
            confidence=round(confidence, 3),
            supporting=supporting,
            contradicting=contradicting,
            source_count=n_sources,
            explanation=explanation,
        )
