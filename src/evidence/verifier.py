"""Chain-of-Verification (CoVe) for geolocation claims.

Adapted from visionbot's answer_verifier pattern. Decomposes location claims
into verifiable sub-claims, cross-references against independent evidence,
and detects hallucinations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger
from openai import AsyncOpenAI

from src.evidence.chain import Evidence, EvidenceChain
from src.scoring.scorer import GeoScorer


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    PARTIALLY_SUPPORTED = "partially_supported"


@dataclass
class Claim:
    """An atomic, verifiable sub-claim extracted from a location prediction."""

    text: str
    status: VerificationStatus = VerificationStatus.UNSUPPORTED
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class VerificationResult:
    """Result of verifying a geolocation prediction."""

    verified: bool
    confidence: float
    reason: str
    claims: list[Claim] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    suggested_corrections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verified": self.verified,
            "confidence": self.confidence,
            "reason": self.reason,
            "claims": [
                {
                    "text": c.text,
                    "status": c.status.value,
                    "supporting": c.supporting_evidence,
                    "contradicting": c.contradicting_evidence,
                    "confidence": c.confidence,
                }
                for c in self.claims
            ],
            "contradictions": self.contradictions,
            "suggested_corrections": self.suggested_corrections,
        }


class LocationVerifier:
    """Verifies geolocation predictions using CoVe (Chain-of-Verification).

    Pipeline:
    1. Decompose location prediction into atomic claims
    2. Generate verification questions for each claim
    3. Answer questions using only the evidence chain
    4. Detect contradictions
    5. Calculate final verification result
    """

    def __init__(self, client: AsyncOpenAI, model: str, scorer: GeoScorer | None = None):
        self.client = client
        self.model = model
        self.scorer = scorer or GeoScorer()

    async def verify(
        self,
        prediction: dict,
        evidence_chain: EvidenceChain,
    ) -> VerificationResult:
        """Full CoVe verification of a location prediction.

        Args:
            prediction: {"name": str, "lat": float, "lon": float, "confidence": float, "reasoning": str}
            evidence_chain: All collected evidence
        """
        try:
            # Step 1: Decompose into claims
            claims = await self._decompose_into_claims(prediction)
            if not claims:
                return VerificationResult(
                    verified=False, confidence=0.0, reason="Could not decompose prediction into verifiable claims"
                )

            # Steps 2-4: Verify claims and detect contradictions
            evidence_context = evidence_chain.to_prompt_context()
            verified_claims = await self._verify_claims(claims, evidence_context)

            # Skip contradiction detection with too few claims
            if len(verified_claims) < 2:
                contradictions = []
            else:
                contradictions = self._detect_contradictions(verified_claims, evidence_chain)

            # Step 5: Aggregate
            return self._calculate_result(prediction, verified_claims, contradictions)

        except Exception as e:
            logger.error("Verification failed: {}", str(e))
            return VerificationResult(
                verified=False, confidence=prediction.get("confidence", 0.0), reason=f"Verification error: {str(e)}"
            )

    async def quick_verify(
        self,
        prediction: dict,
        evidence_chain: EvidenceChain,
    ) -> tuple[bool, float, str]:
        """Fast binary verification. Returns (is_plausible, adjusted_confidence, reason)."""
        evidence_context = evidence_chain.to_prompt_context()

        prompt = f"""You are verifying a geolocation prediction. Answer with ONLY "SUPPORTED", "CONTRADICTED", or "UNCERTAIN" followed by a brief reason.

Prediction: {prediction.get('name', 'Unknown')} at ({prediction.get('lat', 0)}, {prediction.get('lon', 0)})
Confidence claimed: {prediction.get('confidence', 0)}

Evidence collected:
{evidence_context}

Is this prediction supported by the evidence? Consider:
1. Do coordinates match the evidence locations?
2. Does the country/region match?
3. Are there contradicting evidence points?
"""
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            answer = resp.choices[0].message.content.strip()

            conf = prediction.get("confidence", 0.5)
            if answer.startswith("SUPPORTED"):
                return True, self.scorer.verification_supported(conf), answer
            elif answer.startswith("CONTRADICTED"):
                return False, self.scorer.verification_contradicted(conf), answer
            else:
                return True, self.scorer.verification_uncertain(conf), answer

        except Exception as e:
            logger.warning("Quick verify failed: {}", str(e))
            return True, prediction.get("confidence", 0.5), f"Verification skipped: {str(e)}"

    async def _decompose_into_claims(self, prediction: dict) -> list[str]:
        """Extract atomic verifiable claims from the prediction."""
        prompt = f"""Decompose this geolocation prediction into atomic, verifiable claims. Each claim should be independently checkable.

Prediction: {prediction.get('name', 'Unknown')}
Coordinates: ({prediction.get('lat', 0)}, {prediction.get('lon', 0)})
Reasoning: {prediction.get('reasoning', 'None provided')}

List each claim on a separate line, prefixed with "- ". Example:
- The location is in Germany
- The city is Berlin
- The coordinates are in the Mitte district
- There is a church visible matching Berliner Dom
"""
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            text = resp.choices[0].message.content.strip()
            claims = [line.lstrip("- ").strip() for line in text.split("\n") if line.strip().startswith("-")]
            return claims
        except Exception as e:
            logger.warning("Claim decomposition failed: {}", str(e))
            # Fallback: create basic claims from prediction fields
            claims = []
            if prediction.get("name"):
                claims.append(f"The location is {prediction['name']}")
            return claims

    async def _verify_claims(self, claims: list[str], evidence_context: str) -> list[Claim]:
        """Verify each claim against the evidence."""
        prompt = f"""For each claim below, determine if it is VERIFIED, CONTRADICTED, UNSUPPORTED, or PARTIALLY_SUPPORTED based ONLY on the evidence provided. For each claim, cite the specific evidence that supports or contradicts it.

Evidence:
{evidence_context}

Claims to verify:
{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(claims))}

For each claim, respond in this exact format:
CLAIM: [claim text]
STATUS: [VERIFIED/CONTRADICTED/UNSUPPORTED/PARTIALLY_SUPPORTED]
SUPPORTING: [list supporting evidence or "none"]
CONTRADICTING: [list contradicting evidence or "none"]
CONFIDENCE: [0.0-1.0]
"""
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1500,
            )
            return self._parse_verification_response(resp.choices[0].message.content, claims)
        except Exception as e:
            logger.warning("Claim verification failed: {}", str(e))
            return [Claim(text=c) for c in claims]

    def _parse_verification_response(self, response: str, original_claims: list[str]) -> list[Claim]:
        """Parse the LLM verification response into Claim objects."""
        results = []
        current_claim = None

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("CLAIM:"):
                if current_claim:
                    results.append(current_claim)
                current_claim = Claim(text=line.split(":", 1)[1].strip())
            elif line.startswith("STATUS:") and current_claim:
                status_str = line.split(":", 1)[1].strip().upper()
                try:
                    current_claim.status = VerificationStatus(status_str.lower())
                except ValueError:
                    current_claim.status = VerificationStatus.UNSUPPORTED
            elif line.startswith("SUPPORTING:") and current_claim:
                val = line.split(":", 1)[1].strip()
                if val.lower() != "none":
                    current_claim.supporting_evidence = [s.strip() for s in val.split(",")]
            elif line.startswith("CONTRADICTING:") and current_claim:
                val = line.split(":", 1)[1].strip()
                if val.lower() != "none":
                    current_claim.contradicting_evidence = [s.strip() for s in val.split(",")]
            elif line.startswith("CONFIDENCE:") and current_claim:
                try:
                    current_claim.confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

        if current_claim:
            results.append(current_claim)

        # If parsing failed, create default claims
        if not results:
            results = [Claim(text=c) for c in original_claims]

        return results

    def _detect_contradictions(self, claims: list[Claim], evidence_chain: EvidenceChain) -> list[str]:
        """Find contradictions between claims and evidence."""
        contradictions = []
        for claim in claims:
            if claim.status == VerificationStatus.CONTRADICTED:
                contradictions.append(
                    f"Claim '{claim.text}' contradicted by: {', '.join(claim.contradicting_evidence)}"
                )
        return contradictions

    def _calculate_result(
        self,
        prediction: dict,
        claims: list[Claim],
        contradictions: list[str],
    ) -> VerificationResult:
        """Aggregate claim verification into final result."""
        if not claims:
            return VerificationResult(
                verified=False, confidence=0.0, reason="No claims to verify"
            )

        verified_count = sum(1 for c in claims if c.status == VerificationStatus.VERIFIED)
        contradicted_count = sum(1 for c in claims if c.status == VerificationStatus.CONTRADICTED)
        partial_count = sum(1 for c in claims if c.status == VerificationStatus.PARTIALLY_SUPPORTED)
        total = len(claims)

        # Weighted confidence from claims
        if total > 0:
            avg_claim_confidence = sum(c.confidence for c in claims) / total
        else:
            avg_claim_confidence = 0.0

        # Determine overall verification
        if contradicted_count > total / 2:
            verified = False
            confidence = self.scorer.verification_majority_contradicted(prediction.get("confidence", 0.5))
            reason = f"{contradicted_count}/{total} claims contradicted"
        elif verified_count > total / 2:
            verified = True
            confidence = self.scorer.verification_majority_verified(avg_claim_confidence)
            reason = f"{verified_count}/{total} claims verified"
        else:
            verified = True
            confidence = self.scorer.verification_partial(avg_claim_confidence)
            reason = f"{verified_count} verified, {partial_count} partial, {contradicted_count} contradicted out of {total}"

        # Generate suggested corrections from contradictions
        corrections = []
        for claim in claims:
            if claim.status == VerificationStatus.CONTRADICTED and claim.contradicting_evidence:
                corrections.append(f"Reconsider: {claim.text} (contradicted by {claim.contradicting_evidence[0]})")

        return VerificationResult(
            verified=verified,
            confidence=round(confidence, 3),
            reason=reason,
            claims=claims,
            contradictions=contradictions,
            suggested_corrections=corrections,
        )
