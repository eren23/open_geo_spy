"""Single Pydantic model holding ALL tunable scoring parameters.

Default values reproduce the current hardcoded behavior exactly (zero behavioral
change on upgrade).  Load overrides from JSON via ``ScoringConfig.from_file()``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class GeoAgreementThresholds(BaseModel):
    """Thresholds for mapping geographic spread (km) to agreement score."""

    tight_km: float = 50.0
    tight_score: float = 1.0
    good_km: float = 200.0
    good_score: float = 0.7
    weak_km: float = 500.0
    weak_score: float = 0.4
    poor_score: float = 0.2
    single_evidence_score: float = 0.3
    no_evidence_score: float = 0.0

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "GeoAgreementThresholds":
        if self.tight_km >= self.good_km:
            raise ValueError(f"tight_km ({self.tight_km}) must be < good_km ({self.good_km})")
        if self.good_km >= self.weak_km:
            raise ValueError(f"good_km ({self.good_km}) must be < weak_km ({self.weak_km})")
        return self

    def score(self, spread_km: float) -> float:
        """Continuous interpolation from spread distance to agreement score."""
        if spread_km < self.tight_km:
            return self.tight_score
        if spread_km < self.good_km:
            t = (spread_km - self.tight_km) / (self.good_km - self.tight_km)
            return self.tight_score + t * (self.good_score - self.tight_score)
        if spread_km < self.weak_km:
            t = (spread_km - self.good_km) / (self.weak_km - self.good_km)
            return self.good_score + t * (self.weak_score - self.good_score)
        return self.poor_score


class AgreementBlend(BaseModel):
    """Weights for blending geographic and country agreement."""

    geo_weight: float = 0.6
    country_weight: float = 0.4


class EnsembleBlend(BaseModel):
    """Weights for ML ensemble agreement blending."""

    country_weight: float = 0.5
    geo_weight: float = 0.5


class CandidateRankingWeights(BaseModel):
    """Weights for the composite candidate ranking score."""

    confidence: float = 0.40
    evidence_count: float = 0.10
    source_diversity: float = 0.10
    visual_match: float = 0.15
    country_match: float = 0.25
    evidence_count_cap: int = 5
    evidence_count_normalizer: float = 5.0
    source_diversity_normalizer: float = 5.0


class CountryPenalty(BaseModel):
    """Parameters for penalizing wrong-country candidates."""

    consensus_threshold: float = 0.3
    penalty_factor: float = 0.7
    hint_vote_multiplier: int = 3


class HintAdjustment(BaseModel):
    """Confidence adjustments for location hint matching."""

    match_boost: float = 1.5
    no_match_penalty: float = 0.5


class VerificationAdjustment(BaseModel):
    """Confidence adjustments from CoVe verification."""

    supported_boost: float = 1.1
    contradicted_penalty: float = 0.5
    uncertain_penalty: float = 0.9
    # Full CoVe claim-level adjustments
    majority_contradicted_penalty: float = 0.3
    majority_contradicted_floor: float = 0.1
    majority_verified_boost: float = 1.1
    partial_verification_factor: float = 0.8


class SourceConfidence(BaseModel):
    """Default confidence values for different evidence sources."""

    # Feature extraction (features.py)
    landmark: float = 0.8
    architecture: float = 0.6
    traffic_side: float = 0.7
    country_clue: float = 0.65
    cultural_indicator: float = 0.55
    environment_type: float = 0.7

    # OCR (ocr.py)
    license_plate: float = 0.8
    street_sign: float = 0.75
    business_name: float = 0.7
    language_detected: float = 0.6

    # VLM geo (vlm_geo.py)
    vlm_country_boost: float = 0.1
    vlm_alternative_penalty: float = 0.2
    vlm_alternative_floor: float = 0.1

    # Metadata (metadata.py)
    exif_gps: float = 0.95
    timestamp: float = 0.5

    # User hint (feature_agent.py)
    user_hint: float = 0.8


class VisualMatchMapping(BaseModel):
    """Maps CLIP cosine similarity to confidence."""

    low_similarity: float = 0.5
    low_confidence: float = 0.1
    high_similarity: float = 0.98
    high_confidence: float = 0.95


class EnvironmentEvidenceWeights(BaseModel):
    """Environment-aware evidence weights per evidence type per environment."""

    weights: dict[str, dict[str, float]] = Field(default_factory=lambda: {
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
    })

    def get(self, env_type: str) -> dict[str, float]:
        return self.weights.get(env_type, {})


class EnvironmentBlend(BaseModel):
    """Blending ratio for environment adjustment."""

    llm_weight: float = 0.6
    env_weight: float = 0.4


class RefinementThresholds(BaseModel):
    """Thresholds that trigger pipeline refinement loops."""

    min_geographic_agreement: float = 0.5
    min_country_agreement: float = 0.6
    min_top_confidence: float = 0.5
    min_evidence_sources: int = 3
    max_iterations: int = 2


class ClusteringParams(BaseModel):
    """Geographic clustering parameters."""

    eps_km: float = 50.0
    cluster_confidence_decay: float = 0.8
    evidence_redistribution_radius_km: float = 200.0
    max_evidence_per_candidate: int = 15


class FallbackParams(BaseModel):
    """Parameters for fallback prediction from evidence centroid."""

    agreement_discount: float = 0.7


class CandidateVerificationParams(BaseModel):
    """Parameters for candidate verification agent."""

    max_candidates: int = 8
    max_refs_per_candidate: int = 5
    mapillary_radius_m: int = 500


# ---------------------------------------------------------------------------
# Root ScoringConfig
# ---------------------------------------------------------------------------


class ScoringConfig(BaseModel):
    """Single model holding ALL tunable scoring parameters."""

    geo_agreement: GeoAgreementThresholds = GeoAgreementThresholds()
    agreement_blend: AgreementBlend = AgreementBlend()
    ensemble_blend: EnsembleBlend = EnsembleBlend()
    candidate_ranking: CandidateRankingWeights = CandidateRankingWeights()
    country_penalty: CountryPenalty = CountryPenalty()
    hint: HintAdjustment = HintAdjustment()
    verification: VerificationAdjustment = VerificationAdjustment()
    source_confidence: SourceConfidence = SourceConfidence()
    visual_match: VisualMatchMapping = VisualMatchMapping()
    environment_weights: EnvironmentEvidenceWeights = EnvironmentEvidenceWeights()
    environment_blend: EnvironmentBlend = EnvironmentBlend()
    refinement: RefinementThresholds = RefinementThresholds()
    clustering: ClusteringParams = ClusteringParams()
    fallback: FallbackParams = FallbackParams()
    candidate_verification: CandidateVerificationParams = CandidateVerificationParams()

    @classmethod
    def from_file(cls, path: str | Path) -> ScoringConfig:
        """Load config from a JSON file."""
        with open(path) as f:
            return cls.model_validate(json.load(f))

    def to_file(self, path: str | Path) -> None:
        """Save config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    def diff(self, other: ScoringConfig) -> dict[str, Any]:
        """Return fields that differ between two configs."""
        a = self.model_dump()
        b = other.model_dump()
        return _diff_dicts(a, b)


def _diff_dicts(a: dict, b: dict, prefix: str = "") -> dict[str, Any]:
    diffs = {}
    for key in set(a) | set(b):
        path = f"{prefix}.{key}" if prefix else key
        va, vb = a.get(key), b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            sub = _diff_dicts(va, vb, path)
            diffs.update(sub)
        elif va != vb:
            diffs[path] = {"old": va, "new": vb}
    return diffs
