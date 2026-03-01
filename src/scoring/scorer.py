"""Unified scoring API — all scoring methods read from ScoringConfig."""

from __future__ import annotations

from src.scoring.config import ScoringConfig


class GeoScorer:
    """Provides scoring methods backed by a single ScoringConfig."""

    def __init__(self, config: ScoringConfig | None = None):
        self.config = config or ScoringConfig()
        sc = self.config.source_confidence
        self._source_conf_map: dict[str, float] = {
            "landmark": sc.landmark,
            "architecture": sc.architecture,
            "traffic_side": sc.traffic_side,
            "country_clue": sc.country_clue,
            "cultural_indicator": sc.cultural_indicator,
            "cultural": sc.cultural_indicator,
            "environment_type": sc.environment_type,
            "environment": sc.environment_type,
            "license_plate": sc.license_plate,
            "street_sign": sc.street_sign,
            "business_name": sc.business_name,
            "language_detected": sc.language_detected,
            "language": sc.language_detected,
            "exif_gps": sc.exif_gps,
            "timestamp": sc.timestamp,
            "user_hint": sc.user_hint,
        }

    # ------------------------------------------------------------------
    # Geographic agreement
    # ------------------------------------------------------------------

    def geo_agreement_score(self, spread_km: float) -> float:
        """Continuous agreement score from geographic spread distance."""
        return self.config.geo_agreement.score(spread_km)

    def single_evidence_agreement(self) -> float:
        return self.config.geo_agreement.single_evidence_score

    def no_evidence_agreement(self) -> float:
        return self.config.geo_agreement.no_evidence_score

    def blend_agreement(self, geo_agreement: float, country_agreement: float) -> float:
        b = self.config.agreement_blend
        return b.geo_weight * geo_agreement + b.country_weight * country_agreement

    def blend_ensemble(self, country_agree: float, geo_agree: float) -> float:
        b = self.config.ensemble_blend
        return b.country_weight * country_agree + b.geo_weight * geo_agree

    # ------------------------------------------------------------------
    # Candidate ranking
    # ------------------------------------------------------------------

    def rank_score(
        self,
        raw_confidence: float,
        evidence_count: int,
        source_diversity: int,
        visual_match: float,
        country_match: float,
    ) -> float:
        w = self.config.candidate_ranking
        return (
            w.confidence * raw_confidence
            + w.evidence_count * min(evidence_count / w.evidence_count_normalizer, 1.0)
            + w.source_diversity * min(source_diversity / w.source_diversity_normalizer, 1.0)
            + w.visual_match * visual_match
            + w.country_match * country_match
        )

    # ------------------------------------------------------------------
    # Country consensus
    # ------------------------------------------------------------------

    def country_penalty(
        self,
        confidence: float,
        candidate_country: str | None,
        dominant_country: str | None,
        consensus_strength: float,
    ) -> float:
        if not dominant_country or not candidate_country:
            return confidence
        if candidate_country.lower() == dominant_country.lower():
            return confidence
        cp = self.config.country_penalty
        if consensus_strength < cp.consensus_threshold:
            return confidence
        return max(0.0, confidence * (1.0 - consensus_strength * cp.penalty_factor))

    def country_match_score(
        self,
        candidate_country: str | None,
        dominant_country: str | None,
        consensus_strength: float,
    ) -> float:
        cp = self.config.country_penalty
        c_country = (candidate_country or "").lower()
        if dominant_country and c_country and consensus_strength >= cp.consensus_threshold:
            return 1.0 if c_country == dominant_country else 0.0
        return 1.0

    @property
    def hint_vote_multiplier(self) -> int:
        return self.config.country_penalty.hint_vote_multiplier

    # ------------------------------------------------------------------
    # Hint adjustment
    # ------------------------------------------------------------------

    def hint_boost(self, confidence: float) -> float:
        return min(1.0, confidence * self.config.hint.match_boost)

    def hint_penalty(self, confidence: float) -> float:
        return confidence * self.config.hint.no_match_penalty

    # ------------------------------------------------------------------
    # Verification adjustment
    # ------------------------------------------------------------------

    def verification_supported(self, confidence: float) -> float:
        return min(1.0, confidence * self.config.verification.supported_boost)

    def verification_contradicted(self, confidence: float) -> float:
        return confidence * self.config.verification.contradicted_penalty

    def verification_uncertain(self, confidence: float) -> float:
        return confidence * self.config.verification.uncertain_penalty

    def verification_majority_contradicted(self, confidence: float) -> float:
        v = self.config.verification
        return max(v.majority_contradicted_floor, confidence * v.majority_contradicted_penalty)

    def verification_majority_verified(self, avg_claim_confidence: float) -> float:
        return min(1.0, avg_claim_confidence * self.config.verification.majority_verified_boost)

    def verification_partial(self, avg_claim_confidence: float) -> float:
        return avg_claim_confidence * self.config.verification.partial_verification_factor

    # ------------------------------------------------------------------
    # Source confidence
    # ------------------------------------------------------------------

    def source_conf(self, source_type: str) -> float:
        return self._source_conf_map.get(source_type, 0.5)

    @property
    def vlm_country_boost(self) -> float:
        return self.config.source_confidence.vlm_country_boost

    @property
    def vlm_alternative_penalty(self) -> float:
        return self.config.source_confidence.vlm_alternative_penalty

    @property
    def vlm_alternative_floor(self) -> float:
        return self.config.source_confidence.vlm_alternative_floor

    # ------------------------------------------------------------------
    # Visual match
    # ------------------------------------------------------------------

    def similarity_to_confidence(self, similarity: float) -> float:
        vm = self.config.visual_match
        if similarity <= vm.low_similarity:
            return vm.low_confidence
        if similarity >= vm.high_similarity:
            return vm.high_confidence
        range_sim = vm.high_similarity - vm.low_similarity
        range_conf = vm.high_confidence - vm.low_confidence
        return vm.low_confidence + (similarity - vm.low_similarity) * (range_conf / range_sim)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def env_weights(self, env_type: str) -> dict[str, float]:
        return self.config.environment_weights.get(env_type)

    def blend_env_confidence(self, llm_conf: float, env_conf: float) -> float:
        b = self.config.environment_blend
        return b.llm_weight * llm_conf + b.env_weight * env_conf

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    @property
    def cluster_eps_km(self) -> float:
        return self.config.clustering.eps_km

    @property
    def cluster_confidence_decay(self) -> float:
        return self.config.clustering.cluster_confidence_decay

    @property
    def evidence_redistribution_radius_km(self) -> float:
        return self.config.clustering.evidence_redistribution_radius_km

    @property
    def max_evidence_per_candidate(self) -> int:
        return self.config.clustering.max_evidence_per_candidate

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    @property
    def refinement(self):
        return self.config.refinement

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def fallback_confidence(self, agreement_score: float) -> float:
        return agreement_score * self.config.fallback.agreement_discount

    # ------------------------------------------------------------------
    # Candidate verification
    # ------------------------------------------------------------------

    @property
    def max_candidates(self) -> int:
        return self.config.candidate_verification.max_candidates

    @property
    def max_refs_per_candidate(self) -> int:
        return self.config.candidate_verification.max_refs_per_candidate

    @property
    def mapillary_radius_m(self) -> int:
        return self.config.candidate_verification.mapillary_radius_m
