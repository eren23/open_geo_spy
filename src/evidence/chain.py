"""Evidence tracking with source hashing and deduplication.

Every piece of information discovered during geolocation is recorded as an Evidence
object with a content hash for deduplication and a source for traceability.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from loguru import logger

from src.utils.geo_math import haversine_distance, validate_coordinates, weighted_centroid


class EvidenceSource(str, Enum):
    EXIF = "exif"
    VLM_ANALYSIS = "vlm_analysis"
    VLM_GEO = "vlm_geo"
    OCR = "ocr"
    GEOCLIP = "geoclip"
    STREETCLIP = "streetclip"
    PIGEON = "pigeon"
    SERPER = "serper"
    BRAVE = "brave"
    SEARXNG = "searxng"
    GOOGLE_MAPS = "google_maps"
    OSM = "osm"
    BROWSER = "browser"
    GEONAMES = "geonames"
    USER_HINT = "user_hint"
    REASONING = "reasoning"
    VISUAL_MATCH = "visual_match"
    MAPILLARY = "mapillary"


@dataclass
class Evidence:
    """A single piece of geolocation evidence."""

    source: EvidenceSource
    content: str
    confidence: float  # 0-1, derived from source reliability
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    url: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str = ""
    metadata: dict = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)  # Agent/step path
    derived_from: list[str] = field(default_factory=list)  # Parent evidence hashes
    is_negative: bool = False  # P2.4: Mark evidence that contradicts hypotheses

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                f"{self.source.value}:{self.content}:{self.is_negative}".encode()
            ).hexdigest()[:16]

        # Validate coordinates if provided
        if self.latitude is not None and self.longitude is not None:
            if not validate_coordinates(self.latitude, self.longitude):
                self.latitude = None
                self.longitude = None

        self.confidence = max(0.0, min(1.0, self.confidence))

    @property
    def has_coordinates(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    def to_dict(self) -> dict:
        return {
            "source": self.source.value,
            "content": self.content,
            "confidence": self.confidence,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "country": self.country,
            "region": self.region,
            "city": self.city,
            "url": self.url,
            "timestamp": self.timestamp.isoformat(),
            "content_hash": self.content_hash,
            "metadata": self.metadata,
            "provenance": self.provenance,
            "derived_from": self.derived_from,
            "is_negative": self.is_negative,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Evidence:
        data = data.copy()
        data["source"] = EvidenceSource(data["source"])
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class EvidenceChain:
    """Collects and aggregates evidence from all agents."""

    evidences: list[Evidence] = field(default_factory=list)
    _hashes: set[str] = field(default_factory=set, repr=False)
    _geo_evidences: list[Evidence] = field(default_factory=list, repr=False)

    def add(self, evidence: Evidence) -> bool:
        """Add evidence if not a duplicate. Returns True if added."""
        if evidence.content_hash in self._hashes:
            return False
        self._hashes.add(evidence.content_hash)
        self.evidences.append(evidence)
        if evidence.has_coordinates:
            self._geo_evidences.append(evidence)
        return True

    def add_many(self, evidences: list[Evidence]) -> int:
        """Add multiple evidences, return count of new ones added."""
        return sum(1 for e in evidences if self.add(e))

    @property
    def geo_evidences(self) -> list[Evidence]:
        """Evidences that have coordinates."""
        return self._geo_evidences

    @property
    def country_predictions(self) -> list[str]:
        """All country predictions from evidence."""
        return [e.country for e in self.evidences if e.country]

    def location_cluster(self) -> Optional[tuple[float, float]]:
        """Weighted centroid of all coordinate evidence."""
        points = [
            (e.latitude, e.longitude, e.confidence)
            for e in self.geo_evidences
        ]
        return weighted_centroid(points)

    def agreement_score(self, scorer=None) -> float:
        """How much the evidence agrees (0-1).

        Based on geographic spread of coordinate predictions and country agreement.
        Accepts an optional GeoScorer for configurable thresholds.
        """
        geo = self.geo_evidences
        if scorer:
            if len(geo) < 2:
                return scorer.single_evidence_agreement() if geo else scorer.no_evidence_agreement()
        else:
            if len(geo) < 2:
                return 0.3 if geo else 0.0

        # P2.5: Temporal weighting - newer evidence gets higher weight
        now = datetime.now(timezone.utc)
        max_age_seconds = 3600  # 1 hour max age for weighting
        
        weighted_coords = []
        for e in geo:
            age_seconds = (now - e.timestamp).total_seconds()
            recency_weight = max(0.5, 1.0 - (age_seconds / max_age_seconds) * 0.5)
            weighted_coords.append((e.latitude, e.longitude, e.confidence * recency_weight))
        
        from src.utils.geo_math import geographic_spread
        spread = geographic_spread([(c[0], c[1]) for c in weighted_coords])

        if scorer:
            geo_agreement = scorer.geo_agreement_score(spread)
        else:
            # Legacy step-function behavior (kept for backward compat when no scorer)
            if spread < 50:
                geo_agreement = 1.0
            elif spread < 200:
                geo_agreement = 0.7
            elif spread < 500:
                geo_agreement = 0.4
            else:
                geo_agreement = 0.2

        # P2.4: Reduce confidence if negative evidence present
        negative_count = sum(1 for e in self.evidences if e.is_negative)
        if negative_count > 0:
            negative_penalty = min(0.3, negative_count * 0.1)  # Cap at 0.3 penalty
            geo_agreement = max(0.0, geo_agreement - negative_penalty)

        countries = self.country_predictions
        if countries:
            from src.utils.geo_math import country_level_agreement

            country_agree = country_level_agreement(countries)
            if scorer:
                return scorer.blend_agreement(geo_agreement, country_agree)
            return 0.6 * geo_agreement + 0.4 * country_agree

        return geo_agreement

    def negative_evidences(self) -> list[Evidence]:
        """Get evidences marked as negative/contradicting (P2.4)."""
        return [e for e in self.evidences if e.is_negative]

    def recency_weighted_confidence(self) -> float:
        """Get average confidence weighted by recency (P2.5)."""
        if not self.evidences:
            return 0.0
        
        now = datetime.now(timezone.utc)
        max_age_seconds = 3600
        
        weighted_sum = 0.0
        weight_sum = 0.0
        for e in self.evidences:
            age_seconds = (now - e.timestamp).total_seconds()
            recency_weight = max(0.5, 1.0 - (age_seconds / max_age_seconds) * 0.5)
            weighted_sum += e.confidence * recency_weight
            weight_sum += recency_weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    def by_source(self, source: EvidenceSource) -> list[Evidence]:
        """Filter evidences by source."""
        return [e for e in self.evidences if e.source == source]

    def top_evidence(self, n: int = 5) -> list[Evidence]:
        """Get top N evidences by confidence."""
        return sorted(self.evidences, key=lambda e: e.confidence, reverse=True)[:n]

    def find_supporting(self, claim_country: str, radius_km: float = 200) -> list[Evidence]:
        """Find evidences that support a location claim."""
        supporting = []
        for e in self.evidences:
            if e.country and e.country.lower() == claim_country.lower():
                supporting.append(e)
        return supporting

    def summary(self) -> dict:
        """Produce a summary of the evidence chain for logging/display."""
        cluster = self.location_cluster()
        return {
            "total_evidences": len(self.evidences),
            "geo_evidences": len(self.geo_evidences),
            "sources": list({e.source.value for e in self.evidences}),
            "countries_mentioned": list(set(self.country_predictions)),
            "agreement_score": round(self.agreement_score(), 3),
            "centroid": {"lat": cluster[0], "lon": cluster[1]} if cluster else None,
            "top_evidence": [e.to_dict() for e in self.top_evidence(10)],
        }

    def to_prompt_context(self) -> str:
        """Format evidence chain as text for LLM prompts."""
        lines = []
        for e in sorted(self.evidences, key=lambda x: x.confidence, reverse=True):
            loc_str = ""
            if e.has_coordinates:
                loc_str = f" [{e.latitude:.4f}, {e.longitude:.4f}]"
            geo_str = ""
            parts = [p for p in [e.city, e.region, e.country] if p]
            if parts:
                geo_str = f" -> {', '.join(parts)}"
            lines.append(
                f"- [{e.source.value}] (conf={e.confidence:.2f}){loc_str}{geo_str}: {e.content}"
            )
        return "\n".join(lines)

    def clear(self) -> None:
        self.evidences.clear()
        self._hashes.clear()
        self._geo_evidences.clear()
    
    def filter_by_hint(self, hint_country: str, keep_non_geo: bool = True) -> "EvidenceChain":
        """Filter evidence to only include items matching or not contradicting the hint country.
        
        This is used to remove evidence from wrong countries when the user provides
        a strong location hint.
        
        Args:
            hint_country: ISO country code or country name from user hint
            keep_non_geo: Whether to keep evidence without country info (default True)
            
        Returns:
            New filtered EvidenceChain
        """
        from src.geo.country_matcher import countries_match
        
        filtered = EvidenceChain()
        for e in self.evidences:
            # Always keep user hints
            if e.source == EvidenceSource.USER_HINT:
                filtered.add(e)
                continue
            
            # Keep evidence without country if keep_non_geo is True
            if not e.country:
                if keep_non_geo:
                    filtered.add(e)
                continue
            
            # Keep evidence that matches the hint country
            if countries_match(hint_country, e.country):
                filtered.add(e)
            else:
                # Mark as filtered out for debugging
                logger.debug(
                    "Filtered out evidence from {} (hint={})",
                    e.country, hint_country
                )
        
        logger.info(
            "Filtered evidence chain: {} -> {} evidences (hint={})",
            len(self.evidences),
            len(filtered.evidences),
            hint_country,
        )
        
        return filtered
    
    def get_hint_from_evidence(self) -> Optional[str]:
        """Extract user hint from evidence chain if present.
        
        Returns:
            The hint text or None
        """
        for e in self.evidences:
            if e.source == EvidenceSource.USER_HINT:
                return e.metadata.get("hint", e.content)
        return None
