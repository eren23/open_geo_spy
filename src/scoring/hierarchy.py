"""Hierarchical geographic resolution with per-level grounding.

Inspired by LocationAgent's RER hierarchy and GeoToken's S2 cell
coarse-to-fine approach.  Resolves predictions from continent down
to coordinates, grounding at each level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

from src.evidence.chain import EvidenceChain
from src.scoring.grounding import GroundingEngine, GroundingResult, GroundingVerdict


class GeoLevel(IntEnum):
    """Geographic resolution levels, coarse to fine."""

    CONTINENT = 0
    COUNTRY = 1
    REGION = 2
    CITY = 3
    COORDINATES = 4


# Maps continent names to common aliases for matching
CONTINENT_COUNTRIES: dict[str, list[str]] = {
    "europe": ["germany", "france", "italy", "spain", "uk", "united kingdom", "poland", "netherlands", "belgium", "austria", "switzerland", "portugal", "greece", "sweden", "norway", "denmark", "finland", "ireland", "czech republic", "romania", "hungary", "croatia", "turkey"],
    "asia": ["china", "japan", "india", "south korea", "thailand", "vietnam", "indonesia", "malaysia", "philippines", "singapore", "taiwan", "pakistan", "bangladesh"],
    "north america": ["united states", "usa", "canada", "mexico"],
    "south america": ["brazil", "argentina", "chile", "colombia", "peru", "venezuela", "ecuador"],
    "africa": ["south africa", "nigeria", "egypt", "kenya", "morocco", "ethiopia", "tanzania", "ghana"],
    "oceania": ["australia", "new zealand"],
}


@dataclass
class LevelGrounding:
    """Grounding result for a specific geographic level."""

    level: GeoLevel
    value: str | None  # The predicted value at this level
    grounding: GroundingResult | None = None

    @property
    def is_grounded(self) -> bool:
        if not self.grounding:
            return False
        return self.grounding.verdict in (GroundingVerdict.GROUNDED, GroundingVerdict.SUPPORTED)


@dataclass
class HierarchicalPrediction:
    """A prediction grounded at multiple geographic levels."""

    # Raw prediction fields
    country: str | None = None
    region: str | None = None
    city: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    confidence: float = 0.0

    # Per-level grounding
    groundings: dict[str, LevelGrounding] = field(default_factory=dict)

    @property
    def resolved_level(self) -> GeoLevel:
        """Finest level that is GROUNDED or SUPPORTED."""
        for level in reversed(GeoLevel):
            key = level.name.lower()
            lg = self.groundings.get(key)
            if lg and lg.is_grounded:
                return level
        return GeoLevel.CONTINENT

    @property
    def effective_confidence(self) -> float:
        """Weighted blend of per-level grounding confidences."""
        weights = {
            GeoLevel.COUNTRY: 0.3,
            GeoLevel.REGION: 0.15,
            GeoLevel.CITY: 0.25,
            GeoLevel.COORDINATES: 0.3,
        }
        total_weight = 0.0
        weighted_conf = 0.0

        for level, w in weights.items():
            key = level.name.lower()
            lg = self.groundings.get(key)
            if lg and lg.grounding:
                weighted_conf += w * lg.grounding.confidence
                total_weight += w

        if total_weight == 0:
            return self.confidence
        return round(weighted_conf / total_weight, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "country": self.country,
            "region": self.region,
            "city": self.city,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "confidence": self.confidence,
            "effective_confidence": self.effective_confidence,
            "resolved_level": self.resolved_level.name.lower(),
            "groundings": {
                k: {
                    "level": v.level.name.lower(),
                    "value": v.value,
                    "grounding": v.grounding.to_dict() if v.grounding else None,
                }
                for k, v in self.groundings.items()
            },
        }


class HierarchicalResolver:
    """Resolves a raw LLM prediction into a hierarchical, grounded prediction.

    Grounds at each geographic level (country, region, city, coordinates)
    and exposes the finest grounded level.
    """

    def __init__(self, engine: GroundingEngine | None = None):
        self.engine = engine or GroundingEngine()

    def resolve(
        self,
        prediction: dict[str, Any],
        chain: EvidenceChain,
    ) -> HierarchicalPrediction:
        """Take a raw prediction dict and ground it at each level.

        Args:
            prediction: {"country", "region", "city", "lat"/"latitude",
                         "lon"/"longitude", "confidence", ...}
            chain: Evidence chain to ground against.

        Returns:
            HierarchicalPrediction with per-level grounding results.
        """
        country = prediction.get("country")
        region = prediction.get("region")
        city = prediction.get("city")
        lat = prediction.get("lat")
        if lat is None:
            lat = prediction.get("latitude")
        lon = prediction.get("lon")
        if lon is None:
            lon = prediction.get("longitude")
        confidence = prediction.get("confidence", 0.0)

        hp = HierarchicalPrediction(
            country=country,
            region=region,
            city=city,
            latitude=lat,
            longitude=lon,
            confidence=confidence,
        )

        # Ground each level
        if country:
            result = self.engine.ground_country(country, chain)
            hp.groundings["country"] = LevelGrounding(
                level=GeoLevel.COUNTRY, value=country, grounding=result,
            )

        if region:
            result = self.engine.ground_region(region, country, chain)
            hp.groundings["region"] = LevelGrounding(
                level=GeoLevel.REGION, value=region, grounding=result,
            )

        if city:
            result = self.engine.ground_city(city, chain)
            hp.groundings["city"] = LevelGrounding(
                level=GeoLevel.CITY, value=city, grounding=result,
            )

        if lat is not None and lon is not None:
            result = self.engine.ground_coordinates(lat, lon, chain)
            hp.groundings["coordinates"] = LevelGrounding(
                level=GeoLevel.COORDINATES,
                value=f"({lat:.4f}, {lon:.4f})",
                grounding=result,
            )

        return hp
