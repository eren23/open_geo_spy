"""Geolocation Reranker - re-ranks search results for geolocation relevance.

P2.7: Prioritizes results with coordinates, addresses, and location mentions.
"""

from __future__ import annotations

from typing import Any

from src.geo.confidence import safe_coords


class GeolocationReranker:
    """Reranks search results based on geolocation-specific signals."""

    def __init__(
        self,
        coord_weight: float = 0.4,
        address_weight: float = 0.3,
        location_mention_weight: float = 0.2,
        source_weight: float = 0.1,
    ):
        self.coord_weight = coord_weight
        self.address_weight = address_weight
        self.location_mention_weight = location_mention_weight
        self.source_weight = source_weight

        # Location keywords that boost relevance
        self.location_keywords = {
            "coordinates", "latitude", "longitude", "gps",
            "located", "location", "address", "street", "avenue",
            "city", "town", "village", "country", "region", "state",
            "near", "at", "in", "map", "place",
        }

        # High-value sources for geolocation
        self.geo_sources = {"osm", "google_maps", "mapillary", "geonames"}

    def rerank(
        self,
        results: list[dict[str, Any]],
        query: str = "",
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank results by geolocation relevance.

        Args:
            results: List of search result dicts
            query: Original query (for context)
            top_n: Return only top N results (None = all)

        Returns:
            Results sorted by geolocation relevance score
        """
        scored = []
        for i, result in enumerate(results):
            score = self._compute_geo_score(result, query, i)
            scored.append((score, result))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [r for _, r in scored]

        if top_n is not None:
            return reranked[:top_n]
        return reranked

    def _compute_geo_score(
        self,
        result: dict[str, Any],
        query: str,
        position: int,
    ) -> float:
        """Compute geolocation relevance score for a single result."""
        score = 0.0

        # Coordinate presence (strongest signal)
        lat, lon = safe_coords(result.get("latitude"), result.get("longitude"))
        if lat is not None and lon is not None:
            score += self.coord_weight

        # Address presence
        address = result.get("address", "") or ""
        if address:
            # Longer addresses are better
            addr_score = min(1.0, len(address) / 50)
            score += self.address_weight * addr_score

        # Location mentions in title/snippet
        title = result.get("title", "") or ""
        snippet = result.get("snippet", "") or result.get("description", "") or ""
        text = f"{title} {snippet}".lower()

        mention_count = sum(1 for kw in self.location_keywords if kw in text)
        mention_score = min(1.0, mention_count / 5)
        score += self.location_mention_weight * mention_score

        # Source quality
        source = result.get("source", "") or ""
        if source.lower() in self.geo_sources:
            score += self.source_weight

        # Position decay (original ranking still matters)
        position_decay = max(0.0, 1.0 - position / 20)
        score = score * 0.8 + position_decay * 0.2

        return round(score, 3)

    def rerank_evidences(
        self,
        evidences: list[Any],
        top_n: int | None = None,
    ) -> list[Any]:
        """Rerank Evidence objects by geolocation relevance.

        Args:
            evidences: List of Evidence objects
            top_n: Return only top N (None = all)

        Returns:
            Evidences sorted by geolocation relevance
        """
        scored = []
        for i, ev in enumerate(evidences):
            result = {
                "latitude": ev.latitude,
                "longitude": ev.longitude,
                "address": ev.metadata.get("address", ""),
                "title": ev.content[:100],
                "snippet": ev.content,
                "source": ev.source.value,
            }
            score = self._compute_geo_score(result, "", i)
            scored.append((score, ev))

        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [e for _, e in scored]

        if top_n is not None:
            return reranked[:top_n]
        return reranked
