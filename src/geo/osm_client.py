"""OpenStreetMap / Overpass API client.

Adapted from original geo_interface.py, keeping the valuable
adaptive radius logic and multi-source search patterns.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import overpy
from loguru import logger

from src.cache.decorators import cached
from src.cache.store import CacheStore
from src.evidence.chain import Evidence, EvidenceSource
from src.utils.geo_math import bounding_box, haversine_distance


class OSMClient:
    """OpenStreetMap Overpass API client for POI and place search."""

    def __init__(self, cache: Optional[CacheStore] = None):
        self._api = overpy.Overpass()
        self._cache = cache

    @cached("osm", 86400)
    async def search_nearby(
        self,
        lat: float,
        lon: float,
        radius_km: float = 5.0,
        query_type: str = "all",
    ) -> list[dict[str, Any]]:
        """Search for POIs near coordinates.

        Args:
            lat, lon: Center coordinates
            radius_km: Search radius
            query_type: "all", "business", "transport", "landmark", "street"
        """
        bbox = bounding_box(lat, lon, radius_km)
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        queries = self._build_queries(query_type, bbox_str)
        all_results = []

        for query in queries:
            try:
                result = await asyncio.to_thread(self._api.query, query)
                for node in result.nodes:
                    tags = node.tags
                    name = tags.get("name", "")
                    if name:
                        dist = haversine_distance(lat, lon, float(node.lat), float(node.lon))
                        all_results.append({
                            "name": name,
                            "lat": float(node.lat),
                            "lon": float(node.lon),
                            "type": tags.get("amenity") or tags.get("shop") or tags.get("tourism") or "place",
                            "distance_km": round(dist, 2),
                            "tags": dict(tags),
                        })
            except Exception as e:
                logger.debug("OSM query failed: {}", str(e)[:100])

        # Sort by distance and deduplicate
        seen = set()
        unique = []
        for r in sorted(all_results, key=lambda x: x["distance_km"]):
            key = r["name"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique[:50]

    @cached("osm", 86400)
    async def search_by_name(self, name: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search OSM Nominatim for a place by name."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": name, "format": "json", "limit": limit},
                    headers={"User-Agent": "OpenGeoSpy/2.0"},
                )
                resp.raise_for_status()
                data = resp.json()

            results = []
            for item in data:
                results.append({
                    "name": item.get("display_name", ""),
                    "lat": float(item["lat"]),
                    "lon": float(item["lon"]),
                    "type": item.get("type", "place"),
                    "importance": float(item.get("importance", 0)),
                })
            return results

        except Exception as e:
            logger.error("OSM Nominatim search failed for '{}': {}", name, e)
            return []

    async def reverse_geocode(self, lat: float, lon: float) -> dict[str, Any] | None:
        """Reverse geocode coordinates to address."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://nominatim.openstreetmap.org/reverse",
                    params={"lat": lat, "lon": lon, "format": "json"},
                    headers={"User-Agent": "OpenGeoSpy/2.0"},
                )
                resp.raise_for_status()
                data = resp.json()

            address = data.get("address", {})
            return {
                "display_name": data.get("display_name", ""),
                "country": address.get("country", ""),
                "country_code": address.get("country_code", ""),
                "state": address.get("state", ""),
                "city": address.get("city") or address.get("town") or address.get("village", ""),
                "road": address.get("road", ""),
                "postcode": address.get("postcode", ""),
            }

        except Exception as e:
            logger.error("Reverse geocode failed for ({}, {}): {}", lat, lon, e)
            return None

    def to_evidence(self, results: list[dict], search_context: str = "") -> list[Evidence]:
        """Convert OSM results to Evidence objects."""
        evidences = []
        for r in results[:10]:
            evidences.append(
                Evidence(
                    source=EvidenceSource.OSM,
                    content=f"OSM POI: {r['name']} ({r['type']})",
                    confidence=0.6,
                    latitude=r["lat"],
                    longitude=r["lon"],
                    metadata={
                        "distance_km": r.get("distance_km"),
                        "osm_type": r.get("type"),
                        "search_context": search_context,
                    },
                )
            )
        return evidences

    def _build_queries(self, query_type: str, bbox: str) -> list[str]:
        """Build Overpass QL queries."""
        queries = []

        if query_type in ("all", "business"):
            queries.append(f"""
                [out:json][timeout:10];
                (
                    node["amenity"]["name"]({bbox});
                    node["shop"]["name"]({bbox});
                );
                out body 20;
            """)

        if query_type in ("all", "transport"):
            queries.append(f"""
                [out:json][timeout:10];
                (
                    node["highway"="bus_stop"]["name"]({bbox});
                    node["railway"="station"]["name"]({bbox});
                    node["aeroway"="aerodrome"]["name"]({bbox});
                );
                out body 10;
            """)

        if query_type in ("all", "landmark"):
            queries.append(f"""
                [out:json][timeout:10];
                (
                    node["tourism"]["name"]({bbox});
                    node["historic"]["name"]({bbox});
                );
                out body 10;
            """)

        return queries
