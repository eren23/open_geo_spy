"""Serper SERP API client for structured Google search results.

Replaces googlesearch-python (gets blocked) with Serper's reliable API.
Returns structured JSON results without browser scraping.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx
from loguru import logger

from src.cache.decorators import cached
from src.cache.store import CacheStore
from src.evidence.chain import Evidence, EvidenceSource
from src.geo.provider_base import SearchProvider


class SerperClient(SearchProvider):
    """Serper.dev SERP API for Google search results."""

    BASE_URL = "https://google.serper.dev"

    def __init__(self, api_key: str, cache: Optional[CacheStore] = None):
        self.api_key = api_key
        self._cache = cache
        self._client = httpx.AsyncClient(
            timeout=15.0,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            transport=httpx.AsyncHTTPTransport(retries=2),
        )

    @property
    def name(self) -> str:
        return "serper"

    @cached("serper", 7200)
    async def search(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Google web search via Serper API.

        Returns list of result dicts with: title, link, snippet, position.
        """
        try:
            resp = await self._client.post(
                f"{self.BASE_URL}/search",
                json={"q": query, "num": num_results},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("organic", [])
        except Exception as e:
            logger.error("Serper search failed for '{}': {}", query, e)
            return []

    @cached("serper", 7200)
    async def search_places(self, query: str) -> list[dict[str, Any]]:
        """Google Places search via Serper API.

        Returns list of place dicts with: title, address, latitude, longitude, rating, etc.
        """
        try:
            resp = await self._client.post(
                f"{self.BASE_URL}/places",
                json={"q": query},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("places", [])
        except Exception as e:
            logger.error("Serper places search failed for '{}': {}", query, e)
            return []

    async def search_maps(self, query: str) -> list[dict[str, Any]]:
        """Google Maps search via Serper API."""
        try:
            resp = await self._client.post(
                f"{self.BASE_URL}/maps",
                json={"q": query},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("places", [])
        except Exception as e:
            logger.error("Serper maps search failed for '{}': {}", query, e)
            return []

    @cached("serper", 7200)
    async def search_images(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Google Image search via Serper /images endpoint.

        Returns list of image dicts with: imageUrl, thumbnailUrl, title, link, source.
        """
        try:
            resp = await self._client.post(
                f"{self.BASE_URL}/images",
                json={"q": query, "num": num_results},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("images", [])
        except Exception as e:
            logger.error("Serper image search failed for '{}': {}", query, e)
            return []

    def results_to_evidence(self, results: list[dict], query: str) -> list[Evidence]:
        """Convert Serper results to Evidence objects."""
        evidences = []
        for r in results:
            # Extract location mentions from snippets
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            lat = r.get("latitude")
            lon = r.get("longitude")
            address = r.get("address")

            content = f"Search result for '{query}': {title}"
            if snippet:
                content += f" - {snippet[:200]}"
            if address:
                content += f" (Address: {address})"

            evidences.append(
                Evidence(
                    source=EvidenceSource.SERPER,
                    content=content,
                    confidence=0.5,
                    latitude=float(lat) if lat else None,
                    longitude=float(lon) if lon else None,
                    url=link,
                    metadata={"query": query, "title": title},
                )
            )
        return evidences

    async def close(self):
        await self._client.aclose()
