"""Brave Search API client."""

from __future__ import annotations

from typing import Any, Optional

import httpx
from loguru import logger

from src.cache.decorators import cached
from src.cache.store import CacheStore
from src.evidence.chain import Evidence, EvidenceSource
from src.geo.provider_base import SearchProvider


class BraveClient(SearchProvider):
    """Brave Search API for web and image search."""

    BASE_URL = "https://api.search.brave.com/res/v1"

    def __init__(self, api_key: str, cache: Optional[CacheStore] = None):
        self.api_key = api_key
        self._cache = cache
        self._client = httpx.AsyncClient(
            timeout=15.0,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
        )

    @property
    def name(self) -> str:
        return "brave"

    @cached("brave", 7200)
    async def search(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Brave web search."""
        try:
            resp = await self._client.get(
                f"{self.BASE_URL}/web/search",
                params={"q": query, "count": num_results},
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("url", ""),
                    "snippet": item.get("description", ""),
                })
            return results

        except Exception as e:
            logger.error("Brave search failed for '{}': {}", query, e)
            return []

    @cached("brave", 7200)
    async def search_images(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Brave image search."""
        try:
            resp = await self._client.get(
                f"{self.BASE_URL}/images/search",
                params={"q": query, "count": num_results},
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("results", []):
                results.append({
                    "imageUrl": item.get("properties", {}).get("url", ""),
                    "thumbnailUrl": item.get("thumbnail", {}).get("src", ""),
                    "title": item.get("title", ""),
                    "link": item.get("url", ""),
                    "source": item.get("source", ""),
                })
            return results

        except Exception as e:
            logger.error("Brave image search failed for '{}': {}", query, e)
            return []

    def results_to_evidence(self, results: list[dict], query: str) -> list[Evidence]:
        """Convert Brave results to Evidence objects."""
        evidences = []
        for r in results:
            title = r.get("title", "")
            snippet = r.get("snippet", r.get("description", ""))
            link = r.get("link", r.get("url", ""))

            content = f"Brave search for '{query}': {title}"
            if snippet:
                content += f" - {snippet[:200]}"

            evidences.append(
                Evidence(
                    source=EvidenceSource.SERPER,  # Re-use source for web searches
                    content=content,
                    confidence=0.5,
                    url=link,
                    metadata={"query": query, "provider": "brave", "title": title},
                )
            )
        return evidences

    async def close(self):
        await self._client.aclose()
