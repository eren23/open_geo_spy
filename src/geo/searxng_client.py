"""SearXNG self-hosted search client."""

from __future__ import annotations

from typing import Any, Optional

import httpx
from loguru import logger

from src.cache.decorators import cached
from src.cache.store import CacheStore
from src.evidence.chain import Evidence, EvidenceSource
from src.geo.provider_base import SearchProvider


class SearXNGClient(SearchProvider):
    """SearXNG metasearch engine client."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        cache: Optional[CacheStore] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._cache = cache
        self._client = httpx.AsyncClient(
            timeout=15.0,
            transport=httpx.AsyncHTTPTransport(retries=2),
        )

    @property
    def name(self) -> str:
        return "searxng"

    @cached("searxng", 3600)
    async def search(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Search via SearXNG API."""
        try:
            resp = await self._client.get(
                f"{self.base_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "general",
                    "engines": "google,bing,duckduckgo",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "engine": item.get("engine", ""),
                })
            return results

        except Exception as e:
            logger.error("SearXNG search failed for '{}': {}", query, e)
            return []

    @cached("searxng", 3600)
    async def search_images(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Image search via SearXNG."""
        try:
            resp = await self._client.get(
                f"{self.base_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "images",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("results", [])[:num_results]:
                results.append({
                    "imageUrl": item.get("img_src", ""),
                    "thumbnailUrl": item.get("thumbnail_src", ""),
                    "title": item.get("title", ""),
                    "link": item.get("url", ""),
                    "source": item.get("source", ""),
                })
            return results

        except Exception as e:
            logger.error("SearXNG image search failed for '{}': {}", query, e)
            return []

    def results_to_evidence(self, results: list[dict], query: str) -> list[Evidence]:
        """Convert SearXNG results to Evidence objects."""
        evidences = []
        for r in results:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")

            content = f"SearXNG search for '{query}': {title}"
            if snippet:
                content += f" - {snippet[:200]}"

            evidences.append(
                Evidence(
                    source=EvidenceSource.SERPER,
                    content=content,
                    confidence=0.45,
                    url=link,
                    metadata={
                        "query": query,
                        "provider": "searxng",
                        "engine": r.get("engine", ""),
                        "title": title,
                    },
                )
            )
        return evidences

    async def close(self):
        await self._client.aclose()
