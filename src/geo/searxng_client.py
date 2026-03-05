"""SearXNG self-hosted search client."""

from __future__ import annotations

from typing import Any, Optional

import httpx
from loguru import logger

from src.cache.decorators import cached
from src.cache.store import CacheStore
from src.evidence.chain import Evidence, EvidenceSource
from src.geo.confidence import calculate_search_confidence, safe_coords
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
        """Convert SearXNG results to Evidence objects with dynamic confidence."""
        evidences = []
        for position, r in enumerate(results):
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            engine = r.get("engine", "")

            content = f"SearXNG search for '{query}': {title}"
            if snippet:
                content += f" - {snippet[:200]}"
            if engine:
                content += f" [via {engine}]"

            # Dynamic confidence based on result quality
            confidence = calculate_search_confidence(
                result=r,
                query=query,
                base_confidence=0.30,  # Self-hosted, variable quality
                position=position,
            )

            evidences.append(
                Evidence(
                    source=EvidenceSource.SEARXNG,  # Fixed: use correct source
                    content=content,
                    confidence=confidence,
                    url=link,
                    metadata={
                        "query": query,
                        "provider": "searxng",
                        "engine": engine,
                        "title": title,
                        "position": position,
                    },
                )
            )
        return evidences

    def filter_by_country_hint(
        self,
        results: list[dict[str, Any]],
        country_hint: str,
        boost_factor: float = 1.5,
    ) -> list[dict[str, Any]]:
        """Filter and boost results based on country hint.
        
        Args:
            results: Search results to filter
            country_hint: Country name or ISO code from user hint
            boost_factor: Confidence multiplier for matching results
            
        Returns:
            Filtered and reranked results
        """
        from src.geo.country_matcher import countries_match, extract_country_from_location, get_all_names
        
        target_iso = extract_country_from_location(country_hint)
        if not target_iso:
            return results
        
        scored_results = []
        for r in results:
            # Extract country from result if available
            result_country = r.get("country")
            snippet = (r.get("snippet", "") + " " + r.get("title", "")).lower()
            
            # Use robust country matching
            country_match = False
            if result_country:
                country_match = countries_match(country_hint, result_country)
            else:
                # Check if country name/ISO appears in snippet/title
                all_names = get_all_names(target_iso)
                for name in [target_iso.lower(), country_hint.lower()] + [n.lower() for n in all_names]:
                    if name in snippet:
                        country_match = True
                        break
            
            # Score based on match
            if country_match:
                r["_country_match"] = True
                r["_score_boost"] = boost_factor
            else:
                r["_country_match"] = False
                r["_score_boost"] = 0.5  # Penalty for non-matching
            
            scored_results.append(r)
        
        # Sort by country match first, then by original position
        scored_results.sort(key=lambda x: (not x.get("_country_match", False), results.index(x) if x in results else 999))
        return scored_results

    async def close(self):
        await self._client.aclose()
