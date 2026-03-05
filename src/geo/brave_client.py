"""Brave Search API client."""

from __future__ import annotations

from typing import Any, Optional

import httpx
from loguru import logger

from src.cache.decorators import cached
from src.cache.store import CacheStore
from src.evidence.chain import Evidence, EvidenceSource
from src.geo.confidence import calculate_search_confidence, safe_coords
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
            transport=httpx.AsyncHTTPTransport(retries=2),
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
        """Convert Brave results to Evidence objects with dynamic confidence."""
        evidences = []
        for position, r in enumerate(results):
            title = r.get("title", "")
            snippet = r.get("snippet", r.get("description", ""))
            link = r.get("link", r.get("url", ""))

            content = f"Brave search for '{query}': {title}"
            if snippet:
                content += f" - {snippet[:200]}"

            # Dynamic confidence based on result quality
            confidence = calculate_search_confidence(
                result=r,
                query=query,
                base_confidence=0.32,  # Slightly lower than Serper (less geo-specific)
                position=position,
            )

            evidences.append(
                Evidence(
                    source=EvidenceSource.BRAVE,  # Fixed: use correct source
                    content=content,
                    confidence=confidence,
                    url=link,
                    metadata={"query": query, "provider": "brave", "title": title, "position": position},
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
        from src.geo.country_matcher import countries_match, extract_country_from_location
        
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
                from src.geo.country_matcher import get_all_names
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
