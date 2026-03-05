"""Serper SERP API client for structured Google search results.

Replaces googlesearch-python (gets blocked) with Serper's reliable API.
Returns structured JSON results without browser scraping.

Supports geographic constraints via gl (geolocation) and cr (country restrict)
parameters to prioritize results from specific countries.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx
from loguru import logger

from src.cache.decorators import cached
from src.cache.store import CacheStore
from src.evidence.chain import Evidence, EvidenceSource
from src.geo.confidence import calculate_search_confidence, safe_coords
from src.geo.country_codes import extract_country_from_hint, get_google_cr, get_google_gl
from src.geo.provider_base import SearchProvider


class SerperClient(SearchProvider):
    """Serper.dev SERP API for Google search results.
    
    Supports geographic constraints to prioritize results from specific countries.
    When a country_hint is provided:
    - gl parameter sets Google's geolocation to that country
    - cr parameter restricts results to pages from that country
    - Results are post-filtered and reranked by country match
    """

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
    async def search(
        self, 
        query: str, 
        num_results: int = 10,
        country_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        """Google web search via Serper API with optional geographic constraints.

        Args:
            query: Search query string
            num_results: Maximum number of results to return
            country_hint: Country name or ISO code to prioritize results from
            
        Returns:
            List of result dicts with: title, link, snippet, position.
        """
        try:
            payload: dict[str, Any] = {"q": query, "num": num_results}
            
            # Apply geographic constraints if country hint provided
            if country_hint:
                iso_code = extract_country_from_hint(country_hint)
                if iso_code:
                    # Set geolocation to prioritize local results
                    payload["gl"] = get_google_gl(iso_code)
                    # Restrict to pages from this country
                    cr = get_google_cr(iso_code)
                    if cr:
                        payload["cr"] = cr
                    logger.debug(
                        "Serper search with geo constraint: gl={}, cr={}",
                        payload.get("gl"), payload.get("cr")
                    )
            
            resp = await self._client.post(
                f"{self.BASE_URL}/search",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("organic", [])
            
            # Post-filter and boost results matching country hint
            if country_hint and results:
                results = self._filter_by_country(results, country_hint)
            
            return results
        except Exception as e:
            logger.error("Serper search failed for '{}': {}", query, e)
            return []

    def _filter_by_country(
        self, 
        results: list[dict], 
        country_hint: str,
        boost_factor: float = 1.5,
        penalty_factor: float = 0.3,
    ) -> list[dict]:
        """Filter and rerank results based on country match.
        
        Args:
            results: Raw search results
            country_hint: Country to prioritize
            boost_factor: Confidence boost for matching results
            penalty_factor: Confidence penalty for non-matching results
            
        Returns:
            Filtered and reranked results
        """
        from src.geo.country_matcher import countries_match, extract_country_from_location
        
        target_iso = extract_country_from_location(country_hint)
        if not target_iso:
            return results
        
        scored_results = []
        for r in results:
            result_country = r.get("country") or self._extract_country_from_result(r)
            
            # Use robust country matching
            country_match = False
            if result_country:
                country_match = countries_match(country_hint, result_country)
            elif target_iso:
                # Check if target ISO appears in snippet/title
                snippet = (r.get("snippet", "") + " " + r.get("title", "")).lower()
                for name in [target_iso, country_hint.lower()]:
                    if name in snippet:
                        country_match = True
                        break
            
            # Score based on match
            if country_match:
                r["_country_match"] = True
                r["_score_boost"] = boost_factor
            else:
                r["_country_match"] = False
                r["_score_boost"] = penalty_factor
            
            scored_results.append(r)
        
        # Sort: matches first, then by original position
        scored_results.sort(key=lambda x: (
            not x.get("_country_match", False),
            x.get("position", 999)
        ))
        
        return scored_results

    def _extract_country_from_result(self, result: dict) -> str | None:
        """Extract country name from result fields."""
        # Check address field
        address = result.get("address", "")
        if address:
            # Usually "City, Country" format
            parts = address.split(",")
            if len(parts) >= 2:
                return parts[-1].strip()
        
        # Check snippet for location patterns
        snippet = result.get("snippet", "")
        if snippet:
            import re
            # Look for "in Country" or "Country" at end
            match = re.search(r'(?:in|at|near)\s+([A-Z][a-z]+)(?:\s*[,.]|$)', snippet)
            if match:
                return match.group(1)
        
        return None

    @cached("serper", 7200)
    async def search_places(
        self, 
        query: str,
        country_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        """Google Places search via Serper API with geographic constraints.

        Args:
            query: Place search query
            country_hint: Country to prioritize results from
            
        Returns:
            List of place dicts with: title, address, latitude, longitude, rating, etc.
        """
        try:
            payload: dict[str, Any] = {"q": query}
            
            # Apply geographic constraints
            if country_hint:
                iso_code = extract_country_from_hint(country_hint)
                if iso_code:
                    payload["gl"] = get_google_gl(iso_code)
                    cr = get_google_cr(iso_code)
                    if cr:
                        payload["cr"] = cr
            
            resp = await self._client.post(
                f"{self.BASE_URL}/places",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("places", [])
            
            # Filter by country if hint provided
            if country_hint and results:
                results = self._filter_by_country(results, country_hint)
            
            return results
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
        """Convert Serper results to Evidence objects with dynamic confidence."""
        evidences = []
        for position, r in enumerate(results):
            # Extract location mentions from snippets
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            
            # Use safe coordinate extraction with validation
            lat, lon = safe_coords(r.get("latitude"), r.get("longitude"))
            address = r.get("address")

            content = f"Search result for '{query}': {title}"
            if snippet:
                content += f" - {snippet[:200]}"
            if address:
                content += f" (Address: {address})"

            # Dynamic confidence based on result quality
            confidence = calculate_search_confidence(
                result=r,
                query=query,
                base_confidence=0.35,  # Serper base confidence
                position=position,
            )

            evidences.append(
                Evidence(
                    source=EvidenceSource.SERPER,
                    content=content,
                    confidence=confidence,
                    latitude=lat,
                    longitude=lon,
                    url=link,
                    metadata={"query": query, "title": title, "position": position},
                )
            )
        return evidences

    async def close(self):
        await self._client.aclose()
