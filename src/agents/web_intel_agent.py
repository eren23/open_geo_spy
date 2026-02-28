"""Web Intelligence Agent - tiered search from cheapest to most expensive.

Tier 1: API calls (Serper, OSM) - parallel, <2s
Tier 2: Stealth browser (Google Maps scraping, Street View) - only if Tier 1 insufficient

Adapted search query building from enhanced_search.py and multi-source
patterns from geo_interface.py.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from loguru import logger

from src.cache import CacheStore
from src.config.settings import Settings
from src.evidence.chain import Evidence, EvidenceChain, EvidenceSource
from src.geo.geocoding import reverse_geocode
from src.geo.osm_client import OSMClient
from src.utils.geo_math import validate_coordinates


class WebIntelAgent:
    """Tiered web search and verification agent."""

    def __init__(self, settings: Settings, cache: CacheStore | None = None):
        self.settings = settings
        self._cache = cache
        self._serper = None
        self._osm = OSMClient(cache=cache)
        self._browser_pool = None
        self._browser_search = None

    @property
    def serper(self):
        if self._serper is None and self.settings.geo.serper_api_key:
            from src.geo.serper_client import SerperClient
            self._serper = SerperClient(self.settings.geo.serper_api_key, cache=self._cache)
        return self._serper

    async def search(
        self,
        evidence_chain: EvidenceChain,
        features: dict[str, Any] | None = None,
        ocr_result: dict[str, list[str]] | None = None,
        weak_areas: list[str] | None = None,
    ) -> EvidenceChain:
        """Run tiered search based on available evidence.

        Args:
            evidence_chain: Evidence from feature extraction and ML models
            features: Raw visual features dict
            ocr_result: Raw OCR results dict
            weak_areas: Weakness areas from refinement check (triggers targeted queries)
        """
        chain = EvidenceChain()

        # Build search queries from evidence
        queries = self._build_search_queries(evidence_chain, features, ocr_result)
        if not queries:
            logger.warning("No search queries generated from evidence")
            return chain

        logger.info("Generated {} search queries", len(queries))

        # --- Tier 1: API searches (parallel) ---
        tier1_tasks = []

        for query in queries[:5]:  # Limit to 5 queries
            if self.serper:
                tier1_tasks.append(self._serper_search(query))

        # OSM search if we have coordinates from evidence
        cluster = evidence_chain.location_cluster()
        if cluster:
            tier1_tasks.append(self._osm_search(cluster[0], cluster[1]))

        if tier1_tasks:
            results = await asyncio.gather(*tier1_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    chain.add_many(result)
                elif isinstance(result, Exception):
                    logger.debug("Tier 1 search task failed: {}", result)

        # Reverse geocode the cluster point
        if cluster:
            geo_result = await reverse_geocode(cluster[0], cluster[1])
            if geo_result:
                chain.add(
                    Evidence(
                        source=EvidenceSource.OSM,
                        content=f"Reverse geocode: {geo_result.get('display_name', '')}",
                        confidence=0.7,
                        latitude=cluster[0],
                        longitude=cluster[1],
                        country=geo_result.get("country"),
                        region=geo_result.get("state"),
                        city=geo_result.get("city"),
                        metadata={"reverse_geocode": geo_result},
                    )
                )

        logger.info("Tier 1 search complete: {} new evidences", len(chain.evidences))

        # --- Tier 2: Browser search (only if Tier 1 insufficient) ---
        if len(chain.geo_evidences) < 3 and self.settings.browser.enabled:
            logger.info("Tier 1 insufficient, escalating to browser search")
            browser_evidences = await self._browser_tier(queries[:3])
            chain.add_many(browser_evidences)
            logger.info("Tier 2 added {} evidences", len(browser_evidences))

        return chain

    def _build_search_queries(
        self,
        evidence_chain: EvidenceChain,
        features: dict | None,
        ocr_result: dict | None,
    ) -> list[str]:
        """Build optimized search queries from evidence.

        Adapted from enhanced_search.py query building pattern.
        """
        queries = []

        # From OCR results
        if ocr_result:
            businesses = ocr_result.get("business_names", [])
            streets = ocr_result.get("street_signs", [])
            plates = ocr_result.get("license_plates", [])

            # Business + street combinations (most specific)
            for biz in businesses[:3]:
                for street in streets[:2]:
                    queries.append(f"{biz} {street}")
                if not streets:
                    queries.append(f"{biz} location address")

            # Street names alone
            for street in streets[:3]:
                queries.append(f'"{street}" location')

            # License plates (region identification)
            for plate in plates[:2]:
                queries.append(f"license plate format {plate}")

        # From visual features
        if features:
            landmarks = features.get("landmarks", [])
            for landmark in landmarks[:3]:
                queries.append(f"{landmark} location")

            country_clues = features.get("country_clues", [])
            if country_clues:
                short_clues = [" ".join(c.split()[:5]) for c in country_clues[:3]]
                context = " ".join(short_clues)[:60]
                queries.append(f"location {context}")

        # From evidence chain (e.g., ML predictions)
        for e in evidence_chain.evidences:
            if e.source in (EvidenceSource.VLM_GEO, EvidenceSource.STREETCLIP):
                if e.country and e.city:
                    queries.append(f"{e.city} {e.country}")
                elif e.country:
                    # Try to narrow down with OCR data
                    if ocr_result and ocr_result.get("business_names"):
                        queries.append(f"{ocr_result['business_names'][0]} {e.country}")

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in seen and len(q_lower) > 3:
                seen.add(q_lower)
                unique.append(q)

        return unique

    async def _serper_search(self, query: str) -> list[Evidence]:
        """Search via Serper API."""
        results = await self.serper.search(query, num_results=5)
        evidences = self.serper.results_to_evidence(results, query)

        # Also try places search
        places = await self.serper.search_places(query)
        evidences.extend(self.serper.results_to_evidence(places, query))

        return evidences

    async def _osm_search(self, lat: float, lon: float) -> list[Evidence]:
        """Search OSM for nearby POIs."""
        results = await self._osm.search_nearby(lat, lon, radius_km=5.0)
        return self._osm.to_evidence(results, f"nearby ({lat:.4f}, {lon:.4f})")

    async def _browser_tier(self, queries: list[str]) -> list[Evidence]:
        """Tier 2: Browser-based search."""
        try:
            if self._browser_pool is None:
                from src.browser.browser_pool import BrowserPool
                self._browser_pool = BrowserPool(self.settings.browser)
                await self._browser_pool.initialize()

            if self._browser_search is None:
                from src.browser.search import BrowserSearch
                self._browser_search = BrowserSearch(self._browser_pool, cache=self._cache)

            all_evidences = []
            for query in queries:
                results = await self._browser_search.search_google_maps(query)
                all_evidences.extend(self._browser_search.to_evidence(results, query))

            return all_evidences

        except Exception as e:
            logger.error("Browser tier failed: {}", e)
            return []

    async def close(self):
        """Clean up resources."""
        if self._serper:
            await self._serper.close()
        if self._browser_pool:
            await self._browser_pool.close()
