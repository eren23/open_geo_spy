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
from openai import AsyncOpenAI

from src.cache import CacheStore
from src.config.settings import Settings
from src.evidence.chain import Evidence, EvidenceChain, EvidenceSource
from src.geo.geocoding import reverse_geocode
from src.geo.osm_client import OSMClient
from src.geo.provider_base import SearchProvider
from src.search.graph import QueryIntent, SearchGraph, SearchNodeStatus
from src.search.smart_expander import SmartQueryExpander
from src.utils.geo_math import validate_coordinates


class WebIntelAgent:
    """Tiered web search and verification agent."""

    def __init__(self, settings: Settings, cache: CacheStore | None = None, client: Any = None):
        self.settings = settings
        self._cache = cache
        self._providers: list[SearchProvider] = []
        self._osm = OSMClient(cache=cache)
        self._browser_pool = None
        self._browser_search = None
        self._api_semaphore = asyncio.Semaphore(10)

        # Smart query expansion via LLM
        self.client = client or AsyncOpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
        )
        self._expander = SmartQueryExpander(self.client, settings.llm.fast_model)

        # Initialize search providers from settings
        self._init_providers()

    def _init_providers(self) -> None:
        """Instantiate search providers based on settings.geo.search_providers."""
        for provider_name in self.settings.geo.search_providers:
            try:
                if provider_name == "serper" and self.settings.geo.serper_api_key:
                    from src.geo.serper_client import SerperClient
                    self._providers.append(
                        SerperClient(self.settings.geo.serper_api_key, cache=self._cache)
                    )
                elif provider_name == "brave" and self.settings.geo.brave_api_key:
                    from src.geo.brave_client import BraveClient
                    self._providers.append(
                        BraveClient(self.settings.geo.brave_api_key, cache=self._cache)
                    )
                elif provider_name == "searxng" and self.settings.geo.searxng_url:
                    from src.geo.searxng_client import SearXNGClient
                    self._providers.append(
                        SearXNGClient(self.settings.geo.searxng_url, cache=self._cache)
                    )
            except Exception as e:
                logger.warning("Failed to init provider '{}': {}", provider_name, e)

        if not self._providers:
            logger.warning("No search providers configured")

    @property
    def serper(self):
        """Legacy accessor — returns the first Serper provider if available."""
        from src.geo.serper_client import SerperClient
        for p in self._providers:
            if isinstance(p, SerperClient):
                return p
        return None

    async def search(
        self,
        evidence_chain: EvidenceChain,
        features: dict[str, Any] | None = None,
        ocr_result: dict[str, list[str]] | None = None,
        weak_areas: list[str] | None = None,
        started_at_monotonic: float = 0.0,
    ) -> tuple[EvidenceChain, SearchGraph]:
        """Run tiered search based on available evidence.

        Args:
            evidence_chain: Evidence from feature extraction and ML models
            features: Raw visual features dict
            ocr_result: Raw OCR results dict
            weak_areas: Weakness areas from refinement check (triggers targeted queries)
            started_at_monotonic: Pipeline start time for latency budgeting

        Returns:
            Tuple of (evidence_chain, search_graph) for pipeline state.
        """
        chain = EvidenceChain()
        graph = SearchGraph()

        # Build search queries from evidence
        queries = self._build_search_queries(evidence_chain, features, ocr_result)
        if not queries:
            logger.warning("No search queries generated from evidence")
            return chain, graph

        logger.info("Generated {} search queries", len(queries))

        # Seed the search graph with initial queries
        for query in queries[:5]:
            graph.add_node(query, intent=QueryIntent.INITIAL)

        # --- Tier 1: Execute pending graph nodes (parallel) ---
        await self._execute_pending_nodes(graph, chain)

        # OSM search if we have coordinates from evidence
        cluster = evidence_chain.location_cluster()
        if cluster:
            osm_evidences = await self._osm_search(cluster[0], cluster[1])
            chain.add_many(osm_evidences)

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

        # --- Tier 1.5: Smart query expansion if evidence is weak ---
        if len(chain.geo_evidences) < 3:
            try:
                suggestions = await self._expander.suggest(graph, chain, weak_areas)
                for s in suggestions:
                    graph.add_node(
                        s["query"],
                        intent=s.get("intent", QueryIntent.REFINE),
                        parent_id=s.get("parent_id"),
                        language=s.get("language", "en"),
                    )
                # Execute the newly added pending nodes
                if suggestions:
                    logger.info("Smart expander added {} queries, executing...", len(suggestions))
                    await self._execute_pending_nodes(graph, chain)
            except Exception as e:
                logger.warning("Smart expansion failed: {}", e)

        # Prune dead ends in the graph
        for dead_id in graph.dead_ends():
            graph.prune_branch(dead_id)

        # --- Tier 2: Browser search (only if still insufficient and budget allows) ---
        if len(chain.geo_evidences) < 3 and self.settings.browser.enabled:
            # Latency guard: skip browser tier if we've used too much of the budget
            skip_browser = False
            if started_at_monotonic > 0 and self.settings.pipeline.max_total_latency_ms > 0:
                import time as _time
                elapsed_ms = (_time.monotonic() - started_at_monotonic) * 1000.0
                if elapsed_ms >= self.settings.pipeline.max_total_latency_ms * 0.5:
                    logger.info(
                        "Skipping Tier 2 browser search: {:.0f}ms elapsed (>50% of budget)",
                        elapsed_ms,
                    )
                    skip_browser = True

            if not skip_browser:
                logger.info("Tier 1 insufficient, escalating to browser search")
                browser_evidences = await self._browser_tier(queries[:3])
                chain.add_many(browser_evidences)
                logger.info("Tier 2 added {} evidences", len(browser_evidences))

        return chain, graph

    async def _execute_pending_nodes(self, graph: SearchGraph, chain: EvidenceChain) -> None:
        """Execute all pending search nodes in the graph in parallel."""
        import time as _time
        import hashlib

        pending = graph.pending_nodes()
        if not pending or not self._providers:
            return

        # P1.8: Filter out queries matching failed patterns
        pending = [n for n in pending if not graph.matches_failed_pattern(n.query)]

        async def _run_node(node):
            node.status = SearchNodeStatus.RUNNING
            start = _time.monotonic()
            try:
                evidences = await self._provider_search(node.query)
                node.status = SearchNodeStatus.COMPLETED
                node.evidence_count = len(evidences)
                node.best_confidence = max((e.confidence for e in evidences), default=0.0)
                node.duration_ms = round((_time.monotonic() - start) * 1000, 1)
                # P2.6: Track cost effectiveness
                if node.duration_ms > 0:
                    node.cost_effectiveness = round(node.evidence_count / (node.duration_ms / 1000), 2)
                # P1.8: Record failed pattern if no evidence
                if node.evidence_count == 0:
                    graph.record_failed_pattern(node.query)
                return evidences
            except Exception as e:
                node.status = SearchNodeStatus.FAILED
                node.error = str(e)
                node.duration_ms = round((_time.monotonic() - start) * 1000, 1)
                node.retry_count += 1  # P1.7: Increment retry count
                graph.record_failed_pattern(node.query)  # P1.8: Record failure
                return []

        results = await asyncio.gather(*[_run_node(n) for n in pending], return_exceptions=True)
        
        # P2.1: Cross-provider deduplication using content hashes
        seen_hashes: set[str] = set()
        for result in results:
            if isinstance(result, list):
                for evidence in result:
                    # Create hash from title + url + coordinates for dedup
                    dedup_key = hashlib.sha256(
                        f"{evidence.content}:{evidence.url}:{evidence.latitude}:{evidence.longitude}".encode()
                    ).hexdigest()[:16]
                    if dedup_key not in seen_hashes:
                        seen_hashes.add(dedup_key)
                        chain.add(evidence)
            elif isinstance(result, Exception):
                logger.debug("Search node failed: {}", result)

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

        # Prioritize location hint from evidence chain
        hint = None
        for e in evidence_chain.evidences:
            if e.source == EvidenceSource.USER_HINT:
                hint = e.metadata.get("hint", "").strip()
                break

        if hint:
            # Prepend hint to top queries for higher relevance
            if ocr_result:
                businesses = ocr_result.get("business_names", [])
                for biz in businesses[:2]:
                    queries.append(f"{biz} {hint}")
            if features:
                landmarks = features.get("landmarks", [])
                for lm in landmarks[:2]:
                    queries.append(f"{lm} {hint}")
            # Generic hint query
            queries.append(f"location {hint}")

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

    async def _provider_search(self, query: str) -> list[Evidence]:
        """Search across all configured providers in parallel (capped by semaphore)."""
        if not self._providers:
            return []

        async def _search_one(provider: SearchProvider) -> list[Evidence]:
            async with self._api_semaphore:
                results = await provider.search(query, num_results=5)
                evidences = provider.results_to_evidence(results, query)

                # Serper-specific: also try places search for geo data
                if hasattr(provider, "search_places"):
                    places = await provider.search_places(query)
                    evidences.extend(provider.results_to_evidence(places, query))

                return evidences

        all_results = await asyncio.gather(
            *[_search_one(p) for p in self._providers],
            return_exceptions=True,
        )

        evidences: list[Evidence] = []
        for result in all_results:
            if isinstance(result, list):
                evidences.extend(result)
            elif isinstance(result, Exception):
                logger.debug("Provider search failed: {}", result)

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

            async def _single_browser_search(query: str) -> list:
                try:
                    results = await self._browser_search.search_google_maps(query)
                    return self._browser_search.to_evidence(results, query)
                except Exception as e:
                    logger.debug("Browser search failed for '{}': {}", query[:50], e)
                    return []

            tasks = [_single_browser_search(q) for q in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_evidences = []
            for result in results:
                if isinstance(result, list):
                    all_evidences.extend(result)

            return all_evidences

        except Exception as e:
            logger.error("Browser tier failed: {}", e)
            return []

    async def close(self):
        """Clean up resources."""
        for provider in self._providers:
            try:
                await provider.close()
            except Exception:
                pass
        if self._browser_pool:
            await self._browser_pool.close()
