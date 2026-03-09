"""Search graph executor.

Dispatches search nodes to appropriate providers (Serper, OSM, Browser)
and collects evidence. Supports iterative expansion.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from loguru import logger

from src.cache.store import CacheStore
from src.config.settings import Settings
from src.evidence.chain import Evidence, EvidenceChain
from src.search.graph import QueryIntent, SearchGraph, SearchNode, SearchNodeStatus


class SearchGraphExecutor:
    """Execute search graph nodes against external providers."""

    def __init__(self, settings: Settings, cache: Optional[CacheStore] = None):
        self.settings = settings
        self._cache = cache
        self._serper = None
        self._osm = None
        self._browser = None

    @property
    def serper(self):
        if self._serper is None and self.settings.geo.serper_api_key:
            from src.geo.serper_client import SerperClient
            self._serper = SerperClient(self.settings.geo.serper_api_key, cache=self._cache)
        return self._serper

    @property
    def osm(self):
        if self._osm is None:
            from src.geo.osm_client import OSMClient
            self._osm = OSMClient(cache=self._cache)
        return self._osm

    async def execute_node(self, node: SearchNode) -> list[Evidence]:
        """Execute a single search node and return evidence."""
        node.status = SearchNodeStatus.RUNNING
        start = time.monotonic()
        evidences: list[Evidence] = []

        try:
            if node.provider == "serper" and self.serper:
                results = await self.serper.search(node.query)
                evidences = self.serper.results_to_evidence(results, node.query)

                # Also try places search for location queries
                places = await self.serper.search_places(node.query)
                evidences.extend(self.serper.results_to_evidence(places, node.query))

            elif node.provider == "osm":
                # OSM needs coordinates - try search_by_name
                results = await self.osm.search_by_name(node.query)
                evidences = self.osm.to_evidence(results, node.query)

            elif node.provider == "browser":
                if self.settings.browser.enabled:
                    from src.browser.browser_pool import BrowserPool
                    from src.browser.search import BrowserSearch

                    pool = BrowserPool(self.settings)
                    browser = BrowserSearch(pool, cache=self._cache)
                    results = await browser.search_google_maps(node.query)
                    evidences = browser.to_evidence(results, node.query)
                    await pool.close()

            node.status = SearchNodeStatus.COMPLETED
            node.evidence_count = len(evidences)
            node.evidence_ids = [e.content_hash for e in evidences]
            if evidences:
                node.best_confidence = max(e.confidence for e in evidences)

        except Exception as e:
            node.status = SearchNodeStatus.FAILED
            node.error = str(e)[:200]
            logger.error("Search node {} failed: {}", node.id, e)

        node.duration_ms = round((time.monotonic() - start) * 1000, 1)
        return evidences

    async def execute_layer(
        self,
        graph: SearchGraph,
        node_ids: list[str],
    ) -> EvidenceChain:
        """Execute a batch of nodes in parallel."""
        chain = EvidenceChain()
        nodes = [graph.nodes[nid] for nid in node_ids if nid in graph.nodes]

        tasks = [self.execute_node(node) for node in nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                chain.add_many(result)
            elif isinstance(result, Exception):
                logger.error("Layer execution error: {}", result)

        return chain

    async def expand_and_execute(
        self,
        graph: SearchGraph,
        evidence_chain: EvidenceChain,
        max_depth: int = 3,
        expander=None,
    ) -> EvidenceChain:
        """Iterative expand-execute loop.

        1. Execute all pending nodes
        2. Analyze results, generate child nodes via expander
        3. Repeat until max_depth or no new productive nodes
        """
        combined = EvidenceChain()
        combined.add_many(evidence_chain.evidences)

        for depth in range(max_depth):
            pending = graph.pending_nodes()
            if not pending:
                break

            logger.info("Search depth {}: executing {} nodes", depth, len(pending))
            layer_chain = await self.execute_layer(
                graph, [n.id for n in pending]
            )
            combined.add_many(layer_chain.evidences)

            # Expand first (including BROADEN for 0-evidence nodes) so we try
            # broader queries before pruning obscure-place results
            if expander and depth < max_depth - 1:
                suggestions = expander.suggest(graph, combined)
            else:
                suggestions = graph.suggest_expansions(max_suggestions=3)

            for s in suggestions:
                if expander:
                    graph.add_node(
                        query=s["query"],
                        intent=s.get("intent", QueryIntent.REFINE),
                        provider=s.get("provider", "serper"),
                        parent_id=s.get("parent_id"),
                    )
                else:
                    graph.add_node(
                        query=s["query_template"],
                        intent=s["intent"],
                        parent_id=s["parent_id"],
                    )

            # Prune dead ends (after expansion so BROADEN children get a chance)
            for dead_id in graph.dead_ends():
                node = graph.nodes.get(dead_id)
                if node and node.intent != QueryIntent.INITIAL:
                    graph.prune_branch(dead_id)

        return combined

    async def close(self):
        if self._serper:
            await self._serper.close()
