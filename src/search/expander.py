"""Query expansion strategies.

Phase 1 stub: simple heuristic rules.
Phase 2 replaces with LLM-powered expansion (see 2.2).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.evidence.chain import EvidenceChain
from src.search.graph import QueryIntent, SearchGraph


class QueryExpander:
    """Heuristic query expander (Phase 1).

    Generates child queries based on simple rules:
    - Refine: add "exact location" / "street address"
    - Broaden: add region/country context
    - Translate: wrap in local language (placeholder)
    """

    def suggest(
        self,
        graph: SearchGraph,
        evidence_chain: EvidenceChain,
        weak_areas: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate expansion suggestions.

        Returns list of dicts: {"query", "intent", "parent_id", "provider", "reason"}
        """
        suggestions: list[dict] = []

        # Find productive completed nodes
        productive = [
            n for n in graph.nodes.values()
            if n.status.value == "completed" and n.evidence_count > 0
        ]

        if not productive:
            return suggestions

        # Country info from evidence
        countries = list(set(evidence_chain.country_predictions))
        top_country = countries[0] if countries else None

        for node in productive[:3]:
            children = graph.get_children(node.id)
            existing_intents = {c.intent for c in children}

            # Refine: if not already refined
            if QueryIntent.REFINE not in existing_intents:
                suggestions.append({
                    "query": f"{node.query} exact location street address",
                    "intent": QueryIntent.REFINE,
                    "parent_id": node.id,
                    "provider": node.provider,
                    "reason": "Refine productive query",
                })

            # Broaden: if few results
            if node.evidence_count < 3 and QueryIntent.BROADEN not in existing_intents:
                broader = f"{node.query} {top_country}" if top_country else f"{node.query} area"
                suggestions.append({
                    "query": broader,
                    "intent": QueryIntent.BROADEN,
                    "parent_id": node.id,
                    "provider": node.provider,
                    "reason": "Broaden low-result query",
                })

        # Handle weak areas from refinement
        if weak_areas:
            if "no_web_corroboration" in weak_areas and top_country:
                suggestions.append({
                    "query": f"landmarks in {top_country}",
                    "intent": QueryIntent.VERIFY,
                    "parent_id": None,
                    "provider": "serper",
                    "reason": "No web corroboration, search for landmarks",
                })

            if "country_disagreement" in weak_areas:
                for country in countries[:2]:
                    suggestions.append({
                        "query": f"typical scenery {country}",
                        "intent": QueryIntent.VERIFY,
                        "parent_id": None,
                        "provider": "serper",
                        "reason": f"Verify country candidate: {country}",
                    })

        return suggestions[:5]
