"""LLM-powered smart query expansion (Phase 2.2).

Replaces the heuristic expander with 6 expansion strategies:
local language translation, synonyms, nearby landmarks,
reverse search, more specific, broader.
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from src.evidence.chain import EvidenceChain
from src.search.graph import QueryIntent, SearchGraph, SearchNode


EXPANSION_PROMPT = """You are a geolocation search query generator. Given a parent search query and evidence context, generate improved follow-up search queries.

## Parent query
{parent_query}

## Parent results
Produced {evidence_count} evidence items with best confidence {best_confidence:.2f}.

## Current evidence context
{evidence_summary}

## Weaknesses to address
{weaknesses}

## Strategies to use
Generate queries using these strategies (pick the most relevant 3-5):
1. LOCAL_LANGUAGE: Translate key terms into the likely local language
2. SYNONYMS: Use alternative names or spellings for places/landmarks
3. LANDMARKS: Search for nearby landmarks or distinctive features
4. MORE_SPECIFIC: Add street names, neighborhoods, or building numbers
5. BROADER: Expand to region or country level
6. VERIFY: Search for evidence that confirms or denies the current hypothesis

Return a JSON array of query objects:
[
  {{"query": "...", "intent": "refine|broaden|pivot|translate|verify", "reason": "...", "language": "en"}}
]

Return ONLY the JSON array, no other text."""


class SmartQueryExpander:
    """LLM-powered query expansion using Gemini Flash for speed."""

    def __init__(self, client: AsyncOpenAI, model: str = "google/gemini-2.5-flash"):
        self.client = client
        self.model = model

    async def suggest(
        self,
        graph: SearchGraph,
        evidence_chain: EvidenceChain,
        weak_areas: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate smart expansion queries using LLM."""
        # Find productive completed nodes to expand from
        productive = [
            n for n in graph.nodes.values()
            if n.status.value == "completed" and n.evidence_count > 0
        ]

        if not productive:
            return []

        # Pick the best parent node to expand
        parent = max(productive, key=lambda n: n.best_confidence)
        summary = evidence_chain.summary()

        prompt = EXPANSION_PROMPT.format(
            parent_query=parent.query,
            evidence_count=parent.evidence_count,
            best_confidence=parent.best_confidence,
            evidence_summary=json.dumps(summary, default=str)[:500],
            weaknesses=", ".join(weak_areas) if weak_areas else "none identified",
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content
            suggestions = self._parse_suggestions(raw, parent.id)
            logger.info("Smart expander generated {} queries", len(suggestions))
            return suggestions

        except Exception as e:
            logger.warning("Smart expansion failed, falling back to heuristic: {}", e)
            from src.search.expander import QueryExpander
            return QueryExpander().suggest(graph, evidence_chain, weak_areas)

    def _parse_suggestions(
        self, raw: str, parent_id: str
    ) -> list[dict[str, Any]]:
        """Parse LLM response into suggestion dicts."""
        try:
            match = re.search(r"\[[\s\S]*\]", raw)
            if match:
                items = json.loads(match.group())
                suggestions = []
                intent_map = {
                    "refine": QueryIntent.REFINE,
                    "broaden": QueryIntent.BROADEN,
                    "pivot": QueryIntent.PIVOT,
                    "translate": QueryIntent.TRANSLATE,
                    "verify": QueryIntent.VERIFY,
                }
                for item in items[:5]:
                    suggestions.append({
                        "query": item.get("query", ""),
                        "intent": intent_map.get(
                            item.get("intent", "refine"), QueryIntent.REFINE
                        ),
                        "parent_id": parent_id,
                        "provider": "serper",
                        "reason": item.get("reason", ""),
                        "language": item.get("language", "en"),
                    })
                return suggestions
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to parse expansion response: {}", e)

        return []
