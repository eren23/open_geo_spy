"""Search graph data model.

Replaces flat query lists with a directed graph where nodes are queries,
edges represent refine/broaden/pivot/translate relationships, and each
node tracks which evidence it produced.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class QueryIntent(str, Enum):
    INITIAL = "initial"
    REFINE = "refine"
    BROADEN = "broaden"
    PIVOT = "pivot"
    TRANSLATE = "translate"
    VERIFY = "verify"


class SearchNodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass
class SearchNode:
    """A single search query in the graph."""

    id: str
    query: str
    intent: QueryIntent
    status: SearchNodeStatus = SearchNodeStatus.PENDING
    provider: str = "serper"  # serper, osm, browser, brave, searxng
    parent_id: Optional[str] = None
    evidence_ids: list[str] = field(default_factory=list)
    evidence_count: int = 0
    best_confidence: float = 0.0
    duration_ms: float = 0.0
    language: str = "en"
    error: Optional[str] = None
    retry_count: int = 0  # P1.7: Track retries for smarter pruning
    cost_effectiveness: float = 0.0  # P2.6: evidence_count / duration_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "intent": self.intent.value,
            "status": self.status.value,
            "provider": self.provider,
            "parent_id": self.parent_id,
            "evidence_count": self.evidence_count,
            "best_confidence": self.best_confidence,
            "duration_ms": self.duration_ms,
            "language": self.language,
            "error": self.error,
            "retry_count": self.retry_count,
            "cost_effectiveness": self.cost_effectiveness,
        }


@dataclass
class SearchEdge:
    """Directed edge between search nodes."""

    source_id: str
    target_id: str
    relationship: QueryIntent

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship.value,
        }


@dataclass
class SearchGraph:
    """Directed graph of search queries and their relationships."""

    nodes: dict[str, SearchNode] = field(default_factory=dict)
    edges: list[SearchEdge] = field(default_factory=list)
    root_ids: list[str] = field(default_factory=list)
    failed_patterns: list[str] = field(default_factory=list)  # P1.8: Track failed query patterns
    max_retries: int = 2  # P1.7: Max retries before pruning
    max_retries_initial: int = 3  # More retries for INITIAL nodes (obscure places often return 0)
    metadata: dict[str, Any] = field(default_factory=dict)  # Store hints and other context

    def add_node(
        self,
        query: str,
        intent: QueryIntent = QueryIntent.INITIAL,
        provider: str = "serper",
        parent_id: Optional[str] = None,
        language: str = "en",
    ) -> SearchNode:
        """Create and add a new search node."""
        node_id = str(uuid.uuid4())[:8]
        node = SearchNode(
            id=node_id,
            query=query,
            intent=intent,
            provider=provider,
            parent_id=parent_id,
            language=language,
        )
        self.nodes[node_id] = node

        if parent_id and parent_id in self.nodes:
            self.edges.append(SearchEdge(
                source_id=parent_id,
                target_id=node_id,
                relationship=intent,
            ))
        else:
            self.root_ids.append(node_id)

        return node

    def get_children(self, node_id: str) -> list[SearchNode]:
        """Get direct child nodes."""
        child_ids = [e.target_id for e in self.edges if e.source_id == node_id]
        return [self.nodes[cid] for cid in child_ids if cid in self.nodes]

    def get_path_to_root(self, node_id: str) -> list[SearchNode]:
        """Trace path from node back to its root."""
        path = []
        current = node_id
        visited = set()
        while current and current not in visited:
            visited.add(current)
            if current in self.nodes:
                path.append(self.nodes[current])
            node = self.nodes.get(current)
            current = node.parent_id if node else None
        return list(reversed(path))

    def pending_nodes(self) -> list[SearchNode]:
        """Get all nodes that haven't been executed yet."""
        return [n for n in self.nodes.values() if n.status == SearchNodeStatus.PENDING]

    def productive_branches(self, min_conf: float = 0.3) -> list[str]:
        """Return root IDs whose subtrees have produced high-confidence evidence."""
        productive = []
        for root_id in self.root_ids:
            if self._subtree_max_confidence(root_id) >= min_conf:
                productive.append(root_id)
        return productive

    def dead_ends(self) -> list[str]:
        """Return node IDs that completed with zero evidence and exceeded retries.

        INITIAL nodes get more retries (max_retries_initial) since queries for
        obscure places often return 0 results initially.
        Does not prune nodes that have pending children (let them run first).
        """
        dead = []
        for n in self.nodes.values():
            if n.status != SearchNodeStatus.COMPLETED or n.evidence_count != 0:
                continue
            # Don't prune if we have pending children (BROADEN may find results)
            children = self.get_children(n.id)
            if any(c.status == SearchNodeStatus.PENDING for c in children):
                continue
            max_allowed = self.max_retries_initial if n.intent == QueryIntent.INITIAL else self.max_retries
            if n.retry_count >= max_allowed:
                dead.append(n.id)
        return dead

    def retryable_dead_ends(self) -> list[str]:
        """Return node IDs that completed with zero evidence but can be retried (P1.7)."""
        return [
            n.id for n in self.nodes.values()
            if n.status == SearchNodeStatus.COMPLETED
            and n.evidence_count == 0
            and n.retry_count < (
                self.max_retries_initial if n.intent == QueryIntent.INITIAL else self.max_retries
            )
        ]

    def prune_branch(self, node_id: str) -> None:
        """Mark a node and all its descendants as pruned."""
        if node_id in self.nodes:
            self.nodes[node_id].status = SearchNodeStatus.PRUNED
        for child in self.get_children(node_id):
            self.prune_branch(child.id)

    def record_failed_pattern(self, query: str) -> None:
        """Record a failed query pattern to avoid similar queries (P1.8)."""
        # Extract pattern: first few words normalized
        words = query.lower().split()[:4]
        pattern = " ".join(words)
        if pattern and pattern not in self.failed_patterns:
            self.failed_patterns.append(pattern)

    def matches_failed_pattern(self, query: str) -> bool:
        """Check if query matches a known failed pattern (P1.8)."""
        words = query.lower().split()[:4]
        pattern = " ".join(words)
        return pattern in self.failed_patterns

    def suggest_expansions(self, max_suggestions: int = 5) -> list[dict]:
        """Suggest new queries based on graph analysis.

        Returns list of dicts: {"parent_id", "query_template", "intent", "reason"}.
        Includes BROADEN for 0-evidence nodes so obscure places get a broader
        query before pruning (reduces urban bias from over-pruning sparse results).
        """
        suggestions = []

        for node in self.nodes.values():
            if node.status != SearchNodeStatus.COMPLETED:
                continue

            children = self.get_children(node.id)
            if len(children) >= 3:
                continue  # Already well-expanded

            # For 0-evidence nodes: suggest BROADEN to try broader query before pruning
            if node.evidence_count == 0:
                if not any(c.intent == QueryIntent.BROADEN for c in children):
                    # Broader query: drop last token or add "area region"
                    words = node.query.split()
                    broader = f"{node.query} area region" if len(words) <= 3 else " ".join(words[:-1])
                    suggestions.append({
                        "parent_id": node.id,
                        "query_template": broader,
                        "intent": QueryIntent.BROADEN,
                        "reason": f"Zero results, try broader query before pruning",
                    })
                if len(suggestions) >= max_suggestions:
                    break
                continue

            # Suggest refinement for high-evidence nodes
            if node.best_confidence > 0.5 and not any(
                c.intent == QueryIntent.REFINE for c in children
            ):
                suggestions.append({
                    "parent_id": node.id,
                    "query_template": f"{node.query} specific location",
                    "intent": QueryIntent.REFINE,
                    "reason": f"High-confidence node ({node.best_confidence:.2f}) not yet refined",
                })

            # Suggest broadening for narrow results (1 evidence)
            if node.evidence_count < 2 and not any(
                c.intent == QueryIntent.BROADEN for c in children
            ):
                suggestions.append({
                    "parent_id": node.id,
                    "query_template": f"{node.query} area region",
                    "intent": QueryIntent.BROADEN,
                    "reason": f"Few results ({node.evidence_count}), try broader query",
                })

            if len(suggestions) >= max_suggestions:
                break

        return suggestions[:max_suggestions]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for frontend / SSE."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "root_ids": self.root_ids,
            "stats": {
                "total_nodes": len(self.nodes),
                "completed": sum(1 for n in self.nodes.values() if n.status == SearchNodeStatus.COMPLETED),
                "pending": sum(1 for n in self.nodes.values() if n.status == SearchNodeStatus.PENDING),
                "failed": sum(1 for n in self.nodes.values() if n.status == SearchNodeStatus.FAILED),
                "pruned": sum(1 for n in self.nodes.values() if n.status == SearchNodeStatus.PRUNED),
                "total_evidence": sum(n.evidence_count for n in self.nodes.values()),
            },
        }

    def _subtree_max_confidence(self, node_id: str) -> float:
        """Max confidence across a node and its descendants."""
        node = self.nodes.get(node_id)
        if not node:
            return 0.0
        best = node.best_confidence
        for child in self.get_children(node_id):
            best = max(best, self._subtree_max_confidence(child.id))
        return best
