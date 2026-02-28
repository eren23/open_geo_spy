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
        """Return node IDs that completed with zero evidence."""
        return [
            n.id for n in self.nodes.values()
            if n.status == SearchNodeStatus.COMPLETED and n.evidence_count == 0
        ]

    def prune_branch(self, node_id: str) -> None:
        """Mark a node and all its descendants as pruned."""
        if node_id in self.nodes:
            self.nodes[node_id].status = SearchNodeStatus.PRUNED
        for child in self.get_children(node_id):
            self.prune_branch(child.id)

    def suggest_expansions(self, max_suggestions: int = 5) -> list[dict]:
        """Suggest new queries based on graph analysis.

        Returns list of dicts: {"parent_id", "query_template", "intent", "reason"}
        """
        suggestions = []

        # Find productive nodes that haven't been expanded
        for node in self.nodes.values():
            if node.status != SearchNodeStatus.COMPLETED:
                continue
            if node.evidence_count == 0:
                continue

            children = self.get_children(node.id)
            if len(children) >= 3:
                continue  # Already well-expanded

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

            # Suggest broadening for narrow results
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
