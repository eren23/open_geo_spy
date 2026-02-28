"""Tests for the SearchGraph data model."""

from src.search.graph import QueryIntent, SearchGraph, SearchNodeStatus


class TestSearchGraph:
    def test_add_root_node(self):
        g = SearchGraph()
        node = g.add_node("eiffel tower paris", QueryIntent.INITIAL)
        assert node.id in g.nodes
        assert node.id in g.root_ids

    def test_add_child_node(self):
        g = SearchGraph()
        parent = g.add_node("paris", QueryIntent.INITIAL)
        child = g.add_node("eiffel tower", QueryIntent.REFINE, parent_id=parent.id)
        assert child.parent_id == parent.id
        assert child.id not in g.root_ids
        assert len(g.edges) == 1

    def test_get_children(self):
        g = SearchGraph()
        parent = g.add_node("root", QueryIntent.INITIAL)
        c1 = g.add_node("child1", QueryIntent.REFINE, parent_id=parent.id)
        c2 = g.add_node("child2", QueryIntent.BROADEN, parent_id=parent.id)
        children = g.get_children(parent.id)
        assert len(children) == 2

    def test_path_to_root(self):
        g = SearchGraph()
        root = g.add_node("root", QueryIntent.INITIAL)
        mid = g.add_node("mid", QueryIntent.REFINE, parent_id=root.id)
        leaf = g.add_node("leaf", QueryIntent.REFINE, parent_id=mid.id)
        path = g.get_path_to_root(leaf.id)
        assert len(path) == 3
        assert path[0].id == root.id
        assert path[-1].id == leaf.id

    def test_pending_nodes(self):
        g = SearchGraph()
        n1 = g.add_node("a", QueryIntent.INITIAL)
        n2 = g.add_node("b", QueryIntent.INITIAL)
        n1.status = SearchNodeStatus.COMPLETED
        assert len(g.pending_nodes()) == 1

    def test_dead_ends(self):
        g = SearchGraph()
        n = g.add_node("dead", QueryIntent.INITIAL)
        n.status = SearchNodeStatus.COMPLETED
        n.evidence_count = 0
        assert n.id in g.dead_ends()

    def test_prune_branch(self):
        g = SearchGraph()
        root = g.add_node("root", QueryIntent.INITIAL)
        child = g.add_node("child", QueryIntent.REFINE, parent_id=root.id)
        g.prune_branch(root.id)
        assert g.nodes[root.id].status == SearchNodeStatus.PRUNED
        assert g.nodes[child.id].status == SearchNodeStatus.PRUNED

    def test_suggest_expansions(self):
        g = SearchGraph()
        n = g.add_node("productive query", QueryIntent.INITIAL)
        n.status = SearchNodeStatus.COMPLETED
        n.evidence_count = 5
        n.best_confidence = 0.7
        suggestions = g.suggest_expansions()
        assert len(suggestions) > 0
        assert suggestions[0]["parent_id"] == n.id

    def test_to_dict(self):
        g = SearchGraph()
        g.add_node("test", QueryIntent.INITIAL)
        d = g.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "stats" in d
        assert d["stats"]["total_nodes"] == 1
