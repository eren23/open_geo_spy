"""LangGraph StateGraph builder for the geolocation pipeline.

Graph topology::

    START -> feature_extraction
    feature_extraction -> [ml_ensemble, web_intelligence]  (parallel fan-out)
    [ml_ensemble, web_intelligence] -> candidate_verification  (fan-in)
    candidate_verification -> reasoning
    reasoning -> refinement_check
    refinement_check --(should_refine)--> web_intelligence  (loop back)
    refinement_check --(done)--> END
"""

from __future__ import annotations

from functools import partial
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agents.nodes import (
    candidate_verification_node,
    feature_extraction_node,
    ml_ensemble_node,
    reasoning_node,
    refinement_check_node,
    web_intelligence_node,
)
from src.agents.state import PipelineState
from src.cache import CacheStore
from src.config.settings import Settings


def _should_refine(state: PipelineState) -> Literal["web_intelligence", "__end__"]:
    """Conditional edge: loop back to web_intelligence or finish."""
    if state.get("should_refine", False):
        return "web_intelligence"
    return END


def build_pipeline_graph(settings: Settings | None = None, cache: CacheStore | None = None):
    """Construct and compile the geolocation pipeline graph.

    Returns a compiled ``CompiledGraph`` ready for ``.invoke()`` or
    ``.stream()``.
    """
    # Bind settings and cache into each node via partial so they don't need globals
    fe_node = partial(feature_extraction_node, settings=settings)
    ml_node = partial(ml_ensemble_node, settings=settings)
    wi_node = partial(web_intelligence_node, settings=settings, cache=cache)
    cv_node = partial(candidate_verification_node, settings=settings)
    rs_node = partial(reasoning_node, settings=settings)

    builder = StateGraph(PipelineState)

    # Register nodes
    builder.add_node("feature_extraction", fe_node)
    builder.add_node("ml_ensemble", ml_node)
    builder.add_node("web_intelligence", wi_node)
    builder.add_node("candidate_verification", cv_node)
    builder.add_node("reasoning", rs_node)
    builder.add_node("refinement_check", refinement_check_node)

    # Edges
    builder.add_edge(START, "feature_extraction")

    # Parallel fan-out from feature_extraction
    builder.add_edge("feature_extraction", "ml_ensemble")
    builder.add_edge("feature_extraction", "web_intelligence")

    # Fan-in: both go to candidate_verification
    builder.add_edge("ml_ensemble", "candidate_verification")
    builder.add_edge("web_intelligence", "candidate_verification")

    # Sequential: candidate_verification -> reasoning -> refinement_check
    builder.add_edge("candidate_verification", "reasoning")
    builder.add_edge("reasoning", "refinement_check")

    # Conditional loop
    builder.add_conditional_edges(
        "refinement_check",
        _should_refine,
        {"web_intelligence": "web_intelligence", END: END},
    )

    return builder.compile()
