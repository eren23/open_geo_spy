"""LangGraph pipeline state definition.

Central state schema shared across all graph nodes.
Uses Annotated reducers for safe parallel state merging.
"""

from __future__ import annotations

import operator
from typing import Any, Optional

from typing_extensions import Annotated, TypedDict

from src.evidence.chain import Evidence
from src.search.graph import SearchGraph


def _merge_evidence(left: list[Evidence], right: list[Evidence]) -> list[Evidence]:
    """Merge evidence lists, deduplicating by content_hash."""
    seen = {e.content_hash for e in left}
    merged = list(left)
    for e in right:
        if e.content_hash not in seen:
            seen.add(e.content_hash)
            merged.append(e)
    return merged


class PipelineState(TypedDict, total=False):
    """State that flows through the LangGraph pipeline.

    Annotated fields with reducers allow parallel nodes
    to independently write to the same field.
    """

    # --- Inputs (set once at START) ---
    image_path: str
    location_hint: Optional[str]
    session_id: str
    quality: str

    # --- Evidence (accumulated via reducer) ---
    evidences: Annotated[list[Evidence], _merge_evidence]

    # --- Raw extraction outputs (set by feature_extraction node) ---
    metadata: dict
    features: dict
    ocr_result: dict

    # --- ML results ---
    candidates: Annotated[list[dict], operator.add]

    # --- Search graph (set by web_intel node) ---
    search_graph: Optional[SearchGraph]

    # --- Final output (set by reasoning node) ---
    prediction: dict
    ranked_candidates: list[dict]

    # --- Iterative refinement control ---
    iteration: int
    max_iterations: int
    weak_evidence_areas: list[str]
    should_refine: bool
    skip_full_verification: bool
    fast_path_reason: Optional[str]
    early_exit: bool

    # --- Chat history ---
    messages: Annotated[list[dict], operator.add]

    # --- Tracing ---
    trace_recorder: Optional[Any]  # TraceRecorder (optional, avoids circular import)

    # --- Shared resources ---
    _base_llm_client: Optional[Any]  # AsyncOpenAI shared across nodes

    # --- Pipeline metadata ---
    step_results: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
    started_at_monotonic: float
    execution_policy: dict[str, Any]
