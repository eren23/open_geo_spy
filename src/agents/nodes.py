"""Thin async node wrappers that call existing agent classes.

Each function receives PipelineState, calls the real agent,
and returns a partial state update dict.  Uses StreamWriter
for per-node SSE events when running inside ``graph.stream()``.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.types import StreamWriter
from loguru import logger

from src.agents.state import PipelineState
from src.cache import CacheStore
from src.config.settings import Settings
from src.evidence.chain import EvidenceChain


def _chain_to_evidences(chain: EvidenceChain) -> list:
    """Extract Evidence list from an EvidenceChain."""
    return list(chain.evidences)


def _build_chain_from_state(state: PipelineState) -> EvidenceChain:
    """Reconstruct an EvidenceChain from accumulated state evidences."""
    chain = EvidenceChain()
    for e in state.get("evidences", []):
        chain.add(e)
    return chain


# ---------------------------------------------------------------------------
# Node: feature_extraction
# ---------------------------------------------------------------------------

async def feature_extraction_node(
    state: PipelineState,
    writer: StreamWriter,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Extract EXIF, visual features, and OCR in parallel."""
    from src.agents.feature_agent import FeatureExtractionAgent

    writer({"event": "step_start", "step": "feature_extraction"})
    start = time.monotonic()

    if settings is None:
        from src.config.settings import get_settings
        settings = get_settings()

    agent = FeatureExtractionAgent(settings)

    try:
        chain, metadata, features, ocr_result = await agent.extract_with_raw(
            state["image_path"], state.get("location_hint")
        )
        duration = round((time.monotonic() - start) * 1000, 1)

        writer({
            "event": "step_complete",
            "step": "feature_extraction",
            "duration_ms": duration,
            "evidence_count": len(chain.evidences),
        })

        return {
            "evidences": _chain_to_evidences(chain),
            "metadata": metadata,
            "features": features,
            "ocr_result": ocr_result,
            "step_results": [{
                "name": "feature_extraction",
                "status": "completed",
                "duration_ms": duration,
                "evidence_count": len(chain.evidences),
            }],
        }
    except Exception as e:
        logger.error("Feature extraction failed: {}", e)
        writer({"event": "step_error", "step": "feature_extraction", "error": str(e)})
        return {
            "metadata": {},
            "features": {},
            "ocr_result": {},
            "errors": [f"feature_extraction: {e}"],
            "step_results": [{"name": "feature_extraction", "status": "failed", "error": str(e)}],
        }


# ---------------------------------------------------------------------------
# Node: ml_ensemble
# ---------------------------------------------------------------------------

async def ml_ensemble_node(
    state: PipelineState,
    writer: StreamWriter,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run ML models (GeoCLIP, StreetCLIP, VLM geo) in parallel."""
    from src.agents.ml_ensemble_agent import MLEnsembleAgent

    writer({"event": "step_start", "step": "ml_ensemble"})
    start = time.monotonic()

    if settings is None:
        from src.config.settings import get_settings
        settings = get_settings()

    agent = MLEnsembleAgent(settings)

    try:
        feature_chain = _build_chain_from_state(state)

        # Build candidate cities from existing evidence (VLM Geo predictions, OCR mentions)
        candidate_cities: list[str] = []
        for e in feature_chain.evidences:
            if e.city and e.city not in candidate_cities:
                candidate_cities.append(e.city)
        # Also pull from OCR-detected cities if available
        ocr = state.get("ocr_result", {})
        for city_name in ocr.get("cities", []):
            if city_name not in candidate_cities:
                candidate_cities.append(city_name)

        chain = await agent.predict(
            state["image_path"], feature_chain, candidate_cities=candidate_cities or None,
        )
        duration = round((time.monotonic() - start) * 1000, 1)

        writer({
            "event": "step_complete",
            "step": "ml_ensemble",
            "duration_ms": duration,
            "evidence_count": len(chain.evidences),
        })

        return {
            "evidences": _chain_to_evidences(chain),
            "step_results": [{
                "name": "ml_ensemble",
                "status": "completed",
                "duration_ms": duration,
                "evidence_count": len(chain.evidences),
            }],
        }
    except Exception as e:
        logger.error("ML ensemble failed: {}", e)
        writer({"event": "step_error", "step": "ml_ensemble", "error": str(e)})
        return {
            "errors": [f"ml_ensemble: {e}"],
            "step_results": [{"name": "ml_ensemble", "status": "failed", "error": str(e)}],
        }


# ---------------------------------------------------------------------------
# Node: web_intelligence
# ---------------------------------------------------------------------------

async def web_intelligence_node(
    state: PipelineState,
    writer: StreamWriter,
    *,
    settings: Settings | None = None,
    cache: CacheStore | None = None,
) -> dict[str, Any]:
    """Run web search (Serper + OSM + browser)."""
    from src.agents.web_intel_agent import WebIntelAgent

    writer({"event": "step_start", "step": "web_intelligence"})
    start = time.monotonic()

    if settings is None:
        from src.config.settings import get_settings
        settings = get_settings()

    agent = WebIntelAgent(settings, cache=cache)

    try:
        evidence_chain = _build_chain_from_state(state)
        features = state.get("features", {})
        ocr_result = state.get("ocr_result", {})

        chain, search_graph = await agent.search(
            evidence_chain,
            features,
            ocr_result,
            weak_areas=state.get("weak_evidence_areas"),
        )
        duration = round((time.monotonic() - start) * 1000, 1)

        await agent.close()

        writer({
            "event": "step_complete",
            "step": "web_intelligence",
            "duration_ms": duration,
            "evidence_count": len(chain.evidences),
        })

        return {
            "evidences": _chain_to_evidences(chain),
            "search_graph": search_graph,
            "step_results": [{
                "name": "web_intelligence",
                "status": "completed",
                "duration_ms": duration,
                "evidence_count": len(chain.evidences),
            }],
        }
    except Exception as e:
        logger.error("Web intelligence failed: {}", e)
        writer({"event": "step_error", "step": "web_intelligence", "error": str(e)})
        return {
            "errors": [f"web_intelligence: {e}"],
            "step_results": [{"name": "web_intelligence", "status": "failed", "error": str(e)}],
        }


# ---------------------------------------------------------------------------
# Node: candidate_verification
# ---------------------------------------------------------------------------

async def candidate_verification_node(
    state: PipelineState,
    writer: StreamWriter,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Visual similarity verification of top candidates."""
    from src.agents.candidate_verification_agent import CandidateVerificationAgent

    writer({"event": "step_start", "step": "candidate_verification"})
    start = time.monotonic()

    if settings is None:
        from src.config.settings import get_settings
        settings = get_settings()

    agent = CandidateVerificationAgent(settings)

    try:
        evidence_chain = _build_chain_from_state(state)
        features = state.get("features", {})
        ocr_result = state.get("ocr_result", {})

        chain = await agent.verify_candidates(
            state["image_path"], evidence_chain, features, ocr_result
        )
        duration = round((time.monotonic() - start) * 1000, 1)

        await agent.close()

        writer({
            "event": "step_complete",
            "step": "candidate_verification",
            "duration_ms": duration,
            "evidence_count": len(chain.evidences),
        })

        return {
            "evidences": _chain_to_evidences(chain),
            "step_results": [{
                "name": "candidate_verification",
                "status": "completed",
                "duration_ms": duration,
                "evidence_count": len(chain.evidences),
            }],
        }
    except Exception as e:
        logger.error("Candidate verification failed: {}", e)
        writer({"event": "step_error", "step": "candidate_verification", "error": str(e)})
        return {
            "errors": [f"candidate_verification: {e}"],
            "step_results": [{"name": "candidate_verification", "status": "failed", "error": str(e)}],
        }


# ---------------------------------------------------------------------------
# Node: reasoning
# ---------------------------------------------------------------------------

async def reasoning_node(
    state: PipelineState,
    writer: StreamWriter,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Final synthesis: reason over all evidence to produce prediction."""
    from src.agents.reasoning_agent import ReasoningAgent

    writer({"event": "step_start", "step": "reasoning"})
    start = time.monotonic()

    if settings is None:
        from src.config.settings import get_settings
        settings = get_settings()

    agent = ReasoningAgent(settings)

    try:
        evidence_chain = _build_chain_from_state(state)
        features = state.get("features", {})

        # Produce multi-candidate results and use the top one as primary
        ranked = await agent.reason_multi_candidate(evidence_chain, features)
        prediction = ranked[0] if ranked else await agent.reason(evidence_chain, features)
        duration = round((time.monotonic() - start) * 1000, 1)

        writer({
            "event": "step_complete",
            "step": "reasoning",
            "duration_ms": duration,
        })

        return {
            "prediction": prediction,
            "ranked_candidates": ranked,
            "step_results": [{
                "name": "reasoning",
                "status": "completed",
                "duration_ms": duration,
            }],
        }
    except Exception as e:
        logger.error("Reasoning failed: {}", e)
        writer({"event": "step_error", "step": "reasoning", "error": str(e)})
        return {
            "prediction": {
                "name": "Unknown",
                "lat": 0,
                "lon": 0,
                "confidence": 0,
                "reasoning": str(e),
            },
            "ranked_candidates": [],
            "errors": [f"reasoning: {e}"],
            "step_results": [{"name": "reasoning", "status": "failed", "error": str(e)}],
        }


# ---------------------------------------------------------------------------
# Node: refinement_check  (conditional – decides whether to loop)
# ---------------------------------------------------------------------------

async def refinement_check_node(
    state: PipelineState,
    writer: StreamWriter,
) -> dict[str, Any]:
    """Decide if refinement loop is needed based on evidence quality."""
    iteration = state.get("iteration", 0) + 1
    max_iter = state.get("max_iterations", 2)
    prediction = state.get("prediction", {})

    # Never loop more than max_iterations
    if iteration > max_iter:
        return {"should_refine": False, "iteration": iteration}

    # Weakness detection
    weak_areas: list[str] = []
    evidence_chain = _build_chain_from_state(state)

    if evidence_chain.agreement_score() < 0.5:
        weak_areas.append("low_geographic_agreement")

    sources = {e.source.value for e in evidence_chain.evidences}
    if len(sources) < 3:
        weak_areas.append("few_evidence_sources")

    web_sources = {"serper", "google_maps", "osm", "browser"}
    if not sources & web_sources:
        weak_areas.append("no_web_corroboration")

    countries = evidence_chain.country_predictions
    if countries:
        from src.utils.geo_math import country_level_agreement
        if country_level_agreement(countries) < 0.6:
            weak_areas.append("country_disagreement")

    if prediction.get("confidence", 0) < 0.4:
        weak_areas.append("low_top_confidence")

    should_refine = len(weak_areas) > 0
    if should_refine:
        writer({
            "event": "refinement",
            "iteration": iteration,
            "weak_areas": weak_areas,
        })
        logger.info("Refinement triggered (iter {}): {}", iteration, weak_areas)

    return {
        "should_refine": should_refine,
        "weak_evidence_areas": weak_areas,
        "iteration": iteration,
    }
