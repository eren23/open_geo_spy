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
from openai import AsyncOpenAI

from src.agents.state import PipelineState
from src.cache import CacheStore
from src.config.settings import Settings, get_scoring_config, get_settings
from src.evidence.chain import EvidenceChain
from src.scoring.scorer import GeoScorer
from src.tracing.instrumented_client import InstrumentedOpenAI


def _get_instrumented_client(state: PipelineState, purpose: str) -> Any:
    """Wrap shared base client with instrumentation if a trace recorder is present."""
    base = state.get("_base_llm_client")
    if base is None:
        settings = get_settings()
        base = AsyncOpenAI(base_url=settings.llm.base_url, api_key=settings.llm.api_key)
    recorder = state.get("trace_recorder")
    if recorder:
        return InstrumentedOpenAI(base, recorder, default_purpose=purpose)
    return base


def _chain_to_evidences(chain: EvidenceChain) -> list:
    """Extract Evidence list from an EvidenceChain."""
    return list(chain.evidences)


def _build_chain_from_state(state: PipelineState) -> EvidenceChain:
    """Reconstruct an EvidenceChain from accumulated state evidences."""
    chain = EvidenceChain()
    for e in state.get("evidences", []):
        chain.add(e)
    return chain


def _emit_evidence_events(
    writer: StreamWriter,
    chain: EvidenceChain,
    prev_count: int = 0,
) -> None:
    """Emit evidence_added SSE events only for new items since prev_count."""
    new_evidences = chain.evidences[prev_count:]
    if not new_evidences:
        return
    writer({
        "event": "evidence_batch",
        "items": [
            {
                "event": "evidence_added",
                "source": e.source.value,
                "content_preview": e.content[:120] if e.content else "",
                "confidence": round(e.confidence, 3),
            }
            for e in new_evidences
        ],
    })


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

    recorder = state.get("trace_recorder")
    if recorder:
        recorder.record_step("feature_extraction", "started")

    writer({"event": "step_start", "step": "feature_extraction"})
    start = time.monotonic()

    if settings is None:
        settings = get_settings()

    client = _get_instrumented_client(state, "feature_extraction")
    agent = FeatureExtractionAgent(settings, client=client)

    try:
        chain, metadata, features, ocr_result = await agent.extract_with_raw(
            state["image_path"], state.get("location_hint")
        )
        duration = round((time.monotonic() - start) * 1000, 1)

        _emit_evidence_events(writer, chain)

        if recorder:
            recorder.record_step("feature_extraction", "completed", duration_ms=duration, evidence_count=len(chain.evidences))

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
        if recorder:
            recorder.record_error(str(e), step="feature_extraction")
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

    recorder = state.get("trace_recorder")
    if recorder:
        recorder.record_step("ml_ensemble", "started")

    writer({"event": "step_start", "step": "ml_ensemble"})
    start = time.monotonic()

    if settings is None:
        settings = get_settings()

    client = _get_instrumented_client(state, "ml_ensemble")
    agent = MLEnsembleAgent(settings, client=client)

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

        _emit_evidence_events(writer, chain)

        if recorder:
            recorder.record_step("ml_ensemble", "completed", duration_ms=duration, evidence_count=len(chain.evidences))

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
        if recorder:
            recorder.record_error(str(e), step="ml_ensemble")
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

    recorder = state.get("trace_recorder")
    if recorder:
        recorder.record_step("web_intelligence", "started")

    writer({"event": "step_start", "step": "web_intelligence"})
    start = time.monotonic()

    if settings is None:
        settings = get_settings()

    client = _get_instrumented_client(state, "web_intelligence")
    agent = WebIntelAgent(settings, cache=cache, client=client)

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

        _emit_evidence_events(writer, chain)

        if recorder:
            recorder.record_step("web_intelligence", "completed", duration_ms=duration, evidence_count=len(chain.evidences))

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
        if recorder:
            recorder.record_error(str(e), step="web_intelligence")
        writer({"event": "step_error", "step": "web_intelligence", "error": str(e)})
        return {
            "errors": [f"web_intelligence: {e}"],
            "step_results": [{"name": "web_intelligence", "status": "failed", "error": str(e)}],
        }
    finally:
        await agent.close()


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

    recorder = state.get("trace_recorder")
    if recorder:
        recorder.record_step("candidate_verification", "started")

    writer({"event": "step_start", "step": "candidate_verification"})
    start = time.monotonic()

    if settings is None:
        settings = get_settings()

    agent = CandidateVerificationAgent(settings)

    # Try to share StreetCLIP model from ML registry to avoid loading it twice
    try:
        from src.models.registry import ModelRegistry
        for name, instance in dict(ModelRegistry._instances).items():
            if "streetclip" in name.lower() and hasattr(instance, '_predictor'):
                predictor = instance._predictor
                if hasattr(predictor, 'model') and predictor.model is not None:
                    if hasattr(agent, '_scorer') and agent._scorer is not None:
                        agent._scorer.model = predictor.model
                        agent._scorer.processor = predictor.processor
                        logger.debug("Shared StreetCLIP model with verification agent")
                    break
    except Exception:
        pass  # Fall back to loading independently

    try:
        evidence_chain = _build_chain_from_state(state)
        features = state.get("features", {})
        ocr_result = state.get("ocr_result", {})

        chain = await agent.verify_candidates(
            state["image_path"], evidence_chain, features, ocr_result
        )
        duration = round((time.monotonic() - start) * 1000, 1)

        await agent.close()

        _emit_evidence_events(writer, chain)

        if recorder:
            recorder.record_step("candidate_verification", "completed", duration_ms=duration, evidence_count=len(chain.evidences))

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
        if recorder:
            recorder.record_error(str(e), step="candidate_verification")
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

    recorder = state.get("trace_recorder")
    if recorder:
        recorder.record_step("reasoning", "started")

    writer({"event": "step_start", "step": "reasoning"})
    start = time.monotonic()

    if settings is None:
        settings = get_settings()

    client = _get_instrumented_client(state, "reasoning")
    agent = ReasoningAgent(settings, client=client)

    try:
        evidence_chain = _build_chain_from_state(state)
        features = state.get("features", {})

        # Produce multi-candidate results and use the top one as primary
        ranked = await agent.reason_multi_candidate(evidence_chain, features)
        prediction = ranked[0] if ranked else await agent.reason(evidence_chain, features)
        duration = round((time.monotonic() - start) * 1000, 1)

        # Emit grounding results from hierarchical resolver
        try:
            from src.scoring.grounding import GroundingEngine
            from src.scoring.hierarchy import HierarchicalResolver
            resolver = HierarchicalResolver(engine=GroundingEngine())
            hier_pred = resolver.resolve(prediction, evidence_chain)
            for level_g in hier_pred.groundings.values():
                grounding_event = {
                    "event": "grounding_result",
                    "level": level_g.level.name.lower(),
                    "value": level_g.value or "",
                    "verdict": level_g.grounding.verdict.value if level_g.grounding else "UNCERTAIN",
                    "confidence": round(level_g.grounding.confidence, 3) if level_g.grounding else 0,
                    "supporting_count": len(level_g.grounding.supporting) if level_g.grounding else 0,
                    "contradicting_count": len(level_g.grounding.contradicting) if level_g.grounding else 0,
                }
                writer(grounding_event)
                if recorder:
                    recorder.record_grounding(
                        level=grounding_event["level"],
                        verdict=grounding_event["verdict"],
                        confidence=grounding_event["confidence"],
                    )
        except Exception as grounding_err:
            logger.debug("Grounding emission skipped: {}", grounding_err)

        if recorder:
            recorder.record_step("reasoning", "completed", duration_ms=duration)

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
        if recorder:
            recorder.record_error(str(e), step="reasoning")
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
    scorer = GeoScorer(get_scoring_config())
    thresholds = scorer.refinement

    iteration = state.get("iteration", 0) + 1
    max_iter = state.get("max_iterations", thresholds.max_iterations)
    prediction = state.get("prediction", {})

    # Never loop more than max_iterations
    if iteration > max_iter:
        return {"should_refine": False, "iteration": iteration}

    # Weakness detection
    weak_areas: list[str] = []
    evidence_chain = _build_chain_from_state(state)

    if evidence_chain.agreement_score(scorer) < thresholds.min_geographic_agreement:
        weak_areas.append("low_geographic_agreement")

    sources = {e.source.value for e in evidence_chain.evidences}
    if len(sources) < thresholds.min_evidence_sources:
        weak_areas.append("few_evidence_sources")

    web_sources = {"serper", "google_maps", "osm", "browser"}
    if not sources & web_sources:
        weak_areas.append("no_web_corroboration")

    countries = evidence_chain.country_predictions
    if countries:
        from src.utils.geo_math import country_level_agreement
        if country_level_agreement(countries) < thresholds.min_country_agreement:
            weak_areas.append("country_disagreement")

    if prediction.get("confidence", 0) < thresholds.min_top_confidence:
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
