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


def _should_skip_candidate_verification(
    state: PipelineState,
    evidence_chain: EvidenceChain,
    settings: Settings,
) -> tuple[bool, str | None]:
    """Policy gate for expensive visual candidate verification."""
    quality = (state.get("quality") or "balanced").lower()
    if quality == "fast":
        return True, "quality=fast"
    if quality == "max":
        return False, None
    if not settings.pipeline.fast_path_enabled:
        return False, None

    scorer = GeoScorer(get_scoring_config())
    agreement = evidence_chain.agreement_score(scorer)
    geo_confident = sum(
        1 for e in evidence_chain.geo_evidences
        if e.confidence >= settings.pipeline.fast_path_confidence_threshold
    )
    if (
        agreement >= settings.pipeline.fast_path_agreement_threshold
        and geo_confident >= 2
    ):
        return True, f"agreement={agreement:.2f},geo_confident={geo_confident}"

    # Skip verification during refinement loops (already verified in first pass)
    iteration = state.get("iteration", 0)
    if iteration > 0:
        return True, f"refinement_loop_iteration={iteration}"

    started = state.get("started_at_monotonic", 0.0) or 0.0
    if started > 0 and settings.pipeline.max_total_latency_ms > 0:
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if elapsed_ms >= settings.pipeline.max_total_latency_ms * 0.6:
            return True, f"latency_guard={elapsed_ms:.0f}ms"

    return False, None


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

        # Get location hint from state and resolve to country code
        location_hint = state.get("location_hint")
        country_hint = None
        if location_hint:
            from src.geo.country_matcher import extract_country_from_location
            country_hint = extract_country_from_location(location_hint)
            logger.info(
                "ML ensemble using hint: '{}' (country={})",
                location_hint,
                country_hint or "unknown",
            )

        chain = await agent.predict(
            state["image_path"],
            feature_chain,
            candidate_cities=candidate_cities or None,
            location_hint=location_hint,
            country_hint=country_hint,
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
            started_at_monotonic=state.get("started_at_monotonic", 0.0) or 0.0,
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
# Node: early_exit_check  (conditional – decides whether to skip expensive steps)
# ---------------------------------------------------------------------------


async def early_exit_check_node(
    state: PipelineState,
    writer: StreamWriter,
) -> dict[str, Any]:
    """Check if models agree well enough to skip candidate_verification + refinement.

    When VLM Geo, GeoCLIP, and web evidence converge within a tight radius
    with reasonable confidence, we can jump straight to reasoning and skip
    the expensive candidate_verification and refinement loop entirely.
    """
    settings = get_settings()
    if not settings.pipeline.early_exit_enabled:
        return {"early_exit": False}

    quality = (state.get("quality") or "balanced").lower()
    if quality == "max":
        return {"early_exit": False}

    evidence_chain = _build_chain_from_state(state)
    geo_evs = evidence_chain.geo_evidences
    if len(geo_evs) < 2:
        return {"early_exit": False}

    # Check geographic spread: all geo evidence within threshold km
    from src.utils.geo_math import haversine_distance

    coords = [
        (e.latitude, e.longitude)
        for e in geo_evs
        if e.latitude is not None and e.longitude is not None
    ]
    if len(coords) < 2:
        return {"early_exit": False}

    max_spread = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            max_spread = max(max_spread, d)

    agreement_km = settings.pipeline.early_exit_agreement_km
    min_confidence = settings.pipeline.early_exit_min_confidence

    # Check country agreement
    countries = evidence_chain.country_predictions
    country_agree = True
    if countries:
        from src.utils.geo_math import country_level_agreement
        country_agree = country_level_agreement(countries) >= 0.6

    # Check average confidence
    confidences = [e.confidence for e in geo_evs if e.confidence > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    if max_spread <= agreement_km and country_agree and avg_conf >= min_confidence:
        logger.info(
            "Early exit: spread={:.1f}km, avg_conf={:.2f}, {} geo evidences",
            max_spread, avg_conf, len(geo_evs),
        )
        writer({
            "event": "early_exit",
            "spread_km": round(max_spread, 1),
            "avg_confidence": round(avg_conf, 3),
            "geo_evidence_count": len(geo_evs),
        })
        return {
            "early_exit": True,
            "skip_full_verification": True,
            "fast_path_reason": f"early_exit:spread={max_spread:.0f}km,conf={avg_conf:.2f}",
        }

    return {"early_exit": False}


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

    try:
        evidence_chain = _build_chain_from_state(state)
        skip_candidate_verification, reason = _should_skip_candidate_verification(
            state, evidence_chain, settings,
        )
        if skip_candidate_verification:
            duration = round((time.monotonic() - start) * 1000, 1)
            if recorder:
                recorder.record_step(
                    "candidate_verification",
                    "skipped",
                    duration_ms=duration,
                    evidence_count=len(evidence_chain.evidences),
                )
            writer({
                "event": "step_complete",
                "step": "candidate_verification",
                "duration_ms": duration,
                "evidence_count": len(evidence_chain.evidences),
            })
            return {
                "skip_full_verification": True,
                "fast_path_reason": reason or "candidate_verification_skipped",
                "step_results": [{
                    "name": "candidate_verification",
                    "status": "skipped",
                    "duration_ms": duration,
                    "evidence_count": len(evidence_chain.evidences),
                    "error": reason,
                }],
            }

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
            "skip_full_verification": False,
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
        
        # Get location hint and optionally filter evidence
        location_hint = state.get("location_hint")
        if location_hint:
            # Log that we have a hint
            writer({
                "event": "hint_applied",
                "hint": location_hint,
            })
            
            # Filter evidence to only include matching countries
            # This prevents wrong-country ML predictions from polluting results
            from src.geo.country_matcher import extract_country_from_location
            hint_country = extract_country_from_location(location_hint)
            if hint_country:
                logger.info(
                    "Filtering evidence by hint country: {} -> {}",
                    location_hint,
                    hint_country,
                )
                # Keep non-geo evidence but filter geo evidence from wrong countries
                evidence_chain = evidence_chain.filter_by_hint(
                    hint_country,
                    keep_non_geo=True,
                )

        # Produce multi-candidate results and use the top one as primary
        quality = (state.get("quality") or "balanced").lower()
        skip_verification = bool(state.get("skip_full_verification", False))
        if quality == "fast":
            skip_verification = True
        if (
            settings.pipeline.skip_visual_verification_if_confident
            and evidence_chain.agreement_score() >= settings.pipeline.fast_path_agreement_threshold
        ):
            skip_verification = True

        recorder = state.get("trace_recorder")
        if recorder and len(recorder.llm_calls) >= settings.pipeline.max_llm_calls:
            skip_verification = True

        ranked = await agent.reason_multi_candidate(
            evidence_chain,
            features,
            skip_verification=skip_verification,
        )
        prediction = ranked[0] if ranked else await agent.reason(
            evidence_chain,
            features,
            skip_verification=skip_verification,
        )
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
            "skip_full_verification": skip_verification,
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
    quality = (state.get("quality") or "balanced").lower()
    started = state.get("started_at_monotonic", 0.0) or 0.0

    if quality == "fast":
        return {"should_refine": False, "iteration": iteration}

    if state.get("early_exit", False):
        return {"should_refine": False, "iteration": iteration}

    pipeline_settings = get_settings().pipeline
    if started > 0 and pipeline_settings.max_total_latency_ms > 0:
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if elapsed_ms >= pipeline_settings.max_total_latency_ms:
            return {
                "should_refine": False,
                "iteration": iteration,
                "fast_path_reason": f"latency_budget_exceeded:{elapsed_ms:.0f}ms",
            }
        # Also check per-iteration refinement budget
        if pipeline_settings.max_refinement_latency_ms > 0:
            remaining_ms = pipeline_settings.max_total_latency_ms - elapsed_ms
            if remaining_ms < pipeline_settings.max_refinement_latency_ms:
                return {
                    "should_refine": False,
                    "iteration": iteration,
                    "fast_path_reason": f"insufficient_budget_for_refinement:{remaining_ms:.0f}ms_remaining",
                }

    recorder = state.get("trace_recorder")
    if recorder and len(recorder.llm_calls) >= pipeline_settings.max_llm_calls:
        return {
            "should_refine": False,
            "iteration": iteration,
            "fast_path_reason": f"llm_call_budget_exceeded:{len(recorder.llm_calls)}",
        }

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
