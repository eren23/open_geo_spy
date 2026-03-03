"""Orchestrator - wraps the LangGraph pipeline.

Provides the same ``locate()`` and ``locate_stream()`` API as v0.1.0
but delegates to the compiled LangGraph StateGraph internally.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator

from loguru import logger

from openai import AsyncOpenAI

from src.agents.graph import build_pipeline_graph
from src.agents.state import PipelineState
from src.cache import CacheStore
from src.config.settings import Settings, get_settings
from src.evidence.chain import EvidenceChain
from src.tracing.recorder import TraceRecorder
from src.tracing.index import TraceIndex


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result from a pipeline step."""

    name: str
    status: StepStatus
    duration_ms: float = 0.0
    evidence_count: int = 0
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineProgress:
    """Tracks overall pipeline progress for SSE streaming."""

    steps: list[StepResult] = field(default_factory=list)
    current_step: str = ""
    total_evidence: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "steps": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "duration_ms": s.duration_ms,
                    "evidence_count": s.evidence_count,
                    "error": s.error,
                }
                for s in self.steps
            ],
            "current_step": self.current_step,
            "total_evidence": self.total_evidence,
            "elapsed_ms": self.elapsed_ms,
        }


class GeoLocatorOrchestrator:
    """Main orchestrator wrapping the LangGraph pipeline.

    Backward-compatible API: ``locate()`` and ``locate_stream()``.
    """

    def __init__(self, settings: Settings | None = None, cache: CacheStore | None = None):
        self.settings = settings or get_settings()
        self.cache = cache
        self._graph = build_pipeline_graph(self.settings, cache=cache)

    def _initial_state(
        self,
        image_path: str,
        location_hint: str | None = None,
        session_id: str | None = None,
        quality: str = "balanced",
    ) -> PipelineState:
        sid = session_id or str(uuid.uuid4())
        quality = quality.lower().strip() if quality else "balanced"
        if quality not in {"fast", "balanced", "max"}:
            quality = "balanced"

        # Create trace recorder for this pipeline run
        try:
            image_hash = TraceRecorder.hash_image(image_path)
        except Exception:
            image_hash = ""

        recorder = TraceRecorder(
            session_id=sid,
            version=self.settings.version if hasattr(self.settings, "version") else "0.3.0",
            image_hash=image_hash,
        )

        # Single base LLM client shared across all pipeline nodes
        base_llm_client = AsyncOpenAI(
            base_url=self.settings.llm.base_url,
            api_key=self.settings.llm.api_key,
        )

        return {
            "image_path": image_path,
            "location_hint": location_hint,
            "session_id": sid,
            "quality": quality,
            "evidences": [],
            "metadata": {},
            "features": {},
            "ocr_result": {},
            "candidates": [],
            "search_graph": None,
            "prediction": {},
            "ranked_candidates": [],
            "iteration": 0,
            "max_iterations": 2,
            "weak_evidence_areas": [],
            "should_refine": False,
            "early_exit": False,
            "skip_full_verification": quality == "fast",
            "fast_path_reason": "quality=fast" if quality == "fast" else None,
            "messages": [],
            "step_results": [],
            "errors": [],
            "started_at_monotonic": time.monotonic(),
            "execution_policy": {
                "quality": quality,
                "fast_path_enabled": self.settings.pipeline.fast_path_enabled,
                "max_total_latency_ms": self.settings.pipeline.max_total_latency_ms,
                "max_llm_calls": self.settings.pipeline.max_llm_calls,
            },
            "trace_recorder": recorder,
            "_base_llm_client": base_llm_client,
        }

    async def locate(
        self,
        image_path: str,
        location_hint: str | None = None,
        quality: str = "balanced",
    ) -> dict[str, Any]:
        """Run the full geolocation pipeline (blocking).

        Returns final result dict with location, confidence, evidence, reasoning.
        """
        start = time.monotonic()
        state = self._initial_state(image_path, location_hint, quality=quality)

        try:
            # Run graph to completion
            final_state = await self._graph.ainvoke(state)

            elapsed = (time.monotonic() - start) * 1000
            prediction = final_state.get("prediction", {})

            # Build progress from step_results
            progress = PipelineProgress()
            for sr in final_state.get("step_results", []):
                progress.steps.append(
                    StepResult(
                        name=sr.get("name", ""),
                        status=StepStatus(sr.get("status", "completed")),
                        duration_ms=sr.get("duration_ms", 0),
                        evidence_count=sr.get("evidence_count", 0),
                        error=sr.get("error"),
                    )
                )
            progress.elapsed_ms = elapsed
            progress.total_evidence = len(final_state.get("evidences", []))

            result = {
                **prediction,
                "pipeline_progress": progress.to_dict(),
                "total_evidence_count": progress.total_evidence,
                "elapsed_ms": round(elapsed, 1),
                "execution_policy": final_state.get("execution_policy", state.get("execution_policy", {})),
                "quality": final_state.get("quality", quality),
                "fast_path_reason": final_state.get("fast_path_reason"),
            }

            # Finalize trace recording
            recorder = final_state.get("trace_recorder")
            if recorder:
                try:
                    trace_path = recorder.finalize(
                        prediction=prediction,
                        candidates=final_state.get("ranked_candidates", []),
                        total_duration_ms=elapsed,
                    )
                    index = TraceIndex()
                    try:
                        index.index_trace(trace_path)
                    finally:
                        index.close()
                except Exception as e:
                    logger.warning("Trace finalization failed: {}", e)

            logger.info(
                "Pipeline complete: {} in {:.0f}ms with {} evidences (conf={:.2f})",
                prediction.get("name", "Unknown"),
                elapsed,
                progress.total_evidence,
                prediction.get("confidence", 0),
            )

            return result
        finally:
            recorder = state.get("trace_recorder")
            if recorder:
                recorder._close_file()

    async def locate_stream(
        self,
        image_path: str,
        location_hint: str | None = None,
        session_id: str | None = None,
        quality: str = "balanced",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run pipeline with SSE progress streaming.

        Yields progress events during execution:
          {"event": "step_start",    "step": "feature_extraction"}
          {"event": "step_complete", "step": "feature_extraction", "duration_ms": 1234}
          {"event": "refinement",    "iteration": 1, "weak_areas": [...]}
          {"event": "result",        "data": {...}}
        """
        start = time.monotonic()
        sid = session_id or str(uuid.uuid4())
        state = self._initial_state(image_path, location_hint, sid, quality=quality)

        try:
            # Use both "custom" and "values" stream modes to get SSE events
            # AND final state in a single pass (no double invocation).
            final_state = None
            async for mode, chunk in self._graph.astream(
                state,
                stream_mode=["custom", "values"],
            ):
                if mode == "custom":
                    if isinstance(chunk, dict) and "event" in chunk:
                        yield chunk
                elif mode == "values":
                    # Each values chunk is a full state snapshot;
                    # keep the latest one as the final state.
                    final_state = chunk

            if final_state is None:
                raise RuntimeError("Pipeline produced no final state")

            elapsed = (time.monotonic() - start) * 1000
            prediction = final_state.get("prediction", {})

            # Finalize trace recording
            recorder = final_state.get("trace_recorder")
            if recorder:
                try:
                    trace_path = recorder.finalize(
                        prediction=prediction,
                        candidates=final_state.get("ranked_candidates", []),
                        total_duration_ms=elapsed,
                    )
                    index = TraceIndex()
                    try:
                        index.index_trace(trace_path)
                    finally:
                        index.close()
                except Exception as e:
                    logger.warning("Trace finalization failed: {}", e)

            yield {
                "event": "result",
                "data": {
                    **prediction,
                    "session_id": sid,
                    "total_evidence_count": len(final_state.get("evidences", [])),
                    "elapsed_ms": round(elapsed, 1),
                    "execution_policy": final_state.get("execution_policy", state.get("execution_policy", {})),
                    "quality": final_state.get("quality", quality),
                    "fast_path_reason": final_state.get("fast_path_reason"),
                },
            }

        except Exception as e:
            logger.error("Stream pipeline failed: {}", e)
            yield {"event": "error", "error": str(e)}
        finally:
            recorder = state.get("trace_recorder")
            if recorder:
                recorder._close_file()

    async def locate_stream_v2(
        self,
        image_path: str,
        location_hint: str | None = None,
        session_id: str | None = None,
        quality: str = "balanced",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """V2 streaming: produces multi-candidate results + search graph.

        Same SSE events as ``locate_stream()`` plus:
          {"event": "result", "data": {candidates: [...], search_graph: {...}, ...}}
        """
        start = time.monotonic()
        sid = session_id or str(uuid.uuid4())
        state = self._initial_state(image_path, location_hint, sid, quality=quality)

        try:
            final_state = None
            async for mode, chunk in self._graph.astream(
                state,
                stream_mode=["custom", "values"],
            ):
                if mode == "custom":
                    if isinstance(chunk, dict) and "event" in chunk:
                        yield chunk
                elif mode == "values":
                    final_state = chunk

            if final_state is None:
                raise RuntimeError("Pipeline produced no final state")

            elapsed = (time.monotonic() - start) * 1000
            prediction = final_state.get("prediction", {})
            ranked = final_state.get("ranked_candidates", [])

            # Finalize trace recording
            recorder = final_state.get("trace_recorder")
            if recorder:
                try:
                    trace_path = recorder.finalize(
                        prediction=prediction,
                        candidates=ranked,
                        total_duration_ms=elapsed,
                    )
                    index = TraceIndex()
                    try:
                        index.index_trace(trace_path)
                    finally:
                        index.close()
                except Exception as e:
                    logger.warning("Trace finalization failed: {}", e)

            # If the pipeline produced ranked candidates, use them.
            # Otherwise, wrap the single prediction as the sole candidate.
            if not ranked and prediction:
                ranked = [{
                    **prediction,
                    "rank": 1,
                    "latitude": prediction.get("lat"),
                    "longitude": prediction.get("lon"),
                    "evidence_trail": prediction.get("evidence_trail", []),
                    "source_diversity": len({
                        e.get("source", "") for e in prediction.get("evidence_trail", [])
                        if isinstance(e, dict)
                    }),
                }]

            yield {
                "event": "result",
                "data": {
                    **prediction,
                    "candidates": ranked,
                    "session_id": sid,
                    "search_graph": (
                        final_state.get("search_graph").to_dict()
                        if final_state.get("search_graph")
                        and hasattr(final_state["search_graph"], "to_dict")
                        else None
                    ),
                    "total_evidence_count": len(final_state.get("evidences", [])),
                    "elapsed_ms": round(elapsed, 1),
                    "execution_policy": final_state.get("execution_policy", state.get("execution_policy", {})),
                    "quality": final_state.get("quality", quality),
                    "fast_path_reason": final_state.get("fast_path_reason"),
                },
            }

        except Exception as e:
            logger.error("V2 stream pipeline failed: {}", e)
            yield {"event": "error", "error": str(e)}
        finally:
            recorder = state.get("trace_recorder")
            if recorder:
                recorder._close_file()

    async def close(self):
        """Clean up resources."""
        pass


# Alias for CLI and eval imports
Orchestrator = GeoLocatorOrchestrator
