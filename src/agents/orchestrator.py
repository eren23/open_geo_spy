"""Orchestrator - LangGraph workflow with AoT task decomposition.

Decomposes geolocation into atomic subtasks running as a DAG:

[EXIF + VLM + OCR] -> [Evidence Merge] -> [ML Ensemble]  -> [Reasoning]
                                       -> [Web Intel]     -> [Verification]
                                                          -> [Result]

Independent nodes run in parallel via asyncio.gather().
Each node produces typed Evidence with source tracking.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable

from loguru import logger

from src.agents.candidate_verification_agent import CandidateVerificationAgent
from src.agents.feature_agent import FeatureExtractionAgent
from src.agents.ml_ensemble_agent import MLEnsembleAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.web_intel_agent import WebIntelAgent
from src.config.settings import Settings, get_settings
from src.evidence.chain import EvidenceChain


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
    """Main orchestrator for the geolocation pipeline.

    Manages the DAG execution of all agents and produces the final result.
    Supports SSE progress streaming via async generator.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.feature_agent = FeatureExtractionAgent(self.settings)
        self.ml_agent = MLEnsembleAgent(self.settings)
        self.web_agent = WebIntelAgent(self.settings)
        self.candidate_agent = CandidateVerificationAgent(self.settings)
        self.reasoning_agent = ReasoningAgent(self.settings)

    async def locate(
        self,
        image_path: str,
        location_hint: str | None = None,
    ) -> dict[str, Any]:
        """Run the full geolocation pipeline.

        Returns final result dict with location, confidence, evidence, reasoning.
        """
        start = time.monotonic()
        evidence_chain = EvidenceChain()
        progress = PipelineProgress()

        # --- Step 1: Feature Extraction (EXIF + VLM + OCR in parallel) ---
        step = await self._run_step(
            "feature_extraction",
            self._extract_features,
            image_path, location_hint,
        )
        progress.steps.append(step)
        feature_chain = step.data.get("chain", EvidenceChain())
        metadata = step.data.get("metadata", {})
        features = step.data.get("features", {})
        ocr_result = step.data.get("ocr_result", {})
        evidence_chain.add_many(feature_chain.evidences)

        # --- Step 2: ML Ensemble + Web Intel (parallel) ---
        ml_step, web_step = await asyncio.gather(
            self._run_step(
                "ml_ensemble",
                self._run_ml_ensemble,
                image_path, feature_chain,
            ),
            self._run_step(
                "web_intelligence",
                self._run_web_intel,
                evidence_chain, features, ocr_result,
            ),
        )
        progress.steps.extend([ml_step, web_step])
        ml_chain = ml_step.data.get("chain", EvidenceChain())
        web_chain = web_step.data.get("chain", EvidenceChain())
        evidence_chain.add_many(ml_chain.evidences)
        evidence_chain.add_many(web_chain.evidences)

        # Share StreetCLIP model with candidate agent to avoid reloading
        model_pair = self.ml_agent.streetclip_model_and_processor
        if model_pair:
            self.candidate_agent._scorer.model = model_pair[0]
            self.candidate_agent._scorer.processor = model_pair[1]

        # --- Step 2.5: Candidate Visual Verification ---
        if self.settings.ml.enable_visual_verification:
            verify_step = await self._run_step(
                "candidate_verification",
                self._run_candidate_verification,
                image_path, evidence_chain, features, ocr_result,
            )
            progress.steps.append(verify_step)
            verify_chain = verify_step.data.get("chain", EvidenceChain())
            evidence_chain.add_many(verify_chain.evidences)

        # --- Step 3: Reasoning + Verification ---
        reasoning_step = await self._run_step(
            "reasoning",
            self._run_reasoning,
            evidence_chain, features,
        )
        progress.steps.append(reasoning_step)
        prediction = reasoning_step.data.get("prediction", {})

        # Finalize
        elapsed = (time.monotonic() - start) * 1000
        progress.elapsed_ms = elapsed
        progress.total_evidence = len(evidence_chain.evidences)

        result = {
            **prediction,
            "pipeline_progress": progress.to_dict(),
            "total_evidence_count": len(evidence_chain.evidences),
            "elapsed_ms": round(elapsed, 1),
        }

        logger.info(
            "Pipeline complete: {} in {:.0f}ms with {} evidences (conf={:.2f})",
            prediction.get("name", "Unknown"),
            elapsed,
            len(evidence_chain.evidences),
            prediction.get("confidence", 0),
        )

        return result

    async def locate_stream(
        self,
        image_path: str,
        location_hint: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run pipeline with SSE progress streaming.

        Yields progress events during execution:
        {"event": "step_start", "step": "feature_extraction"}
        {"event": "step_complete", "step": "feature_extraction", "duration_ms": 1234}
        {"event": "result", "data": {...}}
        """
        start = time.monotonic()
        evidence_chain = EvidenceChain()

        # Step 1: Feature Extraction
        yield {"event": "step_start", "step": "feature_extraction"}
        try:
            chain, metadata, features, ocr_result = await self.feature_agent.extract_with_raw(
                image_path, location_hint
            )
            evidence_chain.add_many(chain.evidences)
            yield {
                "event": "step_complete",
                "step": "feature_extraction",
                "duration_ms": round((time.monotonic() - start) * 1000),
                "evidence_count": len(chain.evidences),
            }
        except Exception as e:
            yield {"event": "step_error", "step": "feature_extraction", "error": str(e)}
            chain, metadata, features, ocr_result = EvidenceChain(), {}, {}, {}

        # Step 2: ML + Web (parallel)
        yield {"event": "step_start", "step": "ml_ensemble"}
        yield {"event": "step_start", "step": "web_intelligence"}

        ml_task = self._run_ml_ensemble(image_path, chain)
        web_task = self._run_web_intel(evidence_chain, features, ocr_result)
        ml_result, web_result = await asyncio.gather(ml_task, web_task, return_exceptions=True)

        if isinstance(ml_result, dict):
            ml_chain = ml_result.get("chain", EvidenceChain())
            evidence_chain.add_many(ml_chain.evidences)
            yield {"event": "step_complete", "step": "ml_ensemble", "evidence_count": len(ml_chain.evidences)}
        else:
            yield {"event": "step_error", "step": "ml_ensemble", "error": str(ml_result)}

        if isinstance(web_result, dict):
            web_chain = web_result.get("chain", EvidenceChain())
            evidence_chain.add_many(web_chain.evidences)
            yield {"event": "step_complete", "step": "web_intelligence", "evidence_count": len(web_chain.evidences)}
        else:
            yield {"event": "step_error", "step": "web_intelligence", "error": str(web_result)}

        # Step 2.5: Candidate Visual Verification
        if self.settings.ml.enable_visual_verification:
            # Share StreetCLIP model
            model_pair = self.ml_agent.streetclip_model_and_processor
            if model_pair:
                self.candidate_agent._scorer.model = model_pair[0]
                self.candidate_agent._scorer.processor = model_pair[1]

            yield {"event": "step_start", "step": "candidate_verification"}
            try:
                verify_chain = await self.candidate_agent.verify_candidates(
                    image_path, evidence_chain, features, ocr_result
                )
                evidence_chain.add_many(verify_chain.evidences)
                yield {
                    "event": "step_complete",
                    "step": "candidate_verification",
                    "evidence_count": len(verify_chain.evidences),
                }
            except Exception as e:
                yield {"event": "step_error", "step": "candidate_verification", "error": str(e)}

        # Step 3: Reasoning
        yield {"event": "step_start", "step": "reasoning"}
        try:
            prediction = await self.reasoning_agent.reason(evidence_chain, features)
            yield {"event": "step_complete", "step": "reasoning"}
        except Exception as e:
            yield {"event": "step_error", "step": "reasoning", "error": str(e)}
            prediction = {"name": "Unknown", "lat": 0, "lon": 0, "confidence": 0, "reasoning": str(e)}

        elapsed = (time.monotonic() - start) * 1000
        yield {
            "event": "result",
            "data": {
                **prediction,
                "total_evidence_count": len(evidence_chain.evidences),
                "elapsed_ms": round(elapsed, 1),
            },
        }

    # --- Private step runners ---

    async def _run_step(
        self,
        name: str,
        func: Callable,
        *args,
    ) -> StepResult:
        """Run a pipeline step with timing and error handling."""
        start = time.monotonic()
        try:
            result_data = await func(*args)
            duration = (time.monotonic() - start) * 1000
            evidence_count = 0
            if isinstance(result_data, dict) and "chain" in result_data:
                evidence_count = len(result_data["chain"].evidences)
            logger.info("Step '{}' completed in {:.0f}ms ({} evidences)", name, duration, evidence_count)
            return StepResult(
                name=name,
                status=StepStatus.COMPLETED,
                duration_ms=round(duration, 1),
                evidence_count=evidence_count,
                data=result_data,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            logger.error("Step '{}' failed after {:.0f}ms: {}", name, duration, e)
            return StepResult(
                name=name,
                status=StepStatus.FAILED,
                duration_ms=round(duration, 1),
                error=str(e),
            )

    async def _extract_features(self, image_path: str, location_hint: str | None) -> dict:
        chain, metadata, features, ocr_result = await self.feature_agent.extract_with_raw(
            image_path, location_hint
        )
        return {"chain": chain, "metadata": metadata, "features": features, "ocr_result": ocr_result}

    async def _run_ml_ensemble(self, image_path: str, feature_chain: EvidenceChain) -> dict:
        chain = await self.ml_agent.predict(image_path, feature_chain)
        return {"chain": chain}

    async def _run_web_intel(self, evidence_chain: EvidenceChain, features: dict, ocr_result: dict) -> dict:
        chain = await self.web_agent.search(evidence_chain, features, ocr_result)
        return {"chain": chain}

    async def _run_candidate_verification(
        self, image_path: str, evidence_chain: EvidenceChain, features: dict, ocr_result: dict
    ) -> dict:
        chain = await self.candidate_agent.verify_candidates(
            image_path, evidence_chain, features, ocr_result
        )
        return {"chain": chain}

    async def _run_reasoning(self, evidence_chain: EvidenceChain, features: dict) -> dict:
        prediction = await self.reasoning_agent.reason(evidence_chain, features)
        return {"prediction": prediction}

    async def close(self):
        """Clean up all agent resources."""
        await self.web_agent.close()
        await self.candidate_agent.close()
