"""Per-run JSONL trace writer, injected via PipelineState."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import orjson
from loguru import logger

from src.tracing.schema import (
    LLMCallEvent,
    StepEvent,
    TraceEvent,
    TraceEventType,
    TraceHeader,
    TraceResult,
)


class TraceRecorder:
    """Writes trace events to a JSONL file for a single pipeline run."""

    def __init__(
        self,
        session_id: str,
        output_dir: str = "data/traces",
        version: str = "",
        settings_snapshot: dict | None = None,
        image_hash: str = "",
        on_event: Callable[[dict], None] | None = None,
    ):
        self.session_id = session_id
        self._events: list[dict] = []
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._llm_calls: list[dict] = []
        self._on_event = on_event

        # Build output path: data/traces/YYYY-MM-DD/{session_id}.jsonl
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._output_dir = Path(output_dir) / date_str
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._output_path = self._output_dir / f"{session_id}.jsonl"
        self._file = open(self._output_path, "ab")

        # Write header
        header = TraceHeader(
            session_id=session_id,
            image_hash=image_hash,
            version=version,
            settings_snapshot=settings_snapshot or {},
        )
        self._write_event(header.to_dict())

    def record_step(self, step_name: str, status: str, **kwargs: Any) -> None:
        """Record a pipeline step start or completion."""
        event = StepEvent(step_name=step_name, status=status, **kwargs)
        self._write_event(event.to_trace_event().to_dict())
        if self._on_event:
            try:
                self._on_event({
                    "event": f"step_{status}",
                    "step": step_name,
                    **{k: v for k, v in kwargs.items() if k in ("duration_ms", "evidence_count", "error")},
                })
            except Exception:
                pass

    def record_llm_call(
        self,
        model: str,
        purpose: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        temperature: float = 0.0,
    ) -> None:
        """Record an LLM API call."""
        total_tokens = input_tokens + output_tokens
        event = LLMCallEvent(
            model=model,
            purpose=purpose,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            temperature=temperature,
        )
        self._total_cost += cost_usd
        self._total_tokens += total_tokens
        self._llm_calls.append(event.__dict__)
        self._write_event(event.to_trace_event().to_dict())
        if self._on_event:
            try:
                self._on_event({
                    "event": "llm_call_complete",
                    "model": model,
                    "purpose": purpose,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost_usd,
                    "latency_ms": latency_ms,
                })
                self._on_event({
                    "event": "cost_update",
                    "total_cost_usd": self._total_cost,
                    "total_tokens": self._total_tokens,
                    "llm_call_count": len(self._llm_calls),
                })
            except Exception:
                pass

    def record_evidence(
        self,
        source: str,
        content_preview: str,
        confidence: float,
    ) -> None:
        """Record evidence being added to the chain."""
        self._write_event(TraceEvent(
            event_type=TraceEventType.EVIDENCE_ADDED,
            data={
                "source": source,
                "content_preview": content_preview[:200],
                "confidence": confidence,
            },
        ).to_dict())

    def record_grounding(self, level: str, verdict: str, confidence: float, explanation: str = "") -> None:
        """Record a grounding result."""
        self._write_event(TraceEvent(
            event_type=TraceEventType.GROUNDING_RESULT,
            data={
                "level": level,
                "verdict": verdict,
                "confidence": confidence,
                "explanation": explanation,
            },
        ).to_dict())

    def record_error(self, error: str, step: str = "") -> None:
        """Record an error."""
        self._write_event(TraceEvent(
            event_type=TraceEventType.ERROR,
            data={"error": error, "step": step},
        ).to_dict())

    def finalize(
        self,
        prediction: dict | None = None,
        candidates: list[dict] | None = None,
        total_duration_ms: float = 0.0,
        ground_truth: dict | None = None,
    ) -> Path:
        """Write final result and return the trace file path."""
        result = TraceResult(
            prediction=prediction or {},
            candidates=candidates or [],
            total_cost_usd=self._total_cost,
            total_tokens=self._total_tokens,
            total_duration_ms=total_duration_ms,
            ground_truth=ground_truth,
        )
        self._write_event(result.to_dict())
        self._close_file()
        logger.info("Trace saved: {} ({} events)", self._output_path, len(self._events))
        return self._output_path

    def _close_file(self) -> None:
        """Flush and close the persistent file handle."""
        if self._file and not self._file.closed:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def llm_calls(self) -> list[dict]:
        return list(self._llm_calls)

    @property
    def output_path(self) -> Path:
        return self._output_path

    def _write_event(self, event_dict: dict) -> None:
        """Append an event to the JSONL file."""
        self._events.append(event_dict)
        try:
            self._file.write(orjson.dumps(event_dict) + b"\n")
        except Exception as e:
            logger.warning("Failed to write trace event: {}", e)

    @staticmethod
    def hash_image(image_path: str) -> str:
        """Compute SHA-256 hash of an image file."""
        h = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
