"""Trace event dataclasses for pipeline run persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class TraceEventType(str, Enum):
    HEADER = "header"
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    LLM_CALL = "llm_call"
    EVIDENCE_ADDED = "evidence_added"
    SEARCH_QUERY = "search_query"
    CANDIDATE_SNAPSHOT = "candidate_snapshot"
    FINAL_SELECTION = "final_selection"
    ANOMALY_FLAG = "anomaly_flag"
    COST_UPDATE = "cost_update"
    GROUNDING_RESULT = "grounding_result"
    ERROR = "error"
    RESULT = "result"


@dataclass
class TraceHeader:
    """First line of a trace JSONL file."""

    session_id: str
    image_path: str = ""
    image_hash: str = ""
    version: str = ""
    settings_snapshot: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "type": TraceEventType.HEADER.value,
            "session_id": self.session_id,
            "image_path": self.image_path,
            "image_hash": self.image_hash,
            "version": self.version,
            "settings_snapshot": self.settings_snapshot,
            "timestamp": self.timestamp,
        }


@dataclass
class TraceEvent:
    """A single trace event (step, LLM call, evidence, etc.)."""

    event_type: TraceEventType
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.event_type.value,
            "timestamp": self.timestamp,
            **self.data,
        }


@dataclass
class LLMCallEvent:
    """Details of an LLM API call."""

    model: str
    purpose: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    temperature: float = 0.0

    def to_trace_event(self) -> TraceEvent:
        return TraceEvent(
            event_type=TraceEventType.LLM_CALL,
            data={
                "model": self.model,
                "purpose": self.purpose,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "cost_usd": self.cost_usd,
                "latency_ms": self.latency_ms,
                "temperature": self.temperature,
            },
        )


@dataclass
class StepEvent:
    """Pipeline step start/complete event."""

    step_name: str
    status: str  # "started" or "completed"
    duration_ms: float = 0.0
    evidence_count: int = 0
    error: Optional[str] = None

    def to_trace_event(self) -> TraceEvent:
        event_type = TraceEventType.STEP_START if self.status == "started" else TraceEventType.STEP_COMPLETE
        data: dict[str, Any] = {
            "step": self.step_name,
            "status": self.status,
        }
        if self.duration_ms:
            data["duration_ms"] = self.duration_ms
        if self.evidence_count:
            data["evidence_count"] = self.evidence_count
        if self.error:
            data["error"] = self.error
        return TraceEvent(event_type=event_type, data=data)


@dataclass
class TraceResult:
    """Last line of a trace JSONL file — final results."""

    prediction: dict = field(default_factory=dict)
    candidates: list[dict] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    ground_truth: Optional[dict] = None  # Set during eval runs

    def to_dict(self) -> dict:
        return {
            "type": TraceEventType.RESULT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": self.prediction,
            "candidates": self.candidates,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "ground_truth": self.ground_truth,
        }
