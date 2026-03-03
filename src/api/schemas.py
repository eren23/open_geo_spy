"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# --- Requests ---


class LocateRequest(BaseModel):
    """Query parameters for locate endpoint (file comes via form data)."""

    location_hint: Optional[str] = Field(None, description="Optional location hint to bias search")


# --- Responses ---


class EvidenceItem(BaseModel):
    source: str
    content: str
    confidence: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    url: Optional[str] = None


class PipelineStep(BaseModel):
    name: str
    status: str
    duration_ms: float = 0.0
    evidence_count: int = 0
    error: Optional[str] = None


class LocateResponse(BaseModel):
    """Response from the locate endpoint."""

    name: str = "Unknown"
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    lat: Optional[float] = Field(None, alias="latitude")
    lon: Optional[float] = Field(None, alias="longitude")
    confidence: float = 0.0
    reasoning: str = ""
    verified: bool = False
    verification_warning: Optional[str] = None

    evidence_trail: list[EvidenceItem] = []
    evidence_summary: dict[str, Any] = {}
    pipeline_progress: dict[str, Any] = {}
    total_evidence_count: int = 0
    elapsed_ms: float = 0.0
    execution_policy: dict[str, Any] = {}
    quality: str = "balanced"
    fast_path_reason: Optional[str] = None

    model_config = {"populate_by_name": True}


class GroundingInfo(BaseModel):
    """Per-level grounding verdict from the HierarchicalResolver."""

    level: str
    value: Optional[str] = None
    verdict: str = "uncertain"
    confidence: float = 0.0
    supporting_count: int = 0
    contradicting_count: int = 0
    source_count: int = 0
    explanation: str = ""


class CandidateResult(BaseModel):
    """A ranked location candidate."""

    rank: int
    name: str
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    confidence: float = 0.0
    reasoning: str = ""
    evidence_trail: list[EvidenceItem] = []
    visual_match_score: Optional[float] = None
    source_diversity: int = 0
    resolved_level: Optional[str] = None
    groundings: list[GroundingInfo] = []


class LocateResponseV2(BaseModel):
    """V2 response with multi-candidate ranking and search graph."""

    # Primary prediction (same as v1 for backward compat)
    name: str = "Unknown"
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    confidence: float = 0.0
    reasoning: str = ""
    verified: bool = False
    verification_warning: Optional[str] = None

    # V2 additions
    candidates: list[CandidateResult] = []
    search_graph: Optional[dict[str, Any]] = None
    session_id: Optional[str] = None

    # Common
    evidence_trail: list[EvidenceItem] = []
    evidence_summary: dict[str, Any] = {}
    pipeline_progress: dict[str, Any] = {}
    total_evidence_count: int = 0
    elapsed_ms: float = 0.0
    execution_policy: dict[str, Any] = {}
    quality: str = "balanced"
    fast_path_reason: Optional[str] = None


class ChatRequest(BaseModel):
    """Request for chat follow-up."""

    message: str = Field(..., min_length=1, max_length=2000)


class ChatMessageSchema(BaseModel):
    """A single chat message."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class SessionResponse(BaseModel):
    """Response with session state."""

    session_id: str
    candidates: list[CandidateResult] = []
    evidence_count: int = 0
    search_graph: Optional[dict[str, Any]] = None
    messages: list[ChatMessageSchema] = []


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.3.0"
    services: dict[str, bool] = {}


class SSEEvent(BaseModel):
    """Server-Sent Event payload."""

    event: str
    step: Optional[str] = None
    duration_ms: Optional[float] = None
    evidence_count: Optional[int] = None
    error: Optional[str] = None
    data: Optional[dict[str, Any]] = None

    # Tracing-specific fields (v0.3.0)
    model: Optional[str] = None
    tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    source: Optional[str] = None
    content_preview: Optional[str] = None
    confidence: Optional[float] = None
    level: Optional[str] = None
    verdict: Optional[str] = None
