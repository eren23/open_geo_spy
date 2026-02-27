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

    model_config = {"populate_by_name": True}


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"
    services: dict[str, bool] = {}


class SSEEvent(BaseModel):
    """Server-Sent Event payload."""

    event: str
    step: Optional[str] = None
    duration_ms: Optional[float] = None
    evidence_count: Optional[int] = None
    error: Optional[str] = None
    data: Optional[dict[str, Any]] = None
