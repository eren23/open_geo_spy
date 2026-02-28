"""FastAPI application with SSE streaming for real-time progress.

Replaces the old app.py with:
- Pydantic schemas for request/response validation
- SSE streaming for pipeline progress
- Environment-based CORS
- Structured logging
- Health check with dependency status
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from src.agents.orchestrator import GeoLocatorOrchestrator
from src.api.schemas import (
    CandidateResult,
    ChatRequest,
    HealthResponse,
    LocateResponse,
    LocateResponseV2,
    SessionResponse,
)
from src.config.settings import get_settings
from src.utils.logging import setup_logger


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    settings = get_settings()
    setup_logger("DEBUG" if settings.debug else "INFO")
    logger.info("Starting OpenGeoSpy API (env={})", settings.environment.value)

    # Initialize cache
    from src.cache import CacheStore
    cache = CacheStore(
        max_memory_entries=1000,
        disk_path=os.path.join(settings.image_dir, ".cache"),
        default_ttl=3600,
    )
    app.state.cache = cache

    # Initialize orchestrator
    app.state.orchestrator = GeoLocatorOrchestrator(settings, cache=cache)
    app.state.settings = settings

    # Initialize chat subsystem
    from src.chat.session import SessionManager
    from src.chat.handler import ChatHandler

    app.state.session_manager = SessionManager(ttl_seconds=3600)
    app.state.chat_handler = ChatHandler(settings, app.state.session_manager)

    # Initialize batch manager
    from src.batch.manager import BatchManager
    app.state.batch_manager = BatchManager(max_concurrent=3)

    yield

    # Shutdown
    await app.state.orchestrator.close()
    logger.info("OpenGeoSpy API shutdown")


# --- App ---


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="OpenGeoSpy API",
        version="2.0.0",
        description="Multi-agent geolocation system with evidence tracking",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_routes(app)
    return app


def _register_routes(app: FastAPI):
    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        settings = app.state.settings
        return HealthResponse(
            status="ok",
            version="2.0.0",
            services={
                "llm": bool(settings.llm.api_key),
                "serper": bool(settings.geo.serper_api_key),
                "browser": settings.browser.enabled,
                "geoclip": settings.ml.enable_geoclip,
                "streetclip": settings.ml.enable_streetclip,
                "visual_verification": settings.ml.enable_visual_verification,
            },
        )

    @app.post("/api/locate", response_model=LocateResponse)
    async def locate(
        file: UploadFile = File(...),
        location_hint: str | None = Form(None),
    ):
        """Analyze an image to determine its location.

        Runs the full multi-agent pipeline:
        1. Feature extraction (EXIF + VLM + OCR)
        2. ML ensemble (GeoCLIP + StreetCLIP + VLM geo)
        3. Web intelligence (Serper + Google + OSM)
        4. Reasoning + verification

        Returns location with confidence, evidence trail, and reasoning.
        """
        settings = app.state.settings
        upload_dir = settings.image_dir
        os.makedirs(upload_dir, exist_ok=True)

        # Save uploaded file
        ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
        file_id = str(uuid.uuid4())
        file_path = os.path.join(upload_dir, f"{file_id}{ext}")

        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            orchestrator: GeoLocatorOrchestrator = app.state.orchestrator
            result = await orchestrator.locate(file_path, location_hint)

            return LocateResponse(**_normalize_result(result))

        except Exception as e:
            logger.error("Locate failed: {}", e)
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)

    @app.post("/api/locate/stream")
    async def locate_stream(
        file: UploadFile = File(...),
        location_hint: str | None = Form(None),
    ):
        """SSE streaming version of locate.

        Streams progress events during pipeline execution:
        - step_start: A pipeline step has started
        - step_complete: A step finished successfully
        - step_error: A step failed
        - result: Final result
        """
        settings = app.state.settings
        upload_dir = settings.image_dir
        os.makedirs(upload_dir, exist_ok=True)

        ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
        file_id = str(uuid.uuid4())
        file_path = os.path.join(upload_dir, f"{file_id}{ext}")

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        async def event_generator() -> AsyncGenerator[dict, None]:
            try:
                orchestrator: GeoLocatorOrchestrator = app.state.orchestrator
                async for event in orchestrator.locate_stream(file_path, location_hint):
                    if event.get("event") == "result":
                        # Normalize the final result
                        event["data"] = _normalize_result(event.get("data", {}))
                    yield {"event": event.get("event", "progress"), "data": json.dumps(event)}
            except Exception as e:
                logger.error("Stream failed: {}", e)
                yield {"event": "error", "data": json.dumps({"error": str(e)})}
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

        return EventSourceResponse(event_generator())

    @app.post("/api/v2/locate/stream")
    async def locate_stream_v2(
        file: UploadFile = File(...),
        location_hint: str | None = Form(None),
    ):
        """V2 SSE streaming: multi-candidate results + session + search graph."""
        settings = app.state.settings
        upload_dir = settings.image_dir
        os.makedirs(upload_dir, exist_ok=True)

        ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
        file_id = str(uuid.uuid4())
        file_path = os.path.join(upload_dir, f"{file_id}{ext}")

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        async def event_generator() -> AsyncGenerator[dict, None]:
            try:
                orchestrator: GeoLocatorOrchestrator = app.state.orchestrator
                async for event in orchestrator.locate_stream_v2(file_path, location_hint):
                    if event.get("event") == "result":
                        event["data"] = _normalize_result_v2(event.get("data", {}))
                    yield {"event": event.get("event", "progress"), "data": json.dumps(event)}
            except Exception as e:
                logger.error("V2 stream failed: {}", e)
                yield {"event": "error", "data": json.dumps({"error": str(e)})}
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

        return EventSourceResponse(event_generator())

    # --- Chat & Session endpoints ---

    @app.post("/api/chat/{session_id}")
    async def chat_followup(session_id: str, request: ChatRequest):
        """Chat follow-up on a locate session (SSE streaming)."""
        chat_handler = getattr(app.state, "chat_handler", None)
        if not chat_handler:
            raise HTTPException(status_code=501, detail="Chat not available")

        async def event_generator() -> AsyncGenerator[dict, None]:
            async for event in chat_handler.handle_message(session_id, request.message):
                yield {"event": event.get("event", "chat"), "data": json.dumps(event)}

        return EventSourceResponse(event_generator())

    @app.get("/api/session/{session_id}", response_model=SessionResponse)
    async def get_session(session_id: str):
        """Get session state (candidates, evidence, search graph)."""
        session_mgr = getattr(app.state, "session_manager", None)
        if not session_mgr:
            raise HTTPException(status_code=501, detail="Sessions not available")

        session = session_mgr.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionResponse(
            session_id=session.id,
            candidates=session.pipeline_state.get("ranked_candidates", []),
            evidence_count=len(session.pipeline_state.get("evidences", [])),
            search_graph=None,
            messages=[],
        )

    @app.get("/api/session/{session_id}/search-graph")
    async def get_search_graph(session_id: str):
        """Get the search graph for a session."""
        session_mgr = getattr(app.state, "session_manager", None)
        if not session_mgr:
            raise HTTPException(status_code=501, detail="Sessions not available")

        session = session_mgr.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        sg = session.pipeline_state.get("search_graph")
        if sg and hasattr(sg, "to_dict"):
            return sg.to_dict()
        return {"nodes": [], "edges": [], "root_ids": [], "stats": {}}

    # --- Batch endpoints ---

    @app.post("/api/batch/locate")
    async def batch_locate(files: list[UploadFile] = File(...)):
        """Upload multiple images for batch geolocation."""
        from src.batch.manager import BatchManager

        settings = app.state.settings
        upload_dir = settings.image_dir
        os.makedirs(upload_dir, exist_ok=True)

        batch_mgr: BatchManager = app.state.batch_manager
        items = []

        for f in files:
            ext = os.path.splitext(f.filename or "image.jpg")[1] or ".jpg"
            file_id = str(uuid.uuid4())
            file_path = os.path.join(upload_dir, f"{file_id}{ext}")
            with open(file_path, "wb") as fp:
                shutil.copyfileobj(f.file, fp)
            items.append({"filename": f.filename or f"image{ext}", "image_path": file_path})

        job = batch_mgr.create_job(items)

        # Start processing in background
        orchestrator: GeoLocatorOrchestrator = app.state.orchestrator
        asyncio.create_task(batch_mgr.process_job(job, orchestrator))

        return {"batch_id": job.id, "total": job.total}

    @app.get("/api/batch/{batch_id}")
    async def batch_status(batch_id: str):
        """Get batch job status."""
        batch_mgr = app.state.batch_manager
        job = batch_mgr.get_job(batch_id)
        if not job:
            raise HTTPException(status_code=404, detail="Batch job not found")
        return job.to_dict()

    @app.get("/api/batch/{batch_id}/export")
    async def batch_export(batch_id: str, format: str = "csv"):
        """Export batch results as CSV or JSON."""
        from fastapi.responses import PlainTextResponse

        from src.batch.export import export_csv, export_json

        batch_mgr = app.state.batch_manager
        job = batch_mgr.get_job(batch_id)
        if not job:
            raise HTTPException(status_code=404, detail="Batch job not found")

        if format == "json":
            return PlainTextResponse(
                export_json(job),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}.json"},
            )
        else:
            return PlainTextResponse(
                export_csv(job),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}.csv"},
            )


def _normalize_result_v2(result: dict) -> dict:
    """Normalize result dict for V2 response (includes candidates)."""
    base = _normalize_result(result)
    # Normalize each candidate's lat/lon → latitude/longitude
    raw_candidates = result.get("candidates", [])
    base["candidates"] = [_normalize_candidate(c, rank=i + 1) for i, c in enumerate(raw_candidates)]
    base["search_graph"] = result.get("search_graph")
    base["session_id"] = result.get("session_id")
    return base


def _normalize_candidate(c: dict, rank: int = 1) -> dict:
    """Normalize a single candidate dict for the API response."""
    return {
        "rank": c.get("rank", rank),
        "name": c.get("name", "Unknown"),
        "country": c.get("country"),
        "region": c.get("region"),
        "city": c.get("city"),
        "latitude": c.get("latitude") or c.get("lat"),
        "longitude": c.get("longitude") or c.get("lon"),
        "confidence": c.get("confidence", 0.0),
        "reasoning": c.get("reasoning", ""),
        "evidence_trail": c.get("evidence_trail", []),
        "visual_match_score": c.get("visual_match_score"),
        "source_diversity": c.get("source_diversity", 0),
    }


def _normalize_result(result: dict) -> dict:
    """Ensure result dict has all expected fields for LocateResponse."""
    return {
        "name": result.get("name", "Unknown"),
        "country": result.get("country"),
        "region": result.get("region"),
        "city": result.get("city"),
        "latitude": result.get("lat") or result.get("latitude"),
        "longitude": result.get("lon") or result.get("longitude"),
        "confidence": result.get("confidence", 0.0),
        "reasoning": result.get("reasoning", ""),
        "verified": result.get("verified", False),
        "verification_warning": result.get("verification_warning"),
        "evidence_trail": result.get("evidence_trail", []),
        "evidence_summary": result.get("evidence_summary", {}),
        "pipeline_progress": result.get("pipeline_progress", {}),
        "total_evidence_count": result.get("total_evidence_count", 0),
        "elapsed_ms": result.get("elapsed_ms", 0.0),
    }


# Default app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.debug,
    )
