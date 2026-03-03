"""FastAPI application with SSE streaming for real-time progress.

Replaces the old app.py with:
- Pydantic schemas for request/response validation
- SSE streaming for pipeline progress
- Environment-based CORS
- Structured logging
- Health check with dependency status
- File upload validation (magic bytes + size)
- Rate limiting middleware
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
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


# --- File Validation ---

# Magic bytes for common image formats
IMAGE_SIGNATURES = {
    b'\xff\xd8\xff': 'jpeg',           # JPEG
    b'\x89PNG\r\n\x1a\n': 'png',       # PNG
    b'GIF87a': 'gif',                  # GIF87a
    b'GIF89a': 'gif',                  # GIF89a
    b'RIFF': 'webp',                   # WebP (needs further check)
    b'\x00\x00\x00\x1cftyp': 'heic',   # HEIC (variant)
    b'\x00\x00\x00\x20ftyp': 'heic',   # HEIC (variant)
    b'ftyp': 'mp4',                    # MP4/MOV (video)
}

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic', '.mp4', '.mov'}


def validate_image_file(file: UploadFile, max_size_mb: int = 50) -> tuple[bool, str]:
    """Validate uploaded file by magic bytes and size.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check extension
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"File type '{ext}' not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check content type
    content_type = file.content_type or ""
    if not content_type.startswith(('image/', 'video/')):
        return False, f"Invalid content type: {content_type}"
    
    return True, ""


async def validate_file_contents(file: UploadFile, max_size_mb: int = 50) -> tuple[bool, str, bytes]:
    """Read and validate file contents including magic bytes.
    
    Returns:
        Tuple of (is_valid, error_message, file_bytes)
    """
    # Read file contents
    contents = await file.read()
    await file.seek(0)  # Reset for later reading
    
    # Check size
    max_bytes = max_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        return False, f"File too large: {len(contents) / 1024 / 1024:.1f}MB (max: {max_size_mb}MB)", contents
    
    # Check magic bytes
    header = contents[:16]
    is_valid_image = False
    
    for sig, fmt in IMAGE_SIGNATURES.items():
        if header.startswith(sig):
            is_valid_image = True
            break
        # Special case: WebP has RIFF....WEBP
        if sig == b'RIFF' and len(header) >= 12 and header[8:12] == b'WEBP':
            is_valid_image = True
            break
        # Special case: ftyp for video containers
        if b'ftyp' in header[:12]:
            is_valid_image = True
            break
    
    if not is_valid_image:
        return False, "Invalid file format: file signature does not match image/video types", contents
    
    return True, "", contents


# --- Rate Limiting ---

class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self._requests: dict[str, list[float]] = {}
    
    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, remaining_requests)."""
        now = time.time()
        window_start = now - 60.0
        
        # Get or create request list
        if client_id not in self._requests:
            self._requests[client_id] = []
        
        # Clean old requests
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > window_start
        ]
        
        remaining = max(0, self.rpm - len(self._requests[client_id]))
        
        if len(self._requests[client_id]) >= self.rpm:
            return False, 0
        
        # Record this request
        self._requests[client_id].append(now)
        return True, remaining - 1
    
    def cleanup_stale(self, max_age: float = 120.0):
        """Remove stale entries to prevent memory leak."""
        cutoff = time.time() - max_age
        stale = [k for k, v in self._requests.items() if not v or v[-1] < cutoff]
        for k in stale:
            del self._requests[k]


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    settings = get_settings()
    setup_logger("DEBUG" if settings.debug else "INFO")
    logger.info("Starting OpenGeoSpy API (env={})", settings.environment.value)

    # Initialize cache
    from src.cache import CacheStore
    cache = (
        CacheStore(
            max_memory_entries=settings.cache.max_memory_entries,
            disk_path=settings.cache.disk_path if settings.cache.backend == "disk" else None,
            default_ttl=settings.cache.serper_ttl,
        )
        if settings.cache.enabled
        else None
    )
    app.state.cache = cache

    # Initialize rate limiter
    app.state.rate_limiter = RateLimiter(requests_per_minute=settings.api.rate_limit_rpm)

    # Initialize orchestrator
    app.state.orchestrator = GeoLocatorOrchestrator(settings, cache=cache)
    app.state.settings = settings

    # Preload ML models in background to avoid cold start on first request
    async def _preload_models():
        try:
            # Import adapters module — @ModelRegistry.register decorators execute on import
            import src.models.adapters  # noqa: F401

            from src.models.registry import ModelRegistry
            models = await asyncio.to_thread(ModelRegistry.get_enabled, settings)
            logger.info("Preloaded {} ML models", len(models))
        except Exception as e:
            logger.warning("Model preload failed (will load on first request): {}", e)

    asyncio.create_task(_preload_models())

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
        version="0.3.0",
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
            version="0.3.0",
            services={
                "llm": bool(settings.llm.api_key),
                "serper": bool(settings.geo.serper_api_key),
                "browser": settings.browser.enabled,
                "geoclip": settings.ml.enable_geoclip,
                "streetclip": settings.ml.enable_streetclip,
                "visual_verification": settings.ml.enable_visual_verification,
            },
        )
    
    @app.get("/api/health/deep", response_model=HealthResponse)
    async def health_deep():
        """Deep health check that actually pings external services."""
        import httpx
        
        settings = app.state.settings
        services = {
            "llm": False,
            "serper": False,
            "browser": settings.browser.enabled,
            "geoclip": settings.ml.enable_geoclip,
            "streetclip": settings.ml.enable_streetclip,
            "visual_verification": settings.ml.enable_visual_verification,
        }
        
        # Quick LLM check
        if settings.llm.api_key:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    # Just check if the API endpoint is reachable
                    resp = await client.get(
                        settings.llm.base_url.rstrip("/") + "/models",
                        headers={"Authorization": f"Bearer {settings.llm.api_key[:10]}..."},
                    )
                    services["llm"] = resp.status_code in (200, 401, 403)
            except Exception:
                services["llm"] = False
        
        # Serper check
        if settings.geo.serper_api_key:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(
                        "https://google.serper.dev/search",
                        headers={"X-API-KEY": settings.geo.serper_api_key},
                        json={"q": "test"},
                    )
                    services["serper"] = resp.status_code == 200
            except Exception:
                services["serper"] = False
        
        all_healthy = all(services.values())
        return HealthResponse(
            status="ok" if all_healthy else "degraded",
            version="0.3.0",
            services=services,
        )

    @app.post("/api/locate", response_model=LocateResponse)
    async def locate(
        request: Request,
        file: UploadFile = File(...),
        location_hint: str | None = Form(None),
        quality: str = Form("fast"),
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
        
        # Rate limiting
        client_id = request.client.host if request.client else "unknown"
        rate_limiter: RateLimiter = app.state.rate_limiter
        allowed, remaining = rate_limiter.is_allowed(client_id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait before making more requests.",
                headers={"Retry-After": "60", "X-RateLimit-Remaining": str(remaining)},
            )
        
        # Validate file type by extension
        is_valid, error_msg = validate_image_file(file, settings.api.max_upload_size_mb)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Validate file contents (magic bytes + size)
        is_valid, error_msg, contents = await validate_file_contents(
            file, settings.api.max_upload_size_mb
        )
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        upload_dir = settings.image_dir
        os.makedirs(upload_dir, exist_ok=True)

        # Save uploaded file with sanitized extension
        ext = os.path.splitext(file.filename or "image.jpg")[1].lower() or ".jpg"
        if ext not in ALLOWED_EXTENSIONS:
            ext = ".jpg"  # Safe default
        file_id = str(uuid.uuid4())
        file_path = os.path.join(upload_dir, f"{file_id}{ext}")

        try:
            with open(file_path, "wb") as f:
                f.write(contents)

            orchestrator: GeoLocatorOrchestrator = app.state.orchestrator
            result = await orchestrator.locate(file_path, location_hint, quality=quality)

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
        request: Request,
        file: UploadFile = File(...),
        location_hint: str | None = Form(None),
        quality: str = Form("fast"),
    ):
        """SSE streaming version of locate.

        Streams progress events during pipeline execution:
        - step_start: A pipeline step has started
        - step_complete: A step finished successfully
        - step_error: A step failed
        - result: Final result
        """
        settings = app.state.settings
        
        # Rate limiting
        client_id = request.client.host if request.client else "unknown"
        rate_limiter: RateLimiter = app.state.rate_limiter
        allowed, remaining = rate_limiter.is_allowed(client_id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded.",
                headers={"Retry-After": "60"},
            )
        
        # Validate file
        is_valid, error_msg = validate_image_file(file, settings.api.max_upload_size_mb)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        is_valid, error_msg, contents = await validate_file_contents(
            file, settings.api.max_upload_size_mb
        )
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        upload_dir = settings.image_dir
        os.makedirs(upload_dir, exist_ok=True)

        ext = os.path.splitext(file.filename or "image.jpg")[1].lower() or ".jpg"
        if ext not in ALLOWED_EXTENSIONS:
            ext = ".jpg"
        file_id = str(uuid.uuid4())
        file_path = os.path.join(upload_dir, f"{file_id}{ext}")

        with open(file_path, "wb") as f:
            f.write(contents)

        async def event_generator() -> AsyncGenerator[dict, None]:
            try:
                orchestrator: GeoLocatorOrchestrator = app.state.orchestrator
                async for event in orchestrator.locate_stream(file_path, location_hint, quality=quality):
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
        quality: str = Form("fast"),
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
                session_mgr = getattr(app.state, "session_manager", None)
                async for event in orchestrator.locate_stream_v2(file_path, location_hint, quality=quality):
                    if event.get("event") == "result":
                        raw_data = event.get("data", {})

                        # Create a session with the full pipeline result
                        if session_mgr:
                            sid = raw_data.get("session_id")
                            session = session_mgr.create(
                                image_path=file_path,
                                pipeline_state={
                                    "ranked_candidates": raw_data.get("candidates", []),
                                    "prediction": raw_data,
                                    "evidences": raw_data.get("evidence_trail", []),
                                    "search_graph": raw_data.get("search_graph"),
                                },
                            )
                            # Override session_id to match the one created by session_manager
                            raw_data["session_id"] = session.id

                        event["data"] = _normalize_result_v2(raw_data)
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

        # Serialize search graph if present
        sg = session.pipeline_state.get("search_graph")
        search_graph_data = None
        if sg and hasattr(sg, "to_dict"):
            search_graph_data = sg.to_dict()
        elif isinstance(sg, dict):
            search_graph_data = sg

        return SessionResponse(
            session_id=session.id,
            candidates=session.pipeline_state.get("ranked_candidates", []),
            evidence_count=len(session.pipeline_state.get("evidences", [])),
            search_graph=search_graph_data,
            messages=session.messages,
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
    base["execution_policy"] = result.get("execution_policy", {})
    base["quality"] = result.get("quality", "balanced")
    base["fast_path_reason"] = result.get("fast_path_reason")
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
        "execution_policy": result.get("execution_policy", {}),
        "quality": result.get("quality", "balanced"),
        "fast_path_reason": result.get("fast_path_reason"),
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
