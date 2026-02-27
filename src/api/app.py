"""FastAPI application with SSE streaming for real-time progress.

Replaces the old app.py with:
- Pydantic schemas for request/response validation
- SSE streaming for pipeline progress
- Environment-based CORS
- Structured logging
- Health check with dependency status
"""

from __future__ import annotations

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
from src.api.schemas import HealthResponse, LocateResponse
from src.config.settings import get_settings
from src.utils.logging import setup_logger


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    settings = get_settings()
    setup_logger("DEBUG" if settings.debug else "INFO")
    logger.info("Starting OpenGeoSpy API (env={})", settings.environment.value)

    # Initialize orchestrator
    app.state.orchestrator = GeoLocatorOrchestrator(settings)
    app.state.settings = settings

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
