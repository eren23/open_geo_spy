"""Registry adapters wrapping existing model predictors.

Each adapter implements :class:`GeoModel`, delegates to the underlying
predictor, and is auto-registered via ``@ModelRegistry.register``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from loguru import logger
from openai import AsyncOpenAI

from src.config.settings import Settings
from src.evidence.chain import Evidence
from src.models.base import GeoModel, ModelCapability, ModelInfo
from src.models.registry import ModelRegistry


@ModelRegistry.register
class GeoCLIPAdapter(GeoModel):
    """Adapter for :class:`GeoCLIPPredictor`."""

    def __init__(self, settings: Settings | None = None, **kwargs):
        self._predictor = None
        self._device = settings.ml.device if settings else "cpu"

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="GeoCLIP",
            version="1.0",
            capabilities=[ModelCapability.COORDINATE_PREDICTION],
            description="CLIP + GPS continuous coordinate prediction (NeurIPS '23)",
            requires_gpu=False,
            is_local=True,
            default_weight=1.0,
        )

    def _ensure_loaded(self):
        if self._predictor is None:
            from src.models.geoclip_predictor import GeoCLIPPredictor
            self._predictor = GeoCLIPPredictor(device=self._device)

    async def predict(
        self, image_path: str, context: Optional[dict[str, Any]] = None
    ) -> list[dict]:
        self._ensure_loaded()
        return await asyncio.to_thread(self._predictor.predict, image_path, 5)

    def to_evidence(self, predictions: list[dict]) -> list[Evidence]:
        self._ensure_loaded()
        return self._predictor.to_evidence(predictions)


@ModelRegistry.register
class StreetCLIPAdapter(GeoModel):
    """Adapter for :class:`StreetCLIPPredictor`."""

    def __init__(self, settings: Settings | None = None, **kwargs):
        self._predictor = None
        self._device = settings.ml.device if settings else "cpu"

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="StreetCLIP",
            version="1.0",
            capabilities=[
                ModelCapability.COUNTRY_CLASSIFICATION,
                ModelCapability.VISUAL_SIMILARITY,
            ],
            description="Zero-shot country/city classification via CLIP",
            requires_gpu=False,
            is_local=True,
            default_weight=1.0,
        )

    def _ensure_loaded(self):
        if self._predictor is None:
            from src.models.streetclip_predictor import StreetCLIPPredictor
            self._predictor = StreetCLIPPredictor(device=self._device)

    async def predict(
        self, image_path: str, context: Optional[dict[str, Any]] = None
    ) -> list[dict]:
        self._ensure_loaded()
        return await asyncio.to_thread(self._predictor.predict_country, image_path, 5)

    def to_evidence(self, predictions: list[dict]) -> list[Evidence]:
        self._ensure_loaded()
        return self._predictor.to_evidence(predictions)

    @property
    def model_and_processor(self):
        """Expose StreetCLIP model/processor for sharing with verification agent."""
        if self._predictor and self._predictor.model:
            return (self._predictor.model, self._predictor.processor)
        return None


@ModelRegistry.register
class VLMGeoAdapter(GeoModel):
    """Adapter for VLM geo-reasoning (API-based, always available)."""

    def __init__(self, settings: Settings | None = None, **kwargs):
        self._settings = settings
        self._client: AsyncOpenAI | None = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="VLM Geo",
            version="1.0",
            capabilities=[
                ModelCapability.GEO_REASONING,
                ModelCapability.COORDINATE_PREDICTION,
                ModelCapability.COUNTRY_CLASSIFICATION,
            ],
            description="VLM chain-of-thought geolocation via Gemini",
            requires_gpu=False,
            is_local=False,
            default_weight=1.5,
        )

    def _ensure_client(self):
        if self._client is None and self._settings:
            self._client = AsyncOpenAI(
                base_url=self._settings.llm.base_url,
                api_key=self._settings.llm.api_key,
            )

    async def predict(
        self, image_path: str, context: Optional[dict[str, Any]] = None
    ) -> list[dict]:
        self._ensure_client()
        from src.models import vlm_geo

        additional_context = ""
        if context and context.get("evidence_text"):
            additional_context = context["evidence_text"]

        model = self._settings.llm.reasoning_model if self._settings else "google/gemini-2.5-pro"
        result = await vlm_geo.predict_location(
            image_path, self._client, model, additional_context
        )
        # Wrap single prediction dict into a list for consistency
        return [result] if result.get("country") else []

    def to_evidence(self, predictions: list[dict]) -> list[Evidence]:
        from src.models import vlm_geo

        evidences = []
        for pred in predictions:
            evidences.extend(vlm_geo.to_evidence(pred))
        return evidences
