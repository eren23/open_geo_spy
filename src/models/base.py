"""Abstract base class for pluggable geolocation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.evidence.chain import Evidence


class ModelCapability(str, Enum):
    """What a geo model can do."""

    COORDINATE_PREDICTION = "coordinate_prediction"
    COUNTRY_CLASSIFICATION = "country_classification"
    GEO_REASONING = "geo_reasoning"
    VISUAL_SIMILARITY = "visual_similarity"


@dataclass
class ModelInfo:
    """Metadata about a registered model."""

    name: str
    version: str
    capabilities: list[ModelCapability]
    description: str = ""
    requires_gpu: bool = False
    is_local: bool = True
    default_weight: float = 1.0
    extra: dict = field(default_factory=dict)


class GeoModel(ABC):
    """Abstract base for all geolocation models."""

    @abstractmethod
    def info(self) -> ModelInfo:
        """Return metadata about this model."""

    @abstractmethod
    async def predict(
        self,
        image_path: str,
        context: Optional[dict[str, Any]] = None,
    ) -> list[dict]:
        """Run prediction on an image.

        Args:
            image_path: Path to the image file.
            context: Optional context from prior pipeline steps
                     (features, OCR, evidence chain text, etc.).

        Returns:
            List of prediction dicts. Schema varies per capability but
            typically includes lat/lon/confidence or country/confidence.
        """

    @abstractmethod
    def to_evidence(self, predictions: list[dict]) -> list[Evidence]:
        """Convert raw predictions to typed Evidence objects."""
