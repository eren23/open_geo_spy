"""Tests for the model registry."""

import pytest

from src.models.base import GeoModel, ModelCapability, ModelInfo
from src.models.registry import ModelRegistry


class DummyModel(GeoModel):
    """Test model for registry tests."""

    def __init__(self, **kwargs):
        pass

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="DummyModel",
            version="0.1",
            capabilities=[ModelCapability.COORDINATE_PREDICTION],
        )

    async def predict(self, image_path, context=None):
        return [{"lat": 48.85, "lon": 2.35, "confidence": 0.9}]

    def to_evidence(self, predictions):
        from src.evidence.chain import Evidence, EvidenceSource
        return [
            Evidence(
                source=EvidenceSource.GEOCLIP,
                content="dummy",
                confidence=p["confidence"],
                latitude=p["lat"],
                longitude=p["lon"],
            )
            for p in predictions
        ]


class TestModelRegistry:
    def setup_method(self):
        ModelRegistry.clear()

    def test_register(self):
        ModelRegistry.register(DummyModel)
        assert "DummyModel" in ModelRegistry.get_all()

    def test_get_all_returns_class(self):
        ModelRegistry.register(DummyModel)
        models = ModelRegistry.get_all()
        assert models["DummyModel"] is DummyModel

    def test_get_by_capability(self, test_settings):
        ModelRegistry.register(DummyModel)
        # Need to instantiate first
        ModelRegistry._instances["DummyModel"] = DummyModel()
        matches = ModelRegistry.get_by_capability(ModelCapability.COORDINATE_PREDICTION)
        assert len(matches) == 1
        assert isinstance(matches[0], DummyModel)

    def test_get_by_capability_no_match(self):
        ModelRegistry.register(DummyModel)
        ModelRegistry._instances["DummyModel"] = DummyModel()
        matches = ModelRegistry.get_by_capability(ModelCapability.VISUAL_SIMILARITY)
        assert len(matches) == 0

    def test_clear(self):
        ModelRegistry.register(DummyModel)
        ModelRegistry.clear()
        assert len(ModelRegistry.get_all()) == 0
