"""Shared test fixtures."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock

import pytest

from src.config.settings import Settings
from src.evidence.chain import Evidence, EvidenceChain, EvidenceSource


@pytest.fixture
def test_settings() -> Settings:
    """Settings instance with dummy keys for testing."""
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key-123")
    os.environ.setdefault("SERPER_API_KEY", "test-serper-key")
    return Settings(
        debug=True,
        image_dir=tempfile.mkdtemp(),
    )


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Mock OpenAI AsyncClient that returns a canned geo prediction."""
    client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.choices = [
        AsyncMock(
            message=AsyncMock(
                content='{"country":"France","region":"Île-de-France","city":"Paris",'
                '"latitude":48.8566,"longitude":2.3522,"confidence":0.85,'
                '"reasoning":"Eiffel Tower visible","alternative_countries":["Belgium"]}'
            )
        )
    ]
    client.chat.completions.create = AsyncMock(return_value=mock_response)
    return client


@pytest.fixture
def sample_evidence() -> list[Evidence]:
    """A handful of sample Evidence objects covering different sources."""
    return [
        Evidence(
            source=EvidenceSource.GEOCLIP,
            content="GeoCLIP prediction #1",
            confidence=0.7,
            latitude=48.8566,
            longitude=2.3522,
            country="France",
        ),
        Evidence(
            source=EvidenceSource.STREETCLIP,
            content="StreetCLIP country: France",
            confidence=0.8,
            country="France",
        ),
        Evidence(
            source=EvidenceSource.VLM_GEO,
            content="VLM geo reasoning: Eiffel Tower visible",
            confidence=0.85,
            latitude=48.8570,
            longitude=2.3510,
            country="France",
            city="Paris",
        ),
        Evidence(
            source=EvidenceSource.SERPER,
            content="Search result: Eiffel Tower Paris",
            confidence=0.6,
            latitude=48.8584,
            longitude=2.2945,
            country="France",
        ),
        Evidence(
            source=EvidenceSource.OCR,
            content="OCR text: TOUR EIFFEL",
            confidence=0.9,
        ),
    ]


@pytest.fixture
def sample_chain(sample_evidence) -> EvidenceChain:
    """EvidenceChain populated with sample evidence."""
    chain = EvidenceChain()
    chain.add_many(sample_evidence)
    return chain


@pytest.fixture
def mock_image_path(tmp_path) -> str:
    """Create a minimal valid JPEG for tests that need a file path."""
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    path = str(tmp_path / "test_image.jpg")
    img.save(path, "JPEG")
    return path
