"""Tests for the FastAPI application endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked orchestrator."""
    with patch("src.api.app.GeoLocatorOrchestrator") as MockOrch:
        mock_orch = AsyncMock()
        mock_orch.locate = AsyncMock(return_value={
            "name": "Eiffel Tower, Paris",
            "country": "France",
            "region": "Île-de-France",
            "city": "Paris",
            "lat": 48.8584,
            "lon": 2.2945,
            "confidence": 0.85,
            "reasoning": "Tower visible in image",
            "verified": True,
            "evidence_trail": [],
            "evidence_summary": {},
        })
        mock_orch.close = AsyncMock()
        MockOrch.return_value = mock_orch

        from src.api.app import create_app
        app = create_app()
        app.state.orchestrator = mock_orch
        yield TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "services" in data


class TestLocateEndpoint:
    def test_locate_requires_file(self, client):
        resp = client.post("/api/locate")
        assert resp.status_code == 422  # Validation error

    def test_locate_with_image(self, client, mock_image_path):
        with open(mock_image_path, "rb") as f:
            resp = client.post(
                "/api/locate",
                files={"file": ("test.jpg", f, "image/jpeg")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Eiffel Tower, Paris"
        assert data["country"] == "France"
        assert data["confidence"] == 0.85
