"""Integration tests for the orchestrator (with mocked agents)."""

from unittest.mock import AsyncMock, patch

import pytest

from src.evidence.chain import Evidence, EvidenceChain, EvidenceSource


@pytest.mark.asyncio
async def test_orchestrator_locate(test_settings, mock_image_path):
    """Test that orchestrator runs through the full pipeline with mocked agents."""
    # Create mock evidence chains for each agent
    feature_chain = EvidenceChain()
    feature_chain.add(Evidence(
        source=EvidenceSource.VLM_ANALYSIS,
        content="Tower visible",
        confidence=0.8,
        country="France",
    ))

    ml_chain = EvidenceChain()
    ml_chain.add(Evidence(
        source=EvidenceSource.GEOCLIP,
        content="GeoCLIP pred",
        confidence=0.7,
        latitude=48.85,
        longitude=2.35,
    ))

    web_chain = EvidenceChain()
    web_chain.add(Evidence(
        source=EvidenceSource.SERPER,
        content="Eiffel Tower",
        confidence=0.6,
        latitude=48.8584,
        longitude=2.2945,
        country="France",
    ))

    verify_chain = EvidenceChain()

    prediction = {
        "name": "Eiffel Tower, Paris",
        "country": "France",
        "region": "Île-de-France",
        "city": "Paris",
        "lat": 48.8584,
        "lon": 2.2945,
        "confidence": 0.85,
        "reasoning": "Tower visible",
        "verified": True,
        "evidence_trail": [],
        "evidence_summary": {},
    }

    with (
        patch("src.agents.nodes.FeatureExtractionAgent") as MockFE,
        patch("src.agents.nodes.MLEnsembleAgent") as MockML,
        patch("src.agents.nodes.WebIntelAgent") as MockWI,
        patch("src.agents.nodes.CandidateVerificationAgent") as MockCV,
        patch("src.agents.nodes.ReasoningAgent") as MockRA,
    ):
        MockFE.return_value.extract_with_raw = AsyncMock(
            return_value=(feature_chain, {}, {"landmarks": ["tower"]}, {})
        )
        MockML.return_value.predict = AsyncMock(return_value=ml_chain)
        MockWI.return_value.search = AsyncMock(return_value=web_chain)
        MockWI.return_value.close = AsyncMock()
        MockCV.return_value.verify_candidates = AsyncMock(return_value=verify_chain)
        MockCV.return_value.close = AsyncMock()
        MockRA.return_value.reason = AsyncMock(return_value=prediction)

        from src.agents.orchestrator import GeoLocatorOrchestrator
        orch = GeoLocatorOrchestrator(test_settings)
        result = await orch.locate(mock_image_path, location_hint="Paris")

        assert result["name"] == "Eiffel Tower, Paris"
        assert result["confidence"] == 0.85
        assert "elapsed_ms" in result
