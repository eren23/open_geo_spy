"""Tests for LangGraph pipeline nodes."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.agents.state import PipelineState
from src.evidence.chain import Evidence, EvidenceSource


@pytest.mark.asyncio
async def test_early_exit_check_node_disabled():
    """Test early exit is skipped when disabled in settings."""
    from src.agents.nodes import early_exit_check_node
    
    state: PipelineState = {
        "image_path": "/tmp/test.jpg",
        "evidences": [],
        "early_exit": False,
        "quality": "balanced",
    }
    
    with patch("src.agents.nodes.get_settings") as mock_settings:
        settings = MagicMock()
        settings.pipeline.early_exit_enabled = False
        mock_settings.return_value = settings
        
        result = await early_exit_check_node(state, MagicMock())
        assert result["early_exit"] is False


@pytest.mark.asyncio
async def test_early_exit_check_node_low_evidence():
    """Test early exit requires at least 2 geo evidences."""
    from src.agents.nodes import early_exit_check_node
    
    state: PipelineState = {
        "image_path": "/tmp/test.jpg",
        "evidences": [
            Evidence(
                source=EvidenceSource.GEOCLIP,
                content="One prediction",
                confidence=0.8,
                latitude=48.85,
                longitude=2.35,
            )
        ],
        "quality": "balanced",
    }
    
    with patch("src.agents.nodes.get_settings") as mock_settings:
        settings = MagicMock()
        settings.pipeline.early_exit_enabled = True
        mock_settings.return_value = settings
        
        result = await early_exit_check_node(state, MagicMock())
        assert result["early_exit"] is False


@pytest.mark.asyncio
async def test_refinement_check_node_max_iterations():
    """Test refinement stops at max iterations."""
    from src.agents.nodes import refinement_check_node
    
    state: PipelineState = {
        "image_path": "/tmp/test.jpg",
        "evidences": [],
        "iteration": 3,
        "max_iterations": 2,
        "prediction": {"confidence": 0.5},
        "quality": "balanced",
        "started_at_monotonic": 0,
    }
    
    result = await refinement_check_node(state, MagicMock())
    assert result["should_refine"] is False


@pytest.mark.asyncio 
async def test_refinement_check_node_fast_quality():
    """Test refinement is skipped for fast quality."""
    from src.agents.nodes import refinement_check_node
    
    state: PipelineState = {
        "image_path": "/tmp/test.jpg",
        "evidences": [],
        "iteration": 0,
        "prediction": {"confidence": 0.3},
        "quality": "fast",
    }
    
    result = await refinement_check_node(state, MagicMock())
    assert result["should_refine"] is False


@pytest.mark.asyncio
async def test_refinement_check_node_early_exit():
    """Test refinement is skipped when early_exit is True."""
    from src.agents.nodes import refinement_check_node
    
    state: PipelineState = {
        "image_path": "/tmp/test.jpg",
        "evidences": [],
        "iteration": 0,
        "prediction": {"confidence": 0.3},
        "quality": "balanced",
        "early_exit": True,
    }
    
    result = await refinement_check_node(state, MagicMock())
    assert result["should_refine"] is False
