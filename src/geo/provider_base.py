"""Abstract search provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.evidence.chain import Evidence


class SearchProvider(ABC):
    """Interface for all search providers (Serper, Brave, SearXNG, etc.)."""

    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Run a web search."""

    @abstractmethod
    async def search_images(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        """Run an image search."""

    @abstractmethod
    def results_to_evidence(self, results: list[dict], query: str) -> list[Evidence]:
        """Convert raw results to Evidence objects."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
