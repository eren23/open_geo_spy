"""Batch processing manager: queue, worker pool, status tracking."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger


class BatchItemStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchItem:
    id: str
    filename: str
    image_path: str
    status: BatchItemStatus = BatchItemStatus.PENDING
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class BatchJob:
    id: str
    items: list[BatchItem] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def completed(self) -> int:
        return sum(1 for i in self.items if i.status == BatchItemStatus.COMPLETED)

    @property
    def failed(self) -> int:
        return sum(1 for i in self.items if i.status == BatchItemStatus.FAILED)

    @property
    def is_done(self) -> bool:
        return all(
            i.status in (BatchItemStatus.COMPLETED, BatchItemStatus.FAILED)
            for i in self.items
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "is_done": self.is_done,
            "items": [
                {
                    "id": i.id,
                    "filename": i.filename,
                    "status": i.status.value,
                    "result": i.result,
                    "error": i.error,
                }
                for i in self.items
            ],
        }


class BatchManager:
    """Manages batch geolocation jobs."""

    def __init__(self, max_concurrent: int = 3):
        self._jobs: dict[str, BatchJob] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def create_job(self, items: list[dict[str, str]]) -> BatchJob:
        """Create a new batch job.

        Args:
            items: List of {"filename": str, "image_path": str}
        """
        job = BatchJob(id=str(uuid.uuid4()))
        for item in items:
            job.items.append(
                BatchItem(
                    id=str(uuid.uuid4())[:8],
                    filename=item["filename"],
                    image_path=item["image_path"],
                )
            )
        self._jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        return self._jobs.get(job_id)

    async def process_job(self, job: BatchJob, orchestrator) -> None:
        """Process all items in a job concurrently (limited by semaphore)."""
        tasks = [
            self._process_item(item, orchestrator)
            for item in job.items
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_item(self, item: BatchItem, orchestrator) -> None:
        """Process a single batch item."""
        async with self._semaphore:
            item.status = BatchItemStatus.PROCESSING
            try:
                result = await orchestrator.locate(item.image_path)
                item.result = {
                    "name": result.get("name", "Unknown"),
                    "country": result.get("country"),
                    "region": result.get("region"),
                    "city": result.get("city"),
                    "latitude": result.get("lat") or result.get("latitude"),
                    "longitude": result.get("lon") or result.get("longitude"),
                    "confidence": result.get("confidence", 0),
                    "reasoning": result.get("reasoning", ""),
                }
                item.status = BatchItemStatus.COMPLETED
                logger.info("Batch item {} completed: {}", item.id, item.result.get("name"))
            except Exception as e:
                item.status = BatchItemStatus.FAILED
                item.error = str(e)
                logger.error("Batch item {} failed: {}", item.id, e)
