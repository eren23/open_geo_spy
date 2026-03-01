"""Runs the pipeline on an evaluation dataset, records traces, computes metrics."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from src.eval.dataset import EvalDataset, GroundTruthSample
from src.eval.metrics import EvalMetrics, SampleResult


class EvalRunner:
    """Runs the geolocation pipeline on a labeled dataset and collects metrics."""

    def __init__(
        self,
        label: str = "",
        max_concurrent: int = 3,
        output_dir: str = "data/eval/results",
    ):
        self.label = label
        self.max_concurrent = max_concurrent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        dataset: EvalDataset,
        scoring_config_path: str | None = None,
    ) -> EvalMetrics:
        """Run pipeline on all samples, return aggregated metrics."""
        logger.info("Starting eval: {} samples, label={}", len(dataset.samples), self.label)

        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: list[SampleResult] = []

        async def _process(sample: GroundTruthSample) -> SampleResult:
            async with semaphore:
                return await self._run_single(sample)

        tasks = [_process(s) for s in dataset.samples]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(completed):
            if isinstance(result, SampleResult):
                results.append(result)
            else:
                logger.error("Sample {} failed: {}", dataset.samples[i].image_path, result)
                results.append(SampleResult(
                    image_path=dataset.samples[i].image_path,
                    gt_lat=dataset.samples[i].latitude,
                    gt_lon=dataset.samples[i].longitude,
                    gt_country=dataset.samples[i].country,
                    gt_city=dataset.samples[i].city,
                    difficulty=dataset.samples[i].difficulty,
                    tags=dataset.samples[i].tags,
                ))

        metrics = EvalMetrics(results=results)
        logger.info(
            "Eval complete: {} samples, median GCD={:.1f}km, country acc={:.1%}",
            metrics.n,
            metrics.median_gcd_km,
            metrics.country_accuracy,
        )
        return metrics

    async def _run_single(self, sample: GroundTruthSample) -> SampleResult:
        """Run pipeline on a single sample."""
        from src.agents.orchestrator import Orchestrator
        from src.config.settings import get_settings

        settings = get_settings()
        orchestrator = Orchestrator(settings)

        start = time.monotonic()
        try:
            result = await orchestrator.locate(sample.image_path)
            duration_ms = round((time.monotonic() - start) * 1000, 1)

            prediction = result.get("prediction", result)
            return SampleResult(
                image_path=sample.image_path,
                pred_lat=prediction.get("lat") or prediction.get("latitude"),
                pred_lon=prediction.get("lon") or prediction.get("longitude"),
                pred_country=prediction.get("country", ""),
                pred_city=prediction.get("city", ""),
                pred_confidence=prediction.get("confidence", 0.0),
                gt_lat=sample.latitude,
                gt_lon=sample.longitude,
                gt_country=sample.country,
                gt_city=sample.city,
                difficulty=sample.difficulty,
                tags=sample.tags,
                latency_ms=duration_ms,
            )
        except Exception as e:
            logger.error("Pipeline failed for {}: {}", sample.image_path, e)
            return SampleResult(
                image_path=sample.image_path,
                gt_lat=sample.latitude,
                gt_lon=sample.longitude,
                gt_country=sample.country,
                gt_city=sample.city,
                difficulty=sample.difficulty,
                tags=sample.tags,
            )
