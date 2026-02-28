"""Calibration benchmark: run pipeline on labeled dataset.

Collects (raw_confidence, actual_distance_km) pairs for calibration fitting.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field

from loguru import logger

from src.calibration.calibrator import ConfidenceCalibrator
from src.utils.geo_math import haversine_distance


@dataclass
class BenchmarkResult:
    image_path: str
    predicted_lat: float
    predicted_lon: float
    actual_lat: float
    actual_lon: float
    raw_confidence: float
    distance_km: float
    correct: bool  # within threshold

    def to_dict(self) -> dict:
        return {
            "image": self.image_path,
            "predicted": [self.predicted_lat, self.predicted_lon],
            "actual": [self.actual_lat, self.actual_lon],
            "raw_confidence": self.raw_confidence,
            "distance_km": self.distance_km,
            "correct": self.correct,
        }


class CalibrationBenchmark:
    """Run the pipeline on labeled images and collect calibration data."""

    def __init__(
        self,
        correct_threshold_km: float = 50.0,
    ):
        self.threshold = correct_threshold_km
        self.results: list[BenchmarkResult] = []

    async def run(
        self,
        labeled_images: list[dict],
        orchestrator=None,
    ) -> list[BenchmarkResult]:
        """Run pipeline on labeled images.

        Args:
            labeled_images: List of {"image_path": str, "lat": float, "lon": float}
            orchestrator: GeoLocatorOrchestrator instance
        """
        if orchestrator is None:
            from src.agents.orchestrator import GeoLocatorOrchestrator
            orchestrator = GeoLocatorOrchestrator()

        for item in labeled_images:
            image_path = item["image_path"]
            actual_lat = item["lat"]
            actual_lon = item["lon"]

            if not os.path.exists(image_path):
                logger.warning("Image not found: {}", image_path)
                continue

            try:
                result = await orchestrator.locate(image_path)
                pred_lat = result.get("lat") or result.get("latitude") or 0
                pred_lon = result.get("lon") or result.get("longitude") or 0
                raw_conf = result.get("confidence", 0)

                dist = haversine_distance(actual_lat, actual_lon, pred_lat, pred_lon)
                correct = dist < self.threshold

                br = BenchmarkResult(
                    image_path=image_path,
                    predicted_lat=pred_lat,
                    predicted_lon=pred_lon,
                    actual_lat=actual_lat,
                    actual_lon=actual_lon,
                    raw_confidence=raw_conf,
                    distance_km=dist,
                    correct=correct,
                )
                self.results.append(br)

                logger.info(
                    "Benchmark {}: dist={:.0f}km, conf={:.2f}, correct={}",
                    os.path.basename(image_path), dist, raw_conf, correct,
                )

            except Exception as e:
                logger.error("Benchmark failed for {}: {}", image_path, e)

        return self.results

    def fit_calibrator(self, output_path: str | None = None) -> ConfidenceCalibrator:
        """Fit a calibrator from benchmark results."""
        if not self.results:
            raise ValueError("No benchmark results to fit on")

        raw_scores = [r.raw_confidence for r in self.results]
        actual_correct = [r.correct for r in self.results]

        calibrator = ConfidenceCalibrator(data_path=output_path)
        calibrator.fit(raw_scores, actual_correct)

        return calibrator

    def save_results(self, path: str) -> None:
        """Save benchmark results to JSON."""
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

    def summary(self) -> dict:
        """Summary statistics."""
        if not self.results:
            return {}

        correct = sum(1 for r in self.results if r.correct)
        total = len(self.results)
        distances = [r.distance_km for r in self.results]

        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
            "median_distance_km": sorted(distances)[total // 2],
            "mean_distance_km": sum(distances) / total,
            "mean_confidence": sum(r.raw_confidence for r in self.results) / total,
        }
