"""Standard geolocation evaluation metrics.

Aligned with SOTA benchmarks: accuracy@{1,25,50,150,750}km, GCD error,
country/city accuracy, ECE, cost stats.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from src.utils.geo_math import haversine_distance


@dataclass
class SampleResult:
    """Result for a single evaluation sample."""

    image_path: str
    pred_lat: float | None = None
    pred_lon: float | None = None
    pred_country: str = ""
    pred_city: str = ""
    pred_confidence: float = 0.0

    gt_lat: float = 0.0
    gt_lon: float = 0.0
    gt_country: str = ""
    gt_city: str = ""
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)

    cost_usd: float = 0.0
    latency_ms: float = 0.0
    tokens: int = 0

    @property
    def gcd_km(self) -> float | None:
        """Great Circle Distance error in km."""
        if self.pred_lat is None or self.pred_lon is None:
            return None
        return haversine_distance(self.pred_lat, self.pred_lon, self.gt_lat, self.gt_lon)

    @property
    def country_correct(self) -> bool:
        return self.pred_country.lower() == self.gt_country.lower() if self.pred_country and self.gt_country else False

    @property
    def city_correct(self) -> bool:
        return self.pred_city.lower() == self.gt_city.lower() if self.pred_city and self.gt_city else False


@dataclass
class EvalMetrics:
    """Aggregated metrics from an evaluation run."""

    results: list[SampleResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def gcd_errors(self) -> list[float]:
        return sorted(r.gcd_km for r in self.results if r.gcd_km is not None)

    def accuracy_at(self, threshold_km: float) -> float:
        """Fraction of predictions within threshold_km of ground truth."""
        errors = self.gcd_errors
        if not errors:
            return 0.0
        return sum(1 for e in errors if e < threshold_km) / len(errors)

    @property
    def accuracy_1km(self) -> float:
        return self.accuracy_at(1)

    @property
    def accuracy_25km(self) -> float:
        return self.accuracy_at(25)

    @property
    def accuracy_50km(self) -> float:
        return self.accuracy_at(50)

    @property
    def accuracy_150km(self) -> float:
        return self.accuracy_at(150)

    @property
    def accuracy_750km(self) -> float:
        return self.accuracy_at(750)

    @property
    def median_gcd_km(self) -> float:
        errors = self.gcd_errors
        if not errors:
            return float("inf")
        return statistics.median(errors)

    @property
    def mean_gcd_km(self) -> float:
        errors = self.gcd_errors
        if not errors:
            return float("inf")
        return sum(errors) / len(errors)

    @property
    def p90_gcd_km(self) -> float:
        errors = self.gcd_errors
        if not errors:
            return float("inf")
        return errors[int((len(errors) - 1) * 0.9)]

    @property
    def country_accuracy(self) -> float:
        valid = [r for r in self.results if r.gt_country]
        if not valid:
            return 0.0
        return sum(1 for r in valid if r.country_correct) / len(valid)

    @property
    def city_accuracy(self) -> float:
        valid = [r for r in self.results if r.gt_city]
        if not valid:
            return 0.0
        return sum(1 for r in valid if r.city_correct) / len(valid)

    def ece(self, n_bins: int = 10) -> float:
        """Expected Calibration Error — confidence vs actual accuracy."""
        valid = [r for r in self.results if r.gcd_km is not None]
        if not valid:
            return 0.0

        bins: list[list[SampleResult]] = [[] for _ in range(n_bins)]
        for r in valid:
            bin_idx = min(int(r.pred_confidence * n_bins), n_bins - 1)
            bins[bin_idx].append(r)

        ece_sum = 0.0
        for i, bin_results in enumerate(bins):
            if not bin_results:
                continue
            avg_conf = sum(r.pred_confidence for r in bin_results) / len(bin_results)
            # "Correct" if within 50km (city-level)
            accuracy = sum(1 for r in bin_results if (r.gcd_km or float("inf")) < 50) / len(bin_results)
            ece_sum += len(bin_results) * abs(accuracy - avg_conf)

        return ece_sum / len(valid) if valid else 0.0

    @property
    def mean_cost_usd(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.cost_usd for r in self.results) / len(self.results)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def mean_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.results)

    def by_difficulty(self) -> dict[str, EvalMetrics]:
        """Break down metrics by difficulty level."""
        groups: dict[str, list[SampleResult]] = {}
        for r in self.results:
            groups.setdefault(r.difficulty, []).append(r)
        return {d: EvalMetrics(results=rs) for d, rs in groups.items()}

    def by_tag(self, tag: str) -> EvalMetrics:
        """Filter metrics to samples with a given tag."""
        return EvalMetrics(results=[r for r in self.results if tag in r.tags])

    def summary(self) -> dict[str, Any]:
        return {
            "count": self.n,
            "accuracy_1km": round(self.accuracy_1km, 4),
            "accuracy_25km": round(self.accuracy_25km, 4),
            "accuracy_50km": round(self.accuracy_50km, 4),
            "accuracy_150km": round(self.accuracy_150km, 4),
            "accuracy_750km": round(self.accuracy_750km, 4),
            "median_gcd_km": round(self.median_gcd_km, 2),
            "mean_gcd_km": round(self.mean_gcd_km, 2),
            "p90_gcd_km": round(self.p90_gcd_km, 2),
            "country_accuracy": round(self.country_accuracy, 4),
            "city_accuracy": round(self.city_accuracy, 4),
            "ece": round(self.ece(), 4),
            "mean_cost_usd": round(self.mean_cost_usd, 5),
            "total_cost_usd": round(self.total_cost_usd, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "total_tokens": self.total_tokens,
        }
