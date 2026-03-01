"""Failure analysis — categorizes prediction failures for auto-evolution."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.eval.metrics import EvalMetrics, SampleResult


class FailureCategory:
    WRONG_COUNTRY = "wrong_country"
    HIGH_CONF_WRONG = "high_confidence_wrong"
    NO_WEB_EVIDENCE = "no_web_evidence"
    MODEL_DISAGREEMENT = "model_disagreement"
    LOW_EVIDENCE = "low_evidence"
    COUNTRY_CONFUSION = "country_confusion"
    CITY_MISS = "city_miss"
    CONTINENT_WRONG = "continent_wrong"
    OVERCONFIDENT = "overconfident"
    UNDERCONFIDENT = "underconfident"


@dataclass
class FailureReport:
    """Categorized failures with counts and example samples."""

    category: str
    count: int = 0
    examples: list[dict] = field(default_factory=list)
    suggestion: str = ""

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "count": self.count,
            "examples": self.examples[:3],
            "suggestion": self.suggestion,
        }


class FailureAnalyzer:
    """Categorizes evaluation failures to guide weight tuning."""

    def __init__(
        self,
        high_confidence_threshold: float = 0.7,
        low_evidence_threshold: int = 5,
    ):
        self.high_confidence_threshold = high_confidence_threshold
        self.low_evidence_threshold = low_evidence_threshold

    def analyze(self, metrics: EvalMetrics) -> list[FailureReport]:
        """Analyze failures and return categorized reports."""
        categories: dict[str, FailureReport] = {}

        for result in metrics.results:
            if result.gcd_km is None:
                continue

            failures = self._categorize_sample(result)
            for cat, suggestion in failures:
                if cat not in categories:
                    categories[cat] = FailureReport(
                        category=cat, suggestion=suggestion,
                    )
                report = categories[cat]
                report.count += 1
                if len(report.examples) < 5:
                    report.examples.append({
                        "image": result.image_path,
                        "gcd_km": round(result.gcd_km, 1),
                        "pred_country": result.pred_country,
                        "gt_country": result.gt_country,
                        "confidence": result.pred_confidence,
                    })

        return sorted(categories.values(), key=lambda r: r.count, reverse=True)

    def _categorize_sample(self, r: SampleResult) -> list[tuple[str, str]]:
        """Assign failure categories to a single sample."""
        cats = []
        gcd = r.gcd_km
        if gcd is None:
            return cats

        # Wrong country
        if not r.country_correct and r.gt_country:
            cats.append((
                FailureCategory.WRONG_COUNTRY,
                "Increase country_penalty.penalty_factor or lower consensus_threshold",
            ))

        # High confidence but wrong (>150km off)
        if r.pred_confidence >= self.high_confidence_threshold and gcd > 150:
            cats.append((
                FailureCategory.HIGH_CONF_WRONG,
                "Strengthen verification adjustments to penalize high-conf wrong predictions",
            ))

        # Overconfident (confidence > 0.8 but > 50km off)
        if r.pred_confidence > 0.8 and gcd > 50:
            cats.append((
                FailureCategory.OVERCONFIDENT,
                "Lower environment_blend.llm_weight to reduce LLM confidence influence",
            ))

        # Underconfident (confidence < 0.3 but within 25km)
        if r.pred_confidence < 0.3 and gcd < 25:
            cats.append((
                FailureCategory.UNDERCONFIDENT,
                "Increase verification.supported_boost or reduce verification penalties",
            ))

        # Wrong continent (>3000km off)
        if gcd > 3000:
            cats.append((
                FailureCategory.CONTINENT_WRONG,
                "Evidence chain is fundamentally misled; check source_confidence values",
            ))

        # City miss (right country, wrong city, >50km)
        if r.country_correct and not r.city_correct and gcd > 50 and r.gt_city:
            cats.append((
                FailureCategory.CITY_MISS,
                "Improve geo_agreement scoring or candidate_ranking weights",
            ))

        return cats

    def summary(self, reports: list[FailureReport]) -> dict[str, Any]:
        """Summarize failure analysis."""
        return {
            "total_categories": len(reports),
            "categories": [r.to_dict() for r in reports],
            "top_issue": reports[0].category if reports else None,
            "top_suggestion": reports[0].suggestion if reports else None,
        }
