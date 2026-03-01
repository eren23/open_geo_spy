"""Bayesian-style weight adjustment from eval results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.eval.evolution import FailureAnalyzer, FailureCategory, FailureReport
from src.eval.metrics import EvalMetrics
from src.scoring.config import ScoringConfig


@dataclass
class WeightAdjustment:
    """A proposed change to a scoring config parameter."""

    path: str  # Dot-separated config path (e.g., "country_penalty.penalty_factor")
    old_value: float
    new_value: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "delta": round(self.new_value - self.old_value, 4),
            "reason": self.reason,
        }


class WeightTuner:
    """Proposes scoring weight adjustments based on failure analysis."""

    def __init__(
        self,
        config: ScoringConfig | None = None,
        step_size: float = 0.05,
    ):
        self.config = config or ScoringConfig()
        self.step_size = step_size
        self._adjustments: list[WeightAdjustment] = []

    def tune(self, metrics: EvalMetrics) -> list[WeightAdjustment]:
        """Analyze failures and propose weight adjustments."""
        analyzer = FailureAnalyzer()
        reports = analyzer.analyze(metrics)
        self._adjustments = []

        for report in reports:
            self._handle_category(report, metrics)

        return self._adjustments

    def _handle_category(self, report: FailureReport, metrics: EvalMetrics) -> None:
        """Generate adjustments for a failure category."""
        # Scale adjustment by severity (more failures = bigger adjustment)
        severity = min(report.count / max(metrics.n, 1), 0.5)  # Cap at 50%
        step = self.step_size * (1 + severity)

        if report.category == FailureCategory.WRONG_COUNTRY:
            self._propose(
                "country_penalty.penalty_factor",
                self.config.country_penalty.penalty_factor,
                min(0.95, self.config.country_penalty.penalty_factor + step),
                f"Wrong country in {report.count} samples — increase penalty",
            )

        elif report.category == FailureCategory.HIGH_CONF_WRONG:
            self._propose(
                "verification.contradicted_penalty",
                self.config.verification.contradicted_penalty,
                max(0.2, self.config.verification.contradicted_penalty - step),
                f"High-conf wrong in {report.count} samples — harsher contradiction penalty",
            )

        elif report.category == FailureCategory.OVERCONFIDENT:
            self._propose(
                "environment_blend.llm_weight",
                self.config.environment_blend.llm_weight,
                max(0.4, self.config.environment_blend.llm_weight - step),
                f"Overconfident in {report.count} samples — reduce LLM confidence weight",
            )
            self._propose(
                "environment_blend.env_weight",
                self.config.environment_blend.env_weight,
                min(0.6, self.config.environment_blend.env_weight + step),
                f"Overconfident in {report.count} samples — increase evidence weight",
            )

        elif report.category == FailureCategory.UNDERCONFIDENT:
            self._propose(
                "verification.supported_boost",
                self.config.verification.supported_boost,
                min(1.3, self.config.verification.supported_boost + step),
                f"Underconfident in {report.count} samples — boost support multiplier",
            )

        elif report.category == FailureCategory.CITY_MISS:
            self._propose(
                "candidate_ranking.evidence_count",
                self.config.candidate_ranking.evidence_count,
                min(0.35, self.config.candidate_ranking.evidence_count + step),
                f"City miss in {report.count} samples — increase evidence count weight",
            )

        elif report.category == FailureCategory.CONTINENT_WRONG:
            self._propose(
                "country_penalty.consensus_threshold",
                self.config.country_penalty.consensus_threshold,
                max(0.3, self.config.country_penalty.consensus_threshold - step),
                f"Wrong continent in {report.count} samples — lower consensus threshold",
            )

    def _propose(self, path: str, old: float, new: float, reason: str) -> None:
        """Add an adjustment if the value actually changed."""
        if abs(new - old) < 0.001:
            return
        self._adjustments.append(WeightAdjustment(
            path=path, old_value=old, new_value=round(new, 4), reason=reason,
        ))

    def apply(self, adjustments: list[WeightAdjustment] | None = None) -> ScoringConfig:
        """Apply adjustments to produce an updated ScoringConfig."""
        adjustments = adjustments or self._adjustments
        config_dict = self.config.model_dump()

        for adj in adjustments:
            parts = adj.path.split(".")
            target = config_dict
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = adj.new_value

        return ScoringConfig.model_validate(config_dict)

    def save_evolution(
        self,
        new_config: ScoringConfig,
        adjustments: list[WeightAdjustment],
        output_dir: str = "data/evolution",
    ) -> Path:
        """Save tuned config with history."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save current weights
        weights_path = out / "weights.json"
        new_config.to_file(weights_path)

        # Save versioned history
        history_dir = out / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        history_path = history_dir / f"weights_{date_str}.json"
        history_data = {
            "config": new_config.model_dump(),
            "adjustments": [a.to_dict() for a in adjustments],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)

        logger.info("Evolution saved: {} ({} adjustments)", weights_path, len(adjustments))
        return weights_path
