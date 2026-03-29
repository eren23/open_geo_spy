"""Rank candidate runs against a baseline using hard gates first."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.settings import get_settings


@dataclass
class RankingResult:
    """Outcome for a single candidate during ranking."""

    candidate_id: str
    status: str
    score: float
    reasons: list[str]
    metrics: dict[str, Any]
    judge_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "score": round(self.score, 4),
            "reasons": self.reasons,
            "metrics": self.metrics,
            "judge_summary": self.judge_summary,
        }


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _metric(metrics: dict[str, Any] | None, key: str) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    value = metrics.get(key)
    return float(value) if value is not None else 0.0


def _dataset_metrics(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        entry["dataset_id"]: entry.get("metrics") or {}
        for entry in manifest.get("datasets", [])
    }


def rank_candidate(
    baseline_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
) -> RankingResult:
    """Compare one candidate run to the baseline."""
    settings = get_settings().improvement
    baseline = baseline_manifest.get("overall_metrics", {})
    candidate = candidate_manifest.get("overall_metrics", {})
    reasons: list[str] = []

    delta_accuracy_25 = _metric(candidate, "accuracy_25km") - _metric(baseline, "accuracy_25km")
    delta_country = _metric(candidate, "country_accuracy") - _metric(baseline, "country_accuracy")
    delta_median = _metric(candidate, "median_gcd_km") - _metric(baseline, "median_gcd_km")
    delta_latency = _metric(candidate, "mean_latency_ms") - _metric(baseline, "mean_latency_ms")
    delta_cost = _metric(candidate, "mean_cost_usd") - _metric(baseline, "mean_cost_usd")

    status = "accepted"
    if delta_accuracy_25 < -settings.hard_regression_accuracy_25km:
        status = "rejected"
        reasons.append("regressed accuracy@25km")
    if delta_country < -settings.hard_regression_country_accuracy:
        status = "rejected"
        reasons.append("regressed country accuracy")
    if delta_median > settings.hard_regression_median_gcd_km:
        status = "rejected"
        reasons.append("regressed median GCD")

    baseline_by_dataset = _dataset_metrics(baseline_manifest)
    candidate_by_dataset = _dataset_metrics(candidate_manifest)
    for dataset in candidate_manifest.get("datasets", []):
        if not dataset.get("protected"):
            continue
        dataset_id = dataset["dataset_id"]
        baseline_metrics = baseline_by_dataset.get(dataset_id, {})
        candidate_metrics = candidate_by_dataset.get(dataset_id, {})
        if _metric(candidate_metrics, "accuracy_25km") + 1e-9 < _metric(baseline_metrics, "accuracy_25km"):
            status = "rejected"
            reasons.append(f"protected dataset regression: {dataset_id}")

    judge_summary = candidate_manifest.get("judge_summary")
    judge_bonus = 0.0
    if judge_summary:
        judge_bonus = float(judge_summary.get("mean_score", 0.0)) * 0.05

    score = (
        delta_accuracy_25 * 100.0
        + delta_country * 75.0
        - delta_median / 25.0
        - delta_latency * settings.soft_latency_penalty
        - delta_cost * settings.soft_cost_penalty
        + judge_bonus
    )

    if status == "accepted" and score <= 0:
        status = "rejected"
        reasons.append("no net improvement after penalties")
    elif status == "accepted":
        reasons.append("passed hard gates")

    return RankingResult(
        candidate_id=Path(candidate_manifest.get("run_dir", candidate_manifest.get("sample_results_path", ""))).name,
        status=status,
        score=score,
        reasons=reasons,
        metrics=candidate,
        judge_summary=judge_summary,
    )


def rank_experiment_dir(experiment_dir: str | Path) -> dict[str, Any]:
    """Rank all evaluated candidates in an experiment directory."""
    experiment_dir = Path(experiment_dir)
    baseline_manifest = _load_json(experiment_dir / "baseline" / "run_manifest.json")

    rankings: list[RankingResult] = []
    for candidate_manifest_path in sorted((experiment_dir / "candidates").glob("*/run_manifest.json")):
        manifest = _load_json(candidate_manifest_path)
        manifest["run_dir"] = str(candidate_manifest_path.parent)
        rankings.append(rank_candidate(baseline_manifest, manifest))

    rankings.sort(key=lambda item: (item.status != "accepted", -item.score, item.candidate_id))
    payload = {
        "baseline_metrics": baseline_manifest.get("overall_metrics", {}),
        "rankings": [item.to_dict() for item in rankings],
        "winner": next((item.candidate_id for item in rankings if item.status == "accepted"), None),
    }
    ranking_path = experiment_dir / "ranking.json"
    with open(ranking_path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload
