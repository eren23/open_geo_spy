"""Tests for candidate ranking and hard gates."""

from __future__ import annotations

import json

from src.improve.ranking import rank_experiment_dir


def test_rank_experiment_prefers_improving_candidate(tmp_path) -> None:
    experiment_dir = tmp_path / "experiment"
    baseline_dir = experiment_dir / "baseline"
    candidate_a_dir = experiment_dir / "candidates" / "candidate_01"
    candidate_b_dir = experiment_dir / "candidates" / "candidate_02"
    baseline_dir.mkdir(parents=True)
    candidate_a_dir.mkdir(parents=True)
    candidate_b_dir.mkdir(parents=True)

    baseline_manifest = {
        "overall_metrics": {
            "accuracy_25km": 0.40,
            "country_accuracy": 0.60,
            "median_gcd_km": 120.0,
            "mean_latency_ms": 1000.0,
            "mean_cost_usd": 0.010,
        },
        "datasets": [
            {
                "dataset_id": "regression",
                "protected": True,
                "metrics": {"accuracy_25km": 0.50},
            }
        ],
    }
    candidate_a_manifest = {
        "overall_metrics": {
            "accuracy_25km": 0.46,
            "country_accuracy": 0.65,
            "median_gcd_km": 90.0,
            "mean_latency_ms": 1100.0,
            "mean_cost_usd": 0.011,
        },
        "datasets": [
            {
                "dataset_id": "regression",
                "protected": True,
                "metrics": {"accuracy_25km": 0.50},
            }
        ],
        "sample_results_path": str(candidate_a_dir / "sample_results.jsonl"),
    }
    candidate_b_manifest = {
        "overall_metrics": {
            "accuracy_25km": 0.35,
            "country_accuracy": 0.58,
            "median_gcd_km": 170.0,
            "mean_latency_ms": 950.0,
            "mean_cost_usd": 0.009,
        },
        "datasets": [
            {
                "dataset_id": "regression",
                "protected": True,
                "metrics": {"accuracy_25km": 0.45},
            }
        ],
        "sample_results_path": str(candidate_b_dir / "sample_results.jsonl"),
    }

    (baseline_dir / "run_manifest.json").write_text(json.dumps(baseline_manifest))
    (candidate_a_dir / "run_manifest.json").write_text(json.dumps(candidate_a_manifest))
    (candidate_b_dir / "run_manifest.json").write_text(json.dumps(candidate_b_manifest))

    ranking = rank_experiment_dir(experiment_dir)

    assert ranking["winner"] == "candidate_01"
    assert ranking["rankings"][0]["status"] == "accepted"
    assert ranking["rankings"][1]["status"] == "rejected"


def test_rank_experiment_tolerates_missing_optional_dataset_metrics(tmp_path) -> None:
    experiment_dir = tmp_path / "experiment"
    baseline_dir = experiment_dir / "baseline"
    candidate_dir = experiment_dir / "candidates" / "candidate_01"
    baseline_dir.mkdir(parents=True)
    candidate_dir.mkdir(parents=True)

    manifest = {
        "overall_metrics": {
            "accuracy_25km": 1.0,
            "country_accuracy": 1.0,
            "median_gcd_km": 0.0,
            "mean_latency_ms": 1000.0,
            "mean_cost_usd": 0.0,
        },
        "datasets": [
            {
                "dataset_id": "seed",
                "protected": True,
                "metrics": {"accuracy_25km": 1.0},
            },
            {
                "dataset_id": "geovistabench_subset",
                "protected": True,
                "metrics": None,
            },
        ],
        "sample_results_path": str(candidate_dir / "sample_results.jsonl"),
    }

    (baseline_dir / "run_manifest.json").write_text(json.dumps(manifest))
    (candidate_dir / "run_manifest.json").write_text(json.dumps(manifest))

    ranking = rank_experiment_dir(experiment_dir)

    assert ranking["rankings"][0]["status"] in {"accepted", "rejected"}


def test_rank_experiment_rejects_non_positive_score_candidates(tmp_path) -> None:
    experiment_dir = tmp_path / "experiment"
    baseline_dir = experiment_dir / "baseline"
    candidate_dir = experiment_dir / "candidates" / "candidate_01"
    baseline_dir.mkdir(parents=True)
    candidate_dir.mkdir(parents=True)

    baseline_manifest = {
        "overall_metrics": {
            "accuracy_25km": 1.0,
            "country_accuracy": 1.0,
            "median_gcd_km": 0.0,
            "mean_latency_ms": 1000.0,
            "mean_cost_usd": 0.0,
        },
        "datasets": [
            {
                "dataset_id": "seed",
                "protected": True,
                "metrics": {"accuracy_25km": 1.0},
            }
        ],
    }
    candidate_manifest = {
        "overall_metrics": {
            "accuracy_25km": 1.0,
            "country_accuracy": 1.0,
            "median_gcd_km": 0.0,
            "mean_latency_ms": 6000.0,
            "mean_cost_usd": 0.0,
        },
        "datasets": [
            {
                "dataset_id": "seed",
                "protected": True,
                "metrics": {"accuracy_25km": 1.0},
            }
        ],
        "sample_results_path": str(candidate_dir / "sample_results.jsonl"),
    }

    (baseline_dir / "run_manifest.json").write_text(json.dumps(baseline_manifest))
    (candidate_dir / "run_manifest.json").write_text(json.dumps(candidate_manifest))

    ranking = rank_experiment_dir(experiment_dir)

    assert ranking["winner"] is None
    assert ranking["rankings"][0]["status"] == "rejected"
    assert "no net improvement after penalties" in ranking["rankings"][0]["reasons"]
