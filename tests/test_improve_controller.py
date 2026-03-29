"""Tests for the improvement controller mutation flow."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.eval.metrics import SampleResult
from src.improve.controller import (
    SCORING_CONFIG_TARGET,
    CandidateContext,
    ImprovementController,
    ProposalFeedback,
    ProposalOutcome,
)


def test_validate_candidate_rewrite_statuses(tmp_path, test_settings) -> None:
    test_settings.improvement.output_dir = str(tmp_path / "improve")
    test_settings.improvement.worktree_dir = str(tmp_path / "worktrees")
    controller = ImprovementController(test_settings)

    worktree_dir = tmp_path / "worktree"
    target_path = worktree_dir / "src" / "module.py"
    target_path.parent.mkdir(parents=True)
    target_path.write_text("VALUE = 1\n")

    accepted = controller._validate_candidate_rewrite(
        worktree_dir,
        "src/module.py",
        {
            "summary": "raise the constant",
            "path": "src/module.py",
            "new_content": "VALUE = 2\n",
        },
    )
    skipped = controller._validate_candidate_rewrite(
        worktree_dir,
        "src/module.py",
        {
            "summary": "no safe edit",
            "path": "src/module.py",
            "new_content": "",
        },
    )
    rejected = controller._validate_candidate_rewrite(
        worktree_dir,
        "src/module.py",
        {
            "summary": "broken rewrite",
            "path": "src/module.py",
            "new_content": "def broken(:\n",
        },
    )

    assert accepted["status"] == "accepted"
    assert skipped["status"] == "skipped"
    assert rejected["status"] == "rejected"


def test_validate_scoring_config_override(tmp_path, test_settings) -> None:
    test_settings.improvement.output_dir = str(tmp_path / "improve")
    test_settings.improvement.worktree_dir = str(tmp_path / "worktrees")
    controller = ImprovementController(test_settings)

    accepted = controller._validate_candidate_rewrite(
        tmp_path,
        SCORING_CONFIG_TARGET,
        {
            "summary": "reduce license plate weight",
            "overrides": {
                "source_confidence": {
                    "license_plate": 0.6,
                }
            },
        },
    )

    assert accepted["status"] == "accepted"
    assert "\"license_plate\": 0.6" in accepted["new_content"]


def test_generate_candidate_rewrite_prefers_heuristic_scoring_override(tmp_path, test_settings) -> None:
    controller = ImprovementController(test_settings)

    class FailingCompletions:
        async def create(self, *args, **kwargs):
            raise AssertionError("LLM mutator should not be called for the heuristic scoring candidate")

    controller._client.chat.completions = FailingCompletions()
    baseline_manifest = {
        "overall_metrics": {
            "city_accuracy": 0.75,
            "country_accuracy": 1.0,
            "mean_latency_ms": 40000.0,
            "ece": 0.7,
        }
    }
    candidate_context = CandidateContext(
        failure_summary=[{"category": "city_miss"}],
        worst_samples=[],
        target_files=[SCORING_CONFIG_TARGET],
    )

    prompt, raw = asyncio.run(
        controller._generate_candidate_rewrite(
            tmp_path,
            baseline_manifest,
            "candidate_01",
            {"path": SCORING_CONFIG_TARGET},
            candidate_context,
        )
    )
    payload = json.loads(raw)

    assert prompt == "heuristic_scoring_candidate"
    assert payload["overrides"]["source_confidence"]["license_plate"] == 0.4


def test_run_marks_invalid_rewrite_as_rejected(tmp_path, monkeypatch, test_settings) -> None:
    test_settings.improvement.output_dir = str(tmp_path / "improve")
    test_settings.improvement.worktree_dir = str(tmp_path / "worktrees")
    test_settings.improvement.candidate_count = 1
    controller = ImprovementController(test_settings)

    sample_results_path = tmp_path / "baseline_sample_results.jsonl"
    sample_results_path.write_text(
        json.dumps(
            SampleResult(
                image_path="images/example.jpg",
                pred_lat=0.0,
                pred_lon=0.0,
                pred_country="France",
                pred_city="Paris",
                gt_lat=48.8584,
                gt_lon=2.2945,
                gt_country="France",
                gt_city="Paris",
            ).to_dict()
        )
        + "\n"
    )

    baseline_manifest = {
        "sample_results_path": str(sample_results_path),
        "overall_metrics": {"accuracy_25km": 0.25, "country_accuracy": 1.0, "median_gcd_km": 100.0},
    }

    def fake_run_suite_sync(suite_path, output_dir, **kwargs):
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_manifest.json").write_text(json.dumps(baseline_manifest))
        return baseline_manifest

    worktree_dir = tmp_path / "candidate_worktree"
    target_file = worktree_dir / "src" / "module.py"
    target_file.parent.mkdir(parents=True)
    target_file.write_text("VALUE = 1\n")

    async def fake_generate(*args, **kwargs):
        return (
            "prompt",
            json.dumps(
                {
                    "summary": "introduce a syntax error",
                    "path": "src/module.py",
                    "new_content": "def broken(:\n",
                }
            ),
        )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("candidate evaluation should not run for rejected rewrites")

    monkeypatch.setattr("src.improve.controller.run_suite_sync", fake_run_suite_sync)
    monkeypatch.setattr(controller, "_build_candidate_context", lambda *args, **kwargs: CandidateContext([], [], ["src/module.py"]))
    monkeypatch.setattr(controller, "_create_worktree", lambda planned, candidate_id: worktree_dir)
    monkeypatch.setattr(controller, "_generate_candidate_rewrite", fake_generate)
    monkeypatch.setattr(controller, "_run_suite_subprocess", fail_if_called)

    experiment_dir = controller.run(tmp_path / "suite.json", candidate_count=1)
    experiment = json.loads((experiment_dir / "experiment.json").read_text())
    candidate = experiment["candidates"][0]
    validation = json.loads((experiment_dir / "candidates" / "candidate_01" / "candidate_validation.json").read_text())

    assert candidate["status"] == "rejected"
    assert "syntax error" in candidate["error"].lower()
    assert validation["status"] == "rejected"


def test_run_passes_scoring_config_override_to_worker(tmp_path, monkeypatch, test_settings) -> None:
    test_settings.improvement.output_dir = str(tmp_path / "improve")
    test_settings.improvement.worktree_dir = str(tmp_path / "worktrees")
    test_settings.improvement.candidate_count = 1
    controller = ImprovementController(test_settings)

    sample_results_path = tmp_path / "baseline_sample_results.jsonl"
    sample_results_path.write_text(
        json.dumps(
            SampleResult(
                image_path="images/example.jpg",
                pred_lat=48.8584,
                pred_lon=2.2945,
                pred_country="France",
                pred_city="Paris",
                gt_lat=48.8584,
                gt_lon=2.2945,
                gt_country="France",
                gt_city="Paris",
            ).to_dict()
        )
        + "\n"
    )
    baseline_manifest = {
        "sample_results_path": str(sample_results_path),
        "overall_metrics": {"accuracy_25km": 0.75, "country_accuracy": 1.0, "median_gcd_km": 10.0},
    }

    base_config_path = tmp_path / "base_scoring_config.json"
    base_config_path.write_text(json.dumps(controller._current_scoring_config().model_dump(), indent=2))

    def fake_run_suite_sync(suite_path, output_dir, **kwargs):
        assert kwargs["scoring_config_path"] == str(base_config_path)
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_manifest.json").write_text(json.dumps(baseline_manifest))
        return baseline_manifest

    worktree_dir = tmp_path / "candidate_worktree"
    worktree_dir.mkdir(parents=True)

    async def fake_generate(*args, **kwargs):
        return (
            "prompt",
            json.dumps(
                {
                    "summary": "reduce license plate weight",
                    "overrides": {
                        "source_confidence": {
                            "license_plate": 0.6,
                        }
                    },
                }
            ),
        )

    def fake_run_suite_subprocess(worktree_dir_arg, suite_path, output_dir, **kwargs):
        config_path = kwargs["extra_env"]["SCORING_CONFIG_PATH"]
        content = json.loads(Path(config_path).read_text())
        assert content["source_confidence"]["license_plate"] == 0.6
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "overall_metrics": {
                        "accuracy_25km": 0.76,
                        "country_accuracy": 1.0,
                        "median_gcd_km": 9.0,
                        "mean_latency_ms": 1000.0,
                        "mean_cost_usd": 0.0,
                    },
                    "datasets": [],
                    "sample_results_path": str(output_dir / "sample_results.jsonl"),
                }
            )
        )

    monkeypatch.setattr("src.improve.controller.run_suite_sync", fake_run_suite_sync)
    monkeypatch.setattr("src.improve.controller.rank_experiment_dir", lambda experiment_dir: {"winner": None, "rankings": []})
    monkeypatch.setattr(controller, "_build_candidate_context", lambda *args, **kwargs: CandidateContext([], [], [SCORING_CONFIG_TARGET]))
    monkeypatch.setattr(controller, "_create_worktree", lambda planned, candidate_id: worktree_dir)
    monkeypatch.setattr(controller, "_generate_candidate_rewrite", fake_generate)
    monkeypatch.setattr(controller, "_run_suite_subprocess", fake_run_suite_subprocess)

    experiment_dir = controller.run(
        tmp_path / "suite.json",
        candidate_count=1,
        base_scoring_config_path=str(base_config_path),
    )
    experiment = json.loads((experiment_dir / "experiment.json").read_text())
    candidate = experiment["candidates"][0]

    assert candidate["status"] == "evaluated"
    assert candidate["runtime_config_path"].endswith("candidate_scoring_config.json")


def test_create_worktree_uses_snapshot_when_repo_dirty(tmp_path, monkeypatch, test_settings) -> None:
    controller = ImprovementController(test_settings)
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()

    monkeypatch.setattr(controller, "_repo_has_local_changes", lambda: True)
    monkeypatch.setattr(controller, "_create_snapshot_checkout", lambda candidate_id: snapshot_dir)

    result = controller._create_worktree(tmp_path / "worktree", "candidate_01")

    assert result == snapshot_dir


def test_build_candidate_context_proposals_are_deterministic_and_diverse(tmp_path, test_settings) -> None:
    controller = ImprovementController(test_settings)
    sample_results_path = tmp_path / "baseline_sample_results.jsonl"
    rows = [
        SampleResult(
            image_path="images/giza.jpg",
            pred_lat=0.0,
            pred_lon=0.0,
            pred_country="France",
            pred_city="",
            pred_confidence=0.92,
            gt_lat=29.9792,
            gt_lon=31.1342,
            gt_country="Egypt",
            gt_city="Giza",
            latency_ms=62000.0,
        ),
        SampleResult(
            image_path="images/paris.jpg",
            pred_lat=48.8584,
            pred_lon=2.2945,
            pred_country="France",
            pred_city="Paris",
            pred_confidence=0.22,
            gt_lat=48.8584,
            gt_lon=2.2945,
            gt_country="France",
            gt_city="Paris",
            latency_ms=18000.0,
        ),
    ]
    sample_results_path.write_text("".join(json.dumps(row.to_dict()) + "\n" for row in rows))
    baseline_manifest = {
        "sample_results_path": str(sample_results_path),
        "overall_metrics": {
            "accuracy_25km": 0.5,
            "country_accuracy": 0.5,
            "city_accuracy": 0.5,
            "mean_latency_ms": 40000.0,
            "ece": 0.7,
        },
        "datasets": [
            {
                "dataset_id": "core",
                "status": "completed",
                "optional": False,
                "metrics": {"accuracy_25km": 0.5},
            }
        ],
    }

    first = controller._build_candidate_context(baseline_manifest, max_scoring_proposals=3)
    second = controller._build_candidate_context(baseline_manifest, max_scoring_proposals=3)

    assert [proposal.fingerprint for proposal in first.scoring_proposals] == [
        proposal.fingerprint for proposal in second.scoring_proposals
    ]
    assert len({proposal.fingerprint for proposal in first.scoring_proposals}) == len(first.scoring_proposals)
    assert len({proposal.family for proposal in first.scoring_proposals[:2]}) >= 2


def test_build_candidate_context_retains_breadth_after_two_wins(tmp_path, test_settings) -> None:
    controller = ImprovementController(test_settings)
    base_config_path = tmp_path / "base_scoring_config.json"
    config = controller._current_scoring_config().model_dump()
    config["source_confidence"]["license_plate"] = 0.4
    config["verification"]["supported_boost"] = 1.05
    base_config_path.write_text(json.dumps(config, indent=2))

    sample_results_path = tmp_path / "baseline_sample_results.jsonl"
    rows = [
        SampleResult(
            image_path="images/sample_a.jpg",
            pred_lat=48.8584,
            pred_lon=2.2945,
            pred_country="France",
            pred_city="Paris",
            pred_confidence=0.84,
            gt_lat=48.8584,
            gt_lon=2.2945,
            gt_country="France",
            gt_city="Paris",
            latency_ms=72000.0,
        ),
        SampleResult(
            image_path="images/sample_b.jpg",
            pred_lat=51.5007,
            pred_lon=-0.1246,
            pred_country="United Kingdom",
            pred_city="London",
            pred_confidence=0.79,
            gt_lat=51.5007,
            gt_lon=-0.1246,
            gt_country="United Kingdom",
            gt_city="London",
            latency_ms=64000.0,
        ),
    ]
    sample_results_path.write_text("".join(json.dumps(row.to_dict()) + "\n" for row in rows))
    baseline_manifest = {
        "sample_results_path": str(sample_results_path),
        "overall_metrics": {
            "accuracy_25km": 1.0,
            "country_accuracy": 1.0,
            "city_accuracy": 1.0,
            "mean_latency_ms": 68000.0,
            "ece": 0.69,
        },
        "datasets": [
            {
                "dataset_id": "core",
                "status": "completed",
                "optional": False,
                "metrics": {"accuracy_25km": 1.0},
            }
        ],
    }

    context = controller._build_candidate_context(
        baseline_manifest,
        base_scoring_config_path=str(base_config_path),
        max_scoring_proposals=5,
    )

    assert len(context.scoring_proposals) >= 4
    assert {proposal.family for proposal in context.scoring_proposals}.issuperset(
        {"heuristic_latency", "heuristic_calibration"}
    )


def test_build_candidate_context_prefers_local_search_neighbors_after_recent_win(tmp_path, test_settings) -> None:
    controller = ImprovementController(test_settings)
    base_config_path = tmp_path / "base_scoring_config.json"
    config = controller._current_scoring_config().model_dump()
    config["verification"]["supported_boost"] = 1.05
    base_config_path.write_text(json.dumps(config, indent=2))
    base_fingerprint = controller._scoring_config_fingerprint(str(base_config_path))

    sample_results_path = tmp_path / "baseline_sample_results.jsonl"
    rows = [
        SampleResult(
            image_path="images/sample_a.jpg",
            pred_lat=48.8584,
            pred_lon=2.2945,
            pred_country="France",
            pred_city="Paris",
            pred_confidence=0.84,
            gt_lat=48.8584,
            gt_lon=2.2945,
            gt_country="France",
            gt_city="Paris",
            latency_ms=69000.0,
        ),
        SampleResult(
            image_path="images/sample_b.jpg",
            pred_lat=51.5007,
            pred_lon=-0.1246,
            pred_country="United Kingdom",
            pred_city="London",
            pred_confidence=0.79,
            gt_lat=51.5007,
            gt_lon=-0.1246,
            gt_country="United Kingdom",
            gt_city="London",
            latency_ms=64000.0,
        ),
    ]
    sample_results_path.write_text("".join(json.dumps(row.to_dict()) + "\n" for row in rows))
    baseline_manifest = {
        "sample_results_path": str(sample_results_path),
        "overall_metrics": {
            "accuracy_25km": 1.0,
            "country_accuracy": 1.0,
            "city_accuracy": 0.875,
            "mean_latency_ms": 66500.0,
            "ece": 0.69,
        },
        "datasets": [
            {
                "dataset_id": "core",
                "status": "completed",
                "optional": False,
                "metrics": {"accuracy_25km": 1.0},
            }
        ],
    }
    feedback = ProposalFeedback(
        recent_wins=[
            ProposalOutcome(
                family="heuristic_calibration",
                fingerprint="winner-fp",
                rationale="heuristic_calibration:supported_boost_down:v1",
                overrides={"verification": {"supported_boost": 1.05}},
                changed_paths=["verification.supported_boost"],
                changes=[
                    {
                        "path": "verification.supported_boost",
                        "old_value": 1.1,
                        "new_value": 1.05,
                    }
                ],
                status="accepted",
                score=2.2,
                delta_accuracy_25km=0.0,
                delta_country_accuracy=0.0,
                delta_city_accuracy=0.0,
                delta_ece=-0.05,
                delta_latency_ms=-7000.0,
                base_config_fingerprint=base_fingerprint,
            )
        ]
    )

    context = controller._build_candidate_context(
        baseline_manifest,
        base_scoring_config_path=str(base_config_path),
        max_scoring_proposals=3,
        proposal_feedback=feedback,
    )

    assert context.scoring_proposals[0].family == "local_search"
    assert context.scoring_proposals[0].rationale.startswith("local_search:verification.supported_boost")


def test_rank_scoring_proposals_filters_repeated_same_base_latency_losers(test_settings) -> None:
    controller = ImprovementController(test_settings)
    bad = controller._proposal(
        family="heuristic_latency",
        summary="cut refinement iterations",
        overrides={"refinement": {"max_iterations": 1}},
        rationale="heuristic_latency:max_iterations_down",
        origin_signals=["heuristic_latency"],
        objective="latency",
        priority=controller._proposal_priority("heuristic_latency", "latency", "max_iterations"),
    )
    good = controller._proposal(
        family="local_search",
        summary="nudge supported boost around recent winner",
        overrides={"verification": {"supported_boost": 1.03}},
        rationale="local_search:verification.supported_boost:continue:w1:v1",
        origin_signals=["verification.supported_boost"],
        objective="latency",
        priority=controller._proposal_priority("local_search", "latency", "verification.supported_boost"),
    )
    feedback = ProposalFeedback(
        recent_outcomes=[
            ProposalOutcome(
                family="heuristic_latency",
                fingerprint="bad-1",
                rationale="heuristic_latency:max_iterations_down",
                overrides={"refinement": {"max_iterations": 1}},
                changed_paths=["refinement.max_iterations"],
                changes=[{"path": "refinement.max_iterations", "old_value": 2, "new_value": 1}],
                status="rejected",
                score=-4.0,
                delta_accuracy_25km=0.0,
                delta_country_accuracy=0.0,
                delta_city_accuracy=-0.25,
                delta_ece=-0.05,
                delta_latency_ms=6500.0,
                base_config_fingerprint="base-fp",
            ),
            ProposalOutcome(
                family="heuristic_latency",
                fingerprint="bad-2",
                rationale="heuristic_latency:max_iterations_down",
                overrides={"refinement": {"max_iterations": 1}},
                changed_paths=["refinement.max_iterations"],
                changes=[{"path": "refinement.max_iterations", "old_value": 2, "new_value": 1}],
                status="rejected",
                score=-2.5,
                delta_accuracy_25km=0.0,
                delta_country_accuracy=0.0,
                delta_city_accuracy=-0.125,
                delta_ece=-0.03,
                delta_latency_ms=2200.0,
                base_config_fingerprint="base-fp",
            ),
        ],
        recent_wins=[],
    )

    ranked = controller._rank_scoring_proposals(
        [bad, good],
        max_candidates=2,
        current_base_fingerprint="base-fp",
        proposal_feedback=feedback,
    )

    assert [proposal.fingerprint for proposal in ranked] == [good.fingerprint]


def test_campaign_promotes_winner_and_tracks_streak(tmp_path, monkeypatch, test_settings) -> None:
    test_settings.improvement.output_dir = str(tmp_path / "improve")
    controller = ImprovementController(test_settings)
    base_paths_seen: list[str] = []

    def fake_run(
        suite_path,
        *,
        experiment_name="",
        candidate_count=None,
        quality="balanced",
        max_concurrent=3,
        judge=False,
        mutator_instructions="",
        base_scoring_config_path=None,
        config_only=False,
        proposal_fingerprints_to_skip=None,
        proposal_feedback=None,
        lineage_id="",
    ):
        wave_index = len(base_paths_seen) + 1
        base_paths_seen.append(base_scoring_config_path or "")
        assert config_only is True
        assert lineage_id
        experiment_dir = tmp_path / f"experiment_wave_{wave_index:02d}"
        baseline_dir = experiment_dir / "baseline"
        candidate_dir = experiment_dir / "candidates" / "candidate_01"
        baseline_dir.mkdir(parents=True)
        candidate_dir.mkdir(parents=True)

        promoted_path = candidate_dir / "candidate_scoring_config.json"
        promoted_path.write_text(json.dumps({"wave": wave_index}))
        selection_path = candidate_dir / "candidate_selection.json"
        selection_path.write_text(
            json.dumps(
                {
                    "proposal_fingerprint": f"fingerprint-{wave_index}",
                    "proposal_family": "heuristic_accuracy",
                }
            )
        )
        (baseline_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "scoring_config_path": base_scoring_config_path or "",
                    "scoring_config_fingerprint": f"basefp-{wave_index}",
                }
            )
        )
        (experiment_dir / "experiment.json").write_text(
            json.dumps(
                {
                    "available_scoring_proposals": 3,
                    "candidates": [
                        {
                            "candidate_id": "candidate_01",
                            "runtime_config_path": str(promoted_path),
                            "selection_path": str(selection_path),
                        }
                    ],
                }
            )
        )
        (experiment_dir / "ranking.json").write_text(
            json.dumps(
                {
                    "winner": "candidate_01",
                    "rankings": [{"candidate_id": "candidate_01", "status": "accepted", "score": 10.0}],
                }
            )
        )
        return experiment_dir

    monkeypatch.setattr(controller, "run", fake_run)

    campaign_dir = controller.campaign(
        str(tmp_path / "suite.json"),
        campaign_name="repeatable",
        candidate_count=3,
        required_streak=3,
        max_waves=5,
    )
    campaign = json.loads((campaign_dir / "campaign.json").read_text())

    assert campaign["status"] == "succeeded"
    assert campaign["current_streak"] == 3
    assert len(campaign["waves"]) == 3
    assert campaign["waves"][0]["winner"] == "candidate_01"
    assert campaign["waves"][1]["base_config_path"].endswith("candidate_scoring_config.json")
    assert base_paths_seen[1].endswith("candidate_scoring_config.json")


def test_campaign_does_not_promote_negative_score_winner(tmp_path, monkeypatch, test_settings) -> None:
    test_settings.improvement.output_dir = str(tmp_path / "improve")
    controller = ImprovementController(test_settings)

    def fake_run(
        suite_path,
        *,
        experiment_name="",
        candidate_count=None,
        quality="balanced",
        max_concurrent=3,
        judge=False,
        mutator_instructions="",
        base_scoring_config_path=None,
        config_only=False,
        proposal_fingerprints_to_skip=None,
        proposal_feedback=None,
        lineage_id="",
    ):
        experiment_dir = tmp_path / "experiment_wave_01"
        baseline_dir = experiment_dir / "baseline"
        candidate_dir = experiment_dir / "candidates" / "candidate_01"
        baseline_dir.mkdir(parents=True)
        candidate_dir.mkdir(parents=True)

        selection_path = candidate_dir / "candidate_selection.json"
        selection_path.write_text(json.dumps({"proposal_fingerprint": "fp-1"}))
        promoted_path = candidate_dir / "candidate_scoring_config.json"
        promoted_path.write_text(json.dumps({"wave": 1}))
        (baseline_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "scoring_config_path": base_scoring_config_path or "",
                    "scoring_config_fingerprint": "basefp-1",
                }
            )
        )
        (experiment_dir / "experiment.json").write_text(
            json.dumps(
                {
                    "available_scoring_proposals": 1,
                    "candidates": [
                        {
                            "candidate_id": "candidate_01",
                            "runtime_config_path": str(promoted_path),
                            "selection_path": str(selection_path),
                        }
                    ],
                }
            )
        )
        (experiment_dir / "ranking.json").write_text(
            json.dumps(
                {
                    "winner": "candidate_01",
                    "rankings": [
                        {
                            "candidate_id": "candidate_01",
                            "status": "accepted",
                            "score": -1.0,
                        }
                    ],
                }
            )
        )
        return experiment_dir

    monkeypatch.setattr(controller, "run", fake_run)

    campaign_dir = controller.campaign(
        str(tmp_path / "suite.json"),
        campaign_name="negative_score_guard",
        candidate_count=1,
        required_streak=1,
        max_waves=1,
    )
    campaign = json.loads((campaign_dir / "campaign.json").read_text())

    assert campaign["status"] == "exhausted"
    assert campaign["current_streak"] == 0
    assert campaign["waves"][0]["winner"] is None
    assert campaign["waves"][0]["status"] == "no_winner"
