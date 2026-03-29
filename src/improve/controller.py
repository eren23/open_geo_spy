"""Experiment controller for the self-improving loop."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from os.path import relpath
from pathlib import Path
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from src.config.settings import Settings, get_scoring_config, get_settings
from src.eval.evolution import FailureAnalyzer
from src.eval.metrics import EvalMetrics, SampleResult
from src.eval.tuner import WeightTuner
from src.improve.execution import build_improvement_capability_snapshot, run_suite_sync
from src.improve.judge import TraceQualityJudge
from src.improve.ranking import rank_experiment_dir
from src.improve.suite import (
    BenchmarkDatasetSpec,
    add_dataset_to_suite,
    build_regression_dataset_from_traces,
    import_dataset,
)
from src.scoring.config import ScoringConfig
from src.improve.trace_analysis import analyze_trace_file
from src.utils.json_utils import find_json_object

SCORING_CONFIG_TARGET = "__scoring_config__"


@dataclass
class CandidateRecord:
    """Persistent candidate metadata for an experiment."""

    candidate_id: str
    worktree_path: str
    output_dir: str
    status: str = "created"
    patch_path: str = ""
    prompt_path: str = ""
    selection_path: str = ""
    validation_path: str = ""
    diff_path: str = ""
    raw_response_path: str = ""
    target_path: str = ""
    runtime_config_path: str = ""
    error: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "worktree_path": self.worktree_path,
            "output_dir": self.output_dir,
            "status": self.status,
            "patch_path": self.patch_path,
            "prompt_path": self.prompt_path,
            "selection_path": self.selection_path,
            "validation_path": self.validation_path,
            "diff_path": self.diff_path,
            "raw_response_path": self.raw_response_path,
            "target_path": self.target_path,
            "runtime_config_path": self.runtime_config_path,
            "error": self.error,
            "summary": self.summary,
        }


@dataclass
class CandidateContext:
    """Summary of baseline failures used to guide one candidate mutation."""

    failure_summary: list[dict[str, Any]]
    worst_samples: list[dict[str, Any]]
    target_files: list[str]
    scoring_proposals: list["ScoringProposal"] = field(default_factory=list)


@dataclass
class ScoringProposal:
    """Deterministic scoring-config proposal candidate."""

    family: str
    summary: str
    overrides: dict[str, Any]
    rationale: str
    origin_signals: list[str]
    objective: str
    priority: float
    fingerprint: str

    def to_rewrite(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "overrides": self.overrides,
        }


@dataclass
class CampaignState:
    """Persistent state for a multi-wave config-only improvement campaign."""

    campaign_id: str
    suite_path: str
    quality: str
    max_concurrent: int
    candidate_count: int
    required_streak: int
    max_waves: int
    created_at: str
    current_streak: int = 0
    wave_count: int = 0
    current_base_config_path: str = ""
    current_base_config_fingerprint: str = ""
    status: str = "running"
    experiment_dirs: list[str] = field(default_factory=list)
    tried_proposals: dict[str, list[str]] = field(default_factory=dict)
    waves: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "suite_path": self.suite_path,
            "quality": self.quality,
            "max_concurrent": self.max_concurrent,
            "candidate_count": self.candidate_count,
            "required_streak": self.required_streak,
            "max_waves": self.max_waves,
            "created_at": self.created_at,
            "current_streak": self.current_streak,
            "wave_count": self.wave_count,
            "current_base_config_path": self.current_base_config_path,
            "current_base_config_fingerprint": self.current_base_config_fingerprint,
            "status": self.status,
            "experiment_dirs": self.experiment_dirs,
            "tried_proposals": self.tried_proposals,
            "waves": self.waves,
        }


@dataclass
class ProposalOutcome:
    """Observed result for one scoring proposal from a prior campaign wave."""

    family: str
    fingerprint: str
    rationale: str
    overrides: dict[str, Any]
    changed_paths: list[str]
    changes: list[dict[str, Any]]
    status: str
    score: float
    delta_accuracy_25km: float
    delta_country_accuracy: float
    delta_city_accuracy: float
    delta_ece: float
    delta_latency_ms: float
    base_config_fingerprint: str


@dataclass
class ProposalFeedback:
    """Recent campaign outcomes used to shape the next proposal portfolio."""

    recent_outcomes: list[ProposalOutcome] = field(default_factory=list)
    recent_wins: list[ProposalOutcome] = field(default_factory=list)


class ImprovementController:
    """Coordinate baseline runs, code mutations, evaluation, and ranking."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = AsyncOpenAI(
            base_url=self.settings.llm.base_url,
            api_key=self.settings.llm.api_key,
            timeout=90.0,
        )

    def import_benchmark(
        self,
        source_path: str,
        output_manifest: str,
        *,
        adapter: str = "auto",
        dataset_id: str = "",
        description: str = "",
        suite_path: str | None = None,
        protected: bool = False,
        optional: bool = False,
        expected_sample_count: int | None = None,
        source_label: str = "",
        limit: int | None = None,
        seed: int = 42,
        stratify_by: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> Path:
        """Normalize a benchmark dataset and optionally add it to a suite."""
        manifest_path = import_dataset(
            source_path,
            output_manifest,
            adapter=adapter,
            dataset_name=dataset_id,
            description=description,
            limit=limit,
            seed=seed,
            stratify_by=stratify_by,
            source_label=source_label,
        )
        if suite_path:
            manifest_ref = self._suite_manifest_ref(Path(output_manifest), Path(suite_path))
            dataset_spec = BenchmarkDatasetSpec(
                dataset_id=dataset_id or Path(output_manifest).stem,
                manifest_path=manifest_ref,
                protected=protected,
                optional=optional,
                expected_sample_count=expected_sample_count if expected_sample_count is not None else limit,
                source_label=source_label,
                tags=tags or [],
            )
            add_dataset_to_suite(suite_path, dataset_spec)
        return manifest_path

    def import_trace_regressions(
        self,
        traces_dir: str,
        output_manifest: str,
        *,
        suite_path: str | None = None,
        dataset_id: str = "trace_regressions",
    ) -> Path:
        """Build a regression dataset from saved traces."""
        manifest_path = build_regression_dataset_from_traces(traces_dir, output_manifest)
        if suite_path:
            manifest_ref = self._suite_manifest_ref(Path(output_manifest), Path(suite_path))
            add_dataset_to_suite(
                suite_path,
                BenchmarkDatasetSpec(
                    dataset_id=dataset_id,
                    manifest_path=manifest_ref,
                    protected=True,
                    optional=True,
                    expected_sample_count=None,
                    source_label="trace_regressions",
                    tags=["regression", "trace_mined"],
                ),
            )
        return manifest_path

    def run(
        self,
        suite_path: str,
        *,
        experiment_name: str = "",
        candidate_count: int | None = None,
        quality: str = "balanced",
        max_concurrent: int = 3,
        judge: bool = False,
        mutator_instructions: str = "",
        base_scoring_config_path: str | None = None,
        config_only: bool = False,
        proposal_fingerprints_to_skip: set[str] | None = None,
        proposal_feedback: ProposalFeedback | None = None,
        lineage_id: str = "",
    ) -> Path:
        """Run a baseline plus N candidate mutations in isolated worktrees."""
        suite_path = str(Path(suite_path).resolve())
        experiment_dir = self._create_experiment_dir(experiment_name)
        capability_snapshot = build_improvement_capability_snapshot(self.settings)
        capability_snapshot_path = experiment_dir / "capability_snapshot.json"
        capability_snapshot_path.write_text(json.dumps(capability_snapshot, indent=2))
        base_config_path = str(Path(base_scoring_config_path).resolve()) if base_scoring_config_path else ""
        base_config_fingerprint = self._scoring_config_fingerprint(base_config_path or None)
        experiment_state = {
            "name": experiment_name or experiment_dir.name,
            "suite_path": suite_path,
            "quality": quality,
            "max_concurrent": max_concurrent,
            "judge": judge,
            "candidate_mode": "config_only" if config_only else "mixed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "baseline_dir": str(experiment_dir / "baseline"),
            "capability_snapshot_path": str(capability_snapshot_path),
            "base_scoring_config_path": base_config_path,
            "base_scoring_config_fingerprint": base_config_fingerprint,
            "baseline_lineage_id": lineage_id,
            "available_scoring_proposals": 0,
            "candidates": [],
        }

        baseline_manifest = run_suite_sync(
            suite_path,
            experiment_dir / "baseline",
            label="baseline",
            quality=quality,
            max_concurrent=max_concurrent,
            capability_snapshot=capability_snapshot,
            scoring_config_path=base_config_path or None,
            baseline_lineage_id=lineage_id,
        )
        if judge:
            self._judge_run(experiment_dir / "baseline" / "run_manifest.json")

        count = candidate_count if candidate_count is not None else self.settings.improvement.candidate_count
        candidate_context = self._build_candidate_context(
            baseline_manifest,
            base_scoring_config_path=base_config_path or None,
            max_scoring_proposals=count,
            proposal_fingerprints_to_skip=proposal_fingerprints_to_skip or set(),
            proposal_feedback=proposal_feedback,
        )
        experiment_state["available_scoring_proposals"] = len(candidate_context.scoring_proposals)
        if config_only:
            count = min(count, len(candidate_context.scoring_proposals))
        self._write_experiment_state(experiment_dir, experiment_state)
        for index in range(1, count + 1):
            candidate_id = f"candidate_{index:02d}"
            candidate_dir = experiment_dir / "candidates" / candidate_id
            planned_worktree_dir = Path(self.settings.improvement.worktree_dir) / f"{experiment_dir.name}_{candidate_id}"
            planned_worktree_dir.parent.mkdir(parents=True, exist_ok=True)
            record = CandidateRecord(
                candidate_id=candidate_id,
                worktree_path=str(planned_worktree_dir),
                output_dir=str(candidate_dir),
            )
            experiment_state["candidates"].append(record.to_dict())
            self._write_experiment_state(experiment_dir, experiment_state)

            worktree_dir = planned_worktree_dir
            try:
                worktree_dir = self._create_worktree(planned_worktree_dir, candidate_id)
                record.worktree_path = str(worktree_dir)
                candidate_dir.mkdir(parents=True, exist_ok=True)

                selection = self._select_candidate_target(
                    candidate_id,
                    index,
                    candidate_context,
                    config_only=config_only,
                )
                selection_path = candidate_dir / "candidate_selection.json"
                selection_path.write_text(json.dumps(selection, indent=2))
                record.selection_path = str(selection_path)
                record.target_path = str(selection["path"])

                prompt, raw_response = asyncio.run(
                    self._generate_candidate_rewrite(
                        worktree_dir,
                        baseline_manifest,
                        candidate_id,
                        selection,
                        candidate_context,
                        base_scoring_config_path=base_config_path or None,
                        mutator_instructions=mutator_instructions,
                    )
                )
                prompt_path = candidate_dir / "mutator_prompt.txt"
                raw_response_path = candidate_dir / "mutator_response.txt"
                patch_path = candidate_dir / "candidate_rewrite.json"
                validation_path = candidate_dir / "candidate_validation.json"
                prompt_path.write_text(prompt)
                raw_response_path.write_text(raw_response)
                record.prompt_path = str(prompt_path)
                record.raw_response_path = str(raw_response_path)
                record.patch_path = str(patch_path)
                record.validation_path = str(validation_path)

                rewrite = self._extract_candidate_rewrite(raw_response)
                patch_path.write_text(json.dumps(rewrite or {}, indent=2))
                validation = self._validate_candidate_rewrite(
                    worktree_dir,
                    selection["path"],
                    rewrite,
                    base_scoring_config_path=base_config_path or None,
                )
                validation_path.write_text(json.dumps(validation, indent=2))
                record.summary = str(validation.get("summary", "")).strip()
                if validation["status"] == "skipped":
                    record.status = "skipped"
                    continue
                if validation["status"] == "rejected":
                    record.status = "rejected"
                    record.error = validation["reason"]
                    continue

                extra_env = {"SCORING_CONFIG_PATH": base_config_path} if base_config_path else {}
                diff_path = candidate_dir / "candidate_diff.patch"
                diff_path.write_text(
                    self._build_diff_preview(
                        selection["path"],
                        validation["old_content"],
                        validation["new_content"],
                    )
                )
                record.diff_path = str(diff_path)

                if selection["path"] == SCORING_CONFIG_TARGET:
                    runtime_config_path = candidate_dir / "candidate_scoring_config.json"
                    runtime_config_path.write_text(validation["new_content"])
                    record.runtime_config_path = str(runtime_config_path)
                    extra_env["SCORING_CONFIG_PATH"] = str(runtime_config_path)
                else:
                    self._apply_candidate_rewrite(
                        worktree_dir,
                        selection["path"],
                        validation["new_content"],
                    )
                record.status = "mutated"

                self._run_suite_subprocess(
                    worktree_dir,
                    suite_path,
                    candidate_dir,
                    label=candidate_id,
                    quality=quality,
                    max_concurrent=max_concurrent,
                    capability_snapshot_path=capability_snapshot_path,
                    baseline_lineage_id=lineage_id,
                    extra_env=extra_env or None,
                )
                if judge:
                    self._judge_run(candidate_dir / "run_manifest.json")
                record.status = "evaluated"
            except Exception as exc:
                logger.warning("Candidate {} failed: {}", candidate_id, exc)
                record.status = "failed"
                record.error = str(exc)
            finally:
                if self.settings.improvement.auto_cleanup_worktrees:
                    self._remove_worktree(worktree_dir)
                self._replace_candidate(experiment_state, record)
                self._write_experiment_state(experiment_dir, experiment_state)

        rank_experiment_dir(experiment_dir)
        return experiment_dir

    def campaign(
        self,
        suite_path: str,
        *,
        campaign_name: str = "",
        candidate_count: int | None = None,
        required_streak: int = 3,
        max_waves: int = 8,
        quality: str = "balanced",
        max_concurrent: int = 3,
        mutator_instructions: str = "",
        base_scoring_config_path: str | None = None,
    ) -> Path:
        """Run a deterministic config-only campaign until the streak target is met or proposals are exhausted."""
        suite_path = str(Path(suite_path).resolve())
        campaign_dir = self._create_campaign_dir(campaign_name or Path(suite_path).stem)
        count = candidate_count if candidate_count is not None else self.settings.improvement.candidate_count
        current_base_config_path = str(Path(base_scoring_config_path).resolve()) if base_scoring_config_path else ""
        state = CampaignState(
            campaign_id=campaign_dir.name,
            suite_path=suite_path,
            quality=quality,
            max_concurrent=max_concurrent,
            candidate_count=count,
            required_streak=required_streak,
            max_waves=max_waves,
            created_at=datetime.now(timezone.utc).isoformat(),
            current_base_config_path=current_base_config_path,
            current_base_config_fingerprint=self._scoring_config_fingerprint(current_base_config_path or None),
        )
        self._write_campaign_state(campaign_dir, state)

        for wave_index in range(1, max_waves + 1):
            base_fingerprint = self._scoring_config_fingerprint(current_base_config_path or None)
            skipped = set(state.tried_proposals.get(base_fingerprint, []))
            proposal_feedback = self._collect_proposal_feedback(state, base_fingerprint)
            experiment_dir = self.run(
                suite_path,
                experiment_name=f"{campaign_name or Path(suite_path).stem}_wave_{wave_index:02d}",
                candidate_count=count,
                quality=quality,
                max_concurrent=max_concurrent,
                judge=False,
                mutator_instructions=mutator_instructions,
                base_scoring_config_path=current_base_config_path or None,
                config_only=True,
                proposal_fingerprints_to_skip=skipped,
                proposal_feedback=proposal_feedback,
                lineage_id=state.campaign_id,
            )

            state.wave_count = wave_index
            state.experiment_dirs.append(str(experiment_dir))
            ranking = json.loads((experiment_dir / "ranking.json").read_text())
            experiment_state = json.loads((experiment_dir / "experiment.json").read_text())
            baseline_manifest = json.loads((experiment_dir / "baseline" / "run_manifest.json").read_text())
            state.current_base_config_fingerprint = baseline_manifest.get("scoring_config_fingerprint", base_fingerprint)

            seen_fingerprints: list[str] = []
            for candidate in experiment_state.get("candidates", []):
                selection_path = candidate.get("selection_path")
                if not selection_path or not Path(selection_path).exists():
                    continue
                selection = json.loads(Path(selection_path).read_text())
                fingerprint = selection.get("proposal_fingerprint")
                if fingerprint:
                    seen_fingerprints.append(str(fingerprint))
            existing = set(state.tried_proposals.get(base_fingerprint, []))
            existing.update(seen_fingerprints)
            state.tried_proposals[base_fingerprint] = sorted(existing)

            winner_id = ranking.get("winner")
            winner_config_path = ""
            winner_score = None
            exhausted = experiment_state.get("available_scoring_proposals", 0) == 0
            ranking_entry = next((item for item in ranking.get("rankings", []) if item.get("candidate_id") == winner_id), None)
            if ranking_entry and (
                ranking_entry.get("status") != "accepted" or float(ranking_entry.get("score", 0.0)) <= 0.0
            ):
                winner_id = None
                ranking_entry = None
            if winner_id:
                winner = next(
                    (candidate for candidate in experiment_state.get("candidates", []) if candidate["candidate_id"] == winner_id),
                    None,
                )
                winner_config_path = winner.get("runtime_config_path", "") if winner else ""
                current_base_config_path = winner_config_path or current_base_config_path
                state.current_base_config_path = current_base_config_path
                state.current_base_config_fingerprint = self._scoring_config_fingerprint(current_base_config_path or None)
                state.current_streak += 1
                winner_score = ranking_entry.get("score") if ranking_entry else None
            else:
                state.current_streak = 0
                if exhausted:
                    state.status = "exhausted"

            state.waves.append(
                {
                    "wave_index": wave_index,
                    "experiment_dir": str(experiment_dir),
                    "base_config_path": baseline_manifest.get("scoring_config_path") or "",
                    "base_config_fingerprint": baseline_manifest.get("scoring_config_fingerprint", base_fingerprint),
                    "winner": winner_id,
                    "winner_score": winner_score,
                    "promoted_config_path": winner_config_path,
                    "streak_after_wave": state.current_streak,
                    "available_scoring_proposals": experiment_state.get("available_scoring_proposals", 0),
                    "status": "accepted" if winner_id else ("exhausted" if exhausted else "no_winner"),
                }
            )
            self._write_campaign_state(campaign_dir, state)

            if state.current_streak >= required_streak:
                state.status = "succeeded"
                self._write_campaign_state(campaign_dir, state)
                break
            if state.status == "exhausted":
                break
        else:
            state.status = "exhausted"
            self._write_campaign_state(campaign_dir, state)

        return campaign_dir

    def resume(self, experiment_dir: str | Path) -> dict[str, Any]:
        """Rerank an existing experiment and fill in any missing candidate runs."""
        experiment_dir = Path(experiment_dir)
        with open(experiment_dir / "experiment.json") as f:
            experiment_state = json.load(f)
        capability_snapshot_path = experiment_state.get("capability_snapshot_path")

        for candidate in experiment_state.get("candidates", []):
            if candidate.get("status") in {"evaluated", "skipped", "rejected"}:
                continue
            worktree_dir = Path(candidate["worktree_path"])
            candidate_dir = Path(candidate["output_dir"])
            rewrite_path = candidate.get("patch_path")
            extra_env = {}
            base_config_path = experiment_state.get("base_scoring_config_path")
            if base_config_path:
                extra_env["SCORING_CONFIG_PATH"] = base_config_path
            if candidate.get("runtime_config_path"):
                extra_env["SCORING_CONFIG_PATH"] = candidate["runtime_config_path"]
            if rewrite_path and worktree_dir.exists() and not (candidate_dir / "run_manifest.json").exists():
                try:
                    self._run_suite_subprocess(
                        worktree_dir,
                        experiment_state["suite_path"],
                        candidate_dir,
                        label=candidate["candidate_id"],
                        quality=experiment_state.get("quality", "balanced"),
                        max_concurrent=experiment_state.get("max_concurrent", 3),
                        capability_snapshot_path=Path(capability_snapshot_path) if capability_snapshot_path else None,
                        baseline_lineage_id=experiment_state.get("baseline_lineage_id", ""),
                        extra_env=extra_env or None,
                    )
                    candidate["status"] = "evaluated"
                except Exception as exc:
                    candidate["status"] = "failed"
                    candidate["error"] = str(exc)
        self._write_experiment_state(experiment_dir, experiment_state)
        return rank_experiment_dir(experiment_dir)

    def rank(self, experiment_dir: str | Path) -> dict[str, Any]:
        """Rank candidates from saved artifacts."""
        return rank_experiment_dir(experiment_dir)

    def replay_trace(self, trace_path: str | Path) -> dict[str, Any]:
        """Analyze a trace and return anomaly diagnostics."""
        return analyze_trace_file(trace_path).to_dict()

    def _create_experiment_dir(self, experiment_name: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug = _slugify(experiment_name or "improve")
        path = Path(self.settings.improvement.output_dir) / "experiments" / f"{timestamp}_{slug}"
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def _create_campaign_dir(self, campaign_name: str) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug = _slugify(campaign_name or "campaign")
        path = Path(self.settings.improvement.output_dir) / "campaigns" / f"{timestamp}_{slug}"
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    async def _generate_candidate_rewrite(
        self,
        worktree_dir: Path,
        baseline_manifest: dict[str, Any],
        candidate_id: str,
        selection: dict[str, Any],
        candidate_context: CandidateContext,
        *,
        base_scoring_config_path: str | None = None,
        mutator_instructions: str = "",
    ) -> tuple[str, str]:
        target_path = str(selection["path"])
        if target_path == SCORING_CONFIG_TARGET:
            proposal = self._proposal_for_selection(selection, candidate_context)
            if proposal is not None:
                return f"deterministic_scoring_candidate:{proposal.family}", json.dumps(proposal.to_rewrite(), indent=2)
            heuristic = self._heuristic_scoring_candidate(
                baseline_manifest,
                candidate_context,
                candidate_id,
                base_scoring_config_path=base_scoring_config_path,
            )
            if heuristic is not None:
                return "heuristic_scoring_candidate", json.dumps(heuristic, indent=2)

        base_prompt = self._build_mutator_prompt(
            worktree_dir,
            baseline_manifest,
            candidate_id,
            target_path,
            candidate_context,
            mutator_instructions=mutator_instructions,
        )
        prompt = base_prompt
        last_error = ""
        for _attempt in range(1, 4):
            try:
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=self.settings.improvement.mutator_model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Return only valid JSON. Do not return markdown fences, "
                                    "diffs, or explanatory prose outside the JSON object."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=12000,
                        response_format={"type": "json_object"},
                    ),
                    timeout=90.0,
                )
                raw = response.choices[0].message.content or ""
            except TimeoutError:
                last_error = "Mutator request timed out after 90 seconds"
                prompt = (
                    f"{base_prompt}\n\n"
                    f"The previous mutator attempt timed out.\nError:\n{last_error}\n\n"
                    "Return a valid JSON object for the selected file only.\n"
                )
                continue

            rewrite = self._extract_candidate_rewrite(raw)
            if rewrite is not None:
                return prompt, raw

            last_error = "Mutator returned no valid JSON rewrite object"
            prompt = (
                f"{base_prompt}\n\n"
                "The previous response was not a valid rewrite.\n"
                f"Validation error:\n{last_error}\n\n"
                "Return only JSON with `summary`, `path`, and `new_content`.\n"
                "Keep `path` equal to the selected file.\n"
                "If no safe edit is justified, return the same `path` and an empty `new_content` string.\n"
            )

        fallback = {
            "summary": f"no safe edit: {last_error or 'mutator did not return a usable rewrite'}",
            "overrides": {},
        }
        if target_path != SCORING_CONFIG_TARGET:
            fallback = {
                "summary": f"no safe edit: {last_error or 'mutator did not return a usable rewrite'}",
                "path": target_path,
                "new_content": "",
            }
        return prompt, json.dumps(fallback, indent=2)

    def _build_mutator_prompt(
        self,
        worktree_dir: Path,
        baseline_manifest: dict[str, Any],
        candidate_id: str,
        target_path: str,
        candidate_context: CandidateContext,
        *,
        mutator_instructions: str = "",
    ) -> str:
        if target_path == SCORING_CONFIG_TARGET:
            return self._build_scoring_config_prompt(
                baseline_manifest,
                candidate_id,
                candidate_context,
                base_scoring_config_path=base_scoring_config_path,
                mutator_instructions=mutator_instructions,
            )

        file_path = worktree_dir / target_path
        current_content = file_path.read_text() if file_path.exists() else ""
        return (
            "You are improving OpenGeoSpy with one surgical backend rewrite.\n"
            "Return only a JSON object. Do not include markdown or explanations outside JSON.\n"
            "You may rewrite exactly one file, and it must be the selected target path.\n\n"
            "JSON format:\n"
            "{\n"
            '  "summary": "one sentence",\n'
            f'  "path": "{target_path}",\n'
            '  "new_content": "full file contents"\n'
            "}\n\n"
            "Rules:\n"
            f"- `path` must be exactly `{target_path}`.\n"
            "- `new_content` must contain the complete rewritten file contents.\n"
            "- Keep the change small and directly related to the benchmark failures.\n"
            "- Do not edit unrelated infrastructure, frontend, deployment, or tests.\n"
            "- If no safe improvement is possible, return the same `path` with an empty `new_content` string.\n\n"
            f"Candidate id: {candidate_id}\n"
            f"Current overall metrics:\n{json.dumps(baseline_manifest.get('overall_metrics', {}), indent=2)}\n\n"
            f"Failure summary:\n{json.dumps(candidate_context.failure_summary, indent=2)}\n\n"
            f"Worst samples:\n{json.dumps(candidate_context.worst_samples, indent=2, default=str)}\n\n"
            f"Additional operator instructions:\n{mutator_instructions or '(none)'}\n\n"
            f"Selected target file: {target_path}\n\n"
            "Current file contents:\n"
            f"```python\n{current_content}\n```"
        )

    def _build_scoring_config_prompt(
        self,
        baseline_manifest: dict[str, Any],
        candidate_id: str,
        candidate_context: CandidateContext,
        *,
        base_scoring_config_path: str | None = None,
        mutator_instructions: str = "",
    ) -> str:
        base_config = self._current_scoring_config(base_scoring_config_path).model_dump()
        prompt_config = self._scoring_prompt_view(base_config)
        return (
            "You are improving OpenGeoSpy by proposing a small scoring-config override.\n"
            "Return only a JSON object. Do not include markdown or prose outside JSON.\n"
            "Do not rewrite Python files. Only return partial scoring config overrides.\n\n"
            "JSON format:\n"
            "{\n"
            '  "summary": "one sentence",\n'
            '  "overrides": {\n'
            '    "section_name": { "field_name": 0.7 }\n'
            "  }\n"
            "}\n\n"
            "Rules:\n"
            "- `overrides` must be a partial JSON object compatible with `ScoringConfig`.\n"
            "- Change only a few fields.\n"
            "- If no safe improvement is justified, return `{ \"summary\": \"no safe edit\", \"overrides\": {} }`.\n"
            "- Prefer latency/confidence tradeoff improvements when accuracy is already strong.\n\n"
            f"Candidate id: {candidate_id}\n"
            f"Current overall metrics:\n{json.dumps(baseline_manifest.get('overall_metrics', {}), indent=2)}\n\n"
            f"Failure summary:\n{json.dumps(candidate_context.failure_summary, indent=2)}\n\n"
            f"Worst samples:\n{json.dumps(candidate_context.worst_samples, indent=2, default=str)}\n\n"
            f"Additional operator instructions:\n{mutator_instructions or '(none)'}\n\n"
            f"Base scoring-config fingerprint: {self._scoring_config_fingerprint(base_scoring_config_path)}\n\n"
            "High-impact scoring sections:\n"
            f"{json.dumps(prompt_config, indent=2)}"
        )

    def _select_target_files(self, failure_reports: list[Any]) -> list[str]:
        mapping = {
            "wrong_country": ["src/scoring/config.py", "src/geo/country_matcher.py", "src/agents/reasoning_agent.py"],
            "high_confidence_wrong": ["src/scoring/config.py", "src/agents/reasoning_agent.py", "src/evidence/verifier.py"],
            "overconfident": ["src/scoring/config.py", "src/agents/reasoning_agent.py"],
            "underconfident": ["src/scoring/config.py", "src/agents/reasoning_agent.py"],
            "city_miss": ["src/agents/reasoning_agent.py", "src/geo/reranker.py", "src/search/graph.py"],
            "continent_wrong": ["src/scoring/config.py", "src/agents/web_intel_agent.py", "src/search/smart_expander.py"],
        }
        selected: list[str] = []
        for report in failure_reports:
            for candidate in mapping.get(report.category, []):
                if candidate not in selected and self._is_small_rewrite_target(candidate):
                    selected.append(candidate)
                if len(selected) >= self.settings.improvement.candidate_file_limit:
                    return selected
        fallback = [
            "src/scoring/config.py",
            "src/agents/web_intel_agent.py",
            "src/search/graph.py",
            "src/agents/reasoning_agent.py",
        ]
        for candidate in fallback:
            if candidate not in selected and self._is_small_rewrite_target(candidate):
                selected.append(candidate)
            if len(selected) >= self.settings.improvement.candidate_file_limit:
                break
        return selected

    def _is_small_rewrite_target(self, rel_path: str, max_chars: int = 12000) -> bool:
        path = Path(__file__).resolve().parents[2] / rel_path
        if not path.exists():
            return False
        try:
            return len(path.read_text()) <= max_chars
        except Exception:
            return False

    def _build_candidate_context(
        self,
        baseline_manifest: dict[str, Any],
        *,
        base_scoring_config_path: str | None = None,
        max_scoring_proposals: int | None = None,
        proposal_fingerprints_to_skip: set[str] | None = None,
        proposal_feedback: ProposalFeedback | None = None,
    ) -> CandidateContext:
        samples = self._load_sample_results(Path(baseline_manifest["sample_results_path"]))
        metrics = EvalMetrics(results=samples)
        failure_reports = FailureAnalyzer().analyze(metrics)
        scoring_proposals = self._build_scoring_proposals(
            baseline_manifest,
            metrics,
            failure_reports,
            base_scoring_config_path=base_scoring_config_path,
            max_candidates=max_scoring_proposals,
            skip_fingerprints=proposal_fingerprints_to_skip or set(),
            proposal_feedback=proposal_feedback,
        )
        return CandidateContext(
            failure_summary=[report.to_dict() for report in failure_reports[:4]],
            worst_samples=[
                sample.to_dict()
                for sample in sorted(
                    samples,
                    key=lambda item: item.gcd_km if item.gcd_km is not None else -1.0,
                    reverse=True,
                )[:5]
            ],
            target_files=[SCORING_CONFIG_TARGET, *self._select_target_files(failure_reports)],
            scoring_proposals=scoring_proposals,
        )

    def _select_candidate_target(
        self,
        candidate_id: str,
        index: int,
        candidate_context: CandidateContext,
        *,
        config_only: bool = False,
    ) -> dict[str, Any]:
        if candidate_context.scoring_proposals and (config_only or index <= len(candidate_context.scoring_proposals)):
            proposal = candidate_context.scoring_proposals[index - 1]
            return {
                "candidate_id": candidate_id,
                "path": SCORING_CONFIG_TARGET,
                "target_kind": "scoring_config",
                "selection_rank": index,
                "available_target_files": [SCORING_CONFIG_TARGET],
                "failure_categories": [
                    str(report.get("category", ""))
                    for report in candidate_context.failure_summary
                    if report.get("category")
                ],
                "rationale": proposal.rationale,
                "selection_policy": "scoring_portfolio",
                "proposal_family": proposal.family,
                "proposal_fingerprint": proposal.fingerprint,
                "proposal_objective": proposal.objective,
                "origin_signals": proposal.origin_signals,
            }
        if not candidate_context.target_files:
            raise RuntimeError("No candidate target files were selected from the baseline failure report")
        target_offset = len(candidate_context.scoring_proposals) if candidate_context.scoring_proposals and not config_only else 0
        target_index = (index - target_offset - 1) % len(candidate_context.target_files)
        failure_categories = [
            str(report.get("category", ""))
            for report in candidate_context.failure_summary
            if report.get("category")
        ]
        return {
            "candidate_id": candidate_id,
            "path": candidate_context.target_files[target_index],
            "target_kind": "scoring_config" if candidate_context.target_files[target_index] == SCORING_CONFIG_TARGET else "file",
            "selection_rank": target_index + 1,
            "available_target_files": candidate_context.target_files,
            "failure_categories": failure_categories,
            "rationale": (
                f"ranked_target_{target_index + 1}"
                + (f" for {failure_categories[0]}" if failure_categories else "")
            ),
            "selection_policy": "file_target_cycle",
        }

    def _extract_candidate_rewrite(self, raw: str) -> dict[str, Any] | None:
        stripped = raw.strip()
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
        if fenced:
            stripped = fenced.group(1).strip()
        json_str = stripped if stripped.startswith("{") and stripped.endswith("}") else find_json_object(stripped)
        if not json_str:
            return None
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _validate_candidate_rewrite(
        self,
        worktree_dir: Path,
        target_path: str,
        rewrite: dict[str, Any] | None,
        *,
        base_scoring_config_path: str | None = None,
    ) -> dict[str, Any]:
        if target_path == SCORING_CONFIG_TARGET:
            return self._validate_scoring_config_rewrite(
                rewrite,
                base_scoring_config_path=base_scoring_config_path,
            )

        file_path = (worktree_dir / target_path).resolve()
        if not file_path.exists():
            return {
                "status": "rejected",
                "reason": f"Target file does not exist: {target_path}",
                "summary": "",
                "path": target_path,
            }

        current_content = file_path.read_text()
        if not rewrite:
            return {
                "status": "rejected",
                "reason": "Mutator returned no valid JSON rewrite object",
                "summary": "",
                "path": target_path,
                "old_content": current_content,
                "new_content": "",
            }

        summary = str(rewrite.get("summary", "")).strip()
        path = str(rewrite.get("path", "")).strip()
        new_content = rewrite.get("new_content")
        if path != target_path:
            return {
                "status": "rejected",
                "reason": f"Mutator targeted {path or 'an empty path'} instead of {target_path}",
                "summary": summary,
                "path": path or target_path,
                "old_content": current_content,
                "new_content": "",
            }
        if not isinstance(new_content, str):
            return {
                "status": "rejected",
                "reason": "Mutator rewrite is missing a string `new_content` field",
                "summary": summary,
                "path": target_path,
                "old_content": current_content,
                "new_content": "",
            }
        if not new_content.strip():
            return {
                "status": "skipped",
                "reason": summary or "Mutator declined to make a safe change",
                "summary": summary or "no safe edit",
                "path": target_path,
                "old_content": current_content,
                "new_content": current_content,
            }
        if new_content == current_content:
            return {
                "status": "skipped",
                "reason": "Rewrite is identical to the current file",
                "summary": summary or "no-op rewrite",
                "path": target_path,
                "old_content": current_content,
                "new_content": current_content,
            }
        if file_path.suffix == ".py":
            try:
                compile(new_content, str(file_path), "exec")
            except SyntaxError as exc:
                return {
                    "status": "rejected",
                    "reason": f"Python syntax error: {exc.msg} (line {exc.lineno})",
                    "summary": summary,
                    "path": target_path,
                    "old_content": current_content,
                    "new_content": new_content,
                }
        return {
            "status": "accepted",
            "reason": "validated",
            "summary": summary or f"rewrite {target_path}",
            "path": target_path,
            "old_content": current_content,
            "new_content": new_content,
        }

    def _build_diff_preview(self, rel_path: str, old_content: str, new_content: str) -> str:
        return "".join(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=rel_path,
                tofile=rel_path,
            )
        )

    def _apply_candidate_rewrite(self, worktree_dir: Path, rel_path: str, new_content: str) -> None:
        file_path = (worktree_dir / rel_path).resolve()
        file_path.write_text(new_content)

    def _validate_scoring_config_rewrite(
        self,
        rewrite: dict[str, Any] | None,
        *,
        base_scoring_config_path: str | None = None,
    ) -> dict[str, Any]:
        current_config = self._current_scoring_config(base_scoring_config_path).model_dump()
        current_content = json.dumps(current_config, indent=2, sort_keys=True)
        if not rewrite:
            return {
                "status": "rejected",
                "reason": "Mutator returned no valid JSON scoring-config override object",
                "summary": "",
                "path": SCORING_CONFIG_TARGET,
                "old_content": current_content,
                "new_content": "",
            }

        summary = str(rewrite.get("summary", "")).strip()
        overrides = rewrite.get("overrides")
        if not isinstance(overrides, dict):
            return {
                "status": "rejected",
                "reason": "Scoring-config rewrite is missing an object `overrides` field",
                "summary": summary,
                "path": SCORING_CONFIG_TARGET,
                "old_content": current_content,
                "new_content": "",
            }
        if not overrides:
            return {
                "status": "skipped",
                "reason": summary or "Mutator declined to make a safe scoring-config change",
                "summary": summary or "no safe edit",
                "path": SCORING_CONFIG_TARGET,
                "old_content": current_content,
                "new_content": current_content,
            }

        merged = _deep_merge_json(current_config, overrides)
        try:
            validated = ScoringConfig.model_validate(merged).model_dump()
        except Exception as exc:
            return {
                "status": "rejected",
                "reason": f"Invalid scoring-config override: {exc}",
                "summary": summary,
                "path": SCORING_CONFIG_TARGET,
                "old_content": current_content,
                "new_content": json.dumps(overrides, indent=2, sort_keys=True),
            }
        new_content = json.dumps(validated, indent=2, sort_keys=True)
        if new_content == current_content:
            return {
                "status": "skipped",
                "reason": "Override produces no scoring-config change",
                "summary": summary or "no-op scoring override",
                "path": SCORING_CONFIG_TARGET,
                "old_content": current_content,
                "new_content": current_content,
            }
        return {
            "status": "accepted",
            "reason": "validated",
            "summary": summary or "scoring config override",
            "path": SCORING_CONFIG_TARGET,
            "old_content": current_content,
            "new_content": new_content,
        }

    def _current_scoring_config(self, scoring_config_path: str | None = None) -> ScoringConfig:
        if scoring_config_path:
            return ScoringConfig.from_file(scoring_config_path)
        return get_scoring_config()

    def _scoring_config_fingerprint(self, scoring_config_path: str | None = None) -> str:
        config = self._current_scoring_config(scoring_config_path).model_dump()
        return _json_fingerprint(config)

    def _proposal_for_selection(
        self,
        selection: dict[str, Any],
        candidate_context: CandidateContext,
    ) -> ScoringProposal | None:
        fingerprint = str(selection.get("proposal_fingerprint", "")).strip()
        if not fingerprint:
            return None
        return next(
            (proposal for proposal in candidate_context.scoring_proposals if proposal.fingerprint == fingerprint),
            None,
        )

    def _scoring_prompt_view(self, config: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "candidate_ranking",
            "country_penalty",
            "hint",
            "verification",
            "source_confidence",
            "visual_match",
            "refinement",
            "clustering",
            "grounding",
        ]
        return {key: config[key] for key in keys if key in config}

    def _heuristic_scoring_candidate(
        self,
        baseline_manifest: dict[str, Any],
        candidate_context: CandidateContext,
        candidate_id: str,
        *,
        base_scoring_config_path: str | None = None,
    ) -> dict[str, Any] | None:
        metrics = baseline_manifest.get("overall_metrics", {})
        config = self._current_scoring_config(base_scoring_config_path).model_dump()
        failure_categories = {
            str(report.get("category", ""))
            for report in candidate_context.failure_summary
            if report.get("category")
        }
        candidates: list[dict[str, Any]] = []

        if (
            failure_categories.intersection({"city_miss", "high_confidence_wrong", "overconfident"})
            or float(metrics.get("city_accuracy", 0.0)) < 0.95
        ):
            current = float(config["source_confidence"]["license_plate"])
            lowered = round(max(0.4, current - 0.4), 2)
            if lowered < current:
                candidates.append(
                    {
                        "summary": "Reduce license plate confidence to limit noisy OCR-driven city errors.",
                        "overrides": {"source_confidence": {"license_plate": lowered}},
                    }
                )

        if (
            failure_categories.intersection({"wrong_country", "high_confidence_wrong"})
            and float(config["country_penalty"]["penalty_factor"]) < 0.85
        ):
            candidates.append(
                {
                    "summary": "Increase the wrong-country penalty to suppress confident cross-country misses.",
                    "overrides": {
                        "country_penalty": {
                            "penalty_factor": round(min(0.85, float(config["country_penalty"]["penalty_factor"]) + 0.1), 2)
                        }
                    },
                }
            )

        if float(metrics.get("mean_latency_ms", 0.0)) > 35_000 and int(config["refinement"]["max_iterations"]) > 1:
            candidates.append(
                {
                    "summary": "Reduce refinement iterations to cut latency on already-strong cases.",
                    "overrides": {"refinement": {"max_iterations": 1}},
                }
            )

        if float(metrics.get("ece", 0.0)) > 0.65 and float(config["verification"]["supported_boost"]) > 1.05:
            candidates.append(
                {
                    "summary": "Reduce verification confidence boosting to improve calibration.",
                    "overrides": {"verification": {"supported_boost": 1.05}},
                }
            )

        if not candidates:
            return None

        ordinal = _candidate_ordinal(candidate_id)
        return candidates[(ordinal - 1) % len(candidates)]

    def _build_scoring_proposals(
        self,
        baseline_manifest: dict[str, Any],
        metrics: EvalMetrics,
        failure_reports: list[Any],
        *,
        base_scoring_config_path: str | None = None,
        max_candidates: int | None = None,
        skip_fingerprints: set[str] | None = None,
        proposal_feedback: ProposalFeedback | None = None,
    ) -> list[ScoringProposal]:
        skip_fingerprints = skip_fingerprints or set()
        config = self._current_scoring_config(base_scoring_config_path)
        current_config = config.model_dump()
        current_base_fingerprint = self._scoring_config_fingerprint(base_scoring_config_path)
        objective = self._proposal_objective(baseline_manifest)
        origin_signals = [report.category for report in failure_reports[:4]]

        proposals: list[ScoringProposal] = []
        tuner = WeightTuner(config=config)
        adjustments = tuner.tune(metrics)
        for adjustment in adjustments:
            proposals.append(
                self._proposal(
                    family="tuner_single",
                    summary=adjustment.reason,
                    overrides=_override_for_path(adjustment.path, adjustment.new_value),
                    rationale=f"tuner_single:{adjustment.path}",
                    origin_signals=origin_signals or [adjustment.path],
                    objective=objective,
                    priority=self._proposal_priority("tuner_single", objective, adjustment.path),
                )
            )

        compound = self._compound_tuner_overrides(adjustments)
        if compound:
            proposals.append(
                self._proposal(
                    family="tuner_compound",
                    summary="Combine the top non-conflicting tuner adjustments from the current failures.",
                    overrides=compound,
                    rationale="tuner_compound:top_two_non_conflicting",
                    origin_signals=origin_signals,
                    objective=objective,
                    priority=self._proposal_priority("tuner_compound", objective),
                )
            )

        for local_search in self._local_search_scoring_candidates(
            current_config,
            proposal_feedback or ProposalFeedback(),
            objective=objective,
        ):
            proposals.append(
                self._proposal(
                    family=local_search["family"],
                    summary=local_search["summary"],
                    overrides=local_search["overrides"],
                    rationale=local_search["rationale"],
                    origin_signals=local_search["origin_signals"],
                    objective=objective,
                    priority=float(local_search["priority"]),
                )
            )

        for heuristic in self._heuristic_scoring_candidates(
            baseline_manifest,
            failure_reports,
            current_config,
            objective=objective,
        ):
            proposals.append(
                self._proposal(
                    family=heuristic["family"],
                    summary=heuristic["summary"],
                    overrides=heuristic["overrides"],
                    rationale=heuristic["rationale"],
                    origin_signals=heuristic["origin_signals"],
                    objective=objective,
                    priority=float(
                        heuristic.get(
                            "priority",
                            self._proposal_priority(heuristic["family"], objective, heuristic.get("rationale", "")),
                        )
                    ),
                )
            )

        deduped: list[ScoringProposal] = []
        seen: set[str] = set()
        for proposal in proposals:
            if proposal.fingerprint in seen or proposal.fingerprint in skip_fingerprints:
                continue
            seen.add(proposal.fingerprint)
            deduped.append(proposal)

        ranked = self._rank_scoring_proposals(
            deduped,
            max_candidates=max_candidates,
            current_base_fingerprint=current_base_fingerprint,
            proposal_feedback=proposal_feedback,
        )
        return ranked

    def _proposal_objective(self, baseline_manifest: dict[str, Any]) -> str:
        required_dataset = next(
            (
                dataset.get("metrics") or {}
                for dataset in baseline_manifest.get("datasets", [])
                if dataset.get("status") == "completed" and not dataset.get("optional")
            ),
            {},
        )
        accuracy = float(required_dataset.get("accuracy_25km", baseline_manifest.get("overall_metrics", {}).get("accuracy_25km", 0.0)))
        return "latency" if accuracy >= 0.999 else "accuracy"

    def _proposal(
        self,
        *,
        family: str,
        summary: str,
        overrides: dict[str, Any],
        rationale: str,
        origin_signals: list[str],
        objective: str,
        priority: float,
    ) -> ScoringProposal:
        return ScoringProposal(
            family=family,
            summary=summary,
            overrides=overrides,
            rationale=rationale,
            origin_signals=origin_signals,
            objective=objective,
            priority=priority,
            fingerprint=_json_fingerprint(overrides),
        )

    def _proposal_priority(self, family: str, objective: str, detail: str = "") -> float:
        family_priority = {
            "accuracy": {
                "local_search": 130.0,
                "tuner_compound": 120.0,
                "tuner_single": 110.0,
                "heuristic_accuracy": 100.0,
                "heuristic_ranking": 85.0,
                "heuristic_calibration": 70.0,
                "heuristic_latency": 50.0,
            },
            "latency": {
                "local_search": 135.0,
                "heuristic_latency": 120.0,
                "heuristic_calibration": 110.0,
                "heuristic_ranking": 95.0,
                "tuner_single": 100.0,
                "tuner_compound": 95.0,
                "heuristic_accuracy": 60.0,
            },
        }
        score = family_priority[objective].get(family, 10.0)
        if "license_plate" in detail:
            score += 2.0
        if "supported_boost" in detail:
            score += 1.5
        if "max_iterations" in detail:
            score += 1.0
        if "country_penalty" in detail:
            score += 1.0
        return score

    def _local_search_scoring_candidates(
        self,
        config: dict[str, Any],
        proposal_feedback: ProposalFeedback,
        *,
        objective: str,
    ) -> list[dict[str, Any]]:
        if not proposal_feedback.recent_wins:
            return []

        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for win_index, outcome in enumerate(proposal_feedback.recent_wins[:2], start=1):
            for change_index, change in enumerate(outcome.changes, start=1):
                path = str(change.get("path", "")).strip()
                current_value = _nested_value(config, path)
                old_value = change.get("old_value")
                new_value = change.get("new_value")
                if not path or not isinstance(current_value, (int, float)) or not isinstance(old_value, (int, float)) or not isinstance(new_value, (int, float)):
                    continue
                if abs(float(current_value) - float(new_value)) > 1e-6:
                    continue

                neighbor_values = _neighbor_values(path, float(old_value), float(new_value), float(current_value))
                for variant_index, neighbor in enumerate(neighbor_values, start=1):
                    if neighbor == current_value:
                        continue
                    if isinstance(current_value, int):
                        neighbor = int(round(neighbor))
                    fingerprint = _json_fingerprint(_override_for_path(path, neighbor))
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    direction = "continue" if abs(neighbor - float(old_value)) > abs(float(new_value) - float(old_value)) else "settle"
                    candidates.append(
                        {
                            "family": "local_search",
                            "summary": f"Explore a smaller local-search variant around the recent winning change to {path}.",
                            "overrides": _override_for_path(path, neighbor),
                            "rationale": f"local_search:{path}:{direction}:w{win_index}:v{variant_index}",
                            "origin_signals": [path, outcome.family],
                            "priority": self._proposal_priority("local_search", objective, path) - (win_index - 1) * 1.5 - (variant_index - 1) * 0.5,
                        }
                    )
        return candidates

    def _compound_tuner_overrides(self, adjustments: list[Any]) -> dict[str, Any]:
        compound: dict[str, Any] = {}
        seen_paths: set[str] = set()
        for adjustment in adjustments:
            root = adjustment.path.split(".", 1)[0]
            if root in seen_paths:
                continue
            compound = _deep_merge_json(compound, _override_for_path(adjustment.path, adjustment.new_value))
            seen_paths.add(root)
            if len(seen_paths) >= 2:
                break
        return compound

    def _heuristic_scoring_candidates(
        self,
        baseline_manifest: dict[str, Any],
        failure_reports: list[Any],
        config: dict[str, Any],
        *,
        objective: str,
    ) -> list[dict[str, Any]]:
        metrics = baseline_manifest.get("overall_metrics", {})
        failure_categories = {report.category for report in failure_reports}
        origin_signals = [report.category for report in failure_reports[:4]]
        candidates: list[dict[str, Any]] = []
        latency_ms = float(metrics.get("mean_latency_ms", 0.0))
        calibration_error = float(metrics.get("ece", 0.0))
        city_accuracy = float(metrics.get("city_accuracy", 0.0))

        def add_candidate(
            *,
            family: str,
            summary: str,
            overrides: dict[str, Any],
            rationale: str,
            priority: float | None = None,
        ) -> None:
            candidates.append(
                {
                    "family": family,
                    "summary": summary,
                    "overrides": overrides,
                    "rationale": rationale,
                    "origin_signals": origin_signals or [family],
                    "priority": priority if priority is not None else self._proposal_priority(family, objective, rationale),
                }
            )

        if (
            failure_categories.intersection({"city_miss", "high_confidence_wrong", "overconfident"})
            or city_accuracy < 0.95
        ):
            current = float(config["source_confidence"]["license_plate"])
            for idx, lowered in enumerate(_stepped_values(current, [current - 0.2, current - 0.4], minimum=0.2), start=1):
                add_candidate(
                    family="heuristic_accuracy",
                    summary="Reduce license plate confidence to limit noisy OCR-driven city errors.",
                    overrides={"source_confidence": {"license_plate": lowered}},
                    rationale=f"heuristic_accuracy:license_plate_down:v{idx}",
                    priority=self._proposal_priority("heuristic_accuracy", objective, "license_plate") - (idx - 1) * 0.5,
                )
            street_sign = float(config["source_confidence"]["street_sign"])
            for idx, lowered in enumerate(_stepped_values(street_sign, [street_sign - 0.05, street_sign - 0.1], minimum=0.55), start=1):
                add_candidate(
                    family="heuristic_accuracy",
                    summary="Slightly reduce street-sign confidence to avoid over-weighting ambiguous OCR clues.",
                    overrides={"source_confidence": {"street_sign": lowered}},
                    rationale=f"heuristic_accuracy:street_sign_down:v{idx}",
                    priority=self._proposal_priority("heuristic_accuracy", objective, "street_sign") - (idx - 1) * 0.5,
                )

        if (
            failure_categories.intersection({"wrong_country", "high_confidence_wrong"})
            and float(config["country_penalty"]["penalty_factor"]) < 0.85
        ):
            current = float(config["country_penalty"]["penalty_factor"])
            for idx, stronger in enumerate(_stepped_values(current, [current + 0.05, current + 0.1], maximum=0.9), start=1):
                add_candidate(
                    family="heuristic_accuracy",
                    summary="Increase the wrong-country penalty to suppress confident cross-country misses.",
                    overrides={"country_penalty": {"penalty_factor": stronger}},
                    rationale=f"heuristic_accuracy:country_penalty_up:v{idx}",
                    priority=self._proposal_priority("heuristic_accuracy", objective, "country_penalty") - (idx - 1) * 0.5,
                )
            threshold = float(config["country_penalty"]["consensus_threshold_with_hint"])
            for idx, lowered in enumerate(_stepped_values(threshold, [threshold - 0.04], minimum=0.45), start=1):
                add_candidate(
                    family="heuristic_accuracy",
                    summary="Lower the hint-weighted consensus threshold so wrong-country alternatives are penalized earlier.",
                    overrides={"country_penalty": {"consensus_threshold_with_hint": lowered}},
                    rationale=f"heuristic_accuracy:consensus_threshold_with_hint_down:v{idx}",
                )

        if latency_ms > 35_000:
            if int(config["refinement"]["max_iterations"]) > 1:
                add_candidate(
                    family="heuristic_latency",
                    summary="Reduce refinement iterations to cut latency on already-strong cases.",
                    overrides={"refinement": {"max_iterations": 1}},
                    rationale="heuristic_latency:max_iterations_down",
                )
            min_sources = int(config["refinement"]["min_evidence_sources"])
            if min_sources > 2:
                add_candidate(
                    family="heuristic_latency",
                    summary="Require fewer evidence sources before skipping refinement to reduce unnecessary extra loops.",
                    overrides={"refinement": {"min_evidence_sources": 2}},
                    rationale="heuristic_latency:min_evidence_sources_down",
                    priority=self._proposal_priority("heuristic_latency", objective, "min_evidence_sources") - 0.5,
                )
            geo_agreement = float(config["refinement"]["min_geographic_agreement"])
            for idx, lowered in enumerate(_stepped_values(geo_agreement, [geo_agreement - 0.05, geo_agreement - 0.1], minimum=0.35), start=1):
                add_candidate(
                    family="heuristic_latency",
                    summary="Relax the refinement geographic-agreement threshold so high-quality cases exit earlier.",
                    overrides={"refinement": {"min_geographic_agreement": lowered}},
                    rationale=f"heuristic_latency:min_geographic_agreement_down:v{idx}",
                    priority=self._proposal_priority("heuristic_latency", objective, "min_geographic_agreement") - idx,
                )
            if int(config["refinement"]["max_iterations"]) > 1 and min_sources > 2:
                add_candidate(
                    family="heuristic_latency",
                    summary="Combine a lower refinement cap with a looser evidence threshold to attack latency directly.",
                    overrides={"refinement": {"max_iterations": 1, "min_evidence_sources": 2}},
                    rationale="heuristic_latency:max_iterations_and_min_sources_down",
                    priority=self._proposal_priority("heuristic_latency", objective, "max_iterations") + 0.25,
                )

        if calibration_error > 0.6:
            supported_boost = float(config["verification"]["supported_boost"])
            for idx, lowered in enumerate(_stepped_values(supported_boost, [supported_boost - 0.05, supported_boost - 0.1], minimum=1.0), start=1):
                add_candidate(
                    family="heuristic_calibration",
                    summary="Reduce verification confidence boosting to improve calibration.",
                    overrides={"verification": {"supported_boost": lowered}},
                    rationale=f"heuristic_calibration:supported_boost_down:v{idx}",
                    priority=self._proposal_priority("heuristic_calibration", objective, "supported_boost") - (idx - 1) * 0.5,
                )
            majority_boost = float(config["verification"]["majority_verified_boost"])
            for idx, lowered in enumerate(_stepped_values(majority_boost, [majority_boost - 0.05], minimum=1.0), start=1):
                add_candidate(
                    family="heuristic_calibration",
                    summary="Trim the majority-verification boost to keep confidence better calibrated.",
                    overrides={"verification": {"majority_verified_boost": lowered}},
                    rationale=f"heuristic_calibration:majority_verified_boost_down:v{idx}",
                    priority=self._proposal_priority("heuristic_calibration", objective, "majority_verified_boost") - 0.25,
                )
            partial_factor = float(config["verification"]["partial_verification_factor"])
            for idx, lowered in enumerate(_stepped_values(partial_factor, [partial_factor - 0.05, partial_factor - 0.1], minimum=0.65), start=1):
                add_candidate(
                    family="heuristic_calibration",
                    summary="Reduce the partial-verification factor so mixed evidence does not inflate final confidence.",
                    overrides={"verification": {"partial_verification_factor": lowered}},
                    rationale=f"heuristic_calibration:partial_verification_factor_down:v{idx}",
                    priority=self._proposal_priority("heuristic_calibration", objective, "partial_verification_factor") - idx,
                )

        if objective == "latency" or city_accuracy < 0.9:
            confidence_weight = float(config["candidate_ranking"]["confidence"])
            for idx, raised in enumerate(_stepped_values(confidence_weight, [confidence_weight + 0.05], maximum=0.6), start=1):
                add_candidate(
                    family="heuristic_ranking",
                    summary="Increase direct confidence weighting so strong candidates separate earlier.",
                    overrides={"candidate_ranking": {"confidence": raised}},
                    rationale=f"heuristic_ranking:confidence_up:v{idx}",
                )
            evidence_count = float(config["candidate_ranking"]["evidence_count"])
            for idx, lowered in enumerate(_stepped_values(evidence_count, [evidence_count - 0.02], minimum=0.01), start=1):
                add_candidate(
                    family="heuristic_ranking",
                    summary="Reduce evidence-count reward to limit urban/content-density bias.",
                    overrides={"candidate_ranking": {"evidence_count": lowered}},
                    rationale=f"heuristic_ranking:evidence_count_down:v{idx}",
                    priority=self._proposal_priority("heuristic_ranking", objective, "evidence_count") - 0.5,
                )
            source_diversity = float(config["candidate_ranking"]["source_diversity"])
            for idx, lowered in enumerate(_stepped_values(source_diversity, [source_diversity - 0.02], minimum=0.01), start=1):
                add_candidate(
                    family="heuristic_ranking",
                    summary="Reduce source-diversity reward to avoid over-favoring dense urban evidence clusters.",
                    overrides={"candidate_ranking": {"source_diversity": lowered}},
                    rationale=f"heuristic_ranking:source_diversity_down:v{idx}",
                    priority=self._proposal_priority("heuristic_ranking", objective, "source_diversity") - 0.75,
                )

        if objective == "latency":
            candidates.sort(key=lambda item: (-float(item["priority"]), item["family"], item["rationale"]))
        return candidates

    def _rank_scoring_proposals(
        self,
        proposals: list[ScoringProposal],
        *,
        max_candidates: int | None = None,
        current_base_fingerprint: str = "",
        proposal_feedback: ProposalFeedback | None = None,
    ) -> list[ScoringProposal]:
        proposal_feedback = proposal_feedback or ProposalFeedback()
        recent_outcomes = proposal_feedback.recent_outcomes
        same_base_outcomes = [
            outcome for outcome in recent_outcomes if outcome.base_config_fingerprint == current_base_fingerprint
        ]

        def adjusted_priority(proposal: ScoringProposal) -> float | None:
            score = proposal.priority
            changed_paths = _flatten_override_paths(proposal.overrides)
            family_outcomes = [outcome for outcome in same_base_outcomes if outcome.family == proposal.family]
            if family_outcomes:
                average_family_score = sum(outcome.score for outcome in family_outcomes) / len(family_outcomes)
                score += max(-10.0, min(4.0, average_family_score / 2.0))

            path_outcomes: list[ProposalOutcome] = []
            for path in changed_paths:
                path_outcomes.extend(
                    outcome for outcome in same_base_outcomes if path in outcome.changed_paths
                )
                if any(path in win.changed_paths for win in proposal_feedback.recent_wins):
                    score += 6.0
            if path_outcomes:
                average_score = sum(outcome.score for outcome in path_outcomes) / len(path_outcomes)
                average_latency_delta = sum(outcome.delta_latency_ms for outcome in path_outcomes) / len(path_outcomes)
                score += max(-14.0, min(4.0, average_score / 1.5))
                if average_latency_delta > 1500:
                    score -= min(12.0, average_latency_delta / 800.0)
                if len(path_outcomes) >= 2 and all(outcome.score <= 0 for outcome in path_outcomes) and average_latency_delta > 500:
                    return None

            return score

        scored: list[tuple[float, ScoringProposal]] = []
        for proposal in proposals:
            score = adjusted_priority(proposal)
            if score is None:
                continue
            scored.append((score, proposal))

        proposals = [proposal for _, proposal in sorted(scored, key=lambda item: (-item[0], item[1].family, item[1].fingerprint))]
        if not max_candidates or len(proposals) <= 1:
            return proposals[:max_candidates] if max_candidates else proposals

        selected: list[ScoringProposal] = []
        selected_fingerprints: set[str] = set()
        selected_families: set[str] = set()

        first = proposals[0]
        selected.append(first)
        selected_fingerprints.add(first.fingerprint)
        selected_families.add(first.family)

        if max_candidates > 1:
            diverse = next(
                (
                    proposal
                    for proposal in proposals[1:]
                    if proposal.fingerprint not in selected_fingerprints and proposal.family not in selected_families
                ),
                None,
            )
            if diverse is not None:
                selected.append(diverse)
                selected_fingerprints.add(diverse.fingerprint)
                selected_families.add(diverse.family)

        for proposal in proposals:
            if len(selected) >= max_candidates:
                break
            if proposal.fingerprint in selected_fingerprints:
                continue
            selected.append(proposal)
            selected_fingerprints.add(proposal.fingerprint)
        return selected

    def _collect_proposal_feedback(
        self,
        state: CampaignState,
        current_base_fingerprint: str,
    ) -> ProposalFeedback:
        recent_outcomes: list[ProposalOutcome] = []
        recent_wins: list[ProposalOutcome] = []

        for wave in reversed(state.waves[-6:]):
            experiment_dir = Path(str(wave.get("experiment_dir", "")))
            if not experiment_dir.exists():
                continue
            ranking_path = experiment_dir / "ranking.json"
            experiment_path = experiment_dir / "experiment.json"
            if not ranking_path.exists() or not experiment_path.exists():
                continue

            ranking = json.loads(ranking_path.read_text())
            experiment = json.loads(experiment_path.read_text())
            baseline_metrics = ranking.get("baseline_metrics", {})
            ranking_by_id = {
                entry.get("candidate_id"): entry
                for entry in ranking.get("rankings", [])
                if entry.get("candidate_id")
            }

            base_config_path = str(wave.get("base_config_path", "")).strip()
            base_config = self._current_scoring_config(base_config_path or None).model_dump()
            base_fingerprint = str(wave.get("base_config_fingerprint", ""))

            for candidate in experiment.get("candidates", []):
                selection_path = candidate.get("selection_path")
                patch_path = candidate.get("patch_path")
                if not selection_path or not patch_path:
                    continue
                selection_file = Path(selection_path)
                patch_file = Path(patch_path)
                if not selection_file.exists() or not patch_file.exists():
                    continue
                selection = json.loads(selection_file.read_text())
                rewrite = json.loads(patch_file.read_text())
                fingerprint = str(selection.get("proposal_fingerprint", "")).strip()
                family = str(selection.get("proposal_family", "")).strip()
                if not fingerprint or not family:
                    continue

                ranking_entry = ranking_by_id.get(candidate.get("candidate_id"))
                if ranking_entry is None:
                    continue
                overrides = rewrite.get("overrides")
                if not isinstance(overrides, dict) or not overrides:
                    continue
                changes = _override_changes(base_config, overrides)
                changed_paths = [str(change["path"]) for change in changes]
                metrics = ranking_entry.get("metrics") or {}
                outcome = ProposalOutcome(
                    family=family,
                    fingerprint=fingerprint,
                    rationale=str(selection.get("rationale", "")),
                    overrides=overrides,
                    changed_paths=changed_paths,
                    changes=changes,
                    status=str(ranking_entry.get("status", "")),
                    score=float(ranking_entry.get("score", 0.0)),
                    delta_accuracy_25km=_metric(metrics, "accuracy_25km") - _metric(baseline_metrics, "accuracy_25km"),
                    delta_country_accuracy=_metric(metrics, "country_accuracy") - _metric(baseline_metrics, "country_accuracy"),
                    delta_city_accuracy=_metric(metrics, "city_accuracy") - _metric(baseline_metrics, "city_accuracy"),
                    delta_ece=_metric(metrics, "ece") - _metric(baseline_metrics, "ece"),
                    delta_latency_ms=_metric(metrics, "mean_latency_ms") - _metric(baseline_metrics, "mean_latency_ms"),
                    base_config_fingerprint=base_fingerprint,
                )
                recent_outcomes.append(outcome)
                if (
                    base_fingerprint == current_base_fingerprint
                    and outcome.status == "accepted"
                    and outcome.score > 0
                ):
                    recent_wins.append(outcome)
                elif outcome.status == "accepted" and outcome.score > 0 and len(recent_wins) < 2:
                    recent_wins.append(outcome)

        return ProposalFeedback(
            recent_outcomes=recent_outcomes[:18],
            recent_wins=recent_wins[:4],
        )

    def _create_worktree(self, worktree_dir: Path, candidate_id: str) -> Path:
        if self._repo_has_local_changes():
            logger.info(
                "Local workspace has uncommitted changes; using a filesystem snapshot for {}",
                candidate_id,
            )
            return self._create_snapshot_checkout(candidate_id)
        if worktree_dir.exists():
            shutil.rmtree(worktree_dir)
        try:
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree_dir), "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            return worktree_dir
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            if "Operation not permitted" not in stderr and "Permission denied" not in stderr:
                raise
            logger.warning(
                "git worktree add failed for {} ({}); falling back to a filesystem snapshot",
                candidate_id,
                stderr or exc,
            )
            return self._create_snapshot_checkout(candidate_id)

    def _repo_has_local_changes(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return True
        return bool(result.stdout.strip())

    def _create_snapshot_checkout(self, candidate_id: str) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        snapshot_root = Path(tempfile.mkdtemp(prefix=f"open_geo_spy_{candidate_id}_"))
        shutil.copytree(
            repo_root,
            snapshot_root,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
                "node_modules",
                "data/traces",
                "data/improve",
            ),
        )
        return snapshot_root

    def _remove_worktree(self, worktree_dir: Path) -> None:
        if not worktree_dir.exists():
            return
        git_dir = worktree_dir / ".git"
        if git_dir.exists():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_dir)],
                check=False,
                capture_output=True,
                text=True,
            )
            return
        shutil.rmtree(worktree_dir, ignore_errors=True)

    def _run_suite_subprocess(
        self,
        worktree_dir: Path,
        suite_path: str,
        output_dir: Path,
        *,
        label: str,
        quality: str,
        max_concurrent: int,
        capability_snapshot_path: Path | None = None,
        baseline_lineage_id: str = "",
        extra_env: dict[str, str] | None = None,
    ) -> None:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        env = None
        if extra_env:
            env = dict(os.environ)
            env.update(extra_env)
        command = [
            sys.executable,
            "-m",
            "src.improve.worker",
            "--suite",
            suite_path,
            "--output-dir",
            str(output_dir),
            "--label",
            label,
            "--quality",
            quality,
            "--max-concurrent",
            str(max_concurrent),
        ]
        if capability_snapshot_path:
            command.extend(["--capability-snapshot", str(capability_snapshot_path)])
        if baseline_lineage_id:
            command.extend(["--baseline-lineage-id", baseline_lineage_id])
        try:
            subprocess.run(
                command,
                cwd=str(worktree_dir),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(
                f"Candidate worker failed for {label}: {detail or exc}"
            ) from exc

    def _judge_run(self, run_manifest_path: str | Path) -> dict[str, Any]:
        judge = TraceQualityJudge(self._client, self.settings.improvement.judge_model)
        summary = asyncio.run(judge.judge_run(run_manifest_path))
        run_manifest_path = Path(run_manifest_path)
        with open(run_manifest_path) as f:
            manifest = json.load(f)
        manifest["judge_summary"] = summary.to_dict()
        with open(run_manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest["judge_summary"]

    def _write_experiment_state(self, experiment_dir: Path, state: dict[str, Any]) -> None:
        with open(experiment_dir / "experiment.json", "w") as f:
            json.dump(state, f, indent=2)

    def _write_campaign_state(self, campaign_dir: Path, state: CampaignState) -> None:
        with open(campaign_dir / "campaign.json", "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def _suite_manifest_ref(self, output_manifest: Path, suite_path: Path) -> str:
        """Store manifest references relative to the suite file when possible."""
        output_manifest = output_manifest.resolve()
        suite_dir = suite_path.resolve().parent
        return relpath(output_manifest, suite_dir)

    def _replace_candidate(self, state: dict[str, Any], record: CandidateRecord) -> None:
        state["candidates"] = [
            record.to_dict() if candidate["candidate_id"] == record.candidate_id else candidate
            for candidate in state["candidates"]
        ]

    def _load_sample_results(self, path: Path) -> list[SampleResult]:
        samples: list[SampleResult] = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                samples.append(SampleResult(
                    image_path=row["image_path"],
                    pred_lat=row.get("pred_lat"),
                    pred_lon=row.get("pred_lon"),
                    pred_country=row.get("pred_country", ""),
                    pred_city=row.get("pred_city", ""),
                    pred_confidence=row.get("pred_confidence", 0.0),
                    gt_lat=row.get("gt_lat", 0.0),
                    gt_lon=row.get("gt_lon", 0.0),
                    gt_country=row.get("gt_country", ""),
                    gt_city=row.get("gt_city", ""),
                    difficulty=row.get("difficulty", "medium"),
                    urban_rural=row.get("urban_rural", ""),
                    tags=row.get("tags", []),
                    cost_usd=row.get("cost_usd", 0.0),
                    latency_ms=row.get("latency_ms", 0.0),
                    tokens=row.get("tokens", 0),
                    session_id=row.get("session_id", ""),
                    trace_path=row.get("trace_path", ""),
                    candidate_count=row.get("candidate_count", 0),
                    reasoning=row.get("reasoning", ""),
                    prediction=row.get("prediction", {}),
                    trace_anomalies=row.get("trace_anomalies", []),
                    benchmark_source=row.get("benchmark_source", ""),
                ))
        return samples


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return text or "improve"


def _candidate_ordinal(candidate_id: str) -> int:
    match = re.search(r"(\d+)$", candidate_id)
    if not match:
        return 1
    return max(1, int(match.group(1)))


def _deep_merge_json(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_json(merged[key], value)
        else:
            merged[key] = value
    return merged


def _override_for_path(path: str, value: Any) -> dict[str, Any]:
    parts = path.split(".")
    nested: dict[str, Any] = value
    for part in reversed(parts):
        nested = {part: nested}
    return nested


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _stepped_values(
    current: float,
    targets: list[float],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[float]:
    values: list[float] = []
    seen: set[float] = set()
    for raw in targets:
        value = raw
        if minimum is not None:
            value = max(minimum, value)
        if maximum is not None:
            value = min(maximum, value)
        value = round(value, 2)
        if abs(value - current) < 1e-6 or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _nested_value(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _flatten_override_paths(overrides: dict[str, Any], prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, value in overrides.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            paths.extend(_flatten_override_paths(value, path))
        else:
            paths.append(path)
    return paths


def _override_changes(base_config: dict[str, Any], overrides: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for path in _flatten_override_paths(overrides):
        changes.append(
            {
                "path": path,
                "old_value": _nested_value(base_config, path),
                "new_value": _nested_value(overrides, path),
            }
        )
    return changes


def _neighbor_values(path: str, old_value: float, new_value: float, current_value: float) -> list[float]:
    delta = current_value - old_value
    if abs(delta) < 1e-6:
        return []
    magnitude = max(0.01, round(abs(delta) / 2.0, 2))
    minimum, maximum = _path_bounds(path, current_value)
    candidates = _stepped_values(
        current_value,
        [
            current_value + magnitude if delta > 0 else current_value - magnitude,
            current_value - magnitude if delta > 0 else current_value + magnitude,
        ],
        minimum=minimum,
        maximum=maximum,
    )
    return candidates


def _path_bounds(path: str, current_value: float) -> tuple[float | None, float | None]:
    bounds: dict[str, tuple[float | None, float | None]] = {
        "verification.supported_boost": (1.0, 1.3),
        "verification.majority_verified_boost": (1.0, 1.3),
        "verification.partial_verification_factor": (0.5, 1.0),
        "source_confidence.license_plate": (0.2, 0.95),
        "source_confidence.street_sign": (0.4, 0.95),
        "country_penalty.penalty_factor": (0.4, 0.95),
        "country_penalty.consensus_threshold_with_hint": (0.3, 0.8),
        "candidate_ranking.confidence": (0.2, 0.7),
        "candidate_ranking.evidence_count": (0.01, 0.35),
        "candidate_ranking.source_diversity": (0.01, 0.35),
        "refinement.min_geographic_agreement": (0.2, 0.8),
    }
    return bounds.get(path, (0.0 if current_value >= 0 else None, None))
