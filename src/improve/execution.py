"""Suite execution and artifact collation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.config.settings import get_scoring_config, get_settings
from src.eval.metrics import EvalMetrics
from src.eval.runner import EvalRunner
from src.improve.suite import BenchmarkSuite
from src.improve.trace_analysis import analyze_trace_file


def _git_value(args: list[str], cwd: str | Path) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


async def run_suite(
    suite_path: str | Path,
    output_dir: str | Path,
    *,
    label: str = "",
    quality: str = "balanced",
    max_concurrent: int = 3,
    capability_snapshot: dict[str, Any] | None = None,
    scoring_config_path: str | None = None,
    baseline_lineage_id: str = "",
) -> dict[str, Any]:
    """Run every dataset in a suite and write reproducible artifacts."""
    suite_path = Path(suite_path).resolve()
    suite = BenchmarkSuite.load(suite_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    dataset_runs = []
    missing_optional_datasets = []
    active_scoring_config_path = ""
    active_scoring_config_fingerprint = ""
    with _scoring_config_context(scoring_config_path), _capability_snapshot_context(capability_snapshot):
        for dataset_spec in suite.datasets:
            manifest_path = dataset_spec.resolve_manifest_path(suite_path.parent)
            if not manifest_path.exists():
                if dataset_spec.optional:
                    logger.warning("Skipping optional dataset '{}' (missing {})", dataset_spec.dataset_id, manifest_path)
                    missing_entry = {
                        "dataset_id": dataset_spec.dataset_id,
                        "manifest_path": dataset_spec.manifest_path,
                        "weight": dataset_spec.weight,
                        "protected": dataset_spec.protected,
                        "optional": True,
                        "expected_sample_count": dataset_spec.expected_sample_count,
                        "source_label": dataset_spec.source_label,
                        "tags": dataset_spec.tags,
                        "status": "missing_optional",
                        "artifact_dir": None,
                        "metrics": None,
                        "sample_results_path": None,
                    }
                    dataset_runs.append(missing_entry)
                    missing_optional_datasets.append(dataset_spec.dataset_id)
                    continue
                raise FileNotFoundError(f"Required dataset manifest not found: {manifest_path}")

            dataset = dataset_spec.load_dataset(suite_path.parent)
            runner = EvalRunner(
                label=f"{label}:{dataset_spec.dataset_id}" if label else dataset_spec.dataset_id,
                max_concurrent=max_concurrent,
                quality=quality,
            )
            metrics = await runner.run(dataset)
            for result in metrics.results:
                result.benchmark_source = dataset_spec.dataset_id
                if result.trace_path:
                    diagnostics = analyze_trace_file(result.trace_path)
                    result.trace_anomalies = diagnostics.flags
            dataset_dir = output_dir / "datasets" / dataset_spec.dataset_id
            runner.save_artifacts(
                dataset,
                metrics,
                dataset_dir,
                metadata={
                    "dataset_id": dataset_spec.dataset_id,
                    "weight": dataset_spec.weight,
                    "protected": dataset_spec.protected,
                    "tags": dataset_spec.tags,
                },
            )
            dataset_runs.append(
                {
                    "dataset_id": dataset_spec.dataset_id,
                    "manifest_path": dataset_spec.manifest_path,
                    "weight": dataset_spec.weight,
                    "protected": dataset_spec.protected,
                    "optional": dataset_spec.optional,
                    "expected_sample_count": dataset_spec.expected_sample_count,
                    "source_label": dataset_spec.source_label,
                    "tags": dataset_spec.tags,
                    "status": "completed",
                    "artifact_dir": str(dataset_dir),
                    "metrics": metrics.summary(),
                    "sample_results_path": str(dataset_dir / "sample_results.jsonl"),
                }
            )
            all_results.extend(metrics.results)
        scoring_config = get_scoring_config()
        active_scoring_config_path = _active_scoring_config_path(scoring_config_path)
        active_scoring_config_fingerprint = _json_fingerprint(scoring_config.model_dump())

    overall_metrics = EvalMetrics(results=all_results)
    sample_results_path = output_dir / "sample_results.jsonl"
    with open(sample_results_path, "w") as f:
        for result in all_results:
            f.write(json.dumps(result.to_dict(), default=str) + "\n")

    manifest = {
        "suite": suite.name,
        "suite_path": str(suite_path),
        "run_dir": str(output_dir),
        "label": label or suite.name,
        "quality": quality,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "sha": _git_value(["git", "rev-parse", "HEAD"], Path.cwd()),
            "branch": _git_value(["git", "branch", "--show-current"], Path.cwd()),
        },
        "overall_metrics": overall_metrics.summary(),
        "datasets": dataset_runs,
        "sample_results_path": str(sample_results_path),
        "scoring_config_path": active_scoring_config_path,
        "scoring_config_fingerprint": active_scoring_config_fingerprint,
        "baseline_lineage_id": baseline_lineage_id,
        "protected_tags": suite.protected_tags,
        "trace_anomaly_counts": _count_trace_anomalies(all_results),
        "missing_optional_datasets": missing_optional_datasets,
        "capability_snapshot": capability_snapshot or {},
        "status": "completed",
    }
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _count_trace_anomalies(results: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for flag in result.trace_anomalies:
            counts[flag] = counts.get(flag, 0) + 1
    return counts


def _active_scoring_config_path(explicit_path: str | None) -> str:
    if explicit_path:
        return str(Path(explicit_path).resolve())
    env_path = os.environ.get("SCORING_CONFIG_PATH", "").strip()
    if env_path:
        return str(Path(env_path).resolve())
    return ""


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def run_suite_sync(
    suite_path: str | Path,
    output_dir: str | Path,
    *,
    label: str = "",
    quality: str = "balanced",
    max_concurrent: int = 3,
    capability_snapshot: dict[str, Any] | None = None,
    scoring_config_path: str | None = None,
    baseline_lineage_id: str = "",
) -> dict[str, Any]:
    """Synchronous wrapper for CLI and subprocess entrypoints."""
    return asyncio.run(
        run_suite(
            suite_path,
            output_dir,
            label=label,
            quality=quality,
            max_concurrent=max_concurrent,
            capability_snapshot=capability_snapshot,
            scoring_config_path=scoring_config_path,
            baseline_lineage_id=baseline_lineage_id,
        )
    )


def build_improvement_capability_snapshot(settings: Any | None = None) -> dict[str, Any]:
    """Build a stable runtime profile for improvement-loop runs."""
    settings = settings or get_settings()
    requested = {
        "geoclip": bool(settings.ml.enable_geoclip),
        "streetclip": bool(settings.ml.enable_streetclip),
        "visual_verification": bool(settings.ml.enable_visual_verification),
    }
    effective = dict(requested)
    disabled: list[dict[str, str]] = []

    if effective["visual_verification"]:
        effective["visual_verification"] = False
        disabled.append(
            {
                "feature": "visual_verification",
                "reason": "disabled_for_improvement_reliability_profile",
            }
        )

    for feature, module_name in (("geoclip", "geoclip"), ("streetclip", "transformers")):
        if not effective.get(feature):
            continue
        try:
            __import__(module_name)
        except Exception as exc:
            effective[feature] = False
            disabled.append({"feature": feature, "reason": f"{module_name}_unavailable: {exc}"})

    return {
        "profile": "improve_reliable_v1",
        "requested": requested,
        "effective": effective,
        "disabled": disabled,
        "llm_api_configured": bool(settings.llm.api_key),
        "search_providers": list(settings.geo.search_providers),
    }


@contextmanager
def _capability_snapshot_context(capability_snapshot: dict[str, Any] | None):
    if not capability_snapshot:
        yield
        return

    effective = capability_snapshot.get("effective", {})
    overrides = {
        "ML__ENABLE_GEOCLIP": str(bool(effective.get("geoclip", True))).lower(),
        "ML__ENABLE_STREETCLIP": str(bool(effective.get("streetclip", True))).lower(),
        "ML__ENABLE_VISUAL_VERIFICATION": str(bool(effective.get("visual_verification", True))).lower(),
    }
    previous = {key: os.environ.get(key) for key in overrides}
    for key, value in overrides.items():
        os.environ[key] = value
    _reset_runtime_settings()
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _reset_runtime_settings()


@contextmanager
def _scoring_config_context(scoring_config_path: str | None):
    previous = os.environ.get("SCORING_CONFIG_PATH")
    if scoring_config_path:
        os.environ["SCORING_CONFIG_PATH"] = str(Path(scoring_config_path).resolve())
    else:
        os.environ.pop("SCORING_CONFIG_PATH", None)
    get_scoring_config.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SCORING_CONFIG_PATH", None)
        else:
            os.environ["SCORING_CONFIG_PATH"] = previous
        get_scoring_config.cache_clear()


def _reset_runtime_settings() -> None:
    get_settings.cache_clear()
    get_scoring_config.cache_clear()
    try:
        from src.models.registry import ModelRegistry

        ModelRegistry._instances.clear()
    except Exception:
        pass
