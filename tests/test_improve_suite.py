"""Tests for benchmark suite normalization."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

from src.config.settings import get_settings
from src.eval.metrics import EvalMetrics, SampleResult
from src.eval.dataset import EvalDataset
from src.improve.execution import build_improvement_capability_snapshot, run_suite_sync
from src.improve.suite import BenchmarkDatasetSpec, BenchmarkSuite, import_dataset
from src.scoring.config import ScoringConfig


def test_import_dataset_from_jsonl(tmp_path) -> None:
    source = tmp_path / "geobench.jsonl"
    rows = [
        {
            "image_path": "images/sample.jpg",
            "latitude": 48.8584,
            "longitude": 2.2945,
            "country": "France",
            "city": "Paris",
            "tags": ["landmark", "europe"],
        }
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows))

    output_manifest = tmp_path / "normalized" / "manifest.json"
    import_dataset(source, output_manifest, adapter="jsonl", dataset_name="public_suite")

    dataset = EvalDataset.from_manifest(output_manifest)
    assert dataset.name == "public_suite"
    assert len(dataset.samples) == 1
    assert dataset.samples[0].country == "France"
    assert dataset.samples[0].city == "Paris"
    assert "landmark" in dataset.samples[0].tags


def test_import_dataset_subset_is_deterministic(tmp_path) -> None:
    source = tmp_path / "osv5m.jsonl"
    rows = []
    for idx in range(8):
        rows.append(
            {
                "image_path": f"images/sample_{idx}.jpg",
                "latitude": 40.0 + idx,
                "longitude": -70.0 - idx,
                "country": "France" if idx % 2 == 0 else "Italy",
                "city": f"City {idx}",
                "difficulty": "hard" if idx < 4 else "medium",
            }
        )
    source.write_text("\n".join(json.dumps(row) for row in rows))

    output_a = tmp_path / "normalized_a" / "manifest.json"
    output_b = tmp_path / "normalized_b" / "manifest.json"
    import_dataset(
        source,
        output_a,
        adapter="jsonl",
        dataset_name="subset_a",
        limit=4,
        seed=7,
        stratify_by=["difficulty", "country"],
        source_label="OSV5M",
    )
    import_dataset(
        source,
        output_b,
        adapter="jsonl",
        dataset_name="subset_b",
        limit=4,
        seed=7,
        stratify_by=["difficulty", "country"],
        source_label="OSV5M",
    )

    dataset_a = EvalDataset.from_manifest(output_a)
    dataset_b = EvalDataset.from_manifest(output_b)
    assert [sample.image_path for sample in dataset_a.samples] == [sample.image_path for sample in dataset_b.samples]
    assert len(dataset_a.samples) == 4
    assert all(sample.metadata.get("source_label") == "OSV5M" for sample in dataset_a.samples)


def test_run_suite_skips_missing_optional_dataset(tmp_path, monkeypatch) -> None:
    smoke_manifest = tmp_path / "sample" / "manifest.json"
    smoke_manifest.parent.mkdir(parents=True)
    smoke_manifest.write_text(
        json.dumps(
            {
                "name": "smoke",
                "samples": [
                    {
                        "image_path": "images/sample.jpg",
                        "latitude": 48.8584,
                        "longitude": 2.2945,
                        "country": "France",
                        "city": "Paris",
                    }
                ],
            }
        )
    )

    suite_path = tmp_path / "suites" / "core.json"
    BenchmarkSuite(
        name="core",
        datasets=[
            BenchmarkDatasetSpec(dataset_id="smoke", manifest_path="../sample/manifest.json", protected=True),
            BenchmarkDatasetSpec(
                dataset_id="geovistabench_subset",
                manifest_path="../imports/geovistabench_subset/manifest.json",
                protected=True,
                optional=True,
            ),
        ],
    ).save(suite_path)

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, dataset):
            return EvalMetrics(
                results=[
                    SampleResult(
                        image_path=dataset.samples[0].image_path,
                        pred_lat=48.8584,
                        pred_lon=2.2945,
                        pred_country="France",
                        pred_city="Paris",
                        gt_lat=48.8584,
                        gt_lon=2.2945,
                        gt_country="France",
                        gt_city="Paris",
                    )
                ]
            )

        def save_artifacts(self, dataset, metrics, output_dir, metadata=None):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "sample_results.jsonl").write_text("")
            (output_dir / "metrics.json").write_text(json.dumps({"metrics": metrics.summary()}))
            (output_dir / "dataset_manifest.json").write_text(json.dumps({"dataset": dataset.name}))
            return {}

    monkeypatch.setattr("src.improve.execution.EvalRunner", FakeRunner)
    manifest = run_suite_sync(suite_path, tmp_path / "run")

    assert manifest["missing_optional_datasets"] == ["geovistabench_subset"]
    assert manifest["datasets"][1]["status"] == "missing_optional"


def test_run_suite_applies_capability_snapshot(tmp_path, monkeypatch, test_settings) -> None:
    manifest_path = tmp_path / "sample" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "name": "seed",
                "samples": [
                    {
                        "image_path": "images/sample.jpg",
                        "latitude": 48.8584,
                        "longitude": 2.2945,
                        "country": "France",
                        "city": "Paris",
                    }
                ],
            }
        )
    )

    suite_path = tmp_path / "suites" / "core.json"
    BenchmarkSuite(
        name="core",
        datasets=[BenchmarkDatasetSpec(dataset_id="seed", manifest_path="../sample/manifest.json", protected=True)],
    ).save(suite_path)

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, dataset):
            settings = get_settings()
            assert settings.ml.enable_visual_verification is False
            return EvalMetrics(
                results=[
                    SampleResult(
                        image_path=dataset.samples[0].image_path,
                        pred_lat=48.8584,
                        pred_lon=2.2945,
                        pred_country="France",
                        pred_city="Paris",
                        gt_lat=48.8584,
                        gt_lon=2.2945,
                        gt_country="France",
                        gt_city="Paris",
                    )
                ]
            )

        def save_artifacts(self, dataset, metrics, output_dir, metadata=None):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "sample_results.jsonl").write_text("")
            (output_dir / "metrics.json").write_text(json.dumps({"metrics": metrics.summary()}))
            (output_dir / "dataset_manifest.json").write_text(json.dumps({"dataset": dataset.name}))
            return {}

    monkeypatch.setattr("src.improve.execution.EvalRunner", FakeRunner)
    snapshot = build_improvement_capability_snapshot(test_settings)
    manifest = run_suite_sync(suite_path, tmp_path / "run", capability_snapshot=snapshot)

    assert manifest["capability_snapshot"]["profile"] == "improve_reliable_v1"
    assert manifest["capability_snapshot"]["effective"]["visual_verification"] is False


def test_run_suite_records_scoring_config_metadata(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "sample" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "name": "seed",
                "samples": [
                    {
                        "image_path": "images/sample.jpg",
                        "latitude": 48.8584,
                        "longitude": 2.2945,
                        "country": "France",
                        "city": "Paris",
                    }
                ],
            }
        )
    )

    suite_path = tmp_path / "suites" / "core.json"
    BenchmarkSuite(
        name="core",
        datasets=[BenchmarkDatasetSpec(dataset_id="seed", manifest_path="../sample/manifest.json", protected=True)],
    ).save(suite_path)

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, dataset):
            return EvalMetrics(
                results=[
                    SampleResult(
                        image_path=dataset.samples[0].image_path,
                        pred_lat=48.8584,
                        pred_lon=2.2945,
                        pred_country="France",
                        pred_city="Paris",
                        gt_lat=48.8584,
                        gt_lon=2.2945,
                        gt_country="France",
                        gt_city="Paris",
                    )
                ]
            )

        def save_artifacts(self, dataset, metrics, output_dir, metadata=None):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "sample_results.jsonl").write_text("")
            (output_dir / "metrics.json").write_text(json.dumps({"metrics": metrics.summary()}))
            (output_dir / "dataset_manifest.json").write_text(json.dumps({"dataset": dataset.name}))
            return {}

    monkeypatch.setattr("src.improve.execution.EvalRunner", FakeRunner)
    scoring_config_path = tmp_path / "campaign_config.json"
    scoring_config_path.write_text(json.dumps(ScoringConfig().model_dump(), indent=2))

    manifest = run_suite_sync(
        suite_path,
        tmp_path / "run",
        scoring_config_path=str(scoring_config_path),
        baseline_lineage_id="campaign-123",
    )
    payload = json.dumps(ScoringConfig().model_dump(), sort_keys=True, separators=(",", ":"))

    assert manifest["scoring_config_path"] == str(scoring_config_path.resolve())
    assert manifest["scoring_config_fingerprint"] == hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    assert manifest["baseline_lineage_id"] == "campaign-123"
