"""Benchmark suite manifests and import helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.eval.dataset import EvalDataset, GroundTruthSample
from src.improve.trace_analysis import analyze_trace_file, extract_trace_context


@dataclass
class BenchmarkDatasetSpec:
    """A dataset entry in a benchmark suite manifest."""

    dataset_id: str
    manifest_path: str
    weight: float = 1.0
    protected: bool = False
    optional: bool = False
    expected_sample_count: int | None = None
    source_label: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "manifest_path": self.manifest_path,
            "weight": self.weight,
            "protected": self.protected,
            "optional": self.optional,
            "expected_sample_count": self.expected_sample_count,
            "source_label": self.source_label,
            "tags": self.tags,
            "notes": self.notes,
        }

    def resolve_manifest_path(self, base_dir: str | Path) -> Path:
        path = Path(self.manifest_path)
        if not path.is_absolute():
            path = Path(base_dir) / path
        return path.resolve()

    def load_dataset(self, base_dir: str | Path) -> EvalDataset:
        path = self.resolve_manifest_path(base_dir)
        if path.suffix == ".csv":
            return EvalDataset.from_csv(path, name=self.dataset_id)
        return EvalDataset.from_manifest(path)


@dataclass
class BenchmarkSuite:
    """A collection of benchmark datasets and gating hints."""

    name: str
    datasets: list[BenchmarkDatasetSpec] = field(default_factory=list)
    description: str = ""
    protected_tags: list[str] = field(default_factory=lambda: ["regression", "found_then_lost"])
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "protected_tags": self.protected_tags,
            "metadata": self.metadata,
            "datasets": [dataset.to_dict() for dataset in self.datasets],
        }

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkSuite":
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        datasets = [BenchmarkDatasetSpec(**item) for item in data.get("datasets", [])]
        return cls(
            name=data.get("name", path.stem),
            datasets=datasets,
            description=data.get("description", ""),
            protected_tags=data.get("protected_tags", ["regression", "found_then_lost"]),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


def _coerce_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value)]


def _sample_field(sample: GroundTruthSample, field_name: str) -> str:
    value = getattr(sample, field_name, None)
    if value not in (None, "", []):
        if isinstance(value, list):
            return "|".join(sorted(str(item) for item in value))
        return str(value)

    metadata_value = sample.metadata.get(field_name)
    if metadata_value not in (None, "", []):
        if isinstance(metadata_value, list):
            return "|".join(sorted(str(item) for item in metadata_value))
        return str(metadata_value)

    if field_name == "country" and sample.country:
        return sample.country
    if field_name == "city" and sample.city:
        return sample.city
    return "__missing__"


def _stratified_subset(
    samples: list[GroundTruthSample],
    *,
    limit: int | None = None,
    seed: int = 42,
    stratify_by: list[str] | None = None,
) -> list[GroundTruthSample]:
    if limit is None or limit <= 0 or len(samples) <= limit:
        return list(samples)

    import random

    rng = random.Random(seed)
    ordered = sorted(samples, key=lambda item: item.image_path)
    requested_fields = [field for field in (stratify_by or []) if field]

    if not requested_fields:
        shuffled = list(ordered)
        rng.shuffle(shuffled)
        return shuffled[:limit]

    usable_fields = []
    for field_name in requested_fields:
        values = {_sample_field(sample, field_name) for sample in ordered}
        if len(values - {"__missing__"}) > 1:
            usable_fields.append(field_name)
    if not usable_fields:
        shuffled = list(ordered)
        rng.shuffle(shuffled)
        return shuffled[:limit]

    buckets: dict[tuple[str, ...], list[GroundTruthSample]] = {}
    for sample in ordered:
        key = tuple(_sample_field(sample, field_name) for field_name in usable_fields)
        buckets.setdefault(key, []).append(sample)

    for bucket_samples in buckets.values():
        rng.shuffle(bucket_samples)

    selected: list[GroundTruthSample] = []
    bucket_keys = sorted(buckets.keys())
    while len(selected) < limit:
        progressed = False
        for bucket_key in bucket_keys:
            bucket = buckets[bucket_key]
            if bucket:
                selected.append(bucket.pop(0))
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break

    return selected[:limit]


def _normalize_row(row: dict[str, Any], base_dir: Path) -> GroundTruthSample:
    image_value = (
        row.get("image_path")
        or row.get("image")
        or row.get("file_path")
        or row.get("filepath")
        or row.get("filename")
    )
    if not image_value:
        raise ValueError(f"Row missing image path: {row}")
    image_path = Path(str(image_value))
    if not image_path.is_absolute():
        image_path = (base_dir / image_path).resolve()

    latitude = row.get("latitude", row.get("lat", row.get("actual_lat", row.get("gt_lat"))))
    longitude = row.get("longitude", row.get("lon", row.get("actual_lon", row.get("gt_lon"))))
    if latitude is None or longitude is None:
        raise ValueError(f"Row missing latitude/longitude: {row}")

    return GroundTruthSample(
        image_path=str(image_path),
        latitude=float(latitude),
        longitude=float(longitude),
        country=str(row.get("country", row.get("country_name", row.get("actual_country", "")))),
        city=str(row.get("city", row.get("city_name", row.get("actual_city", "")))),
        region=str(row.get("region", row.get("state", ""))),
        difficulty=str(row.get("difficulty", row.get("split", "medium"))),
        urban_rural=str(row.get("urban_rural", row.get("environment", ""))),
        tags=_coerce_tags(row.get("tags", row.get("benchmark_tags", row.get("split", "")))),
        metadata={
            key: value
            for key, value in row.items()
            if key
            not in {
                "image_path",
                "image",
                "file_path",
                "filepath",
                "filename",
                "latitude",
                "lat",
                "actual_lat",
                "gt_lat",
                "longitude",
                "lon",
                "actual_lon",
                "gt_lon",
                "country",
                "country_name",
                "actual_country",
                "city",
                "city_name",
                "actual_city",
                "region",
                "state",
                "difficulty",
                "split",
                "urban_rural",
                "environment",
                "tags",
                "benchmark_tags",
            }
        },
    )


def import_dataset(
    source_path: str | Path,
    output_manifest: str | Path,
    *,
    adapter: str = "auto",
    dataset_name: str = "",
    description: str = "",
    limit: int | None = None,
    seed: int = 42,
    stratify_by: list[str] | None = None,
    source_label: str = "",
) -> Path:
    """Normalize local dataset formats into the repo's manifest schema."""
    source_path = Path(source_path)
    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    selected_adapter = adapter.lower()
    if selected_adapter == "auto":
        if source_path.suffix == ".csv":
            selected_adapter = "csv"
        elif source_path.suffix == ".jsonl":
            selected_adapter = "jsonl"
        elif source_path.suffix == ".parquet":
            selected_adapter = "parquet"
        else:
            selected_adapter = "manifest"

    if selected_adapter == "manifest":
        dataset = EvalDataset.from_manifest(source_path)
        dataset.description = description or dataset.description
        dataset.name = dataset_name or dataset.name
    elif selected_adapter == "csv":
        dataset = EvalDataset.from_csv(source_path, name=dataset_name or source_path.stem)
        dataset.description = description
    else:
        rows: list[dict[str, Any]]
        if selected_adapter in {"jsonl", "geobench", "osv5m", "geovistabench"}:
            rows = []
            with open(source_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        elif selected_adapter == "parquet":
            try:
                import pandas as pd
            except ImportError as exc:
                raise ValueError("Parquet import requires pandas + pyarrow installed") from exc
            rows = pd.read_parquet(source_path).to_dict(orient="records")
        else:
            raise ValueError(f"Unsupported dataset adapter: {selected_adapter}")

        samples = [_normalize_row(row, source_path.parent) for row in rows]
        dataset = EvalDataset(
            name=dataset_name or source_path.stem,
            description=description,
            samples=samples,
        )

    if source_label:
        for sample in dataset.samples:
            sample.metadata["source_label"] = source_label

    subset = _stratified_subset(
        dataset.samples,
        limit=limit,
        seed=seed,
        stratify_by=stratify_by,
    )
    dataset.samples = subset
    dataset.to_manifest(output_manifest)
    return output_manifest


def add_dataset_to_suite(
    suite_path: str | Path,
    dataset_spec: BenchmarkDatasetSpec,
) -> Path:
    """Append or replace a dataset entry in a suite manifest."""
    suite_path = Path(suite_path)
    if suite_path.exists():
        suite = BenchmarkSuite.load(suite_path)
    else:
        suite = BenchmarkSuite(name=suite_path.stem)

    suite.datasets = [
        existing
        for existing in suite.datasets
        if existing.dataset_id != dataset_spec.dataset_id
    ]
    suite.datasets.append(dataset_spec)
    suite.datasets.sort(key=lambda item: item.dataset_id)
    suite.save(suite_path)
    return suite_path


def build_regression_dataset_from_traces(
    traces_dir: str | Path,
    output_manifest: str | Path,
    *,
    max_cases: int = 50,
) -> Path:
    """Create a regression dataset from traces with ground truth and notable anomalies."""
    traces_dir = Path(traces_dir)
    samples: list[GroundTruthSample] = []

    for trace_path in sorted(traces_dir.rglob("*.jsonl")):
        context = extract_trace_context(trace_path)
        result = context.get("result") or {}
        header = context.get("header") or {}
        ground_truth = result.get("ground_truth") or {}
        image_path = header.get("image_path")
        if not ground_truth or not image_path:
            continue

        diagnostics = analyze_trace_file(trace_path)
        gcd_km = diagnostics.final_gcd_km if diagnostics.final_gcd_km is not None else float("inf")
        if not diagnostics.flags and gcd_km <= 150:
            continue

        metadata = {
            "source_trace": str(trace_path),
            "trace_anomalies": diagnostics.flags,
            "final_gcd_km": diagnostics.final_gcd_km,
        }
        sample = GroundTruthSample(
            image_path=image_path,
            latitude=float(ground_truth.get("latitude", 0.0)),
            longitude=float(ground_truth.get("longitude", 0.0)),
            country=str(ground_truth.get("country", "")),
            city=str(ground_truth.get("city", "")),
            region=str(ground_truth.get("region", "")),
            difficulty=str(ground_truth.get("difficulty", "hard")),
            urban_rural=str(ground_truth.get("urban_rural", "")),
            tags=["regression", *diagnostics.flags],
            metadata=metadata,
        )
        samples.append(sample)
        if len(samples) >= max_cases:
            break

    dataset = EvalDataset(
        name="trace_regressions",
        description="Failures and late-stage regressions mined from saved traces",
        samples=samples,
    )
    dataset.to_manifest(output_manifest)
    return Path(output_manifest)
