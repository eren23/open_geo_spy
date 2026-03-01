"""Ground truth dataset management for evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class GroundTruthSample:
    """A single labeled evaluation sample."""

    image_path: str
    latitude: float
    longitude: float
    country: str
    city: str = ""
    region: str = ""
    difficulty: str = "medium"  # easy, medium, hard
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "country": self.country,
            "city": self.city,
            "region": self.region,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class EvalDataset:
    """A collection of labeled ground truth samples."""

    name: str
    samples: list[GroundTruthSample] = field(default_factory=list)
    description: str = ""
    version: str = "1.0"

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> EvalDataset:
        """Load dataset from a manifest.json file.

        Expected format:
        {
            "name": "my_dataset",
            "description": "...",
            "version": "1.0",
            "samples": [
                {
                    "image_path": "images/001.jpg",
                    "latitude": 48.8566,
                    "longitude": 2.3522,
                    "country": "France",
                    "city": "Paris",
                    ...
                }
            ]
        }
        """
        manifest_path = Path(manifest_path)
        with open(manifest_path) as f:
            data = json.load(f)

        base_dir = manifest_path.parent
        samples = []
        for s in data.get("samples", []):
            # Resolve relative image paths
            img_path = s["image_path"]
            if not Path(img_path).is_absolute():
                img_path = str(base_dir / img_path)
            samples.append(GroundTruthSample(
                image_path=img_path,
                latitude=s["latitude"],
                longitude=s["longitude"],
                country=s.get("country", ""),
                city=s.get("city", ""),
                region=s.get("region", ""),
                difficulty=s.get("difficulty", "medium"),
                tags=s.get("tags", []),
                metadata=s.get("metadata", {}),
            ))

        return cls(
            name=data.get("name", manifest_path.stem),
            samples=samples,
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
        )

    @classmethod
    def from_csv(cls, csv_path: str | Path, name: str = "") -> EvalDataset:
        """Load from CSV (image_path,latitude,longitude,country,city,difficulty)."""
        import csv

        csv_path = Path(csv_path)
        samples = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(GroundTruthSample(
                    image_path=row["image_path"],
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    country=row.get("country", ""),
                    city=row.get("city", ""),
                    region=row.get("region", ""),
                    difficulty=row.get("difficulty", "medium"),
                    tags=row.get("tags", "").split(",") if row.get("tags") else [],
                ))

        return cls(name=name or csv_path.stem, samples=samples)

    def to_manifest(self, output_path: str | Path) -> None:
        """Save dataset as manifest.json."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "samples": [s.to_dict() for s in self.samples],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def filter_by_difficulty(self, difficulty: str) -> EvalDataset:
        return EvalDataset(
            name=f"{self.name}_{difficulty}",
            samples=[s for s in self.samples if s.difficulty == difficulty],
            description=self.description,
            version=self.version,
        )

    def filter_by_tag(self, tag: str) -> EvalDataset:
        return EvalDataset(
            name=f"{self.name}_{tag}",
            samples=[s for s in self.samples if tag in s.tags],
            description=self.description,
            version=self.version,
        )
