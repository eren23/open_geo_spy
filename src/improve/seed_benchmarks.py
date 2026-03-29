"""Small real-image benchmark seed builders."""

from __future__ import annotations

import json
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.eval.dataset import EvalDataset, GroundTruthSample

WIKIPEDIA_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
USER_AGENT = "OpenGeoSpy/0.1 benchmark seeding"


@dataclass(frozen=True)
class LandmarkSeed:
    page_title: str
    file_stem: str
    latitude: float
    longitude: float
    country: str
    city: str
    region: str
    tags: tuple[str, ...]
    difficulty: str = "easy"


WIKIPEDIA_LANDMARKS_V1: tuple[LandmarkSeed, ...] = (
    LandmarkSeed("Eiffel Tower", "eiffel_tower", 48.8584, 2.2945, "France", "Paris", "Ile-de-France", ("landmark", "europe")),
    LandmarkSeed("Statue of Liberty", "statue_of_liberty", 40.6892, -74.0445, "United States", "New York", "New York", ("landmark", "north_america")),
    LandmarkSeed("Taj Mahal", "taj_mahal", 27.1751, 78.0421, "India", "Agra", "Uttar Pradesh", ("landmark", "asia")),
    LandmarkSeed("Colosseum", "colosseum", 41.8902, 12.4922, "Italy", "Rome", "Lazio", ("landmark", "europe")),
    LandmarkSeed("Sydney Opera House", "sydney_opera_house", -33.8568, 151.2153, "Australia", "Sydney", "New South Wales", ("landmark", "oceania")),
    LandmarkSeed("Christ the Redeemer (statue)", "christ_the_redeemer", -22.9519, -43.2105, "Brazil", "Rio de Janeiro", "Rio de Janeiro", ("landmark", "south_america")),
    LandmarkSeed("Great Pyramid of Giza", "great_pyramid_of_giza", 29.9792, 31.1342, "Egypt", "Giza", "Giza", ("landmark", "africa")),
    LandmarkSeed("Big Ben", "big_ben", 51.5007, -0.1246, "United Kingdom", "London", "England", ("landmark", "europe")),
)


def build_wikipedia_landmarks_benchmark(
    output_manifest: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Fetch a curated real-image landmark benchmark from Wikipedia/Wikimedia."""
    output_manifest = Path(output_manifest)
    image_dir = output_manifest.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    samples: list[GroundTruthSample] = []
    for seed in WIKIPEDIA_LANDMARKS_V1:
        summary = _fetch_summary(seed.page_title)
        image_url = _select_image_url(summary)
        extension = _guess_extension(image_url, summary)
        image_path = image_dir / f"{seed.file_stem}{extension}"
        if force or not image_path.exists():
            image_path.write_bytes(_download_bytes(image_url))

        samples.append(
            GroundTruthSample(
                image_path=str(Path("images") / image_path.name),
                latitude=seed.latitude,
                longitude=seed.longitude,
                country=seed.country,
                city=seed.city,
                region=seed.region,
                difficulty=seed.difficulty,
                tags=list(seed.tags),
                metadata={
                    "page_title": seed.page_title,
                    "source_url": image_url,
                    "page_url": summary.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "source_label": "WikipediaLandmarksV1",
                },
            )
        )

    dataset = EvalDataset(
        name="wikipedia_landmarks_v1",
        description="Curated real-image landmark seed set fetched from Wikipedia/Wikimedia.",
        samples=samples,
    )
    dataset.to_manifest(output_manifest)
    return output_manifest


def _fetch_summary(page_title: str) -> dict[str, Any]:
    url = WIKIPEDIA_SUMMARY_API.format(title=quote(page_title, safe=""))
    return _json_request(url)


def _select_image_url(summary: dict[str, Any]) -> str:
    for key in ("thumbnail", "originalimage"):
        candidate = summary.get(key, {})
        source = candidate.get("source")
        if source:
            return str(source)
    raise ValueError(f"No downloadable image found in page summary for {summary.get('title', 'unknown page')}")


def _download_bytes(url: str) -> bytes:
    return _request_bytes(url)


def _guess_extension(image_url: str, summary: dict[str, Any]) -> str:
    path_suffix = Path(urlparse(image_url).path).suffix.lower()
    if path_suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if path_suffix == ".jpeg" else path_suffix

    mime = summary.get("thumbnail", {}).get("mimetype") or summary.get("originalimage", {}).get("mimetype")
    if mime:
        guessed = mimetypes.guess_extension(mime, strict=False)
        if guessed in {".jpg", ".jpeg", ".png", ".webp"}:
            return ".jpg" if guessed == ".jpeg" else guessed
    return ".jpg"


def _json_request(url: str) -> dict[str, Any]:
    payload = _request_bytes(url, accept="application/json")
    return json.loads(payload.decode("utf-8"))


def _request_bytes(url: str, *, accept: str = "*/*") -> bytes:
    delays = [1.0, 3.0, 6.0]
    last_error: Exception | None = None
    for delay in delays:
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except HTTPError as exc:
            last_error = exc
            if exc.code != 429:
                raise
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to download {url}")
