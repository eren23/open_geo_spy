"""Visual feature extraction using VLM (replaces OpenCV environment classifier).

Uses Gemini 2.5 Flash for fast visual analysis: landmarks, architecture,
environment classification, geographic features.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

from loguru import logger
from PIL import Image

from src.evidence.chain import Evidence, EvidenceSource
from src.scoring.scorer import GeoScorer

FEATURE_EXTRACTION_PROMPT = """Analyze this image for geolocation clues. Extract ALL visual features that could help identify the location.

Return a JSON object with these fields:
{
  "landmarks": ["list of recognizable landmarks, monuments, statues"],
  "architecture_style": "dominant architectural style (e.g., 'Central European', 'Southeast Asian', 'Modern American')",
  "building_types": ["types of buildings: residential, commercial, industrial, religious, government"],
  "vegetation": {
    "type": "tropical/temperate/arid/boreal/none",
    "density": "dense/moderate/sparse/none",
    "notable_species": ["palm trees", "pine trees", etc.]
  },
  "terrain": ["flat", "hilly", "mountainous", "coastal", etc.],
  "water_bodies": ["ocean", "river", "lake", etc.],
  "infrastructure": {
    "road_type": "highway/urban_road/rural_road/path/none",
    "road_markings": "description of lane markings, colors",
    "traffic_side": "left/right/unclear",
    "power_lines": true/false,
    "rail": true/false
  },
  "vehicles": {
    "types": ["car", "truck", "bus", "motorcycle", "bicycle"],
    "notable": "any distinctive vehicle features (brand, taxi color, bus style)"
  },
  "environment_type": "URBAN/SUBURBAN/RURAL/INDUSTRIAL/AIRPORT/COASTAL/FOREST/MOUNTAIN/DESERT/PARK/HIGHWAY",
  "weather_climate": "sunny/cloudy/rainy/snowy/foggy + hot/warm/cool/cold",
  "time_of_day": "morning/midday/afternoon/evening/night",
  "cultural_indicators": ["flags", "writing systems", "clothing styles", "food types"],
  "country_clues": ["specific clues pointing to a country or region"],
  "confidence_notes": "brief note on how diagnostic these features are"
}
"""


async def extract_visual_features(
    image_path: str,
    client: Any,
    model: str = "google/gemini-2.5-flash",
) -> dict[str, Any]:
    """Extract visual features from image using VLM.

    Returns structured feature dict for geolocation analysis.
    """
    try:
        image_url = _encode_image(image_path)

        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": FEATURE_EXTRACTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=2000,
        )

        raw = resp.choices[0].message.content
        return _parse_features(raw)

    except Exception as e:
        logger.error("Visual feature extraction failed: {}", e)
        return _empty_features()


def to_evidence(features: dict[str, Any], scorer: GeoScorer | None = None) -> list[Evidence]:
    """Convert extracted visual features to Evidence objects."""
    if scorer is None:
        scorer = GeoScorer()
    evidences = []

    for landmark in features.get("landmarks", []):
        evidences.append(
            Evidence(
                source=EvidenceSource.VLM_ANALYSIS,
                content=f"Landmark: {landmark}",
                confidence=scorer.source_conf("landmark"),
                metadata={"type": "landmark"},
            )
        )

    arch = features.get("architecture_style")
    if arch and arch.lower() not in ("unknown", "unclear", "none"):
        evidences.append(
            Evidence(
                source=EvidenceSource.VLM_ANALYSIS,
                content=f"Architecture style: {arch}",
                confidence=scorer.source_conf("architecture"),
                metadata={"type": "architecture"},
            )
        )

    infra = features.get("infrastructure", {})
    traffic_side = infra.get("traffic_side")
    if traffic_side and traffic_side not in ("unclear", "none"):
        evidences.append(
            Evidence(
                source=EvidenceSource.VLM_ANALYSIS,
                content=f"Traffic drives on: {traffic_side}",
                confidence=scorer.source_conf("traffic_side"),
                metadata={"type": "traffic_side"},
            )
        )

    for clue in features.get("country_clues", []):
        evidences.append(
            Evidence(
                source=EvidenceSource.VLM_ANALYSIS,
                content=f"Country clue: {clue}",
                confidence=scorer.source_conf("country_clue"),
                metadata={"type": "country_clue"},
            )
        )

    for indicator in features.get("cultural_indicators", []):
        evidences.append(
            Evidence(
                source=EvidenceSource.VLM_ANALYSIS,
                content=f"Cultural indicator: {indicator}",
                confidence=scorer.source_conf("cultural"),
                metadata={"type": "cultural"},
            )
        )

    env_type = features.get("environment_type")
    if env_type and env_type != "UNKNOWN":
        evidences.append(
            Evidence(
                source=EvidenceSource.VLM_ANALYSIS,
                content=f"Environment: {env_type}",
                confidence=scorer.source_conf("environment"),
                metadata={"type": "environment", "environment_type": env_type},
            )
        )

    return evidences


def _encode_image(image_path: str) -> str:
    img = Image.open(image_path)
    max_dim = 2048
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _parse_features(raw: str) -> dict[str, Any]:
    """Parse VLM response, extracting JSON from possible markdown fences."""
    try:
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse features JSON, attempting extraction")
        # Try to find JSON in the response
        import re

        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.error("Failed to parse features from VLM response")
        return _empty_features()


def _empty_features() -> dict[str, Any]:
    return {
        "landmarks": [],
        "architecture_style": "unknown",
        "building_types": [],
        "vegetation": {"type": "unknown", "density": "unknown", "notable_species": []},
        "terrain": [],
        "water_bodies": [],
        "infrastructure": {},
        "vehicles": {},
        "environment_type": "UNKNOWN",
        "weather_climate": "unknown",
        "time_of_day": "unknown",
        "cultural_indicators": [],
        "country_clues": [],
        "confidence_notes": "Feature extraction failed",
    }
