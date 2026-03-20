"""VLM-based geolocation reasoning.

Uses a strong VLM (Gemini 3 Flash Preview) to reason about location
from the image using chain-of-thought, producing country/region/city/coordinates.
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

from loguru import logger
from PIL import Image

from src.evidence.chain import Evidence, EvidenceSource
from src.scoring.scorer import GeoScorer
from src.utils.retry import execute_with_retry

GEO_REASONING_PROMPT = """You are an expert geolocation analyst. Analyze this image and determine the most likely location.

Think step by step:
1. Language/script on signs, text, license plates
2. Architecture style and building materials
3. Road markings, traffic side, road infrastructure
4. Vegetation, terrain, climate indicators
5. Vehicle types and brands common to the area
6. Cultural indicators (clothing, food, customs)
7. Specific landmarks or identifiable features
8. Sun position and shadows for hemisphere/latitude hints

After your analysis, provide your best estimate as JSON:
{
  "country": "most likely country",
  "region": "state/province/region if determinable",
  "city": "city if determinable, null otherwise",
  "latitude": estimated latitude as float,
  "longitude": estimated longitude as float,
  "confidence": 0.0-1.0 based on strength of evidence,
  "reasoning": "brief explanation of key evidence",
  "alternative_countries": ["2nd most likely", "3rd most likely"]
}

Be honest about uncertainty. If you can only determine the continent or region,
say so. A confident wrong answer is worse than an honest uncertain one.
"""


async def predict_location(
    image_path: str,
    client: Any,
    model: str = "google/gemini-3-flash-preview",
    additional_context: str = "",
) -> dict[str, Any]:
    """Use VLM chain-of-thought to predict location.

    Returns dict with: country, region, city, latitude, longitude, confidence, reasoning
    """
    try:
        image_url = _encode_image(image_path)

        prompt = GEO_REASONING_PROMPT
        if additional_context:
            prompt += f"\n\nAdditional context from other analyses:\n{additional_context}"

        resp = await execute_with_retry(
            client.chat.completions.create,
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=2000,
            max_attempts=2,
            total_timeout=45.0,
        )

        raw = resp.choices[0].message.content
        return _parse_response(raw)

    except Exception as e:
        logger.error("VLM geo prediction failed: {}", e)
        return _empty_prediction()


def to_evidence(prediction: dict[str, Any], scorer: GeoScorer | None = None) -> list[Evidence]:
    """Convert VLM geo prediction to Evidence objects."""
    if scorer is None:
        scorer = GeoScorer()
    evidences = []

    lat = prediction.get("latitude")
    lon = prediction.get("longitude")
    conf = prediction.get("confidence", 0.0)
    country = prediction.get("country")
    region = prediction.get("region")
    city = prediction.get("city")

    # Main prediction
    if lat is not None and lon is not None:
        evidences.append(
            Evidence(
                source=EvidenceSource.VLM_GEO,
                content=f"VLM geo reasoning: {prediction.get('reasoning', 'No reasoning')}",
                confidence=conf,
                latitude=lat,
                longitude=lon,
                country=country,
                region=region,
                city=city,
                metadata={"model": "vlm_geo", "type": "primary_prediction"},
            )
        )

    # Country-level evidence (even without coords)
    if country:
        evidences.append(
            Evidence(
                source=EvidenceSource.VLM_GEO,
                content=f"VLM country prediction: {country}",
                confidence=min(conf + scorer.vlm_country_boost, 1.0),
                country=country,
                metadata={"model": "vlm_geo", "type": "country"},
            )
        )

    # Alternative countries
    for alt in prediction.get("alternative_countries", []):
        if alt:
            evidences.append(
                Evidence(
                    source=EvidenceSource.VLM_GEO,
                    content=f"VLM alternative country: {alt}",
                    confidence=max(scorer.vlm_alternative_floor, conf - scorer.vlm_alternative_penalty),
                    country=alt,
                    metadata={"model": "vlm_geo", "type": "alternative_country"},
                )
            )

    return evidences


def _encode_image(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    max_dim = 2048
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _find_json_object(text: str) -> str | None:
    """Find the last balanced JSON object in text.

    Delegates to the shared utility so preamble JSON fragments
    (e.g. ``The hint says {"city":"London"} but...``) are skipped.
    """
    from src.utils.json_utils import find_json_object
    return find_json_object(text)


def _parse_response(raw: str) -> dict[str, Any]:
    """Parse VLM response, extracting JSON from possible markdown/text."""
    try:
        # Try to find JSON in the response
        json_str = _find_json_object(raw)
        if json_str:
            data = json.loads(json_str)
            # Validate and clean
            result = {
                "country": data.get("country"),
                "region": data.get("region"),
                "city": data.get("city"),
                "latitude": _safe_float(data.get("latitude")),
                "longitude": _safe_float(data.get("longitude")),
                "confidence": max(0.0, min(1.0, _safe_float(data.get("confidence"), 0.0))),
                "reasoning": data.get("reasoning", ""),
                "alternative_countries": data.get("alternative_countries", []),
            }
            return result
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("Failed to parse VLM geo JSON: {}", e)

    return _empty_prediction()


def _safe_float(val: Any, default: float | None = None) -> float | None:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _empty_prediction() -> dict[str, Any]:
    return {
        "country": None,
        "region": None,
        "city": None,
        "latitude": None,
        "longitude": None,
        "confidence": 0.0,
        "reasoning": "Prediction failed",
        "alternative_countries": [],
    }
