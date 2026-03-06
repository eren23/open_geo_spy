"""Text extraction from images using VLM (replaces chunk-based OCR).

Uses Gemini 2.5 Flash via OpenRouter for fast, accurate text extraction.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Optional

from loguru import logger
from PIL import Image

from src.config.llm import LLMCallType, get_llm_params
from src.evidence.chain import Evidence, EvidenceSource
from src.scoring.scorer import GeoScorer

OCR_PROMPT = """Extract ALL visible text from this image. Be thorough and include:

1. **Street signs** - road names, highway numbers, directional signs
2. **Business names** - store names, restaurant names, company logos with text
3. **Building info** - addresses, building numbers, floor numbers
4. **License plates** - plate numbers, state/country identifiers
5. **Informational signs** - notices, warnings, descriptions, opening hours
6. **Other text** - graffiti, billboards, banners, phone numbers, URLs

For each piece of text found, categorize it and note the language.

Return in this format:
STREET_SIGNS: [list each on new line with "- "]
BUSINESS_NAMES: [list each on new line with "- "]
BUILDING_INFO: [list each on new line with "- "]
LICENSE_PLATES: [list each on new line with "- ", include region/country if identifiable]
INFORMATIONAL: [list each on new line with "- "]
LANGUAGES_DETECTED: [comma-separated list]
"""


async def extract_text(
    image_path: str,
    client: Any,
    settings: Optional[Any] = None,
) -> dict[str, list[str]]:
    """Extract text from image using VLM.

    Args:
        image_path: Path to the image file
        client: OpenAI-compatible async client
        settings: Settings object with LLM configuration
        
    Returns dict with keys: street_signs, business_names, building_info,
    license_plates, informational, languages.
    """
    from src.utils.retry import execute_with_retry

    try:
        image_url = _encode_image(image_path)
        
        # Get LLM params from centralized config
        if settings is None:
            from src.config.settings import get_settings
            settings = get_settings()
        llm_params = get_llm_params(LLMCallType.OCR, settings)

        resp = await execute_with_retry(
            client.chat.completions.create,
            **llm_params,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_attempts=2,
            base_delay=2.0,
        )

        return _parse_ocr_response(resp.choices[0].message.content)

    except Exception as e:
        logger.error("OCR extraction failed: {}", e)
        return _empty_result()


def to_evidence(ocr_result: dict[str, list[str]], scorer: GeoScorer | None = None) -> list[Evidence]:
    """Convert OCR results to Evidence objects."""
    if scorer is None:
        scorer = GeoScorer()
    evidences = []

    for plate in ocr_result.get("license_plates", []):
        evidences.append(
            Evidence(
                source=EvidenceSource.OCR,
                content=f"License plate: {plate}",
                confidence=scorer.source_conf("license_plate"),
                metadata={"type": "license_plate"},
            )
        )

    for sign in ocr_result.get("street_signs", []):
        evidences.append(
            Evidence(
                source=EvidenceSource.OCR,
                content=f"Street sign: {sign}",
                confidence=scorer.source_conf("street_sign"),
                metadata={"type": "street_sign"},
            )
        )

    for business in ocr_result.get("business_names", []):
        evidences.append(
            Evidence(
                source=EvidenceSource.OCR,
                content=f"Business: {business}",
                confidence=scorer.source_conf("business_name"),
                metadata={"type": "business_name"},
            )
        )

    languages = ocr_result.get("languages", [])
    if languages:
        evidences.append(
            Evidence(
                source=EvidenceSource.OCR,
                content=f"Languages detected: {', '.join(languages)}",
                confidence=scorer.source_conf("language"),
                metadata={"type": "language", "languages": languages},
            )
        )

    return evidences


def _encode_image(image_path: str) -> str:
    """Encode image as base64 data URL."""
    img = Image.open(image_path)

    # Resize if too large (keep under 4MB for API)
    max_dim = 2048
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _parse_ocr_response(text: str) -> dict[str, list[str]]:
    """Parse VLM OCR response into structured categories."""
    result = _empty_result()
    current_category = None

    category_map = {
        "STREET_SIGNS": "street_signs",
        "BUSINESS_NAMES": "business_names",
        "BUILDING_INFO": "building_info",
        "LICENSE_PLATES": "license_plates",
        "INFORMATIONAL": "informational",
        "LANGUAGES_DETECTED": "languages",
    }

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Check for category header
        for header, key in category_map.items():
            if line.upper().startswith(header):
                current_category = key
                # Check if there's content on the same line after ":"
                after_colon = line.split(":", 1)[1].strip() if ":" in line else ""
                if after_colon and not after_colon.startswith("-"):
                    if current_category == "languages":
                        result[current_category] = [l.strip() for l in after_colon.split(",") if l.strip()]
                    else:
                        result[current_category].append(after_colon)
                break
        else:
            # Content line
            if current_category and line.startswith("-"):
                item = line.lstrip("- ").strip()
                if item:
                    if current_category == "languages":
                        result[current_category].extend(l.strip() for l in item.split(",") if l.strip())
                    else:
                        result[current_category].append(item)

    return result


def _empty_result() -> dict[str, list[str]]:
    return {
        "street_signs": [],
        "business_names": [],
        "building_info": [],
        "license_plates": [],
        "informational": [],
        "languages": [],
    }
