"""LLM-as-judge for scoring reasoning quality, evidence usage, etc."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger
from openai import AsyncOpenAI


@dataclass
class JudgeScores:
    """Per-sample judge scores (1-5 per dimension)."""

    reasoning_quality: int = 0
    evidence_usage: int = 0
    confidence_calibration: int = 0
    specificity: int = 0
    explanation: str = ""

    @property
    def mean_score(self) -> float:
        scores = [self.reasoning_quality, self.evidence_usage,
                  self.confidence_calibration, self.specificity]
        return sum(scores) / len(scores) if any(scores) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning_quality": self.reasoning_quality,
            "evidence_usage": self.evidence_usage,
            "confidence_calibration": self.confidence_calibration,
            "specificity": self.specificity,
            "mean_score": round(self.mean_score, 2),
            "explanation": self.explanation,
        }


JUDGE_PROMPT = """You are evaluating a geolocation prediction system. Score the following prediction on 4 dimensions (1-5 scale each).

## Ground Truth
- Country: {gt_country}
- City: {gt_city}
- Coordinates: ({gt_lat}, {gt_lon})

## Prediction
- Country: {pred_country}
- City: {pred_city}
- Coordinates: ({pred_lat}, {pred_lon})
- Confidence: {confidence}
- Reasoning: {reasoning}

## GCD Error: {gcd_km:.1f} km

## Scoring Dimensions

1. **reasoning_quality** (1-5): How logical and well-structured is the reasoning? Does it follow evidence to conclusion?
2. **evidence_usage** (1-5): Does the reasoning cite specific evidence? Does it weigh conflicting evidence?
3. **confidence_calibration** (1-5): Does the confidence match the actual accuracy? (5 = well-calibrated, 1 = wildly miscalibrated)
4. **specificity** (1-5): How specific is the prediction? (5 = exact location, 1 = just a continent)

Return JSON:
{{"reasoning_quality": N, "evidence_usage": N, "confidence_calibration": N, "specificity": N, "explanation": "brief explanation"}}
"""


class LLMJudge:
    """Uses an LLM to score prediction quality beyond distance metrics."""

    def __init__(self, client: AsyncOpenAI, model: str = "google/gemini-2.5-flash"):
        self.client = client
        self.model = model

    async def judge(
        self,
        prediction: dict,
        ground_truth: dict,
        gcd_km: float,
    ) -> JudgeScores:
        """Score a single prediction."""
        prompt = JUDGE_PROMPT.format(
            gt_country=ground_truth.get("country", "unknown"),
            gt_city=ground_truth.get("city", "unknown"),
            gt_lat=ground_truth.get("latitude", 0),
            gt_lon=ground_truth.get("longitude", 0),
            pred_country=prediction.get("country", "unknown"),
            pred_city=prediction.get("city", "unknown"),
            pred_lat=prediction.get("lat") or prediction.get("latitude", 0),
            pred_lon=prediction.get("lon") or prediction.get("longitude", 0),
            confidence=prediction.get("confidence", 0),
            reasoning=prediction.get("reasoning", "none"),
            gcd_km=gcd_km,
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content
            return self._parse_scores(raw)
        except Exception as e:
            logger.warning("Judge failed: {}", e)
            return JudgeScores(explanation=f"Judge error: {e}")

    def _parse_scores(self, raw: str) -> JudgeScores:
        """Parse judge response."""
        try:
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                data = json.loads(match.group())
                return JudgeScores(
                    reasoning_quality=int(data.get("reasoning_quality", 0)),
                    evidence_usage=int(data.get("evidence_usage", 0)),
                    confidence_calibration=int(data.get("confidence_calibration", 0)),
                    specificity=int(data.get("specificity", 0)),
                    explanation=data.get("explanation", ""),
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse judge response: {}", e)

        return JudgeScores(explanation="Parse failed")
