"""LLM-based secondary judging for trace quality and efficiency."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


@dataclass
class TraceJudgeSummary:
    """Summary score for a candidate run."""

    reasoning_quality: int = 0
    recovery_quality: int = 0
    efficiency_quality: int = 0
    explanation: str = ""

    @property
    def mean_score(self) -> float:
        scores = [self.reasoning_quality, self.recovery_quality, self.efficiency_quality]
        return sum(scores) / len(scores) if any(scores) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning_quality": self.reasoning_quality,
            "recovery_quality": self.recovery_quality,
            "efficiency_quality": self.efficiency_quality,
            "mean_score": round(self.mean_score, 2),
            "explanation": self.explanation,
        }


JUDGE_PROMPT = """You are ranking an OpenGeoSpy experiment run.

Focus on whether the run is operationally better, not just more verbose.

Run summary:
{summary}

Samples:
{samples}

Return JSON:
{{"reasoning_quality": N, "recovery_quality": N, "efficiency_quality": N, "explanation": "brief explanation"}}
"""


class TraceQualityJudge:
    """Secondary LLM judge used only after hard metric gating."""

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def judge_run(
        self,
        run_manifest_path: str | Path,
        *,
        max_samples: int = 6,
    ) -> TraceJudgeSummary:
        run_manifest_path = Path(run_manifest_path)
        with open(run_manifest_path) as f:
            manifest = json.load(f)
        sample_results_path = Path(manifest["sample_results_path"])
        rows = []
        with open(sample_results_path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        rows.sort(
            key=lambda row: (
                -len(row.get("trace_anomalies", [])),
                -(row.get("gcd_km") or 0.0),
            )
        )
        sample_lines = []
        for row in rows[:max_samples]:
            sample_lines.append(
                json.dumps(
                    {
                        "image_path": row.get("image_path"),
                        "gcd_km": row.get("gcd_km"),
                        "pred_country": row.get("pred_country"),
                        "gt_country": row.get("gt_country"),
                        "latency_ms": row.get("latency_ms"),
                        "trace_anomalies": row.get("trace_anomalies", []),
                    }
                )
            )

        prompt = JUDGE_PROMPT.format(
            summary=json.dumps(manifest.get("overall_metrics", {}), indent=2),
            samples="\n".join(sample_lines) or "[]",
        )
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        raw = response.choices[0].message.content
        return self._parse(raw)

    def _parse(self, raw: str) -> TraceJudgeSummary:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return TraceJudgeSummary(explanation="Parse failed")
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return TraceJudgeSummary(explanation="Parse failed")
        return TraceJudgeSummary(
            reasoning_quality=int(data.get("reasoning_quality", 0)),
            recovery_quality=int(data.get("recovery_quality", 0)),
            efficiency_quality=int(data.get("efficiency_quality", 0)),
            explanation=str(data.get("explanation", "")),
        )
