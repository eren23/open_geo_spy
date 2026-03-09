"""Eval report generation: console, JSON, and markdown formats."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.eval.metrics import EvalMetrics


class EvalReport:
    """Generates evaluation reports in multiple formats."""

    def __init__(
        self,
        metrics: EvalMetrics,
        label: str = "",
        baseline: EvalMetrics | None = None,
    ):
        self.metrics = metrics
        self.label = label
        self.baseline = baseline

    def to_console(self) -> str:
        """Rich console-formatted report."""
        s = self.metrics.summary()
        lines = [
            f"{'='*60}",
            f"  Evaluation Report: {self.label or 'unlabeled'}",
            f"  Samples: {s['count']}",
            f"{'='*60}",
            "",
            "  Accuracy (within threshold):",
            f"    @1km (street):   {s['accuracy_1km']:.1%}",
            f"    @25km (city):    {s['accuracy_25km']:.1%}",
            f"    @50km:           {s['accuracy_50km']:.1%}",
            f"    @150km (region): {s['accuracy_150km']:.1%}",
            f"    @750km (country):{s['accuracy_750km']:.1%}",
            "",
            "  GCD Error (km):",
            f"    Median:  {s['median_gcd_km']:.1f}",
            f"    Mean:    {s['mean_gcd_km']:.1f}",
            f"    P90:     {s['p90_gcd_km']:.1f}",
            "",
            f"  Country accuracy: {s['country_accuracy']:.1%}",
            f"  City accuracy:    {s['city_accuracy']:.1%}",
            f"  ECE:              {s['ece']:.4f}",
            "",
            "  Cost:",
            f"    Mean:  ${s['mean_cost_usd']:.4f}",
            f"    Total: ${s['total_cost_usd']:.4f}",
            f"    Tokens: {s['total_tokens']:,}",
            f"    Mean latency: {s['mean_latency_ms']:.0f}ms",
        ]

        if self.baseline:
            lines.extend(self._comparison_lines())

        # Per-difficulty breakdown
        by_diff = self.metrics.by_difficulty()
        if len(by_diff) > 1:
            lines.extend(["", "  By Difficulty:"])
            for diff, m in sorted(by_diff.items()):
                ds = m.summary()
                lines.append(
                    f"    {diff:8s}: n={ds['count']:3d}  "
                    f"median={ds['median_gcd_km']:7.1f}km  "
                    f"@25km={ds['accuracy_25km']:.1%}  "
                    f"country={ds['country_accuracy']:.1%}"
                )

        # Per urban/rural breakdown (bias stratification)
        by_ur = self.metrics.by_urban_rural()
        if len(by_ur) > 1 or (len(by_ur) == 1 and "unspecified" not in by_ur):
            lines.extend(["", "  By Urban/Rural (bias):"])
            for tier, m in sorted(by_ur.items()):
                ds = m.summary()
                lines.append(
                    f"    {tier:12s}: n={ds['count']:3d}  "
                    f"median={ds['median_gcd_km']:7.1f}km  "
                    f"@25km={ds['accuracy_25km']:.1%}  "
                    f"@150km={ds['accuracy_150km']:.1%}  "
                    f"country={ds['country_accuracy']:.1%}"
                )

        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def _comparison_lines(self) -> list[str]:
        """Generate comparison lines against baseline."""
        s = self.metrics.summary()
        b = self.baseline.summary()
        lines = ["", "  vs Baseline:"]

        def _delta(key: str, lower_better: bool = True) -> str:
            diff = s[key] - b[key]
            if lower_better:
                arrow = "v" if diff < 0 else "^"
                color = "better" if diff < 0 else "worse"
            else:
                arrow = "^" if diff > 0 else "v"
                color = "better" if diff > 0 else "worse"
            return f"{arrow} {abs(diff):.4f} ({color})"

        lines.append(f"    median_gcd:     {_delta('median_gcd_km', True)}")
        lines.append(f"    @25km:          {_delta('accuracy_25km', False)}")
        lines.append(f"    country_acc:    {_delta('country_accuracy', False)}")
        lines.append(f"    mean_cost:      {_delta('mean_cost_usd', True)}")
        return lines

    def to_json(self) -> dict[str, Any]:
        """Full JSON report."""
        report = {
            "label": self.label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": self.metrics.summary(),
            "by_difficulty": {
                d: m.summary() for d, m in self.metrics.by_difficulty().items()
            },
            "by_urban_rural": {
                k: m.summary() for k, m in self.metrics.by_urban_rural().items()
            },
        }
        if self.baseline:
            report["baseline"] = self.baseline.summary()
        return report

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    def to_markdown(self) -> str:
        """Markdown-formatted report."""
        s = self.metrics.summary()
        lines = [
            f"# Evaluation Report: {self.label or 'unlabeled'}",
            "",
            f"**Samples:** {s['count']}  ",
            f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Accuracy",
            "",
            "| Threshold | Accuracy |",
            "|-----------|----------|",
            f"| @1km (street) | {s['accuracy_1km']:.1%} |",
            f"| @25km (city) | {s['accuracy_25km']:.1%} |",
            f"| @50km | {s['accuracy_50km']:.1%} |",
            f"| @150km (region) | {s['accuracy_150km']:.1%} |",
            f"| @750km (country) | {s['accuracy_750km']:.1%} |",
            "",
            "## Error Distribution",
            "",
            f"- **Median GCD:** {s['median_gcd_km']:.1f} km",
            f"- **Mean GCD:** {s['mean_gcd_km']:.1f} km",
            f"- **P90 GCD:** {s['p90_gcd_km']:.1f} km",
            "",
            "## Additional Metrics",
            "",
            f"- **Country accuracy:** {s['country_accuracy']:.1%}",
            f"- **City accuracy:** {s['city_accuracy']:.1%}",
            f"- **ECE:** {s['ece']:.4f}",
            "",
            "## Cost",
            "",
            f"- **Mean cost:** ${s['mean_cost_usd']:.4f}",
            f"- **Total cost:** ${s['total_cost_usd']:.4f}",
            f"- **Total tokens:** {s['total_tokens']:,}",
            f"- **Mean latency:** {s['mean_latency_ms']:.0f}ms",
        ]
        return "\n".join(lines)

    def save_markdown(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_markdown())
