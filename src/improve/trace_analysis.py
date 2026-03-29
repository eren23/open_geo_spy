"""Trace analysis utilities for anomaly detection and regression mining."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.geo_math import haversine_distance


def extract_trace_context(trace_path: str | Path) -> dict[str, Any]:
    """Read a JSONL trace into a lightweight event bundle."""
    header: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    events: list[dict[str, Any]] = []

    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            events.append(event)
            if event.get("type") == "header":
                header = event
            elif event.get("type") == "result":
                result = event

    return {
        "header": header or {},
        "result": result or {},
        "events": events,
    }


@dataclass
class TraceDiagnostics:
    """Anomaly summary for a single trace."""

    trace_path: str
    flags: list[str] = field(default_factory=list)
    search_query_count: int = 0
    search_evidence_yield: int = 0
    best_snapshot_gcd_km: float | None = None
    final_gcd_km: float | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_path": self.trace_path,
            "flags": self.flags,
            "search_query_count": self.search_query_count,
            "search_evidence_yield": self.search_evidence_yield,
            "best_snapshot_gcd_km": self.best_snapshot_gcd_km,
            "final_gcd_km": self.final_gcd_km,
            "notes": self.notes,
        }


def _candidate_gcd(candidate: dict[str, Any], ground_truth: dict[str, Any]) -> float | None:
    pred_lat = candidate.get("lat", candidate.get("latitude"))
    pred_lon = candidate.get("lon", candidate.get("longitude"))
    gt_lat = ground_truth.get("latitude")
    gt_lon = ground_truth.get("longitude")
    if None in {pred_lat, pred_lon, gt_lat, gt_lon}:
        return None
    return haversine_distance(float(pred_lat), float(pred_lon), float(gt_lat), float(gt_lon))


def analyze_trace_file(trace_path: str | Path) -> TraceDiagnostics:
    """Identify regressions and inefficiencies from a saved trace."""
    trace_path = str(trace_path)
    context = extract_trace_context(trace_path)
    events = context["events"]
    result = context["result"] or {}
    prediction = result.get("prediction") or {}
    ground_truth = result.get("ground_truth") or {}

    diagnostics = TraceDiagnostics(trace_path=trace_path)
    diagnostics.search_query_count = sum(1 for event in events if event.get("type") == "search_query")
    diagnostics.search_evidence_yield = sum(
        int(event.get("evidence_count", 0))
        for event in events
        if event.get("type") == "search_query"
    )

    diagnostics.final_gcd_km = _candidate_gcd(prediction, ground_truth)

    best_snapshot_gcd: float | None = None
    snapshot_country_match = False
    for event in events:
        if event.get("type") != "candidate_snapshot":
            continue
        for candidate in event.get("candidates", []):
            gcd = _candidate_gcd(candidate, ground_truth)
            if gcd is not None and (best_snapshot_gcd is None or gcd < best_snapshot_gcd):
                best_snapshot_gcd = gcd
            gt_country = str(ground_truth.get("country", "")).lower()
            if gt_country and str(candidate.get("country", "")).lower() == gt_country:
                snapshot_country_match = True

    diagnostics.best_snapshot_gcd_km = best_snapshot_gcd

    if diagnostics.final_gcd_km is not None and 25 < diagnostics.final_gcd_km <= 100:
        diagnostics.flags.append("near_miss")

    if (
        diagnostics.search_query_count >= 8
        and diagnostics.final_gcd_km is not None
        and diagnostics.final_gcd_km <= 150
    ):
        diagnostics.flags.append("inefficient_search")

    if (
        best_snapshot_gcd is not None
        and diagnostics.final_gcd_km is not None
        and best_snapshot_gcd + 25 < diagnostics.final_gcd_km
    ):
        diagnostics.flags.append("found_then_lost")

    if (
        best_snapshot_gcd is not None
        and diagnostics.final_gcd_km is not None
        and best_snapshot_gcd + 100 < diagnostics.final_gcd_km
    ) or (
        snapshot_country_match
        and ground_truth.get("country")
        and str(prediction.get("country", "")).lower() != str(ground_truth.get("country", "")).lower()
    ):
        diagnostics.flags.append("late_stage_regression")

    diagnostics.notes = {
        "final_prediction_country": prediction.get("country"),
        "ground_truth_country": ground_truth.get("country"),
        "search_queries": diagnostics.search_query_count,
    }

    diagnostics.flags = sorted(set(diagnostics.flags))
    return diagnostics
