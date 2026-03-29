"""Tests for trace anomaly detection."""

from __future__ import annotations

import json

from src.improve.trace_analysis import analyze_trace_file


def test_analyze_trace_flags_late_regression(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    events = [
        {"type": "header", "session_id": "abc", "image_path": "/tmp/sample.jpg"},
        {
            "type": "search_query",
            "query": "eiffel tower paris",
            "status": "completed",
            "evidence_count": 3,
        },
        {
            "type": "candidate_snapshot",
            "stage": "post_reasoning",
            "candidates": [
                {"country": "France", "city": "Paris", "lat": 48.8584, "lon": 2.2945, "confidence": 0.72},
                {"country": "Belgium", "city": "Brussels", "lat": 50.8503, "lon": 4.3517, "confidence": 0.51},
            ],
        },
        {
            "type": "result",
            "prediction": {"country": "Belgium", "city": "Brussels", "lat": 50.8503, "lon": 4.3517},
            "ground_truth": {"country": "France", "city": "Paris", "latitude": 48.8584, "longitude": 2.2945},
        },
    ]

    trace_path.write_text("\n".join(json.dumps(event) for event in events))
    diagnostics = analyze_trace_file(trace_path)

    assert "found_then_lost" in diagnostics.flags
    assert "late_stage_regression" in diagnostics.flags
    assert diagnostics.search_query_count == 1


def test_analyze_trace_flags_inefficient_search(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    events = [{"type": "header", "session_id": "abc", "image_path": "/tmp/sample.jpg"}]
    for idx in range(9):
        events.append(
            {
                "type": "search_query",
                "query": f"query-{idx}",
                "status": "completed",
                "evidence_count": 1,
            }
        )
    events.append(
        {
            "type": "result",
            "prediction": {"country": "France", "city": "Paris", "lat": 48.86, "lon": 2.29},
            "ground_truth": {"country": "France", "city": "Paris", "latitude": 48.8584, "longitude": 2.2945},
        }
    )

    trace_path.write_text("\n".join(json.dumps(event) for event in events))
    diagnostics = analyze_trace_file(trace_path)

    assert "inefficient_search" in diagnostics.flags
