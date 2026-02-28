"""Batch result export to CSV and JSON."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from src.batch.manager import BatchJob


def export_csv(job: BatchJob) -> str:
    """Export batch results as CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "filename", "status", "name", "country", "region", "city",
        "latitude", "longitude", "confidence", "reasoning",
    ])

    for item in job.items:
        r = item.result or {}
        writer.writerow([
            item.filename,
            item.status.value,
            r.get("name", ""),
            r.get("country", ""),
            r.get("region", ""),
            r.get("city", ""),
            r.get("latitude", ""),
            r.get("longitude", ""),
            r.get("confidence", ""),
            r.get("reasoning", "")[:200],
        ])

    return output.getvalue()


def export_json(job: BatchJob) -> str:
    """Export batch results as JSON string."""
    return json.dumps(job.to_dict(), indent=2, default=str)
