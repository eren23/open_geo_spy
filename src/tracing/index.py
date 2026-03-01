"""SQLite index for cross-run trace queries."""

from __future__ import annotations

import sqlite3
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import orjson
from loguru import logger


_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    session_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    version TEXT,
    image_hash TEXT,
    prediction_country TEXT,
    prediction_city TEXT,
    prediction_lat REAL,
    prediction_lon REAL,
    confidence REAL,
    total_cost_usd REAL DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_duration_ms REAL DEFAULT 0,
    ground_truth_lat REAL,
    ground_truth_lon REAL,
    ground_truth_country TEXT,
    gcd_error_km REAL,
    country_correct INTEGER,
    trace_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);
CREATE INDEX IF NOT EXISTS idx_traces_version ON traces(version);
"""


class TraceIndex:
    """SQLite index over JSONL trace files for aggregate queries."""

    def __init__(self, db_path: str = "data/traces/index.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def index_trace(self, trace_path: str | Path, *, _commit: bool = True) -> None:
        """Index a single JSONL trace file."""
        trace_path = Path(trace_path)
        if not trace_path.exists():
            return

        header = None
        result = None

        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = orjson.loads(line)
                if event.get("type") == "header":
                    header = event
                elif event.get("type") == "result":
                    result = event

        if not header:
            return

        session_id = header.get("session_id", "")
        prediction = (result or {}).get("prediction", {})
        gt = (result or {}).get("ground_truth")

        # Calculate GCD error if ground truth available
        gcd_error = None
        country_correct = None
        if gt and prediction:
            pred_lat = prediction.get("lat") or prediction.get("latitude")
            pred_lon = prediction.get("lon") or prediction.get("longitude")
            gt_lat = gt.get("latitude")
            gt_lon = gt.get("longitude")
            if all(v is not None for v in [pred_lat, pred_lon, gt_lat, gt_lon]):
                from src.utils.geo_math import haversine_distance
                gcd_error = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)
            gt_country = (gt.get("country") or "").lower()
            pred_country = (prediction.get("country") or "").lower()
            if gt_country and pred_country:
                country_correct = 1 if gt_country == pred_country else 0

        self._conn.execute(
            """INSERT OR REPLACE INTO traces
               (session_id, timestamp, version, image_hash,
                prediction_country, prediction_city, prediction_lat, prediction_lon,
                confidence, total_cost_usd, total_tokens, total_duration_ms,
                ground_truth_lat, ground_truth_lon, ground_truth_country,
                gcd_error_km, country_correct, trace_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                header.get("timestamp", ""),
                header.get("version", ""),
                header.get("image_hash", ""),
                prediction.get("country"),
                prediction.get("city"),
                prediction.get("lat") or prediction.get("latitude"),
                prediction.get("lon") or prediction.get("longitude"),
                prediction.get("confidence", 0),
                (result or {}).get("total_cost_usd", 0),
                (result or {}).get("total_tokens", 0),
                (result or {}).get("total_duration_ms", 0),
                gt.get("latitude") if gt else None,
                gt.get("longitude") if gt else None,
                gt.get("country") if gt else None,
                gcd_error,
                country_correct,
                str(trace_path),
            ),
        )
        if _commit:
            self._conn.commit()

    def index_directory(self, traces_dir: str = "data/traces") -> int:
        """Index all JSONL files in the traces directory."""
        count = 0
        for path in Path(traces_dir).rglob("*.jsonl"):
            try:
                self.index_trace(path, _commit=False)
                count += 1
            except Exception as e:
                logger.warning("Failed to index {}: {}", path, e)
        self._conn.commit()
        return count

    def accuracy_stats(
        self,
        version: str | None = None,
        since: str | None = None,
    ) -> dict[str, Any]:
        """Compute accuracy statistics from indexed traces with ground truth."""
        where_clauses = ["gcd_error_km IS NOT NULL"]
        params: list[Any] = []

        if version:
            where_clauses.append("version = ?")
            params.append(version)
        if since:
            where_clauses.append("timestamp >= ?")
            params.append(since)

        where = " AND ".join(where_clauses)

        rows = self._conn.execute(
            f"SELECT gcd_error_km, country_correct, confidence FROM traces WHERE {where}",
            params,
        ).fetchall()

        if not rows:
            return {"count": 0}

        errors = [r["gcd_error_km"] for r in rows]
        errors_sorted = sorted(errors)
        n = len(errors)

        return {
            "count": n,
            "accuracy_at_1km": sum(1 for e in errors if e < 1) / n,
            "accuracy_at_25km": sum(1 for e in errors if e < 25) / n,
            "accuracy_at_50km": sum(1 for e in errors if e < 50) / n,
            "accuracy_at_150km": sum(1 for e in errors if e < 150) / n,
            "accuracy_at_750km": sum(1 for e in errors if e < 750) / n,
            "median_gcd_km": statistics.median(errors_sorted),
            "mean_gcd_km": round(sum(errors) / n, 2),
            "p90_gcd_km": errors_sorted[int((n - 1) * 0.9)],
            "country_accuracy": (
                sum(r["country_correct"] for r in rows if r["country_correct"] is not None)
                / sum(1 for r in rows if r["country_correct"] is not None)
                if any(r["country_correct"] is not None for r in rows)
                else None
            ),
        }

    def cost_stats(
        self,
        version: str | None = None,
        since: str | None = None,
    ) -> dict[str, Any]:
        """Compute cost/token statistics."""
        where_clauses = ["1=1"]
        params: list[Any] = []

        if version:
            where_clauses.append("version = ?")
            params.append(version)
        if since:
            where_clauses.append("timestamp >= ?")
            params.append(since)

        where = " AND ".join(where_clauses)

        row = self._conn.execute(
            f"""SELECT
                COUNT(*) as count,
                AVG(total_cost_usd) as avg_cost,
                SUM(total_cost_usd) as total_cost,
                AVG(total_tokens) as avg_tokens,
                SUM(total_tokens) as total_tokens,
                AVG(total_duration_ms) as avg_duration_ms
               FROM traces WHERE {where}""",
            params,
        ).fetchone()

        return dict(row) if row else {"count": 0}

    def close(self) -> None:
        self._conn.close()
