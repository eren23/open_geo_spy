"""Subprocess entrypoint for suite execution inside a candidate worktree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.improve.execution import run_suite_sync


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an OpenGeoSpy benchmark suite")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--quality", default="balanced")
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--capability-snapshot", default="")
    parser.add_argument("--baseline-lineage-id", default="")
    args = parser.parse_args()
    capability_snapshot = None
    if args.capability_snapshot:
        capability_snapshot = json.loads(Path(args.capability_snapshot).read_text())

    run_suite_sync(
        args.suite,
        args.output_dir,
        label=args.label,
        quality=args.quality,
        max_concurrent=args.max_concurrent,
        capability_snapshot=capability_snapshot,
        scoring_config_path=os.environ.get("SCORING_CONFIG_PATH") or None,
        baseline_lineage_id=args.baseline_lineage_id,
    )


if __name__ == "__main__":
    main()
