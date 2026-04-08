#!/usr/bin/env python
"""GEPA command evaluator for open_geo_spy prompt optimization.

Invoked by gepa-optim with:
    python scripts/gepa_eval_command.py --candidate {candidate} [--task {task}]

Target is selected via the GEPA_TARGET environment variable (or --target flag):
    reasoning | feature_extraction | ocr | expansion

Behavior:
1. Parse the candidate file (Python RHS expression, e.g. triple-quoted string).
2. Monkey-patch the target module constant in process.
3. Run the pipeline on a fixed 3-sample subset of hard_streets_v1.
4. Compute a composite score (GCD + city + country) and emit JSON to stdout.
5. Restore the original constant.
6. Append an entry to data/gepa_runs/<target>/history.jsonl.
7. Abort early if cumulative spend exceeds GEPA_MAX_USD.

Output (single JSON object on stdout, consumed by gepa-optim):
    {"score": <float 0..1>, "side_info": {"feedback": "...", ...}}
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # pipeline resolves image paths relative to CWD

# Map symbolic target names to (module, attribute).
TARGET_MAP: dict[str, tuple[str, str]] = {
    "reasoning": ("src.agents.reasoning_agent", "REASONING_PROMPT"),
    "feature_extraction": ("src.extraction.features", "FEATURE_EXTRACTION_PROMPT"),
    "ocr": ("src.extraction.ocr", "OCR_PROMPT"),
    "expansion": ("src.search.smart_expander", "EXPANSION_PROMPT"),
}

# Fixed 3-sample training subset for determinism. Chosen for visual diversity
# across the Karlsruhe Muhlburger Tor frame sequence.
TRAIN_SAMPLE_NAMES = {"001.jpg", "004.jpg", "007.jpg"}

# Composite score weights.
W_GCD = 0.60       # continuous per-sample distance signal
W_CITY = 0.25      # discrete city-level correctness
W_COUNTRY = 0.15   # discrete country-level correctness (near-ceiling; sanity)

DATASET_MANIFEST = ROOT / "data" / "eval" / "hard_streets_v1" / "dataset_manifest.json"


def parse_candidate(path: Path) -> str:
    """Extract the prompt string from GEPA's candidate file.

    gepa-optim's SliceTarget writes the RHS verbatim (including triple quotes).
    We accept either form:
      (a) Valid Python string-literal expression -> AST-walk to get value
      (b) Raw text -> return as-is
    """
    text = path.read_text()
    stripped = text.strip()
    # Try parsing as a Python string-literal expression by wrapping in an
    # assignment statement. This is strictly safer than invoking a runtime
    # expression interpreter because ast.parse never executes code.
    try:
        tree = ast.parse("_gepa_candidate_value = " + stripped)
        if tree.body and isinstance(tree.body[0], ast.Assign):
            value_node = tree.body[0].value
            if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
                return value_node.value
            # Handle implicit string concatenation like "a" "b"
            if isinstance(value_node, ast.JoinedStr):
                # f-strings not supported — fall through to raw text
                pass
    except (SyntaxError, ValueError):
        pass
    return text


def load_dataset_subset():
    """Load hard_streets_v1 filtered to the training subset."""
    from src.eval.dataset import EvalDataset

    dataset = EvalDataset.from_manifest(DATASET_MANIFEST)
    dataset.samples = [
        s for s in dataset.samples if Path(s.image_path).name in TRAIN_SAMPLE_NAMES
    ]
    if not dataset.samples:
        raise RuntimeError("hard_streets_v1 training subset is empty")
    return dataset


async def run_evaluation(target: str, new_prompt: str):
    """Monkey-patch the target prompt, run the pipeline, restore."""
    # Override pipeline timeout BEFORE instantiating anything.
    # get_settings() is @lru_cache'd so modifying the returned instance
    # propagates to every downstream orchestrator.
    from src.config.settings import get_settings
    settings = get_settings()
    # Raise the per-sample budget from 90s to 240s so the 3-sample run
    # doesn't drop to empty results under load.
    settings.pipeline.max_total_latency_ms = 240_000

    from src.eval.runner import EvalRunner

    module_name, attr = TARGET_MAP[target]
    mod = importlib.import_module(module_name)
    original = getattr(mod, attr)
    setattr(mod, attr, new_prompt)

    dataset = load_dataset_subset()
    runner = EvalRunner(
        label=f"gepa-{target}",
        max_concurrent=2,  # moderate parallelism; quality=fast keeps per-sample latency well under the 240s budget
        output_dir=str(ROOT / f"data/gepa_runs/{target}/latest"),
        quality="fast",  # skip expensive visual verification; focus GEPA signal on the prompt being optimized
    )
    try:
        metrics = await runner.run(dataset)
    finally:
        setattr(mod, attr, original)
    return metrics


def compute_score(metrics) -> tuple[float, dict]:
    """Composite score in [0, 1]. Richer side_info drives GEPA's reflection LM."""
    per_sample = []
    gcd_subscores = []

    for r in metrics.results:
        gcd = r.gcd_km if r.gcd_km is not None else None
        # 1/(1 + km/5) — km=0 -> 1.0, km=1 -> 0.83, km=5 -> 0.50, km=50 -> 0.09
        gcd_sub = 1.0 / (1.0 + (gcd if gcd is not None else 1e6) / 5.0)
        gcd_subscores.append(gcd_sub)
        per_sample.append(
            {
                "image": Path(r.image_path).name,
                "gt": f"{r.gt_city}, {r.gt_country}",
                "pred": f"{r.pred_city or '?'}, {r.pred_country or '?'}",
                "gcd_km": round(gcd, 2) if gcd is not None else None,
                "city_correct": r.city_correct,
                "country_correct": r.country_correct,
                "confidence": round(r.pred_confidence, 2),
            }
        )

    gcd_component = sum(gcd_subscores) / len(gcd_subscores) if gcd_subscores else 0.0
    score = (
        W_GCD * gcd_component
        + W_CITY * metrics.city_accuracy
        + W_COUNTRY * metrics.country_accuracy
    )
    breakdown = {
        "gcd_component": round(gcd_component, 4),
        "city_accuracy": round(metrics.city_accuracy, 4),
        "country_accuracy": round(metrics.country_accuracy, 4),
        "median_gcd_km": round(metrics.median_gcd_km, 2) if metrics.median_gcd_km != float("inf") else None,
    }
    return float(score), {"per_sample": per_sample, "score_breakdown": breakdown}


def build_feedback(per_sample: list[dict], breakdown: dict, metrics) -> str:
    """Structured natural-language feedback for GEPA's reflection LM."""
    lines = [
        "Task: identify the precise location of street scenes from Karlsruhe, Germany (Muhlburger Tor tram stop).",
        f"Score breakdown: GCD={breakdown['gcd_component']:.3f}, city={breakdown['city_accuracy']:.0%}, country={breakdown['country_accuracy']:.0%}, median_gcd={breakdown['median_gcd_km']}km",
        "",
        "Per-sample results:",
    ]
    for p in per_sample:
        status = "PASS" if p["city_correct"] and p["country_correct"] else "MISS"
        gcd = f"{p['gcd_km']}km" if p["gcd_km"] is not None else "no-pred"
        lines.append(
            f"  [{status}] {p['image']}: pred=({p['pred']}) gcd={gcd} conf={p['confidence']}"
        )

    # Highlight failure patterns
    wrong_country = [p for p in per_sample if not p["country_correct"]]
    wrong_city_ok_country = [
        p for p in per_sample if p["country_correct"] and not p["city_correct"]
    ]
    if wrong_country:
        preds = ", ".join(p["pred"] for p in wrong_country)
        lines.append(
            f"\nCOUNTRY MISS on {len(wrong_country)}/{len(per_sample)} samples. "
            f"Wrong predictions: [{preds}]. "
            f"Goal: the prompt should make the model rely on German-specific visual cues "
            f"(street signs, tram systems, architecture, license plates) to lock in country."
        )
    if wrong_city_ok_country:
        preds = ", ".join(p["pred"] for p in wrong_city_ok_country)
        lines.append(
            f"\nCITY MISS on {len(wrong_city_ok_country)}/{len(per_sample)} samples (country correct). "
            f"Wrong cities: [{preds}]. "
            f"Goal: the prompt should push the model to narrow to SPECIFIC German cities "
            f"using tram network identifiers, local street naming, and regional architectural style."
        )
    return "\n".join(lines)


def append_history(target: str, entry: dict) -> float:
    """Append entry and return cumulative cost_usd."""
    log_dir = ROOT / f"data/gepa_runs/{target}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "history.jsonl"

    prior_total = 0.0
    if log_file.exists():
        for line in log_file.read_text().splitlines():
            try:
                prior_total += float(json.loads(line).get("cost_usd", 0) or 0)
            except Exception:
                pass

    entry["cost_usd_cumulative"] = round(prior_total + entry.get("cost_usd", 0), 5)
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry["cost_usd_cumulative"]


def check_cost_cap(target: str) -> None:
    """Abort with a capped failure score if GEPA_MAX_USD exceeded."""
    max_usd = os.environ.get("GEPA_MAX_USD")
    if not max_usd:
        return
    try:
        cap = float(max_usd)
    except ValueError:
        return

    log_file = ROOT / f"data/gepa_runs/{target}/history.jsonl"
    if not log_file.exists():
        return
    total = 0.0
    for line in log_file.read_text().splitlines():
        try:
            total += float(json.loads(line).get("cost_usd", 0) or 0)
        except Exception:
            pass
    if total >= cap:
        emit_and_exit(
            0.0,
            {
                "feedback": f"GEPA_MAX_USD=${cap:.2f} cap reached (cumulative ${total:.2f}). Aborting candidate evaluation.",
                "cost_cap_hit": True,
            },
        )


def emit_and_exit(score: float, side_info: dict, code: int = 0) -> None:
    """Emit the final JSON and exit."""
    print(json.dumps({"score": float(score), "side_info": side_info}))
    sys.exit(code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="Path to candidate file (RHS expression)")
    parser.add_argument("--task", default=None, help="Path to task JSON (unused but accepted)")
    parser.add_argument("--target", default=None, help="reasoning|feature_extraction|ocr|expansion (or GEPA_TARGET env)")
    args = parser.parse_args()

    target = args.target or os.environ.get("GEPA_TARGET")
    if not target or target not in TARGET_MAP:
        emit_and_exit(
            0.0,
            {"feedback": f"Invalid target '{target}'. Use one of: {sorted(TARGET_MAP)}"},
            code=1,
        )

    check_cost_cap(target)

    candidate_path = Path(args.candidate)
    if not candidate_path.exists():
        emit_and_exit(0.0, {"feedback": f"Candidate file not found: {candidate_path}"}, code=1)

    try:
        new_prompt = parse_candidate(candidate_path)
    except Exception as e:
        emit_and_exit(0.0, {"feedback": f"Failed to parse candidate: {e!r}"})

    if not isinstance(new_prompt, str) or len(new_prompt) < 30:
        emit_and_exit(
            0.0,
            {
                "feedback": (
                    f"Candidate is not a non-trivial string "
                    f"(type={type(new_prompt).__name__}, "
                    f"len={len(new_prompt) if isinstance(new_prompt, str) else 'n/a'})"
                )
            },
        )

    t0 = time.monotonic()
    try:
        metrics = asyncio.run(run_evaluation(target, new_prompt))
    except KeyboardInterrupt:
        raise
    except Exception as e:
        emit_and_exit(
            0.0, {"feedback": f"Evaluation crashed: {type(e).__name__}: {e}"}
        )

    score, scoring_extras = compute_score(metrics)
    per_sample = scoring_extras["per_sample"]
    breakdown = scoring_extras["score_breakdown"]
    feedback_text = build_feedback(per_sample, breakdown, metrics)

    elapsed = time.monotonic() - t0
    cost_usd = float(metrics.total_cost_usd or 0.0)

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "score": round(score, 4),
        "score_breakdown": breakdown,
        "cost_usd": round(cost_usd, 5),
        "elapsed_s": round(elapsed, 1),
        "candidate_len": len(new_prompt),
        "candidate_hash": f"{hash(new_prompt) & 0xFFFFFFFF:08x}",
    }
    cumulative = append_history(target, entry)

    side_info = {
        "feedback": feedback_text,
        "score_breakdown": breakdown,
        "per_sample": per_sample,
        "metrics": metrics.summary(),
        "cost_usd": cost_usd,
        "cost_usd_cumulative": cumulative,
        "elapsed_s": round(elapsed, 1),
    }
    emit_and_exit(score, side_info)


if __name__ == "__main__":
    main()
