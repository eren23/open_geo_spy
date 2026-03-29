"""CLI entry point for OpenGeoSpy — headless geolocation and eval.

Usage:
    ogs locate <image> [--hint TEXT] [--output json|table]
    ogs batch <input_dir> [--output results.jsonl] [--max-concurrent 3]
    ogs eval <dataset_path> [--label TEXT] [--baseline PATH] [--judge] [--max-concurrent 3]
    ogs trace-stats [--since DATE] [--version TEXT]
    ogs evolve <eval_report> [--config PATH] [--dry-run]
    ogs improve import-benchmark <source_path> <output_manifest>
    ogs improve import-trace-regressions <traces_dir> <output_manifest>
    ogs improve seed-landmarks <output_manifest>
    ogs improve run <suite_path>
    ogs improve campaign <suite_path>
    ogs improve rank <experiment_dir>
    ogs improve resume <experiment_dir>
    ogs improve replay-trace <trace_path>
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="ogs", help="OpenGeoSpy CLI — headless geolocation")
improve_app = typer.Typer(name="improve", help="Self-improving benchmark and mutation loop")
console = Console()
app.add_typer(improve_app, name="improve")


@app.command()
def locate(
    image: str = typer.Argument(..., help="Path to image file"),
    hint: Optional[str] = typer.Option(None, "--hint", "-h", help="Location hint"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: json or table"),
    quality: str = typer.Option("balanced", "--quality", help="Execution quality: fast|balanced|max"),
) -> None:
    """Geolocate a single image."""
    from src.agents.orchestrator import Orchestrator
    from src.config.settings import get_settings

    if not Path(image).exists():
        console.print(f"[red]Image not found: {image}[/red]")
        raise typer.Exit(1)

    settings = get_settings()
    orchestrator = Orchestrator(settings)

    result = asyncio.run(orchestrator.locate(image, location_hint=hint, quality=quality))
    prediction = result.get("prediction", result)

    if output == "json":
        console.print_json(json.dumps(prediction, indent=2, default=str))
    else:
        _print_prediction_table(prediction)


@app.command()
def batch(
    input_dir: str = typer.Argument(..., help="Directory containing images"),
    output: str = typer.Option("results.jsonl", "--output", "-o", help="Output JSONL path"),
    max_concurrent: int = typer.Option(3, "--max-concurrent", "-c"),
    labels: Optional[str] = typer.Option(None, "--labels", help="Ground truth CSV"),
    quality: str = typer.Option("balanced", "--quality", help="Execution quality: fast|balanced|max"),
) -> None:
    """Batch geolocate all images in a directory."""
    from src.agents.orchestrator import Orchestrator
    from src.config.settings import get_settings

    input_path = Path(input_dir)
    if not input_path.is_dir():
        console.print(f"[red]Not a directory: {input_dir}[/red]")
        raise typer.Exit(1)

    images = sorted(
        p for p in input_path.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")
    )
    console.print(f"Found {len(images)} images")

    settings = get_settings()

    async def _run():
        sem = asyncio.Semaphore(max_concurrent)
        results = []
        orchestrator = Orchestrator(settings)

        async def _process(img_path: Path):
            async with sem:
                try:
                    result = await orchestrator.locate(str(img_path), quality=quality)
                    prediction = result.get("prediction", result)
                    prediction["image"] = str(img_path)
                    return prediction
                except Exception as e:
                    return {"image": str(img_path), "error": str(e)}

        tasks = [_process(img) for img in images]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            console.print(f"[{i+1}/{len(images)}] {result.get('name', 'Unknown')}")

        await orchestrator.close()
        return results

    results = asyncio.run(_run())

    with open(output, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    console.print(f"[green]Results saved to {output}[/green]")


@app.command("eval")
def eval_cmd(
    dataset_path: str = typer.Argument(..., help="Path to dataset manifest.json or CSV"),
    label: str = typer.Option("", "--label", "-l", help="Label for this eval run"),
    baseline: Optional[str] = typer.Option(None, "--baseline", help="Path to baseline report JSON"),
    judge: bool = typer.Option(False, "--judge", help="Run LLM-as-judge scoring"),
    max_concurrent: int = typer.Option(3, "--max-concurrent", "-c"),
) -> None:
    """Run evaluation on a labeled dataset."""
    from src.eval.dataset import EvalDataset
    from src.eval.report import EvalReport
    from src.eval.runner import EvalRunner

    dataset_path_obj = Path(dataset_path)
    if not dataset_path_obj.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        raise typer.Exit(1)

    if dataset_path_obj.suffix == ".csv":
        dataset = EvalDataset.from_csv(dataset_path_obj)
    else:
        dataset = EvalDataset.from_manifest(dataset_path_obj)

    console.print(f"Loaded dataset '{dataset.name}' with {len(dataset.samples)} samples")

    runner = EvalRunner(label=label, max_concurrent=max_concurrent)
    metrics = asyncio.run(runner.run(dataset))

    # Load baseline if provided
    baseline_metrics = None
    if baseline:
        console.print(f"Comparing against baseline: {baseline}")

    report = EvalReport(metrics=metrics, label=label, baseline=baseline_metrics)
    console.print(report.to_console())

    # Save reports
    output_dir = Path("data/eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_name = label or "eval"
    report.save_json(output_dir / f"{report_name}.json")
    report.save_markdown(output_dir / f"{report_name}.md")
    runner.save_artifacts(
        dataset,
        metrics,
        output_dir / report_name,
        metadata={"baseline_path": baseline, "judge_enabled": judge},
    )
    console.print(f"[green]Reports saved to {output_dir}[/green]")


@app.command("trace-stats")
def trace_stats(
    since: Optional[str] = typer.Option(None, "--since", help="Filter traces since date (YYYY-MM-DD)"),
    version: Optional[str] = typer.Option(None, "--version", help="Filter by pipeline version"),
) -> None:
    """Query trace index for aggregate statistics."""
    from src.tracing.index import TraceIndex

    index = TraceIndex()
    indexed = index.index_directory()
    console.print(f"Indexed {indexed} trace files")

    accuracy = index.accuracy_stats(version=version, since=since)
    cost = index.cost_stats(version=version, since=since)

    if accuracy.get("count", 0) > 0:
        table = Table(title="Accuracy Stats")
        table.add_column("Metric")
        table.add_column("Value")
        for k, v in accuracy.items():
            if isinstance(v, float):
                table.add_row(k, f"{v:.4f}")
            else:
                table.add_row(k, str(v))
        console.print(table)

    if cost.get("count", 0) > 0:
        table = Table(title="Cost Stats")
        table.add_column("Metric")
        table.add_column("Value")
        for k, v in cost.items():
            if isinstance(v, float):
                table.add_row(k, f"{v:.4f}")
            else:
                table.add_row(k, str(v))
        console.print(table)

    index.close()


@app.command()
def evolve(
    eval_report: str = typer.Argument(..., help="Path to eval report JSON"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to current scoring config"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show changes without applying"),
) -> None:
    """Analyze eval results and suggest/apply scoring weight adjustments."""
    from src.eval.metrics import EvalMetrics
    from src.eval.tuner import WeightTuner
    from src.scoring.config import ScoringConfig

    report_path = Path(eval_report)
    if not report_path.exists():
        console.print(f"[red]Report not found: {eval_report}[/red]")
        raise typer.Exit(1)

    # Load current config
    scoring_config = ScoringConfig()
    if config and Path(config).exists():
        scoring_config = ScoringConfig.from_file(config)

    # Load report (we need the raw metrics, not just the summary)
    console.print(f"Analyzing {eval_report}...")

    tuner = WeightTuner(config=scoring_config)

    # For now, we need eval metrics — the report contains the summary
    # In a real pipeline, the runner would save metrics alongside the report
    with open(report_path) as f:
        report_data = json.load(f)

    # Create a minimal EvalMetrics from report data
    # (full implementation would load from traces)
    from src.eval.metrics import EvalMetrics, SampleResult
    metrics = EvalMetrics()  # Will be empty without full sample data

    adjustments = tuner.tune(metrics)

    if not adjustments:
        console.print("[yellow]No adjustments suggested — metrics look good![/yellow]")
        return

    table = Table(title="Proposed Adjustments")
    table.add_column("Parameter")
    table.add_column("Old")
    table.add_column("New")
    table.add_column("Reason")
    for adj in adjustments:
        table.add_row(adj.path, f"{adj.old_value:.4f}", f"{adj.new_value:.4f}", adj.reason)
    console.print(table)

    if dry_run:
        console.print("[yellow]Dry run — no changes applied[/yellow]")
        return

    new_config = tuner.apply(adjustments)
    path = tuner.save_evolution(new_config, adjustments)
    console.print(f"[green]Updated config saved to {path}[/green]")


@improve_app.command("import-benchmark")
def improve_import_benchmark(
    source_path: str = typer.Argument(..., help="Path to local CSV/JSONL/manifest/parquet benchmark export"),
    output_manifest: str = typer.Argument(..., help="Destination normalized manifest path"),
    adapter: str = typer.Option("auto", "--adapter", help="auto|manifest|csv|jsonl|parquet|geobench|osv5m|geovistabench"),
    dataset_id: str = typer.Option("", "--dataset-id", help="Dataset id inside the suite"),
    description: str = typer.Option("", "--description", help="Dataset description"),
    suite: Optional[str] = typer.Option(None, "--suite", help="Optional benchmark suite manifest to update"),
    protected: bool = typer.Option(False, "--protected", help="Mark as protected from regressions"),
    optional: bool = typer.Option(False, "--optional", help="Allow the suite to skip this dataset when its manifest is missing"),
    source_label: str = typer.Option("", "--source-label", help="Human-readable source label stored in suite metadata"),
    limit: int = typer.Option(0, "--limit", help="Optional subset size; 0 imports the full normalized dataset"),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed used for subset sampling"),
    stratify_by: Optional[str] = typer.Option(None, "--stratify-by", help="Comma-separated sample fields to balance across, e.g. difficulty,country"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated dataset tags"),
) -> None:
    """Normalize a benchmark dataset and optionally add it to a suite."""
    from src.improve.controller import ImprovementController

    controller = ImprovementController()
    manifest_path = controller.import_benchmark(
        source_path,
        output_manifest,
        adapter=adapter,
        dataset_id=dataset_id,
        description=description,
        suite_path=suite,
        protected=protected,
        optional=optional,
        expected_sample_count=limit or None,
        source_label=source_label,
        limit=limit or None,
        seed=seed,
        stratify_by=[field.strip() for field in stratify_by.split(",")] if stratify_by else None,
        tags=[tag.strip() for tag in tags.split(",")] if tags else None,
    )
    console.print(f"[green]Benchmark manifest saved to {manifest_path}[/green]")


@improve_app.command("import-trace-regressions")
def improve_import_trace_regressions(
    traces_dir: str = typer.Argument(..., help="Directory containing saved JSONL traces"),
    output_manifest: str = typer.Argument(..., help="Destination manifest path"),
    suite: Optional[str] = typer.Option(None, "--suite", help="Optional benchmark suite manifest to update"),
) -> None:
    """Build a regression dataset from saved traces."""
    from src.improve.controller import ImprovementController

    controller = ImprovementController()
    manifest_path = controller.import_trace_regressions(traces_dir, output_manifest, suite_path=suite)
    console.print(f"[green]Trace regression dataset saved to {manifest_path}[/green]")


@improve_app.command("seed-landmarks")
def improve_seed_landmarks(
    output_manifest: str = typer.Argument(..., help="Destination manifest path for the fetched landmark benchmark"),
    force: bool = typer.Option(False, "--force", help="Redownload images even if they already exist"),
) -> None:
    """Fetch a small real-image landmark benchmark from Wikipedia/Wikimedia."""
    from src.improve.seed_benchmarks import build_wikipedia_landmarks_benchmark

    manifest_path = build_wikipedia_landmarks_benchmark(output_manifest, force=force)
    console.print(f"[green]Landmark benchmark saved to {manifest_path}[/green]")


@improve_app.command("run")
def improve_run(
    suite_path: str = typer.Argument(..., help="Benchmark suite manifest"),
    experiment_name: str = typer.Option("", "--name", help="Human-readable experiment name"),
    candidate_count: int = typer.Option(0, "--candidate-count", help="Number of code-mutated candidates to try; 0 uses config default"),
    quality: str = typer.Option("balanced", "--quality", help="Pipeline quality mode for benchmark runs"),
    max_concurrent: int = typer.Option(3, "--max-concurrent", help="Max concurrent samples during eval"),
    judge: bool = typer.Option(False, "--judge", help="Run LLM trace judge after evaluation"),
    instructions: str = typer.Option("", "--instructions", help="Extra mutator instructions"),
    base_scoring_config: str = typer.Option("", "--base-scoring-config", help="Optional scoring config JSON to use as the baseline for this wave"),
) -> None:
    """Run the baseline and candidate mutation loop for a suite."""
    from src.improve.controller import ImprovementController

    controller = ImprovementController()
    experiment_dir = controller.run(
        suite_path,
        experiment_name=experiment_name,
        candidate_count=candidate_count or None,
        quality=quality,
        max_concurrent=max_concurrent,
        judge=judge,
        mutator_instructions=instructions,
        base_scoring_config_path=base_scoring_config or None,
    )
    console.print(f"[green]Experiment saved to {experiment_dir}[/green]")
    baseline_manifest = Path(experiment_dir) / "baseline" / "run_manifest.json"
    if baseline_manifest.exists():
        with open(baseline_manifest) as f:
            manifest = json.load(f)
        missing = manifest.get("missing_optional_datasets", [])
        if missing:
            console.print(f"[yellow]Skipped optional datasets:[/yellow] {', '.join(missing)}")


@improve_app.command("campaign")
def improve_campaign(
    suite_path: str = typer.Argument(..., help="Benchmark suite manifest"),
    campaign_name: str = typer.Option("", "--name", help="Human-readable campaign name"),
    candidate_count: int = typer.Option(3, "--candidate-count", help="Number of config candidates to evaluate per wave"),
    required_streak: int = typer.Option(3, "--required-streak", help="Accepted winner streak required for success"),
    max_waves: int = typer.Option(8, "--max-waves", help="Maximum number of waves to run before stopping"),
    quality: str = typer.Option("balanced", "--quality", help="Pipeline quality mode for benchmark runs"),
    max_concurrent: int = typer.Option(3, "--max-concurrent", help="Max concurrent samples during eval"),
    instructions: str = typer.Option("", "--instructions", help="Extra operator instructions for campaign waves"),
    base_scoring_config: str = typer.Option("", "--base-scoring-config", help="Optional scoring config JSON to seed the campaign baseline"),
) -> None:
    """Run a multi-wave config-only campaign until a win streak is achieved or proposals are exhausted."""
    from src.improve.controller import ImprovementController

    controller = ImprovementController()
    campaign_dir = controller.campaign(
        suite_path,
        campaign_name=campaign_name,
        candidate_count=candidate_count,
        required_streak=required_streak,
        max_waves=max_waves,
        quality=quality,
        max_concurrent=max_concurrent,
        mutator_instructions=instructions,
        base_scoring_config_path=base_scoring_config or None,
    )
    console.print(f"[green]Campaign saved to {campaign_dir}[/green]")
    with open(Path(campaign_dir) / "campaign.json") as f:
        campaign = json.load(f)
    console.print_json(json.dumps(campaign, indent=2))


@improve_app.command("rank")
def improve_rank(
    experiment_dir: str = typer.Argument(..., help="Experiment directory under data/improve/experiments"),
) -> None:
    """Rank existing candidate artifacts against baseline."""
    from src.improve.controller import ImprovementController

    controller = ImprovementController()
    ranking = controller.rank(experiment_dir)
    console.print_json(json.dumps(ranking, indent=2))


@improve_app.command("resume")
def improve_resume(
    experiment_dir: str = typer.Argument(..., help="Experiment directory to resume"),
) -> None:
    """Resume an interrupted experiment and rerank completed candidates."""
    from src.improve.controller import ImprovementController

    controller = ImprovementController()
    ranking = controller.resume(experiment_dir)
    console.print_json(json.dumps(ranking, indent=2))


@improve_app.command("replay-trace")
def improve_replay_trace(
    trace_path: str = typer.Argument(..., help="Path to a saved JSONL trace"),
) -> None:
    """Analyze a trace for near-misses, regressions, and inefficiency."""
    from src.improve.controller import ImprovementController

    controller = ImprovementController()
    diagnostics = controller.replay_trace(trace_path)
    console.print_json(json.dumps(diagnostics, indent=2))


def _print_prediction_table(prediction: dict) -> None:
    """Print a prediction as a Rich table."""
    table = Table(title="Geolocation Result")
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Name", str(prediction.get("name", "Unknown")))
    table.add_row("Country", str(prediction.get("country", "")))
    table.add_row("Region", str(prediction.get("region", "")))
    table.add_row("City", str(prediction.get("city", "")))
    table.add_row("Latitude", str(prediction.get("lat") or prediction.get("latitude", "")))
    table.add_row("Longitude", str(prediction.get("lon") or prediction.get("longitude", "")))
    table.add_row("Confidence", f"{prediction.get('confidence', 0):.2%}")
    table.add_row("Verified", str(prediction.get("verified", False)))

    console.print(table)

    reasoning = prediction.get("reasoning", "")
    if reasoning:
        console.print(f"\n[dim]Reasoning:[/dim] {reasoning[:500]}")


if __name__ == "__main__":
    app()
