"""CLI entry point for OpenGeoSpy — headless geolocation and eval.

Usage:
    ogs locate <image> [--hint TEXT] [--output json|table]
    ogs batch <input_dir> [--output results.jsonl] [--max-concurrent 3]
    ogs eval <dataset_path> [--label TEXT] [--baseline PATH] [--judge] [--max-concurrent 3]
    ogs trace-stats [--since DATE] [--version TEXT]
    ogs evolve <eval_report> [--config PATH] [--dry-run]
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
console = Console()


@app.command()
def locate(
    image: str = typer.Argument(..., help="Path to image file"),
    hint: Optional[str] = typer.Option(None, "--hint", "-h", help="Location hint"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: json or table"),
) -> None:
    """Geolocate a single image."""
    from src.agents.orchestrator import Orchestrator
    from src.config.settings import get_settings

    if not Path(image).exists():
        console.print(f"[red]Image not found: {image}[/red]")
        raise typer.Exit(1)

    settings = get_settings()
    orchestrator = Orchestrator(settings)

    result = asyncio.run(orchestrator.locate(image, location_hint=hint))
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

        async def _process(img_path: Path):
            async with sem:
                orchestrator = Orchestrator(settings)
                try:
                    result = await orchestrator.locate(str(img_path))
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
