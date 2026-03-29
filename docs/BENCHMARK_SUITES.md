# Benchmark Suites

`data/eval/suites/core_v1.json` is the first committed improvement-loop suite.

## Current Behavior

- `wikipedia_landmarks_v1` is the required real-image seed dataset.
- The old `smoke_sample` dataset is no longer part of `core_v1` because its checked-in files are placeholder gray images.
- `geovistabench_subset`, `osv5m_subset`, and `trace_regressions` are optional.
- Running `ogs improve seed-landmarks data/eval/imports/wikipedia_landmarks_v1/manifest.json` materializes the required core dataset.
- Running `ogs improve run data/eval/suites/core_v1.json` then uses the real landmark set and records any still-missing optional datasets in the run manifest.

## Real Seed Command

```bash
ogs improve seed-landmarks \
  data/eval/imports/wikipedia_landmarks_v1/manifest.json
```

## Planned Public Import Targets

The suite expects future normalized manifests at:

- `data/eval/imports/wikipedia_landmarks_v1/manifest.json`
- `data/eval/imports/geovistabench_subset/manifest.json`
- `data/eval/imports/osv5m_subset/manifest.json`
- `data/eval/regressions/trace_regressions.json`

## Example Import Commands

GeoVistaBench, 25-sample deterministic subset:

```bash
ogs improve import-benchmark /path/to/geovistabench_export.jsonl \
  data/eval/imports/geovistabench_subset/manifest.json \
  --adapter geovistabench \
  --dataset-id geovistabench_subset \
  --description "25-sample GeoVistaBench slice for core_v1" \
  --source-label GeoVistaBench \
  --limit 25 \
  --seed 42 \
  --stratify-by difficulty,split,country
```

OSV5M, 25-sample deterministic subset:

```bash
ogs improve import-benchmark /path/to/osv5m_export.jsonl \
  data/eval/imports/osv5m_subset/manifest.json \
  --adapter osv5m \
  --dataset-id osv5m_subset \
  --description "25-sample OSV5M slice for core_v1" \
  --source-label OSV5M \
  --limit 25 \
  --seed 42 \
  --stratify-by country,urban_rural,split
```

Trace regressions:

```bash
ogs improve import-trace-regressions \
  data/traces \
  data/eval/regressions/trace_regressions.json
```

## Running The Suite

```bash
ogs improve run data/eval/suites/core_v1.json --candidate-count 3 --judge
```

If optional public imports are still missing, the run manifest records them under `missing_optional_datasets`.
