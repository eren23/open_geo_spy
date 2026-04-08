#!/usr/bin/env bash
# Run one stage of GEPA optimization on open_geo_spy.
#
# Usage:
#   scripts/run_gepa_stage.sh <target> <max_evals> [max_usd]
#
# Targets: reasoning | feature_extraction | ocr | expansion
#
# Copies scripts/gepa_configs/<target>.yaml to gepa.yaml at the repo root,
# sets GEPA_TARGET and GEPA_MAX_USD env vars, then invokes gepa-optim evolve.

set -euo pipefail

TARGET="${1:-}"
MAX_EVALS="${2:-20}"
MAX_USD="${3:-4.0}"

if [ -z "$TARGET" ]; then
  echo "Usage: $0 <target> <max_evals> [max_usd]" >&2
  echo "Targets: reasoning feature_extraction ocr expansion" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_SRC="scripts/gepa_configs/${TARGET}.yaml"
if [ ! -f "$CONFIG_SRC" ]; then
  echo "No config for target '$TARGET' at $CONFIG_SRC" >&2
  exit 1
fi

# Load OPENROUTER_API_KEY from .env if not already set
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . .env
  set +a
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY not set (checked env and .env)" >&2
  exit 1
fi

cp "$CONFIG_SRC" gepa.yaml

export GEPA_TARGET="$TARGET"
export GEPA_MAX_USD="$MAX_USD"

GEPA_OPTIM_BIN="/Users/eren/Documents/AI/gepa_optim/.venv/bin/gepa-optim"

echo "=== GEPA stage: $TARGET ==="
echo "    max_evals=$MAX_EVALS max_usd=\$$MAX_USD"
echo "    config=$CONFIG_SRC -> gepa.yaml"
echo ""

"$GEPA_OPTIM_BIN" evolve . --max-evals "$MAX_EVALS" --model openrouter/z-ai/glm-5.1
