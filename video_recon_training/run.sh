#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-1}" \
  --module video_recon_training.train_qwen3vl_video_rae_streaming \
  --config video_recon_training/config.yaml \
  "$@"
