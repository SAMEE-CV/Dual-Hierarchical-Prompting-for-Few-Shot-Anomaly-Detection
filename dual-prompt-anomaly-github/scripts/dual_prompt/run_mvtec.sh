#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATA_ROOT [CATEGORY] [EPOCHS] [DEVICE]"
  exit 2
fi

DATA_ROOT="$1"
CATEGORY="${2:-all}"
EPOCHS="${3:-50}"
DEVICE="${4:-auto}"

python -m dual_prompt train \
  --config configs/trainers/DualPrompt/vit_b16_4shot.yaml \
  --data-root "$DATA_ROOT" \
  --category "$CATEGORY" \
  --epochs "$EPOCHS" \
  --device "$DEVICE"

