#!/usr/bin/env bash
# Pull trained weights + run artifacts back from the GPU box to the local repo.
# Run LOCALLY. The box has no persistent volume, so ALWAYS pull before stopping/destroying it.
#
#   bash vast/pull_results.sh root@HOST 29528
set -euo pipefail

TARGET="${1:?usage: pull_results.sh <ssh-target> <port>}"
PORT="${2:?usage: pull_results.sh <ssh-target> <port>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOX_REPO="${3:-/workspace/batterycv}"
SSH="ssh -p $PORT -o ConnectTimeout=25 $TARGET"

mkdir -p "$REPO/runs"
echo ">> pulling runs/detect (weights, curves, val batches) -> $REPO/runs/"
$SSH "cd $BOX_REPO && tar -cf - runs/detect runs/eval 2>/dev/null" | tar -C "$REPO" -xf - || true

BEST="$REPO/runs/detect/battery_yolo11/weights/best.pt"
if [ -s "$BEST" ]; then
  echo ">> got weights: $BEST ($(du -h "$BEST" | cut -f1))"
else
  echo ">> WARN: best.pt not found locally -- check the box's training run"
fi
echo ">> done. (runs/ is gitignored; copy best.pt somewhere durable if you destroy the box.)"
