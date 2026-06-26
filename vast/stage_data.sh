#!/usr/bin/env bash
# Push code + (JPEG) data to the GPU box. Run LOCALLY (Windows git-bash ok; uses tar over ssh,
# no rsync needed). Ships the compact JPEG stage built by scripts/bmp_to_jpg.py (~0.8 GB), not
# the 9 GB of BMPs.
#
#   bash vast/bmp_to_jpg first (once):  python scripts/bmp_to_jpg.py
#   bash vast/stage_data.sh root@HOST 29528
set -euo pipefail

TARGET="${1:?usage: stage_data.sh <ssh-target> <port> [data_root]}"
PORT="${2:?usage: stage_data.sh <ssh-target> <port> [data_root]}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# local data_root from paths.yaml (arg 3 overrides)
LOCAL_DATA="${3:-$(python -c "from batterycv.config import load_paths; print(load_paths()['data_root'])" 2>/dev/null || echo "$HOME/batterycv-data")}"
STAGE_RAW="$LOCAL_DATA/stage/raw"
EVAL_DIR="$LOCAL_DATA/eval"
BOX_DATA="/workspace/batterycv-data"
SSH="ssh -p $PORT -o ConnectTimeout=25 $TARGET"

[ -d "$STAGE_RAW" ] || { echo "no JPEG stage at $STAGE_RAW -- run: python scripts/bmp_to_jpg.py"; exit 1; }

echo ">> creating box dirs"
$SSH "mkdir -p /workspace/batterycv $BOX_DATA"

echo ">> pushing repo code -> /workspace/batterycv"
tar -C "$REPO" --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
    --exclude='runs' --exclude='*.pyc' -cf - . | $SSH "tar -C /workspace/batterycv -xf -"

echo ">> pushing JPEG frames (~0.8 GB) -> $BOX_DATA/raw"
tar -C "$STAGE_RAW" -cf - . | $SSH "mkdir -p $BOX_DATA/raw && tar -C $BOX_DATA/raw -xf -"

if [ -d "$EVAL_DIR" ]; then
  echo ">> pushing eval set (images + any labels) -> $BOX_DATA/eval"
  tar -C "$EVAL_DIR" -cf - . | $SSH "mkdir -p $BOX_DATA/eval && tar -C $BOX_DATA/eval -xf -"
fi

echo ">> verify on box:"
$SSH "echo '  raw frames:' \$(find $BOX_DATA/raw -name '*.jpg' | wc -l); echo '  eval imgs:' \$(find $BOX_DATA/eval/images -name '*.jpg' 2>/dev/null | wc -l); echo '  eval labels:' \$(find $BOX_DATA/eval/labels -name '*.txt' 2>/dev/null | wc -l)"
echo ">> staging done. On the box:  cd /workspace/batterycv && bash vast/setup.sh && bash vast/run_pipeline.sh"
