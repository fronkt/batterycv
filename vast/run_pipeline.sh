#!/usr/bin/env bash
# Run the full detector pipeline ON the GPU box: SAM pseudo-label -> YOLO11 train -> eval.
# Idempotent / resumable (skips a step whose output exists unless --force). Logs to vast/logs/.
#
#   bash vast/run_pipeline.sh                       # full run
#   bash vast/run_pipeline.sh --limit 40            # quick end-to-end smoke test
#   bash vast/run_pipeline.sh --epochs 120 --batch 24
set -euo pipefail

VENV="${VENV:-/venv/main}"
DATA_ROOT="${DATA_ROOT:-/workspace/batterycv-data}"
CKPT="${CKPT:-/workspace/checkpoints/sam_vit_h_4b8939.pth}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="yolo11s.pt"; EPOCHS=80; IMGSZ=1024; BATCH=16; WORKERS=8; LIMIT=0; FORCE=0; SKIP_EVAL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --imgsz) IMGSZ="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    --ckpt) CKPT="$2"; shift 2;;
    --force) FORCE=1; shift;;
    --skip-eval) SKIP_EVAL=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

cd "$REPO"
mkdir -p vast/logs
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export OMP_NUM_THREADS="$WORKERS"   # match dataloader workers; avoids oversubscription on big boxes

# Linux paths for the box, consumed by batterycv.config via $BATTERYCV_PATHS.
RT="$REPO/configs/paths.runtime.yaml"
cat > "$RT" <<YAML
data_root: $DATA_ROOT
raw_dir: $DATA_ROOT/raw
work_dir: $DATA_ROOT/work
manifest: $DATA_ROOT/manifest.csv
yolo_dataset: $DATA_ROOT/yolo
eval_dir: $DATA_ROOT/eval
zip_path: $DATA_ROOT/raw
YAML
export BATTERYCV_PATHS="$RT"
echo ">> BATTERYCV_PATHS=$RT (DATA_ROOT=$DATA_ROOT)"

ts() { date +%H:%M:%S; }

# 1) SAM pseudo-labels -------------------------------------------------------------
DATA_YAML="$DATA_ROOT/yolo/data.yaml"
if [ "$FORCE" = 0 ] && [ -s "$DATA_YAML" ] && ls "$DATA_ROOT"/yolo/labels/train/*.txt >/dev/null 2>&1; then
  echo ">> [$(ts)] pseudo-labels present ($DATA_YAML) -- skip (use --force to redo)"
else
  [ -s "$CKPT" ] || { echo "SAM checkpoint missing: $CKPT (run vast/setup.sh)"; exit 1; }
  echo ">> [$(ts)] STEP 1 SAM pseudo-labeling (limit=$LIMIT) -> vast/logs/pseudo_label.log"
  python scripts/pseudo_label_sam.py --ckpt "$CKPT" --model vit_h --device cuda \
    --limit "$LIMIT" 2>&1 | tee vast/logs/pseudo_label.log
fi

# 2) Train YOLO11 ------------------------------------------------------------------
BEST="runs/detect/battery_yolo11/weights/best.pt"
if [ "$FORCE" = 0 ] && [ -s "$BEST" ]; then
  echo ">> [$(ts)] trained weights present ($BEST) -- skip (use --force to retrain)"
else
  echo ">> [$(ts)] STEP 2 training $MODEL ${EPOCHS}ep imgsz=$IMGSZ batch=$BATCH -> vast/logs/train.log"
  python scripts/train_detector.py --model "$MODEL" --epochs "$EPOCHS" --imgsz "$IMGSZ" \
    --batch "$BATCH" --workers "$WORKERS" --device 0 2>&1 | tee vast/logs/train.log
fi

# 3) Eval vs hand-verified set (only if labeled) -----------------------------------
if [ "$SKIP_EVAL" = 1 ]; then
  echo ">> [$(ts)] eval skipped (--skip-eval)"
elif ls "$DATA_ROOT"/eval/labels/*.txt >/dev/null 2>&1; then
  echo ">> [$(ts)] STEP 3 eval vs hand-verified set -> vast/logs/eval.log"
  python scripts/eval_detection.py --weights "$BEST" --device 0 2>&1 | tee vast/logs/eval.log
else
  echo ">> [$(ts)] STEP 3 SKIPPED: no hand-verified labels in $DATA_ROOT/eval/labels"
  echo "   (label the 72 eval frames, then: python scripts/eval_detection.py --weights $BEST --device 0)"
fi

echo ""
echo ">> [$(ts)] PIPELINE DONE."
echo "   weights : $REPO/$BEST"
echo "   results : $REPO/runs/detect/battery_yolo11/   (curves, val batches)"
echo "   pull back with:  bash vast/pull_results.sh <ssh-target> <port>   (run locally)"
