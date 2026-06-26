#!/usr/bin/env bash
# Provision a Vast.ai GPU box for batterycv (SAM pseudo-labeling + YOLO11 training).
# Idempotent. Designed for the Vast *PyTorch* base image, which already ships a working
# torch+cu128 in /venv/main -- we DO NOT reinstall torch (that risks breaking the tuned
# Blackwell/sm_120 build); we only add the project deps and the SAM checkpoint.
#
#   bash vast/setup.sh
set -euo pipefail

VENV="${VENV:-/venv/main}"
CKPT_DIR="${CKPT_DIR:-/workspace/checkpoints}"
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# shellcheck disable=SC1091
source "$VENV/bin/activate" 2>/dev/null || { echo "no venv at $VENV"; exit 1; }

echo ">> python: $(python --version)"
python - <<'PY'
import torch
print(">> torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY

# Project deps only -- torch is already present and satisfies ultralytics/SAM.
PIP="uv pip"; command -v uv >/dev/null 2>&1 || PIP="pip"
echo ">> installing project deps with: $PIP"
$PIP install ultralytics opencv-python-headless segment-anything supervision pandas pyyaml tqdm

# SAM checkpoint (vit_h ~2.5 GB) -- skip if already downloaded.
mkdir -p "$CKPT_DIR"
CKPT="$CKPT_DIR/sam_vit_h_4b8939.pth"
if [ -s "$CKPT" ]; then
  echo ">> SAM checkpoint present: $CKPT ($(du -h "$CKPT" | cut -f1))"
else
  echo ">> downloading SAM vit_h -> $CKPT"
  wget -q --show-progress -O "$CKPT" "$SAM_URL"
fi

python -c "import ultralytics, segment_anything, cv2; print('>> deps OK:', ultralytics.__version__)"
echo ">> setup complete. Next: stage data, then  bash vast/run_pipeline.sh"
