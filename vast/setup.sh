#!/usr/bin/env bash
# Provision a Vast.ai GPU box (RTX 5090) for batterycv: SAM pseudo-labeling + YOLO training.
# Notes (from prior Vast workflow):
#   - RTX 5090 (Blackwell) needs the cu128 torch wheels.
#   - Confirm Inet >= 200 Mbit/s before moving the ~10 GB dataset.
#   - Cap dataloader workers on high-core boxes (set --workers in train, or OMP threads).
set -euo pipefail

echo ">> python: $(python3 --version)"

# 1) PyTorch for Blackwell (cu128)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) project deps
pip install ultralytics opencv-python-headless numpy pandas pyyaml tqdm supervision
pip install git+https://github.com/facebookresearch/segment-anything.git

# 3) SAM checkpoint (vit_h ~2.5 GB)
mkdir -p checkpoints
if [ ! -f checkpoints/sam_vit_h_4b8939.pth ]; then
  wget -q --show-progress -O checkpoints/sam_vit_h_4b8939.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

# 4) sanity check
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY

cat <<'EOF'

NEXT (on the box, after staging data to ./batterycv-data/raw):
  python scripts/pseudo_label_sam.py --ckpt checkpoints/sam_vit_h_4b8939.pth --model vit_h
  python scripts/train_detector.py --model yolo11s.pt --epochs 80 --imgsz 1024 --batch 16 --device 0
Then pull runs/detect/battery_yolo11/weights/best.pt back to the local repo.
EOF
