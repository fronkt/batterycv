# Vast.ai GPU runbook — detector pipeline

Turnkey path to build the single-class battery detector on a rented GPU box
(SAM pseudo-labels → YOLO11 train → eval). Designed for the Vast **PyTorch** base image
(ships working `torch+cu128` for Blackwell / RTX 5090).

## One-time, locally
Re-encode the 9 GB of BMPs to a ~0.8 GB JPEG stage (15× less to transfer; visually lossless):
```bash
python scripts/bmp_to_jpg.py
```

## Per box
With a box's `ssh -p <PORT> root@<HOST>` handy:

```bash
# 1. push code + JPEG data + eval set to the box  (local; uses tar over ssh, no rsync needed)
bash vast/stage_data.sh root@<HOST> <PORT>

# 2. on the box: install deps + SAM checkpoint, then run the pipeline
ssh -p <PORT> root@<HOST>
cd /workspace/batterycv
bash vast/setup.sh
tmux new -s bcv 'bash vast/run_pipeline.sh 2>&1 | tee vast/logs/pipeline.log'   # detach: Ctrl-b d

# 3. when done: pull weights/artifacts back  (local). The box has NO persistent volume —
#    always pull before you stop/destroy it.
bash vast/pull_results.sh root@<HOST> <PORT>
```

## Notes / knobs
- **Smoke test first:** `bash vast/run_pipeline.sh --limit 40` runs the whole chain on 40 frames
  (~minutes) to prove it end-to-end before committing the full multi-hour run.
- `run_pipeline.sh` is **resumable**: it skips pseudo-labeling if `data.yaml`+train labels exist,
  and skips training if `best.pt` exists. Use `--force` to redo.
- **Eval** auto-runs only if the hand-verified labels exist in `batterycv-data/eval/labels/`;
  otherwise it's skipped with a hint. Training still gets a real val signal from a 10 % held-out
  slice of the SAM pseudo-labels (see `pseudo_label_sam.py --val-frac`).
- Tunables: `--model yolo11{n,s,m,l}.pt --epochs --imgsz --batch --workers`. Keep `--workers`
  modest (≈8) on high-core boxes (dataloader oversubscription, per the Vast workflow notes).
- Paths on the box are injected via `$BATTERYCV_PATHS` (a generated `configs/paths.runtime.yaml`
  with `/workspace` paths) — the committed Windows `paths.yaml` is left untouched.
- `torch` is **not** reinstalled by `setup.sh` (the base image's cu128 build already targets
  sm_120); only project deps + the SAM checkpoint are added.
