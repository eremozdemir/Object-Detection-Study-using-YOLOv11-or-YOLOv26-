"""
train.py
--------
CLI-friendly training script for all experiments in this project.
Wraps Ultralytics YOLO so every run has consistent logging and
saves its config alongside the weights.

Usage examples:
    # Baseline
    python scripts/train.py --model yolo11s --exp baseline --epochs 50

    # Experiment: larger image size
    python scripts/train.py --model yolo11s --exp exp_imgsz1280 --imgsz 1280 --epochs 50

    # Improvement: weight decay regularisation
    python scripts/train.py --model yolo11s --exp exp_wd0.01 --weight_decay 0.01 --epochs 50

    # Multi-version comparison
    python scripts/train.py --model yolov8s  --exp compare_yolov8s  --epochs 50
    python scripts/train.py --model yolo26s  --exp compare_yolo26s   --epochs 50
"""

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


# Map short model names to Ultralytics checkpoint filenames
MODEL_MAP = {
    # YOLO11
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo11x": "yolo11x.pt",
    # YOLO26
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
    "yolo26m": "yolo26m.pt",
    "yolo26l": "yolo26l.pt",
    "yolo26x": "yolo26x.pt",
    # YOLOv8
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    # YOLOv9
    "yolov9c": "yolov9c.pt",
    "yolov9e": "yolov9e.pt",
    # YOLOv10
    "yolov10n": "yolov10n.pt",
    "yolov10s": "yolov10s.pt",
    "yolov10m": "yolov10m.pt",
    # YOLOv5
    "yolov5n": "yolov5nu.pt",
    "yolov5s": "yolov5su.pt",
    "yolov5m": "yolov5mu.pt",
}


def parse_args():
    p = argparse.ArgumentParser(description="Train a YOLO model on VisDrone.")
    p.add_argument("--model",         type=str,   default="yolo11s",       help="Model key (see MODEL_MAP)")
    p.add_argument("--exp",           type=str,   default="baseline",      help="Experiment name (used as run directory)")
    p.add_argument("--data",          type=str,   default="configs/visdrone.yaml", help="Dataset config path")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--imgsz",         type=int,   default=640)
    p.add_argument("--batch",         type=int,   default=16)
    p.add_argument("--lr0",           type=float, default=0.01,            help="Initial learning rate")
    p.add_argument("--lrf",           type=float, default=0.01,            help="Final LR fraction (lr0 * lrf)")
    p.add_argument("--weight_decay",  type=float, default=0.0005)
    p.add_argument("--dropout",       type=float, default=0.0)
    p.add_argument("--patience",      type=int,   default=20,              help="Early stopping patience (0=off)")
    p.add_argument("--augment",       action="store_true",                 help="Enable extra augmentation")
    p.add_argument("--cos_lr",        action="store_true",                 help="Use cosine LR schedule")
    p.add_argument("--freeze",        type=int,   default=0,               help="Freeze first N layers")
    p.add_argument("--device",        type=str,   default="",              help="cuda device, e.g. 0 or cpu")
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--project",       type=str,   default="results",       help="Root directory for run outputs")
    p.add_argument("--resume",        action="store_true",                 help="Resume last interrupted run")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths relative to the project root (parent of scripts/)
    project_root = Path(__file__).resolve().parent.parent
    data_cfg = (project_root / args.data).resolve()

    if not data_cfg.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_cfg}")

    # Select checkpoint
    if args.model not in MODEL_MAP:
        raise ValueError(f"Unknown model '{args.model}'. Choose from: {list(MODEL_MAP.keys())}")
    checkpoint = MODEL_MAP[args.model]

    print(f"\n{'='*60}")
    print(f"  Experiment : {args.exp}")
    print(f"  Model      : {checkpoint}")
    print(f"  Dataset    : {data_cfg}")
    print(f"  Epochs     : {args.epochs}   |  imgsz: {args.imgsz}  |  batch: {args.batch}")
    print(f"{'='*60}\n")

    model = YOLO(checkpoint)

    train_kwargs = dict(
        data=str(data_cfg),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        augment=args.augment,
        cos_lr=args.cos_lr,
        freeze=args.freeze if args.freeze > 0 else None,
        device=args.device if args.device else None,
        workers=args.workers,
        project=str(project_root / args.project),
        name=args.exp,
        exist_ok=args.resume,
        resume=args.resume,
        verbose=True,
        plots=True,
    )

    t0 = time.time()
    results = model.train(**train_kwargs)
    elapsed = time.time() - t0

    # Save a run summary alongside the weights
    run_dir = Path(results.save_dir)
    summary = {
        "experiment": args.exp,
        "model": checkpoint,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "train_time_s": round(elapsed, 1),
        "best_mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "best_mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
    }
    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRun saved to: {run_dir}")
    print(f"Training time: {elapsed/60:.1f} min")
    print(f"Best mAP50: {summary['best_mAP50']:.4f}  |  mAP50-95: {summary['best_mAP50_95']:.4f}")


if __name__ == "__main__":
    main()
