"""
evaluate.py
-----------
Evaluate a trained YOLO model on the VisDrone val or test-dev split,
then print and save a structured metrics report.

Usage:
    python scripts/evaluate.py --weights results/baseline/weights/best.pt
    python scripts/evaluate.py --weights results/baseline/weights/best.pt --split test-dev
    python scripts/evaluate.py --weights results/baseline/weights/best.pt --save_json
"""

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained YOLO checkpoint on VisDrone.")
    p.add_argument("--weights",  type=str, required=True,          help="Path to .pt weights file")
    p.add_argument("--data",     type=str, default="configs/visdrone.yaml")
    p.add_argument("--split",    type=str, default="val",           choices=["val", "test-dev"],
                   help="Dataset split to evaluate on")
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--conf",     type=float, default=0.001,         help="Confidence threshold")
    p.add_argument("--iou",      type=float, default=0.6,           help="IoU threshold for NMS")
    p.add_argument("--device",   type=str, default="",             help="cuda device or cpu")
    p.add_argument("--save_json", action="store_true",             help="Save metrics to JSON")
    return p.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_cfg = (project_root / args.data).resolve()
    weights = Path(args.weights).resolve()

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))

    print(f"\nEvaluating: {weights.name}  |  split: {args.split}  |  imgsz: {args.imgsz}")

    t0 = time.time()
    metrics = model.val(
        data=str(data_cfg),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device if args.device else None,
        plots=True,
        verbose=True,
    )
    elapsed = time.time() - t0

    rd = metrics.results_dict
    report = {
        "weights":      str(weights),
        "split":        args.split,
        "imgsz":        args.imgsz,
        "eval_time_s":  round(elapsed, 1),
        "mAP50":        round(float(rd.get("metrics/mAP50(B)", 0)), 4),
        "mAP50_95":     round(float(rd.get("metrics/mAP50-95(B)", 0)), 4),
        "precision":    round(float(rd.get("metrics/precision(B)", 0)), 4),
        "recall":       round(float(rd.get("metrics/recall(B)", 0)), 4),
        "fitness":      round(float(rd.get("fitness", 0)), 4),
    }

    print("\n" + "="*50)
    print("  Evaluation Results")
    print("="*50)
    for k, v in report.items():
        print(f"  {k:<18}: {v}")
    print("="*50)

    if args.save_json:
        out_path = weights.parent.parent / f"eval_{args.split}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
