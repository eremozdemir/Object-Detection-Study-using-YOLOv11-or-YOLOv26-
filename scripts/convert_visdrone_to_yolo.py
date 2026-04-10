"""
convert_visdrone_to_yolo.py
----------------------------
Converts VisDrone2019-DET annotations (Task 1: Object Detection in Images)
from the native VisDrone format to YOLO format.

VisDrone annotation format (per line):
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

YOLO annotation format (per line, all values normalised 0-1):
    <class_id> <x_center> <y_center> <width> <height>

VisDrone category mapping (original 1-indexed → YOLO 0-indexed):
    0  : ignored regions  → SKIP
    1  : pedestrian       → 0
    2  : people           → 1
    3  : bicycle          → 2
    4  : car              → 3
    5  : van              → 4
    6  : truck            → 5
    7  : tricycle         → 6
    8  : awning-tricycle  → 7
    9  : bus              → 8
    10 : motor            → 9
    11 : others           → SKIP

Usage:
    python scripts/convert_visdrone_to_yolo.py
    python scripts/convert_visdrone_to_yolo.py --data_root Data --splits train val test-dev
"""

import argparse
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# VisDrone category_id → YOLO class_id  (0 and 11 are skipped)
CATEGORY_MAP = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}

SPLIT_DIRS = {
    "train":     "VisDrone2019-DET-train",
    "val":       "VisDrone2019-DET-val",
    "test-dev":  "VisDrone2019-DET-test-dev",
    "test-challenge": "VisDrone2019-DET-test-challenge",
}


def convert_annotation(ann_path: Path, img_path: Path, label_path: Path) -> int:
    """
    Convert a single VisDrone annotation file to YOLO format.

    Returns the number of valid objects written.
    """
    # Get image dimensions for normalisation
    with Image.open(img_path) as img:
        img_w, img_h = img.size

    lines_out = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue

            x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            category = int(parts[5])

            if category not in CATEGORY_MAP:
                continue  # skip ignored / others

            # Convert to YOLO normalised cx, cy, w, h
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Clamp to [0, 1]
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)

            if nw <= 0 or nh <= 0:
                continue

            yolo_class = CATEGORY_MAP[category]
            lines_out.append(f"{yolo_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        f.write("\n".join(lines_out))

    return len(lines_out)


def convert_split(data_root: Path, split_name: str, split_dir: str):
    split_path = data_root / split_dir
    ann_dir = split_path / "annotations"
    img_dir = split_path / "images"
    label_dir = split_path / "labels"

    if not ann_dir.exists():
        print(f"  [SKIP] No annotations directory found at {ann_dir}")
        return

    ann_files = sorted(ann_dir.glob("*.txt"))
    if not ann_files:
        print(f"  [SKIP] No annotation files found in {ann_dir}")
        return

    print(f"\nConverting split: {split_name} ({len(ann_files)} files) ...")

    total_objects = 0
    skipped = 0
    for ann_path in tqdm(ann_files, desc=split_name):
        stem = ann_path.stem
        # Find matching image (jpg or png)
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            candidate = img_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            skipped += 1
            continue

        label_path = label_dir / (stem + ".txt")
        n = convert_annotation(ann_path, img_path, label_path)
        total_objects += n

    print(f"  Done. {len(ann_files) - skipped} files converted, {skipped} skipped, {total_objects} objects written.")


def main():
    parser = argparse.ArgumentParser(description="Convert VisDrone annotations to YOLO format.")
    parser.add_argument("--data_root", type=str, default="Data",
                        help="Path to the Data directory (default: Data)")
    parser.add_argument("--splits", nargs="+",
                        default=["train", "val", "test-dev"],
                        choices=list(SPLIT_DIRS.keys()),
                        help="Which splits to convert")
    args = parser.parse_args()

    # Resolve data_root relative to the script location (project root)
    script_dir = Path(__file__).resolve().parent.parent
    data_root = (script_dir / args.data_root).resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    print(f"Data root: {data_root}")
    for split in args.splits:
        convert_split(data_root, split, SPLIT_DIRS[split])

    print("\nConversion complete.")
    print("Labels written to: <split>/labels/*.txt")


if __name__ == "__main__":
    main()
