#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image


def read_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "image" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("CSV must contain columns: image,text")
        for row in reader:
            image = (row.get("image") or "").strip()
            text = (row.get("text") or "").strip()
            if image and text:
                rows.append({"image": image, "text": text})
    return rows


def validate_rows(rows, image_root: Path):
    valid = []
    skipped = 0
    for row in rows:
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = image_root / image_path
        try:
            with Image.open(image_path) as img:
                img.verify()
            valid.append({"image": str(image_path.resolve()), "text": row["text"]})
        except Exception:
            skipped += 1
    return valid, skipped


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare image/text JSONL splits for VLM OCR fine-tuning.")
    parser.add_argument("--csv", required=True, help="CSV with columns image,text")
    parser.add_argument("--image-root", default=".", help="Root used for relative image paths")
    parser.add_argument("--out-dir", default="training/vlm_ocr/data")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_rows(Path(args.csv))
    rows, skipped = validate_rows(rows, Path(args.image_root))
    random.Random(args.seed).shuffle(rows)

    train_end = int(len(rows) * args.train_ratio)
    val_end = train_end + int(len(rows) * args.val_ratio)
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", rows[:train_end])
    write_jsonl(out_dir / "val.jsonl", rows[train_end:val_end])
    write_jsonl(out_dir / "test.jsonl", rows[val_end:])

    print(json.dumps({
        "total_valid": len(rows),
        "skipped_invalid_images": skipped,
        "train": train_end,
        "val": val_end - train_end,
        "test": len(rows) - val_end,
        "out_dir": str(out_dir),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
