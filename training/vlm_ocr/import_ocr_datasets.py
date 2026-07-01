#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from PIL import Image


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def clean_text(text: str) -> str:
    return (text or "").strip()


def row_to_item(row, task_types, datasets, min_text_len, max_text_len):
    task_type = row.get("task_type", "")
    dataset = row.get("dataset", "")
    if task_types and task_type not in task_types:
        return None
    if datasets and dataset not in datasets:
        return None

    text = clean_text(row.get("text", ""))
    if len(text) < min_text_len or len(text) > max_text_len:
        return None

    image_path = Path(row.get("image_path", ""))
    if not image_path.exists() or not valid_image(image_path):
        return None
    return {"image": str(image_path.resolve()), "text": text, "dataset": dataset, "task_type": task_type}


def convert_split(src: Path, dst: Path, task_types, datasets, limit, min_text_len, max_text_len):
    kept = 0
    skipped = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8-sig", newline="") as f, dst.open("w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        for row in reader:
            item = row_to_item(row, task_types, datasets, min_text_len, max_text_len)
            if item is None:
                skipped += 1
                continue
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
            if limit and kept >= limit:
                break
    return {"source": str(src), "output": str(dst), "kept": kept, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description="Convert /home/benaaf/Desktop/ocr_datasets splits to VLM OCR JSONL.")
    parser.add_argument("--ocr-root", default="/home/benaaf/Desktop/ocr_datasets")
    parser.add_argument("--out-dir", default="training/vlm_ocr/data")
    parser.add_argument(
        "--task-types",
        default="recognition_line_or_paragraph,recognition_word",
        help="Comma-separated task types. Use recognition_line_or_paragraph for VLM line/page OCR; add recognition_word for more volume.",
    )
    parser.add_argument("--datasets", default="", help="Optional comma-separated dataset names")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--min-text-len", type=int, default=1)
    parser.add_argument("--max-text-len", type=int, default=500)
    args = parser.parse_args()

    ocr_root = Path(args.ocr_root)
    splits = ocr_root / "splits"
    out_dir = Path(args.out_dir)
    task_types = {x.strip() for x in args.task_types.split(",") if x.strip()}
    datasets = {x.strip() for x in args.datasets.split(",") if x.strip()}

    results = {
        "train": convert_split(splits / "manifest_train.csv", out_dir / "train.jsonl", task_types, datasets, args.limit_train, args.min_text_len, args.max_text_len),
        "val": convert_split(splits / "manifest_val.csv", out_dir / "val.jsonl", task_types, datasets, args.limit_val, args.min_text_len, args.max_text_len),
        "test": convert_split(splits / "manifest_test.csv", out_dir / "test.jsonl", task_types, datasets, args.limit_test, args.min_text_len, args.max_text_len),
        "task_types": sorted(task_types),
        "datasets": sorted(datasets),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
