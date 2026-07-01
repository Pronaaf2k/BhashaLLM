#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def normalize_row(image_path: Path, text: str):
    text = (text or "").strip()
    if not text or not image_path.exists() or not verify_image(image_path):
        return None
    return {"image": str(image_path.resolve()), "text": text}


def load_csv(path: Path, image_root: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        image_col = next(
            (
                col
                for col in ["image", "image_path", "path", "filename", "file_name", "File name"]
                if col in fields
            ),
            None,
        )
        text_col = next(
            (
                col
                for col in ["text", "label", "transcription", "extracted text", "ground_truth", "gt"]
                if col in fields
            ),
            None,
        )
        if not image_col or not text_col:
            raise ValueError(
                "CSV needs an image column (image/image_path/path/filename/file_name/File name) "
                "and a text column (text/label/transcription/extracted text/ground_truth/gt)"
            )
        for row in reader:
            image_path = Path(row[image_col])
            if not image_path.is_absolute():
                image_path = image_root / image_path
            item = normalize_row(image_path, row[text_col])
            if item:
                rows.append(item)
    return rows


def load_jsonl(path: Path, image_root: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            image_value = row.get("image") or row.get("image_path") or row.get("path")
            text = row.get("text") or row.get("label") or row.get("transcription")
            if not image_value or not text:
                continue
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = image_root / image_path
            item = normalize_row(image_path, text)
            if item:
                rows.append(item)
    return rows


def load_sidecar_folder(path: Path):
    rows = []
    for image_path in sorted(p for p in path.rglob("*") if p.is_file() and is_image(p)):
        candidates = [
            image_path.with_suffix(".txt"),
            image_path.with_suffix(".gt.txt"),
            image_path.parent / f"{image_path.stem}.label.txt",
        ]
        label_path = next((p for p in candidates if p.exists()), None)
        if not label_path:
            continue
        item = normalize_row(image_path, label_path.read_text(encoding="utf-8"))
        if item:
            rows.append(item)
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(rows, seed: int, train_ratio: float, val_ratio: float):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    train_end = int(len(rows) * train_ratio)
    val_end = train_end + int(len(rows) * val_ratio)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def load_input(path: Path, image_root: Path):
    if path.is_file() and path.suffix.lower() == ".csv":
        return load_csv(path, image_root)
    if path.is_file() and path.suffix.lower() in {".jsonl", ".json"}:
        return load_jsonl(path, image_root)
    if path.is_dir():
        return load_sidecar_folder(path)
    raise ValueError(f"Unsupported dataset input: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Import a Bangla OCR dataset into training/vlm_ocr JSONL format. Supports CSV, JSONL, or image folder with .txt sidecar labels."
    )
    parser.add_argument("--input", required=True, help="CSV, JSONL, or labeled image folder")
    parser.add_argument("--image-root", default=None, help="Root for relative image paths; defaults to input parent")
    parser.add_argument("--out-dir", default="training/vlm_ocr/data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else (input_path.parent if input_path.is_file() else input_path)
    rows = load_input(input_path, image_root)
    if not rows:
        raise SystemExit("No valid image/text pairs found. Expected CSV/JSONL columns image,text or image files with matching .txt files.")

    train, val, test = split_rows(rows, args.seed, args.train_ratio, args.val_ratio)
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)

    print(json.dumps({
        "input": str(input_path),
        "image_root": str(image_root),
        "total": len(rows),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "out_dir": str(out_dir.resolve()),
        "sample": rows[0],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
