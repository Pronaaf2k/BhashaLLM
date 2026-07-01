#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Validate a VLM OCR JSONL dataset.")
    parser.add_argument("jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.jsonl)
    count = 0
    bad = []
    lengths = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if args.limit and count >= args.limit:
                break
            count += 1
            try:
                row = json.loads(line)
                image_path = Path(row["image"])
                text = row["text"]
                with Image.open(image_path) as img:
                    width, height = img.size
                lengths.append(len(text))
                if not text.strip() or width <= 0 or height <= 0:
                    bad.append(line_no)
            except Exception:
                bad.append(line_no)
    print(json.dumps({
        "file": str(path),
        "checked": count,
        "bad_rows": bad[:50],
        "bad_count": len(bad),
        "avg_text_len": round(sum(lengths) / max(1, len(lengths)), 2),
        "max_text_len": max(lengths) if lengths else 0,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
