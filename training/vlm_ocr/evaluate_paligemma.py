#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import editdistance
from tqdm import tqdm

from infer_paligemma import load_config, load_model, predict


def cer(pred: str, ref: str) -> float:
    return editdistance.eval(pred, ref) / max(1, len(ref))


def wer(pred: str, ref: str) -> float:
    pred_words = pred.split()
    ref_words = ref.split()
    return editdistance.eval(pred_words, ref_words) / max(1, len(ref_words))


def main():
    parser = argparse.ArgumentParser(description="Evaluate PaliGemma OCR on JSONL with CER/WER.")
    parser.add_argument("--config", default="training/vlm_ocr/config.json")
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", default="training/vlm_ocr/outputs/eval_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    jsonl_path = args.jsonl or cfg["test_jsonl"]
    processor, model = load_model(cfg["base_model"], args.adapter)

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_cer = 0.0
    total_wer = 0.0
    with out_path.open("w", encoding="utf-8") as out:
        for row in tqdm(rows):
            pred = predict(processor, model, row["image"], cfg["prompt"], int(cfg.get("max_new_tokens", 256)))
            c = cer(pred, row["text"])
            w = wer(pred, row["text"])
            total_cer += c
            total_wer += w
            out.write(json.dumps({"image": row["image"], "ref": row["text"], "pred": pred, "cer": c, "wer": w}, ensure_ascii=False) + "\n")

    n = max(1, len(rows))
    print(json.dumps({"samples": len(rows), "cer": total_cer / n, "wer": total_wer / n, "predictions": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
