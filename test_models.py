#!/usr/bin/env python3
"""
test_models.py -- the command-line interface described in paper Sec. III-F.

    "A command-line interface, test_models.py, is the primary production
    interface: it applies the correct ChatML formatting for the grading
    model and the correct resizing and normalisation for the OCR model
    automatically, so the user does not need to know either model's input
    format."

``docs/ERRATA.md`` group C recorded this file as absent: the manuscript
named it as the primary interface and the repository shipped ``main.py``
(FastAPI) instead. This is that file. ``main.py`` is unchanged and remains
the way to serve the API.

Note the collision worth knowing about: ``tests/test_models.py`` is an
unrelated pytest module that predates this. This file is at the repository
root, which is where Sec. III-F implies it lives, and pytest will not
collect it from here because it defines no test functions.

Subcommands
-----------
``generate``   Bangla text generation under Sec. IV-C's fixed decoding
``grade``      short-answer grading with Bangla feedback (Phase 2 adapter)
``ocr``        handwritten Bangla transcription (Phase 3 adapter)
``chat``       interactive REPL with live adapter switching
``status``     resident models, active adapter, VRAM, adapter presence

Examples
--------
    python test_models.py status

    python test_models.py generate \\
        --prompt "বাংলা সাহিত্যের ইতিহাস সম্পর্কে লিখুন।" --adapter bangla

    python test_models.py grade \\
        --question "রবীন্দ্রনাথ ঠাকুর কে ছিলেন?" \\
        --reference "তিনি একজন বাঙালি কবি ও সাহিত্যিক।" \\
        --answer "তিনি একজন লেখক।"

    python test_models.py ocr --image samples/page0001_line03.png --confidence

    python test_models.py ocr --manifest data/handwriting/manifest.csv \\
        --split test --out eval/ocr_predictions.jsonl

The last form writes the JSONL that ``eval/ocr_cer.py`` and
``eval/confidence.py`` consume, so the CLI feeds the evaluation pipeline
directly rather than requiring a separate inference script.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from bhasha.app.adapter_manager import (  # noqa: E402
    OCR_IMAGE_SIZE, TEXT_DECODING, AdapterManager,
)

TASKS = ("bangla", "grading", "ocr")


def _manager(args) -> AdapterManager:
    return AdapterManager(
        base_model=args.base_model,
        ocr_base_model=args.ocr_base_model,
        models_dir=args.models_dir,
        load_in_4bit=not args.no_4bit,
    )


def _emit(payload, as_json: bool, text_key: str = "text") -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(payload.get(text_key, ""))


# ------------------------------------------------------------- subcommands


def cmd_status(args) -> int:
    m = _manager(args)
    st = m.status()
    print(json.dumps(st, indent=2, ensure_ascii=False))
    missing = [k for k, v in st["known_adapters"].items() if not v["present"]]
    if missing:
        print(f"\nAdapters not found on disk: {missing}")
        print("Adapters total ~3.6 GB and are not tracked in git. "
              "See README, section 'Models'.")
    return 0


def cmd_generate(args) -> int:
    m = _manager(args)
    # ChatML applied automatically -- the property Sec. III-F asks for.
    prompt = (
        m.format_chatml([{"role": "user", "content": args.prompt}])
        if not args.raw_prompt else args.prompt
    )
    out = m.generate(
        prompt,
        task=args.adapter,
        return_logprobs=args.confidence,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    if args.confidence and out.get("token_logprobs"):
        import math
        lps = out["token_logprobs"]
        out["confidence"] = round(math.exp(sum(lps) / len(lps)), 4)
    _emit(out, args.json)
    return 0


def cmd_grade(args) -> int:
    m = _manager(args)
    out = m.grade(args.question, args.reference, args.answer,
                  max_new_tokens=args.max_new_tokens)
    _emit(out, args.json)
    return 0


def _ocr_one(m, path, args):
    out = m.ocr(path, image_size=args.image_size,
                return_logprobs=args.confidence)
    out["image"] = str(path)
    return out


def cmd_ocr(args) -> int:
    m = _manager(args)

    if args.image:
        out = _ocr_one(m, Path(args.image), args)
        _emit(out, args.json)
        return 0

    # Batch mode over a manifest, writing the JSONL that eval/ocr_cer.py and
    # eval/confidence.py read. image_id and writer_id are carried through so
    # the writer-disjoint breakdown of Sec. VI-D is possible downstream.
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8", newline="")))
    if args.split:
        rows = [r for r in rows if r.get("split") == args.split]
    if not rows:
        sys.exit(f"no rows in {args.manifest}"
                 + (f" with split={args.split}" if args.split else ""))

    root = Path(args.image_root) if args.image_root else Path(args.manifest).parent
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            image_id = row.get("image_id", "")
            candidate = Path(row.get("image_path") or (root / f"{image_id}.png"))
            try:
                res = _ocr_one(m, candidate, args)
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                print(f"  [{i}/{len(rows)}] {image_id}: {exc}", file=sys.stderr)
                continue
            record = {
                "image_id": image_id,
                "writer_id": row.get("writer_id"),
                "hypothesis": res["text"],
                "latency_ms": res["latency_ms"],
            }
            if row.get("reference") or row.get("text"):
                record["reference"] = row.get("reference") or row.get("text")
            if res.get("token_logprobs"):
                record["token_logprobs"] = res["token_logprobs"]
                record["confidence"] = res.get("confidence")
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"  {i}/{len(rows)} ...", file=sys.stderr)

    print(f"wrote {out_path}  ({n_ok} ok, {n_fail} failed)")
    print("\nNext:")
    print(f"  python eval/ocr_cer.py --pred {out_path} "
          f"--manifest {args.manifest} --group-by writer_id "
          f"--out eval/ocr_cer.json")
    if args.confidence:
        print(f"  python eval/confidence.py --pred {out_path} "
              f"--out eval/confidence.json")
    return 0 if n_fail == 0 else 1


def cmd_chat(args) -> int:
    """Interactive REPL. ``/adapter <name>`` swaps; ``/status`` reports VRAM."""
    m = _manager(args)
    print("BhashaLLM interactive shell. Commands: /adapter <name|none>, "
          "/status, /quit")
    print(f"Adapters: {', '.join(TASKS)}")
    task = args.adapter
    while True:
        try:
            line = input(f"[{task or 'base'}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/status":
            print(json.dumps(m.status(), indent=2, ensure_ascii=False))
            continue
        if line.startswith("/adapter"):
            parts = line.split()
            name = parts[1] if len(parts) > 1 else None
            task = None if name in (None, "none", "null") else name
            try:
                rec = m.use(task)
                print(f"  {rec.from_adapter} -> {rec.to_adapter}  "
                      f"({rec.vram_before_gb} -> {rec.vram_after_gb} GB, "
                      f"{rec.seconds}s)")
            except Exception as exc:  # noqa: BLE001
                print(f"  error: {exc}")
                task = m.status()["active_adapter"]
            continue
        try:
            prompt = m.format_chatml([{"role": "user", "content": line}])
            out = m.generate(prompt, task=task,
                             max_new_tokens=args.max_new_tokens)
            print(out["text"])
            print(f"  ({out['n_new_tokens']} tokens, {out['latency_ms']} ms)")
        except Exception as exc:  # noqa: BLE001
            print(f"  error: {exc}")
    m.unload()
    return 0


# ------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="test_models.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ocr-base-model", default="swapnillo/Bangla-OCR-SFT")
    ap.add_argument("--models-dir", default=str(BASE_DIR / "models"))
    ap.add_argument("--no-4bit", action="store_true",
                    help="load in fp16 instead of Table III's NF4. Will not "
                         "fit an 11B model in 16 GB; for CPU debugging only.")
    ap.add_argument("--json", action="store_true",
                    help="print the full result object, not just the text")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="resident models and adapter presence")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("generate", help="Bangla text generation")
    p.add_argument("--prompt", required=True)
    p.add_argument("--adapter", default=None, choices=[*TASKS, None])
    p.add_argument("--raw-prompt", action="store_true",
                   help="skip ChatML formatting")
    p.add_argument("--max-new-tokens", type=int,
                   default=TEXT_DECODING["max_new_tokens"])
    p.add_argument("--temperature", type=float,
                   default=TEXT_DECODING["temperature"])
    p.add_argument("--top-p", type=float, default=TEXT_DECODING["top_p"])
    p.add_argument("--repetition-penalty", type=float,
                   default=TEXT_DECODING["repetition_penalty"])
    p.add_argument("--confidence", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("grade", help="short-answer grading, Bangla feedback")
    p.add_argument("--question", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--answer", required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.set_defaults(func=cmd_grade)

    p = sub.add_parser("ocr", help="handwritten Bangla transcription")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="single image")
    g.add_argument("--manifest", help="batch over data/handwriting/manifest.csv")
    p.add_argument("--split", default=None, help="train | val | test")
    p.add_argument("--image-root", default=None)
    p.add_argument("--image-size", type=int, default=OCR_IMAGE_SIZE)
    p.add_argument("--confidence", action="store_true",
                   help="also emit token logprobs (see eval/confidence.py)")
    p.add_argument("--out", default="eval/ocr_predictions.jsonl")
    p.set_defaults(func=cmd_ocr)

    p = sub.add_parser("chat", help="interactive shell with adapter switching")
    p.add_argument("--adapter", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.set_defaults(func=cmd_chat)

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
