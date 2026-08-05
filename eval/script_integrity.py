#!/usr/bin/env python3
"""
Script-integrity metric for Bangla generation.

Defines, in code, the metric the paper reports as "script confusion"
(Sec. V-C: 23% for Qwen-1.5B, under 1% for Llama-3.2-11B) and
"language reversion" (Sec. V-A).

Definitions used here, stated explicitly so the numbers are checkable:

  script_confusion  A generation counts as script-confused if it contains
                    at least one codepoint in the Devanagari block
                    (U+0900-U+097F). Presence, not proportion: one stray
                    Devanagari character makes the output wrong for a
                    Bangla-facing system.

  language_reversion
                    A generation counts as reverted if it contains more
                    Latin alphabetic characters than Bangla codepoints.
                    This tolerates incidental Latin (model names, digits,
                    quoted English terms) while catching an answer that is
                    substantively in English.

  clean             Neither of the above, and at least MIN_BANGLA_CHARS
                    Bangla codepoints present. An empty or near-empty
                    generation is not "clean" -- it is degenerate, and is
                    counted separately so it cannot inflate the clean rate.

Input:  one JSONL file per model, records with at least {"output": str}.
        Optional keys "item_id" and "model" are carried through.
Output: JSON with per-model rates, counts, and N.

Usage:
    python eval/script_integrity.py benchmarks/raw/*.jsonl \
        --out eval/script_integrity.json
    python eval/script_integrity.py benchmarks/raw/qwen-1.5b.jsonl --per-item
"""

import argparse
import glob
import json
import os
import sys
import unicodedata

DEVANAGARI = (0x0900, 0x097F)
BENGALI = (0x0980, 0x09FF)
MIN_BANGLA_CHARS = 10

# U+0964 DANDA and U+0965 DOUBLE DANDA live in the Devanagari block but are
# shared Indic punctuation and are the *correct* sentence terminators in
# Bangla. A detector that does not exempt them flags every properly
# punctuated Bangla sentence as script-confused. Excluded deliberately.
SHARED_INDIC_PUNCTUATION = {0x0964, 0x0965}


def _in(cp, block):
    return block[0] <= cp <= block[1]


def analyse(text):
    """Per-generation script report. Pure function, no I/O -- unit-testable."""
    text = unicodedata.normalize("NFC", text or "")
    dev = ban = lat = 0
    for ch in text:
        cp = ord(ch)
        if cp in SHARED_INDIC_PUNCTUATION:
            continue
        if _in(cp, DEVANAGARI):
            dev += 1
        elif _in(cp, BENGALI):
            ban += 1
        elif ch.isascii() and ch.isalpha():
            lat += 1

    degenerate = ban < MIN_BANGLA_CHARS and dev == 0 and lat < MIN_BANGLA_CHARS
    confused = dev > 0
    reverted = lat > ban

    return {
        "devanagari_chars": dev,
        "bangla_chars": ban,
        "latin_alpha_chars": lat,
        "script_confused": confused,
        "language_reverted": reverted,
        "degenerate": degenerate,
        "clean": (not confused) and (not reverted) and (not degenerate),
    }


def summarise(records):
    n = len(records)
    if n == 0:
        return {"n": 0}
    reports = [analyse(r.get("output", "")) for r in records]
    tally = lambda k: sum(1 for x in reports if x[k])
    return {
        "n": n,
        "script_confused": tally("script_confused"),
        "script_confusion_rate": round(tally("script_confused") / n, 4),
        "language_reverted": tally("language_reverted"),
        "language_reversion_rate": round(tally("language_reverted") / n, 4),
        "degenerate": tally("degenerate"),
        "clean": tally("clean"),
        "clean_rate": round(tally("clean") / n, 4),
        "mean_devanagari_chars_when_confused": round(
            sum(x["devanagari_chars"] for x in reports if x["script_confused"])
            / max(tally("script_confused"), 1), 2),
    }


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  skipping {path}:{lineno} -- {exc}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="JSONL files (globs allowed)")
    ap.add_argument("--out", default=None, help="write JSON summary here")
    ap.add_argument("--per-item", action="store_true",
                    help="also emit a per-generation report")
    args = ap.parse_args()

    paths = sorted({p for pat in args.inputs for p in glob.glob(pat)})
    if not paths:
        sys.exit(f"no files matched: {args.inputs}")

    results, per_item = {}, {}
    for path in paths:
        model = os.path.splitext(os.path.basename(path))[0]
        recs = load_jsonl(path)
        results[model] = summarise(recs)
        results[model]["source_file"] = path
        if args.per_item:
            per_item[model] = [
                dict(item_id=r.get("item_id", i), **analyse(r.get("output", "")))
                for i, r in enumerate(recs)
            ]

    payload = {"metric_definition": {
        "script_confusion": "at least one codepoint in U+0900-U+097F, excluding the shared danda punctuation U+0964/U+0965",
        "language_reversion": "latin alphabetic chars > bengali codepoints",
        "degenerate": f"fewer than {MIN_BANGLA_CHARS} bengali codepoints and no devanagari",
        "normalisation": "NFC",
    }, "per_model": results}
    if args.per_item:
        payload["per_item"] = per_item

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()
