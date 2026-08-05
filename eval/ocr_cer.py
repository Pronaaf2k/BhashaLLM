#!/usr/bin/env python3
"""
OCR scoring: character error rate, per-writer breakdown, and the
per-grapheme-category accuracies reported in Sec. V-B.

Sec. V-B of the paper reports accuracy split by grapheme category
(vowels 94%, base consonants 91%, vowel diacritics 85%, consonant
diacritics 79%). Sec. IV-B says ground truth is *line transcriptions*,
i.e. strings. A string does not carry per-character labels, so those
four numbers require an explicit alignment step. This file is that step,
so the breakdown stops being an assertion and becomes a computation.

Method: Levenshtein alignment of hypothesis to reference with backtracking,
then Unicode-category mapping of each aligned reference character, then
aggregation per category. Substitutions and deletions count against the
reference character's category; insertions are reported separately
because they belong to no reference category.

Writer-disjoint evaluation (Sec. VI-D concedes the current 12% CER is
measured on handwriting the model has seen) is supported by passing a
manifest with a writer_id column and using --group-by writer_id.

Input JSONL records: {"image_id":..., "reference": str, "hypothesis": str}
Optional manifest CSV: image_id,writer_id,session,pen,...

Usage:
    python eval/ocr_cer.py --pred eval/ocr_predictions.jsonl \
        --manifest data/handwriting/manifest.csv \
        --group-by writer_id --out eval/ocr_cer.json
"""

import argparse
import csv
import json
import os
import sys
import unicodedata
from collections import defaultdict

# ---------------------------------------------------------------- categories

INDEPENDENT_VOWELS = set(range(0x0985, 0x098D)) | {0x098F, 0x0990, 0x0993, 0x0994} \
    | {0x09E0, 0x09E1}
CONSONANTS = set(range(0x0995, 0x09A9)) | set(range(0x09AA, 0x09B1)) \
    | {0x09B2} | set(range(0x09B6, 0x09BA)) | {0x09CE, 0x09DC, 0x09DD, 0x09DF} \
    | {0x09F0, 0x09F1}
VOWEL_SIGNS = set(range(0x09BE, 0x09C5)) | {0x09C7, 0x09C8, 0x09CB, 0x09CC, 0x09D7} \
    | {0x09E2, 0x09E3}
# hasant/virama, chandrabindu, anusvara, visarga, nukta: marks that attach to
# a consonant and drive conjunct formation
CONSONANT_DIACRITICS = {0x09CD, 0x0981, 0x0982, 0x0983, 0x09BC}
DIGITS = set(range(0x09E6, 0x09F0))


def category(ch):
    cp = ord(ch)
    if cp in INDEPENDENT_VOWELS:
        return "independent_vowel"
    if cp in CONSONANTS:
        return "consonant"
    if cp in VOWEL_SIGNS:
        return "vowel_diacritic"
    if cp in CONSONANT_DIACRITICS:
        return "consonant_diacritic"
    if cp in DIGITS:
        return "digit"
    if ch.isspace():
        return "whitespace"
    if 0x0980 <= cp <= 0x09FF:
        return "other_bengali"
    return "non_bengali"


# ---------------------------------------------------------------- alignment

def align(ref, hyp):
    """Levenshtein alignment with backtracking.

    Returns a list of (op, ref_char, hyp_char) where op is one of
    'match', 'sub', 'del', 'ins'. ref_char/hyp_char are None where the
    op does not consume from that side.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                ops.append(("match" if cost == 0 else "sub", ref[i - 1], hyp[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
            continue
        ops.append(("ins", None, hyp[j - 1]))
        j -= 1
    ops.reverse()
    return ops


def norm(s):
    return unicodedata.normalize("NFC", s or "")


# ---------------------------------------------------------------- scoring

def score(records):
    total_edits = total_ref = 0
    cat_total = defaultdict(int)
    cat_correct = defaultdict(int)
    insertions = 0
    per_item = []

    for rec in records:
        ref, hyp = norm(rec["reference"]), norm(rec["hypothesis"])
        ops = align(ref, hyp)
        edits = sum(1 for op, _, _ in ops if op != "match")
        total_edits += edits
        total_ref += len(ref)
        per_item.append({
            "image_id": rec.get("image_id"),
            "ref_chars": len(ref),
            "edits": edits,
            "cer": round(edits / len(ref), 4) if ref else None,
        })
        for op, rc, _ in ops:
            if op == "ins":
                insertions += 1
                continue
            cat = category(rc)
            cat_total[cat] += 1
            if op == "match":
                cat_correct[cat] += 1

    by_cat = {
        c: {"n": cat_total[c],
            "correct": cat_correct[c],
            "accuracy": round(cat_correct[c] / cat_total[c], 4)}
        for c in sorted(cat_total) if cat_total[c]
    }
    return {
        "n_items": len(records),
        "reference_chars": total_ref,
        "total_edits": total_edits,
        "cer": round(total_edits / total_ref, 4) if total_ref else None,
        "char_accuracy": round(1 - total_edits / total_ref, 4) if total_ref else None,
        "insertions_unattributable_to_a_reference_category": insertions,
        "by_grapheme_category": by_cat,
    }, per_item


def load_manifest(path):
    with open(path, encoding="utf-8") as fh:
        return {r["image_id"]: r for r in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="JSONL of predictions")
    ap.add_argument("--manifest", help="CSV with image_id and grouping columns")
    ap.add_argument("--group-by", help="manifest column to break results down by")
    ap.add_argument("--out", help="write JSON here")
    args = ap.parse_args()

    with open(args.pred, encoding="utf-8") as fh:
        records = [json.loads(l) for l in fh if l.strip()]
    for r in records:
        if "reference" not in r or "hypothesis" not in r:
            sys.exit("every record needs 'reference' and 'hypothesis'")

    overall, per_item = score(records)
    payload = {"overall": overall}

    if args.group_by:
        if not args.manifest:
            sys.exit("--group-by requires --manifest")
        man = load_manifest(args.manifest)
        groups = defaultdict(list)
        missing = 0
        for r in records:
            row = man.get(str(r.get("image_id")))
            if row is None:
                missing += 1
                continue
            groups[row[args.group_by]].append(r)
        payload["by_" + args.group_by] = {
            g: score(rs)[0] for g, rs in sorted(groups.items())
        }
        payload["records_missing_from_manifest"] = missing
        if missing:
            print(f"warning: {missing} predictions had no manifest row",
                  file=sys.stderr)

    payload["per_item"] = per_item
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w", encoding="utf-8").write(text + "\n")
        print(f"wrote {args.out}")
    print(json.dumps({k: v for k, v in payload.items() if k != "per_item"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
