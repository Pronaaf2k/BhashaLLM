#!/usr/bin/env python3
"""
Build the blind rating sheets for the human evaluation of paper Sec. III-E.

    "Two to three native Bangla-speaking raters, none of them project team
    members, scored each output independently under a blind protocol with
    model identity hidden and output order randomised."

``human_eval/rubric.md`` instructs the rater to record the randomisation
seed and a blinding map at ``human_eval/blinding_map.json``. Nothing in the
repository produced either, so the blinding was unverifiable after the
fact -- which is the one property a blind protocol has to have. This script
produces both.

What blinding actually requires
-------------------------------
Three separate things, all of which are easy to get wrong:

1. **Model identity hidden.** Outputs are relabelled with opaque codes
   (``SYS_A``, ``SYS_B``, ...). The codes are assigned by shuffling, so
   ``SYS_A`` is not the first model in Table I.

2. **Output order randomised.** Not once, but *per item*. A single global
   shuffle still leaves the same system in the same position on every
   item, and a rater notices position patterns within a dozen items. This
   script draws a fresh permutation per item.

3. **The map kept separately and not shown to raters.** Written to
   ``blinding_map.json``, which the rater never opens. ``--unblind``
   rejoins ratings to model identities afterwards.

Anti-fingerprinting
-------------------
Blinding fails if an output identifies its own model. Two cheap leaks are
checked and reported rather than silently passed through: an output that
names a model family ("As an AI language model", "Qwen", "Llama"), and
outputs whose lengths are so distinctive that a rater could sort by them.
The script reports these; it does not edit generations, because editing
them would change what is being rated.

Inputs
------
One JSONL per model, as produced by the benchmark run::

    benchmarks/raw/<model>.jsonl
    {"item_id": "...", "output": "...", "prompt": "...", "reference": "..."}

Outputs
-------
``human_eval/rating_sheet.csv``
    What the rater fills in. One row per (item, system, dimension), with a
    blank ``score`` column. Long-form rather than wide, because
    ``eval/aggregate_human_eval.py`` reads
    ``rater_id,item_id,model_id,dimension,score``.

``human_eval/rating_sheet.md``
    The same content as a readable document, since a CSV of Bangla
    paragraphs is unpleasant to score in a spreadsheet.

``human_eval/blinding_map.json``
    ``system_code -> model_id``, the seed, and the per-item ordering.

Usage
-----
    python eval/make_rating_sheets.py --pred benchmarks/raw/*.jsonl \\
        --raters r1 r2 r3 --seed 42

    # after the raters return filled sheets
    python eval/make_rating_sheets.py --unblind human_eval/rating_sheet.csv \\
        --map human_eval/blinding_map.json --out human_eval/ratings.csv
    python eval/aggregate_human_eval.py --ratings human_eval/ratings.csv
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
import unicodedata
from collections import defaultdict

# The three dimensions of paper Sec. III-E, anchored in human_eval/rubric.md.
DIMENSIONS = ["semantic_accuracy", "script_correctness", "naturalness"]

# Substrings that would tell a rater which system produced an output.
FINGERPRINTS = [
    "as an ai", "language model", "i'm an ai", "i am an ai",
    "qwen", "llama", "gemma", "mistral", "nemo", "openai", "anthropic",
    "alibaba", "meta ai", "google",
]


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


def check_fingerprints(model_id, records):
    """Report outputs that identify their own model."""
    hits = []
    for r in records:
        text = (r.get("output") or "").lower()
        for fp in FINGERPRINTS:
            if fp in text:
                hits.append({"item_id": r.get("item_id"), "match": fp})
                break
    return hits


def length_separability(by_model):
    """Warn when output length alone would let a rater sort the systems.

    If one system's median length is more than twice another's, a rater
    scoring a dozen items will notice, and the blinding is weaker than it
    looks. Reported, not corrected.
    """
    medians = {}
    for model, recs in by_model.items():
        lens = sorted(len(r.get("output") or "") for r in recs)
        if lens:
            medians[model] = lens[len(lens) // 2]
    if len(medians) < 2:
        return {"medians": medians, "warning": None}
    lo_model = min(medians, key=medians.get)
    hi_model = max(medians, key=medians.get)
    lo, hi = medians[lo_model], medians[hi_model]
    warning = None
    if lo and hi / lo > 2.0:
        warning = (
            f"median output length ranges from {lo} ({lo_model}) to {hi} "
            f"({hi_model}), a ratio of {hi / lo:.1f}. Length alone may let a "
            f"rater group systems; blinding is weaker than it appears."
        )
    return {"medians": medians, "warning": warning}


def build(by_model, raters, seed, task):
    """Assign system codes, shuffle per item, emit rows and the map."""
    rng = random.Random(seed)

    models = sorted(by_model)
    codes = [f"SYS_{chr(ord('A') + i)}" for i in range(len(models))]
    shuffled = list(models)
    rng.shuffle(shuffled)          # code assignment is itself randomised
    code_of = dict(zip(shuffled, codes))

    # item_id -> {model_id: record}
    items = defaultdict(dict)
    for model, recs in by_model.items():
        for r in recs:
            items[str(r.get("item_id"))][model] = r

    complete = [i for i, d in sorted(items.items()) if len(d) == len(models)]
    incomplete = [i for i, d in sorted(items.items()) if len(d) != len(models)]

    rows, per_item_order, sheet = [], {}, []
    for item_id in complete:
        present = [m for m in models if m in items[item_id]]
        order = list(present)
        rng.shuffle(order)          # fresh permutation for every item
        per_item_order[item_id] = [code_of[m] for m in order]

        first = items[item_id][order[0]]
        sheet.append({
            "item_id": item_id,
            "prompt": first.get("prompt") or first.get("source") or "",
            "reference": first.get("reference") or "",
            "systems": [
                {"code": code_of[m], "output": items[item_id][m].get("output", "")}
                for m in order
            ],
        })

        for m in order:
            for rater in raters:
                for dim in DIMENSIONS:
                    rows.append({
                        "rater_id": rater,
                        "item_id": item_id,
                        "system_code": code_of[m],
                        "dimension": dim,
                        "score": "",          # the rater fills this in
                        "notes": "",
                    })

    blinding_map = {
        "seed": seed,
        "task": task,
        "created_by": "eval/make_rating_sheets.py",
        "paper_section": "III-E",
        "dimensions": DIMENSIONS,
        "scale": "1-5, anchors in human_eval/rubric.md",
        "n_items": len(complete),
        "n_systems": len(models),
        "n_raters": len(raters),
        "raters": list(raters),
        "system_code_to_model": {v: k for k, v in code_of.items()},
        "per_item_system_order": per_item_order,
        "items_dropped_incomplete": incomplete,
        "note":
            "Do not show this file to raters. Codes were assigned by a "
            "seeded shuffle and output order was re-randomised per item, so "
            "neither code order nor position carries information about "
            "model identity.",
    }
    return rows, sheet, blinding_map


def write_markdown(sheet, path, task):
    """A readable scoring document. A CSV of Bangla paragraphs is unusable."""
    lines = [
        f"# Blind rating sheet — {task}",
        "",
        "Score every system on all three dimensions using the anchors in",
        "[`rubric.md`](rubric.md). System codes carry no information: they are",
        "randomised, and the order changes on every item.",
        "",
        "Record your scores in `rating_sheet.csv`. Do not skip items — a",
        "missing rating is handled by Krippendorff's alpha, but a guessed one",
        "is not.",
        "",
        "---",
        "",
    ]
    for n, item in enumerate(sheet, 1):
        lines.append(f"## Item {n} — `{item['item_id']}`")
        lines.append("")
        if item["prompt"]:
            lines += ["**Prompt**", "", "> " + item["prompt"].replace("\n", "\n> "), ""]
        if item["reference"]:
            lines += ["**Reference**", "",
                      "> " + item["reference"].replace("\n", "\n> "), ""]
        for sysinfo in item["systems"]:
            lines.append(f"### {sysinfo['code']}")
            lines.append("")
            lines.append(sysinfo["output"] or "*(empty output)*")
            lines.append("")
            lines.append("| Dimension | Score (1–5) |")
            lines.append("| --- | --- |")
            for d in DIMENSIONS:
                lines.append(f"| {d.replace('_', ' ')} |  |")
            lines.append("")
        lines.append("---")
        lines.append("")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def unblind(sheet_path, map_path, out_path):
    """Join filled ratings back to model identities.

    Emits exactly the columns ``eval/aggregate_human_eval.py`` reads:
    ``rater_id,item_id,model_id,dimension,score``.
    """
    mapping = json.loads(open(map_path, encoding="utf-8").read())
    code_to_model = mapping["system_code_to_model"]

    rows = list(csv.DictReader(open(sheet_path, encoding="utf-8", newline="")))
    out, unscored, unknown = [], 0, set()
    for r in rows:
        score = (r.get("score") or "").strip()
        if not score:
            unscored += 1
            continue
        code = r.get("system_code", "")
        if code not in code_to_model:
            unknown.add(code)
            continue
        out.append({
            "rater_id": r.get("rater_id", ""),
            "item_id": r.get("item_id", ""),
            "model_id": code_to_model[code],
            "dimension": r.get("dimension", ""),
            "score": score,
        })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["rater_id", "item_id", "model_id", "dimension", "score"])
        w.writeheader()
        w.writerows(out)

    print(f"wrote {out_path}  ({len(out)} ratings)")
    if unscored:
        print(f"  {unscored} rows had no score and were skipped")
    if unknown:
        print(f"  WARNING: unrecognised system codes {sorted(unknown)}")
    print(f"\nNext:\n  python eval/aggregate_human_eval.py --ratings {out_path} "
          f"--out human_eval/alpha_by_dimension.json")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="*", default=[],
                    help="JSONL files, one per model (globs allowed)")
    ap.add_argument("--raters", nargs="*", default=["r1", "r2", "r3"],
                    help="rater ids. Sec. III-E says 'two to three'; "
                         "Table V needs the exact number reported.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", default="translation")
    ap.add_argument("--out-dir", default="human_eval")
    ap.add_argument("--unblind", metavar="SHEET_CSV",
                    help="rejoin a filled sheet to model identities")
    ap.add_argument("--map", metavar="BLINDING_MAP_JSON",
                    default="human_eval/blinding_map.json")
    ap.add_argument("--out", default="human_eval/ratings.csv",
                    help="output for --unblind")
    args = ap.parse_args()

    if args.unblind:
        return unblind(args.unblind, args.map, args.out)

    paths = sorted({p for pat in args.pred for p in glob.glob(pat)})
    if not paths:
        sys.exit("no files matched --pred (or pass --unblind)")

    by_model = {}
    for p in paths:
        model = os.path.splitext(os.path.basename(p))[0]
        by_model[model] = load_jsonl(p)

    fingerprints = {m: check_fingerprints(m, r) for m, r in by_model.items()}
    fingerprints = {m: h for m, h in fingerprints.items() if h}
    lengths = length_separability(by_model)

    rows, sheet, blinding_map = build(by_model, args.raters, args.seed, args.task)
    blinding_map["blinding_risks"] = {
        "self_identifying_outputs": fingerprints,
        "length_medians": lengths["medians"],
        "length_warning": lengths["warning"],
    }

    csv_path = os.path.join(args.out_dir, "rating_sheet.csv")
    md_path = os.path.join(args.out_dir, "rating_sheet.md")
    map_path = os.path.join(args.out_dir, "blinding_map.json")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "rater_id", "item_id", "system_code", "dimension", "score", "notes"])
        w.writeheader()
        w.writerows(rows)
    write_markdown(sheet, md_path, args.task)
    with open(map_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(blinding_map, indent=2, ensure_ascii=False) + "\n")

    print(f"  systems  {blinding_map['n_systems']}")
    print(f"  items    {blinding_map['n_items']}"
          + (f"  ({len(blinding_map['items_dropped_incomplete'])} dropped: "
             f"not every system produced an output)"
             if blinding_map["items_dropped_incomplete"] else ""))
    print(f"  raters   {blinding_map['n_raters']}  {args.raters}")
    print(f"  ratings  {len(rows)} rows to fill "
          f"({blinding_map['n_items']} items x {blinding_map['n_systems']} "
          f"systems x {len(DIMENSIONS)} dimensions x "
          f"{blinding_map['n_raters']} raters)")
    if fingerprints:
        print(f"\n  WARNING: {sum(len(v) for v in fingerprints.values())} outputs "
              f"name a model family and would break blinding:")
        for m, hits in fingerprints.items():
            print(f"    {m}: {len(hits)} (e.g. item {hits[0]['item_id']}, "
                  f"matched {hits[0]['match']!r})")
    if lengths["warning"]:
        print(f"\n  WARNING: {lengths['warning']}")

    print(f"\nwrote {csv_path}\n      {md_path}\n      {map_path}")
    print("\nGive raters rating_sheet.md and rating_sheet.csv only. "
          "blinding_map.json must not be shared with them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
