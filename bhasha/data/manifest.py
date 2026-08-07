"""Handwriting manifest: schema, validation, and split auditing.

``data/handwriting/manifest.csv`` is referenced by
``configs/phase3_ocr_sft.yaml`` ("must carry writer_id"), by
``eval/ocr_cer.py --group-by writer_id``, and by three rows of
``docs/TRACEABILITY.md`` — but no schema, template or validator existed.
Without ``writer_id`` the writer-disjoint CER of paper Sec. VI-D cannot be
computed at all, and Sec. VII commits to releasing the dataset with exactly
this metadata:

    "Recording a per-writer identifier with each page matters most, since
    it is what makes writer-disjoint evaluation possible."

This module defines the columns, validates a manifest against them, and —
the part that matters for the paper — reports whether the train/test split
is actually writer-disjoint. Sec. VI-D concedes it is not:

    "The self-collected split is also not writer-disjoint: the same three
    contributors appear in the training and test portions, so the 12% CER
    should be read as an estimate for familiar handwriting."

``audit_writer_disjointness`` turns that concession into a computed fact
rather than a claim, and prints the overlapping writers by name.

Usage
-----
    python -m bhasha.data.manifest --template data/handwriting/manifest.csv
    python -m bhasha.data.manifest --validate data/handwriting/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Required columns. image_id is the join key used by eval/ocr_cer.py; every
# other required column exists because paper Sec. IV-B or VII names it.
REQUIRED_COLUMNS = [
    "image_id",      # join key for eval/ocr_cer.py predictions
    "writer_id",     # Sec. VII: the field that enables writer-disjoint eval
    "split",         # train | val | test  (Sec. IV-C: 1,050 / 150 / 300)
    "source",        # self_collected | ekush | banglawriting | ...
]

# Recommended columns. Sec. IV-B: "deliberate variation in writing speed,
# pen type, and print-versus-cursive style was introduced across collection
# sessions" — none of which is recoverable without recording it.
RECOMMENDED_COLUMNS = [
    "session",           # collection session identifier
    "pen_type",          # ballpoint | gel | pencil | fountain
    "style",             # print | cursive | mixed
    "writing_speed",     # slow | normal | fast
    "page_id",           # page a line was segmented from
    "line_index",        # line number within the page
    "transcriber_id",    # Sec. IV-B: transcribed by one member,
    "checker_id",        #   spot-checked by a second
    "excluded_reason",   # Sec. IV-B: illegible pages excluded before
                         #   line segmentation; record why
]

VALID_SPLITS = {"train", "val", "test"}


def write_template(path: str | Path) -> Path:
    """Write a header-only manifest with one commented example row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = REQUIRED_COLUMNS + RECOMMENDED_COLUMNS
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerow([
            "page0001_line03", "writer_A", "train", "self_collected",
            "session_01", "ballpoint", "cursive", "normal",
            "page0001", "3", "member_1", "member_2", "",
        ])
    return path


def read_manifest(path: str | Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def validate(rows: List[Dict[str, str]]) -> Dict[str, object]:
    """Check required columns, uniqueness of image_id, and split validity."""
    problems: List[str] = []
    if not rows:
        return {"ok": False, "problems": ["manifest is empty"], "n_rows": 0}

    present = set(rows[0])
    missing = [c for c in REQUIRED_COLUMNS if c not in present]
    if missing:
        problems.append(f"missing required columns: {missing}")

    absent_recommended = [c for c in RECOMMENDED_COLUMNS if c not in present]

    ids = Counter(r.get("image_id", "") for r in rows)
    dupes = [k for k, v in ids.items() if v > 1]
    if dupes:
        problems.append(f"duplicate image_id values ({len(dupes)}): {dupes[:10]}")

    blank_writer = sum(1 for r in rows if not (r.get("writer_id") or "").strip())
    if blank_writer:
        problems.append(
            f"{blank_writer} rows have a blank writer_id; writer-disjoint "
            "evaluation (Sec. VI-D) is impossible for those rows"
        )

    bad_split = sorted({
        r.get("split", "") for r in rows
        if (r.get("split") or "") not in VALID_SPLITS
    })
    if bad_split:
        problems.append(f"unrecognised split values: {bad_split}")

    return {
        "ok": not problems,
        "problems": problems,
        "n_rows": len(rows),
        "columns_present": sorted(present),
        "recommended_columns_absent": absent_recommended,
        "n_writers": len({r.get("writer_id") for r in rows}),
        "rows_per_split": dict(Counter(r.get("split", "") for r in rows)),
        "rows_per_source": dict(Counter(r.get("source", "") for r in rows)),
    }


def audit_writer_disjointness(rows: List[Dict[str, str]]) -> Dict[str, object]:
    """Report whether train and test share writers.

    Paper Sec. VI-D states they do. This computes it, so the errata carries
    a measurement rather than a recollection.
    """
    by_split: Dict[str, set] = defaultdict(set)
    for r in rows:
        by_split[r.get("split", "")].add(r.get("writer_id", ""))

    train, val, test = by_split.get("train", set()), by_split.get("val", set()), by_split.get("test", set())
    overlap_test = sorted(train & test)
    overlap_val = sorted(train & val)
    return {
        "writers_train": sorted(train),
        "writers_val": sorted(val),
        "writers_test": sorted(test),
        "train_test_overlap": overlap_test,
        "train_val_overlap": overlap_val,
        "writer_disjoint": not overlap_test,
        "interpretation": (
            "Train and test share writers. CER measured on this split is an "
            "estimate for FAMILIAR handwriting and is optimistic for an "
            "unseen writer (paper Sec. VI-D). Use "
            "`eval/ocr_cer.py --group-by writer_id` and report the "
            "leave-one-writer-out figure alongside it."
            if overlap_test else
            "Train and test are writer-disjoint. CER on this split is a "
            "generalisation estimate for unseen writers."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--template", metavar="PATH",
                   help="write a header-only manifest template here")
    g.add_argument("--validate", metavar="PATH",
                   help="validate an existing manifest")
    ap.add_argument("--out", default=None, help="write the audit JSON here")
    args = ap.parse_args(argv)

    if args.template:
        p = write_template(args.template)
        print(f"wrote template {p}")
        print("Required:    " + ", ".join(REQUIRED_COLUMNS))
        print("Recommended: " + ", ".join(RECOMMENDED_COLUMNS))
        return 0

    rows = read_manifest(args.validate)
    payload = {
        "manifest": args.validate,
        "validation": validate(rows),
        "writer_disjointness": audit_writer_disjointness(rows),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    print(text)
    return 0 if payload["validation"]["ok"] else 1  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
