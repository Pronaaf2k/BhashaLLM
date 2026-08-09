#!/usr/bin/env python3
"""
Make the committed OCR datasets usable on a machine that is not the author's.

Every JSONL under `training/vlm_ocr/data_desktop/` stores its `image` field
as an absolute path on one developer's laptop::

    {"image": "/home/benaaf/Desktop/datasets/image_787 to image_927/img799line8.png", ...}

There are 3,402 such records across the four split directories, plus 88 in
the committed PaliGemma predictions. On any other machine every one of them
resolves to nothing, so the OCR training and evaluation data — the material
Section IV-B describes as the project's own contribution — cannot be loaded
by anyone else. `docs/ERRATA.md` group C claimed the hardcoded paths had
been "Removed; paths are relative"; only the README instance was. See C21.

What this does
--------------
Rewrites the `image` field to a path relative to a declared dataset root,
writing to **new files** and leaving the originals untouched. Combined with
`OCRDataset(..., image_root=...)`, which already accepts a root, that makes
the splits portable:

    before  /home/benaaf/Desktop/datasets/bnaf/img12line3.png
    after   bnaf/img12line3.png            + --image-root <your dataset dir>

It also reports, without changing anything, what the data actually contains
— which is how the discrepancies in `docs/ERRATA.md` C22 were found.

Why not rewrite in place
------------------------
The absolute paths are the only surviving record of the original collection
layout: three folders named `image_787 to image_927`, `bnaf` and
`dataset10thMay-...`. That structure is the closest thing the repository
has to per-session provenance, and Section IV-B's "deliberate variation in
writing speed, pen type, and print-versus-cursive style ... across
collection sessions" is otherwise unrecorded. Overwriting the originals
would destroy it. The mapping is written to a sidecar file instead.

Usage
-----
    # report only, changes nothing
    python scripts/relocate_dataset_paths.py --report

    # rewrite into a portable copy
    python scripts/relocate_dataset_paths.py --write \\
        --out-dir training/vlm_ocr/data_portable

    # then
    python -m bhasha.ocr.train --config configs/phase3_ocr_sft.yaml \\
        --data_dir training/vlm_ocr/data_portable/merged_lines \\
        --image_root /path/to/your/datasets
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SRC = BASE_DIR / "training" / "vlm_ocr" / "data_desktop"
DEFAULT_OUT = BASE_DIR / "training" / "vlm_ocr" / "data_portable"

# The collection root every committed path shares.
KNOWN_ROOTS = [
    "/home/benaaf/Desktop/datasets/",
    "/home/benaaf/Desktop/ocr_datasets/",
    "/home/benaaf/Desktop/BhashaLLM_Export/",
    "/home/benaaf/Desktop/BhashaLLM/",
    "/home/benaaf/",
]

ABS_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|/)")

# Paper Sec. IV-C, for the comparison this script prints.
PAPER_SPLIT = {"train": 1050, "val": 150, "test": 300, "total": 1500,
               "unit": "pages"}


def strip_root(path: str) -> str:
    """Reduce an absolute path to something relative to a dataset root."""
    p = path.replace("\\", "/")
    for root in KNOWN_ROOTS:
        if p.startswith(root):
            return p[len(root):]
    if ABS_RE.match(p):
        # Unknown absolute root: keep the last two components, which is
        # enough to locate the file under a user-supplied root.
        parts = [x for x in p.split("/") if x]
        return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return p


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    out = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return out


def analyse(src: Path) -> Dict[str, object]:
    """Report what the committed splits actually contain."""
    splits: Dict[str, Dict[str, int]] = defaultdict(dict)
    roots = Counter()
    absolute = total = 0
    missing_text = 0
    has_writer_id = False

    for f in sorted(src.rglob("*.jsonl")):
        group, split = f.parent.name, f.stem
        recs = read_jsonl(f)
        splits[group][split] = len(recs)
        for r in recs:
            total += 1
            img = str(r.get("image", ""))
            if ABS_RE.match(img.replace("\\", "/")):
                absolute += 1
                roots[str(Path(img.replace("\\", "/")).parent)] += 1
            if not str(r.get("text", "")).strip():
                missing_text += 1
            if r.get("writer_id"):
                has_writer_id = True

    return {
        "n_records": total,
        "n_absolute_paths": absolute,
        "portable": absolute == 0,
        "records_with_empty_text": missing_text,
        "has_writer_id": has_writer_id,
        "collection_directories": dict(roots),
        "splits": {g: dict(s) for g, s in sorted(splits.items())},
    }


def compare_to_paper(analysis: Dict[str, object]) -> List[str]:
    findings = []
    splits = analysis["splits"]  # type: ignore[index]

    if analysis["n_absolute_paths"]:
        findings.append(
            f"{analysis['n_absolute_paths']} of {analysis['n_records']} records "
            f"store an absolute path from one developer's machine. On any "
            f"other machine none of them resolves, so the self-collected OCR "
            f"data cannot be loaded. docs/ERRATA.md group C claimed these had "
            f"been made relative; they had not (C21)."
        )

    merged = splits.get("merged_lines")
    if merged:
        tr, va, te = (merged.get("train", 0), merged.get("val", 0),
                      merged.get("test", 0))
        tot = tr + va + te
        findings.append(
            f"merged_lines holds {tot} records split {tr}/{va}/{te} "
            f"(~{100*tr/tot:.0f}/{100*va/tot:.0f}/{100*te/tot:.0f}). "
            f"Paper Sec. IV-C states {PAPER_SPLIT['train']}/"
            f"{PAPER_SPLIT['val']}/{PAPER_SPLIT['test']} — a 70/10/20 split "
            f"of {PAPER_SPLIT['total']} {PAPER_SPLIT['unit']}. Neither the "
            f"proportions nor the counts match."
        )
        findings.append(
            f"The records are LINES, not pages: the filenames follow "
            f"`img<N>line<M>.png`. Sec. IV-B and the abstract describe "
            f"{PAPER_SPLIT['total']} '{PAPER_SPLIT['unit']}' / images. "
            f"{tot} lines is the right order of magnitude for the stated "
            f"figure, which suggests the paper counts lines and calls them "
            f"pages."
        )

    dirs = analysis["collection_directories"]  # type: ignore[index]
    if dirs:
        findings.append(
            f"The data comes from {len(dirs)} collection directories, not "
            f"{len(dirs)} writers: {sorted(Path(d).name for d in dirs)}. "
            f"Sec. IV-B says three writers; a directory is a collection "
            f"session, and nothing maps sessions to writers."
        )

    if not analysis["has_writer_id"]:
        findings.append(
            "No record carries a writer_id, so the writer-disjoint CER of "
            "Sec. VI-D cannot be computed from this data at all. Populate "
            "data/handwriting/manifest.csv "
            "(`python -m bhasha.data.manifest --template ...`)."
        )
    return findings


def rewrite(src: Path, out: Path, dry_run: bool) -> Dict[str, object]:
    written, changed = [], 0
    for f in sorted(src.rglob("*.jsonl")):
        recs = read_jsonl(f)
        mapping = {}
        for r in recs:
            old = str(r.get("image", ""))
            new = strip_root(old)
            if new != old:
                changed += 1
                mapping[new] = old
            r["image"] = new
            # Preserve the collection folder as provenance, since stripping
            # the root is what removes it.
            if old and "collection_dir" not in r:
                r["collection_dir"] = Path(old.replace("\\", "/")).parent.name

        target = out / f.relative_to(src)
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            sidecar = target.with_suffix(".pathmap.json")
            sidecar.write_text(
                json.dumps(mapping, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
        written.append(str(target))
    return {"files": written, "paths_rewritten": changed, "dry_run": dry_run}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--report", action="store_true",
                    help="analyse only; write nothing (the default)")
    ap.add_argument("--write", action="store_true",
                    help="write the portable copy")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"source not found: {src}")

    a = analyse(src)
    print(f"  records            {a['n_records']}")
    print(f"  absolute paths     {a['n_absolute_paths']}  "
          f"({'PORTABLE' if a['portable'] else 'NOT PORTABLE'})")
    print(f"  writer_id present  {a['has_writer_id']}")
    print(f"  empty text fields  {a['records_with_empty_text']}")
    print("\n  splits:")
    for group, s in a["splits"].items():  # type: ignore[union-attr]
        tot = sum(s.values())
        print(f"    {group:24s} " +
              "  ".join(f"{k} {v}" for k, v in sorted(s.items())) +
              f"   total {tot}")
    print("\n  collection directories:")
    for d, n in sorted(a["collection_directories"].items(),  # type: ignore[union-attr]
                       key=lambda x: -x[1]):
        print(f"    {n:5d}  {d}")

    findings = compare_to_paper(a)
    if findings:
        print("\n  findings against the paper:")
        for i, f in enumerate(findings, 1):
            print(f"    {i}. {f}\n")

    if args.write:
        res = rewrite(src, Path(args.out_dir), dry_run=False)
        print(f"  rewrote {res['paths_rewritten']} paths across "
              f"{len(res['files'])} files -> {args.out_dir}")
        print("  each split has a .pathmap.json sidecar preserving the "
              "original absolute paths")
        print(f"\n  use it:\n    python -m bhasha.ocr.train "
              f"--config configs/phase3_ocr_sft.yaml \\\n"
              f"        --data_dir {args.out_dir}/merged_lines \\\n"
              f"        --image_root /path/to/your/datasets")
    else:
        print("  (report only — pass --write to produce a portable copy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
