#!/usr/bin/env python3
"""
Extract and audit `text dataset.rar` against the paper's corpus claims.

`text dataset.rar` sits at the repository root with no manifest, no
checksum, no licence and no provenance. `docs/ERRATA.md` group C claimed it
had been "Replaced by extracted files + `data/text_raw.sha256`". It had
not: the archive is still the only form the corpus exists in, nothing
extracts it, and nothing produced that checksum file. That errata row has
been corrected, and this script is what makes the claim true when run.

What it checks
--------------
Three claims in the paper depend on this archive, and none of them could be
verified before:

**Size (Sec. IV-B, Table IV): "roughly 6.6M tokens", split 5.28M / 0.66M /
0.66M.** The archive is 58 KB compressed. Bangla text compresses well, but
6.6M tokens is on the order of 25-30 MB of UTF-8, which would not fit in
58 KB at any plausible ratio. This script reports the actual character and
token counts so the gap is measured rather than argued about.

**Author-disjoint splitting (Sec. IV-C): "split 80/10/10 by document rather
than by sentence, so no work by the same author appears on both sides."**
The archive's entries are named `text dataset/1.txt` … `N.txt`. There are
no author directories and no metadata of any kind, so *there is nothing in
the committed corpus that records which author wrote which document.* The
guarantee in Sec. IV-C cannot be reconstructed from this archive. The
script reports the directory structure so this is visible rather than
inferred, and `bhasha/data/text_corpus.py` already refuses to claim an
author-disjoint split it cannot deliver.

**Provenance (Refs [21], [22]): Kazi Nazrul Islam and Rabindranath Tagore
corpora.** Numbered files carry no attribution. The script samples content
so a human can begin the attribution `DATA_CARD.md` still marks `FILL`.

Extraction
----------
RAR is a proprietary format with no decoder in the Python standard library.
The script tries, in order: `unar`, `unrar`, `7z`, `bsdtar`, then the
`rarfile` package. If none is available it still reports everything
derivable from the archive bytes — size, checksum, entry names — and says
what to install. It never fails silently.

Usage
-----
    python scripts/audit_text_corpus.py
    python scripts/audit_text_corpus.py --extract-to data/raw/text_dataset
    python scripts/audit_text_corpus.py --tokenizer Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = BASE_DIR / "text dataset.rar"
DEFAULT_EXTRACT = BASE_DIR / "data" / "raw" / "text_dataset"
DEFAULT_CHECKSUMS = BASE_DIR / "data" / "text_raw.sha256"
DEFAULT_REPORT = BASE_DIR / "data" / "text_corpus_audit.json"

# Table IV. The figures this audit exists to check.
PAPER_CLAIMS = {
    "total_tokens": 6_600_000,
    "train_tokens": 5_280_000,
    "val_tokens": 660_000,
    "test_tokens": 660_000,
    "split": "80/10/10 by document",
    "authors": ["Kazi Nazrul Islam [21]", "Rabindranath Tagore [22]"],
    "normalisation": "NFKC, stray Latin stripped",
}

BENGALI = (0x0980, 0x09FF)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry_names_from_bytes(data: bytes) -> list[str]:
    """Recover entry names from the raw archive.

    RAR stores filenames as plain bytes in the header, so they are readable
    without a decoder. This is what lets the script report structure even
    on a machine with no RAR tool installed.
    """
    found = re.findall(rb"[\x20-\x7e]{3,80}\.(?:txt|csv|json|md|docx|pdf)", data)
    return sorted({n.decode("utf-8", "replace") for n in found})


def try_extract(archive: Path, dest: Path) -> dict:
    """Extract with whichever tool is present. Report which one was used."""
    dest.mkdir(parents=True, exist_ok=True)
    attempts = [
        ("unar", ["unar", "-quiet", "-force-overwrite", "-output-directory",
                  str(dest), str(archive)]),
        ("unrar", ["unrar", "x", "-o+", "-idq", str(archive), str(dest) + os.sep]),
        ("7z", ["7z", "x", f"-o{dest}", "-y", str(archive)]),
        ("bsdtar", ["bsdtar", "-xf", str(archive), "-C", str(dest)]),
    ]
    for name, cmd in attempts:
        if not shutil.which(name):
            continue
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0:
                return {"extracted": True, "tool": name}
            last = f"{name} exited {proc.returncode}: {proc.stderr.strip()[:300]}"
        except Exception as exc:  # noqa: BLE001
            last = f"{name} failed: {exc}"
    else:
        last = "no rar tool found on PATH"

    # Last resort: the rarfile package, which still shells out for RAR5 but
    # is worth trying because it may be installed where the CLI tools are not.
    try:
        import rarfile
        with rarfile.RarFile(str(archive)) as rf:
            rf.extractall(str(dest))
        return {"extracted": True, "tool": "python-rarfile"}
    except Exception as exc:  # noqa: BLE001
        return {
            "extracted": False,
            "error": last,
            "rarfile_error": str(exc),
            "hint": "install one of: unar, unrar, p7zip-full, bsdtar, "
                    "or `pip install rarfile`",
        }


def bangla_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    n = sum(1 for c in chars if BENGALI[0] <= ord(c) <= BENGALI[1])
    return n / len(chars)


def analyse_extracted(root: Path, tokenizer_id: str | None) -> dict:
    files = sorted(root.rglob("*.txt"))
    if not files:
        return {"n_files": 0,
                "note": f"no .txt files under {root} after extraction"}

    total_chars = 0
    per_file = []
    dirs = Counter()
    checksums = []
    corpus_texts = []

    for f in files:
        raw = f.read_text(encoding="utf-8", errors="replace")
        text = unicodedata.normalize("NFKC", raw)
        total_chars += len(text)
        rel = f.relative_to(root)
        dirs[str(rel.parent)] += 1
        per_file.append({
            "path": str(rel),
            "chars": len(text),
            "bangla_ratio": round(bangla_ratio(text), 3),
        })
        checksums.append((hashlib.sha256(raw.encode("utf-8")).hexdigest(), str(rel)))
        corpus_texts.append(text)

    tokens = None
    if tokenizer_id:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_id)
            tokens = sum(len(tok(t).input_ids) for t in corpus_texts)
        except Exception as exc:  # noqa: BLE001
            tokens = {"error": str(exc)}

    # Estimate tokens without a tokenizer. Bangla runs roughly 2.5-3.5
    # characters per subword token in a multilingual vocabulary; 3.0 is the
    # midpoint and is clearly labelled as an estimate.
    est_tokens = int(total_chars / 3.0)

    # Author metadata: does the layout carry any?
    top_level = sorted({p.split(os.sep)[0] for p in dirs if p != "."})
    numeric_only = all(re.fullmatch(r"\d+", Path(p["path"]).stem) for p in per_file)

    return {
        "n_files": len(files),
        "total_characters": total_chars,
        "tokens_measured": tokens,
        "tokens_estimated_at_3_chars_per_token": est_tokens,
        "directory_structure": dict(dirs),
        "top_level_directories": top_level,
        "filenames_are_numeric_only": numeric_only,
        "mean_bangla_ratio": round(
            sum(p["bangla_ratio"] for p in per_file) / len(per_file), 3),
        "largest_files": sorted(per_file, key=lambda x: -x["chars"])[:5],
        "checksums": checksums,
    }


def compare_to_paper(analysis: dict) -> dict:
    """Set the measured corpus beside Table IV."""
    if not analysis.get("n_files"):
        return {"comparable": False,
                "reason": "archive not extracted; nothing to compare"}

    measured = analysis.get("tokens_measured")
    if isinstance(measured, dict) or measured is None:
        measured = analysis["tokens_estimated_at_3_chars_per_token"]
        basis = "estimated at 3.0 chars/token"
    else:
        basis = "measured with the named tokenizer"

    claimed = PAPER_CLAIMS["total_tokens"]
    ratio = measured / claimed if claimed else None

    findings = []
    if ratio is not None and ratio < 0.5:
        findings.append(
            f"The extracted corpus is {measured:,} tokens ({basis}) against "
            f"Table IV's {claimed:,} — about {ratio * 100:.1f}% of the claimed "
            f"size. Either this archive is a sample rather than the training "
            f"corpus, or the 6.6M figure needs revising. Sec. IV-B's corpus "
            f"is not reproducible from what is committed."
        )
    elif ratio is not None and ratio > 2.0:
        findings.append(
            f"The extracted corpus is {measured:,} tokens, well above Table "
            f"IV's {claimed:,}. Check whether the archive holds more than the "
            f"corpus actually used."
        )

    if analysis.get("filenames_are_numeric_only"):
        findings.append(
            "Every file is numbered (1.txt, 2.txt, ...) with no author "
            "directory or metadata. Sec. IV-C claims the 80/10/10 split was "
            "made 'by document rather than by sentence, so no work by the "
            "same author appears on both sides'. That guarantee cannot be "
            "reconstructed from this archive: nothing records which author "
            "wrote which file. Add an author column before re-splitting, or "
            "withdraw the author-disjointness claim."
        )

    if len(analysis.get("top_level_directories", [])) < 2:
        findings.append(
            f"Refs [21] and [22] name two corpora (Nazrul, Tagore) but the "
            f"archive has {len(analysis.get('top_level_directories', []))} "
            f"top-level directories, so the two sources are not separable as "
            f"committed. DATA_CARD.md's per-source token counts cannot be "
            f"filled from this file."
        )

    return {
        "comparable": True,
        "tokens_measured_or_estimated": measured,
        "basis": basis,
        "paper_claim": claimed,
        "ratio_to_claim": round(ratio, 4) if ratio else None,
        "findings": findings,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--extract-to", default=str(DEFAULT_EXTRACT))
    ap.add_argument("--no-extract", action="store_true",
                    help="report on the archive bytes only")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer id for an exact token count, e.g. "
                         "Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--checksums", default=str(DEFAULT_CHECKSUMS))
    ap.add_argument("--out", default=str(DEFAULT_REPORT))
    args = ap.parse_args()

    archive = Path(args.archive)
    if not archive.exists():
        sys.exit(f"archive not found: {archive}")

    data = archive.read_bytes()
    report = {
        "archive": str(archive.relative_to(BASE_DIR)),
        "archive_bytes": len(data),
        "archive_sha256": sha256_of(archive),
        "archive_format": data[:7].decode("latin-1", "replace"),
        "entry_names": entry_names_from_bytes(data),
        "paper_claims": PAPER_CLAIMS,
    }
    report["n_entries_visible"] = len(report["entry_names"])

    if args.no_extract:
        report["extraction"] = {"skipped": True}
        report["analysis"] = {}
    else:
        dest = Path(args.extract_to)
        report["extraction"] = try_extract(archive, dest)
        report["extract_to"] = str(dest)
        report["analysis"] = (
            analyse_extracted(dest, args.tokenizer)
            if report["extraction"].get("extracted") else {}
        )

    report["comparison_to_paper"] = compare_to_paper(report["analysis"])

    # Write the checksum file docs/ERRATA.md group C referred to.
    checksums = report["analysis"].get("checksums")
    if checksums:
        cs = Path(args.checksums)
        cs.parent.mkdir(parents=True, exist_ok=True)
        cs.write_text(
            "\n".join(f"{h}  {p}" for h, p in checksums) + "\n", encoding="utf-8")
        report["checksums_written_to"] = str(cs)
        report["analysis"].pop("checksums", None)  # keep the JSON readable

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")

    # ---- human-readable summary
    print(f"archive        {report['archive']}")
    print(f"  size         {report['archive_bytes']:,} bytes")
    print(f"  sha256       {report['archive_sha256']}")
    print(f"  entries      {report['n_entries_visible']} visible in the header")
    ex = report["extraction"]
    if ex.get("skipped"):
        print("  extraction   skipped (--no-extract)")
    elif ex.get("extracted"):
        a = report["analysis"]
        print(f"  extraction   ok via {ex['tool']} -> {report['extract_to']}")
        print(f"  files        {a['n_files']}")
        print(f"  characters   {a['total_characters']:,}")
        print(f"  tokens       {a.get('tokens_measured') or a['tokens_estimated_at_3_chars_per_token']:,}"
              f"  ({report['comparison_to_paper']['basis']})")
        print(f"  bangla ratio {a['mean_bangla_ratio']}")
    else:
        print(f"  extraction   FAILED — {ex.get('error')}")
        print(f"               {ex.get('hint')}")

    cmp_ = report["comparison_to_paper"]
    if cmp_.get("findings"):
        print("\nfindings against the paper:")
        for i, f in enumerate(cmp_["findings"], 1):
            print(f"  {i}. {f}")
    elif cmp_.get("comparable"):
        print("\n  no discrepancies found against Table IV.")

    if report.get("checksums_written_to"):
        print(f"\nwrote {report['checksums_written_to']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
