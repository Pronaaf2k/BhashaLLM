"""Stratified sampling of Ekush by grapheme root.

Implements the sampling rule stated in paper Sec. IV-C:

    "For OCR, the 1,500 self-collected pages were split 1,050 / 150 / 300,
    and 6,000 Ekush images were sampled stratified by grapheme root so that
    rare conjuncts survived sampling."

The repository recorded ``ekush_images: 6000`` in
``configs/phase3_ocr_sft.yaml`` and the "stratified by grapheme root"
comment, but contained no code that performs the stratification. Uniform
sampling of 6,000 from Ekush would, by construction, drop the rarest
grapheme roots entirely — which is precisely the outcome the sentence in
the paper claims was avoided.

Why proportional stratification is not enough
---------------------------------------------
Ekush is heavily imbalanced: common roots have thousands of images, rare
conjuncts have tens. Proportional allocation preserves that imbalance, so
a root with 20 images in a 300k-image corpus gets 0 or 1 slots in a 6,000
sample — it does not "survive sampling" in any useful sense.

``allocate`` therefore uses **guaranteed-floor allocation**: every root
that appears at all receives up to ``min_per_class`` images first, and only
the remainder is distributed proportionally. That is the allocation rule
that makes the paper's stated purpose true, and it is stated here rather
than left implicit.

Also note the domain gap recorded in ``docs/ERRATA.md`` B12: Ekush is
isolated handwritten *characters* while the self-collected material is
running text, so roughly 85% of the Phase-3 training set is isolated
characters against a test set that is 100% running text.

Usage
-----
    python -m bhasha.data.ekush_sampling \\
        --input data/processed/ekush_prepared/train.jsonl \\
        --n 6000 --min-per-class 8 \\
        --out data/processed/ekush_sampled_6000.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_SEED = 42  # paper Sec. IV-C
DEFAULT_N = 6000   # configs/phase3_ocr_sft.yaml -> data.ekush_images

# Bangla combining marks: vowel signs (matra), hasant/virama, chandrabindu,
# anusvara, visarga and nukta. Stripping these from a label leaves the
# grapheme *root*, which is the stratification key the paper names.
_MARK_CATEGORIES = {"Mn", "Mc"}
_EXPLICIT_MARKS = {
    "়",  # nukta
    "্",  # hasant / virama
    "ঁ",  # chandrabindu
    "ং",  # anusvara
    "ঃ",  # visarga
}


def grapheme_root(label: str) -> str:
    """Reduce a Bangla label to its grapheme root.

    Drops combining marks and the hasant, then takes the leading base
    character. For a conjunct written as ``<C1> + hasant + <C2>`` this
    returns ``<C1>``, which groups every form of a conjunct family under one
    stratum — the behaviour needed for "rare conjuncts survive sampling" to
    mean anything.

    An empty or non-Bangla label falls back to the label itself, so
    unexpected data forms a visible stratum rather than being silently
    merged into another.
    """
    text = unicodedata.normalize("NFC", label or "")
    base = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in _MARK_CATEGORIES
        and ch not in _EXPLICIT_MARKS
    ).strip()
    return base[0] if base else (text or "<empty>")


def allocate(
    counts: Dict[str, int], n_total: int, min_per_class: int = 8
) -> Dict[str, int]:
    """Guaranteed-floor then proportional allocation across strata.

    Each stratum first receives ``min(min_per_class, available)``. Whatever
    budget remains is allocated proportionally to the residual availability.
    Returns a mapping stratum -> number of images to draw.

    If the floors alone exceed ``n_total``, the floors are scaled down
    proportionally rather than truncating the stratum list, so no root is
    dropped to zero while another keeps its full floor.
    """
    if n_total <= 0:
        return {k: 0 for k in counts}

    floor_demand = {k: min(min_per_class, v) for k, v in counts.items()}
    total_floor = sum(floor_demand.values())

    if total_floor >= n_total:
        scale = n_total / total_floor
        alloc = {k: int(v * scale) for k, v in floor_demand.items()}
        # Distribute the rounding remainder to the rarest strata first, which
        # is where a lost image costs the most.
        remainder = n_total - sum(alloc.values())
        for k in sorted(counts, key=lambda x: counts[x]):
            if remainder <= 0:
                break
            if alloc[k] < counts[k]:
                alloc[k] += 1
                remainder -= 1
        return alloc

    alloc = dict(floor_demand)
    residual = {k: counts[k] - alloc[k] for k in counts}
    residual_total = sum(residual.values())
    budget = n_total - total_floor

    if residual_total > 0:
        for k, r in residual.items():
            alloc[k] += int(budget * r / residual_total)

    # Hand out any remaining slots to the strata with the most left over.
    remainder = n_total - sum(alloc.values())
    for k in sorted(counts, key=lambda x: counts[x] - alloc[x], reverse=True):
        if remainder <= 0:
            break
        if alloc[k] < counts[k]:
            alloc[k] += 1
            remainder -= 1
    return alloc


def stratified_sample(
    records: List[Dict[str, object]],
    n_total: int = DEFAULT_N,
    min_per_class: int = 8,
    seed: int = DEFAULT_SEED,
    label_key: str = "text",
) -> Dict[str, object]:
    """Draw ``n_total`` records stratified by grapheme root.

    Returns a dict with the sampled records and a report describing the
    stratification, so the sample is reproducible and auditable rather than
    an unexplained file.
    """
    strata: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for rec in records:
        strata[grapheme_root(str(rec.get(label_key, "")))].append(rec)

    counts = {k: len(v) for k, v in strata.items()}
    alloc = allocate(counts, n_total, min_per_class)

    rng = random.Random(seed)
    sampled: List[Dict[str, object]] = []
    for root in sorted(strata):
        pool = list(strata[root])
        rng.shuffle(pool)
        sampled.extend(pool[: alloc[root]])
    rng.shuffle(sampled)

    covered = sum(1 for k in counts if alloc[k] > 0)
    return {
        "records": sampled,
        "report": {
            "seed": seed,
            "requested": n_total,
            "drawn": len(sampled),
            "n_strata_in_source": len(counts),
            "n_strata_represented": covered,
            "strata_lost": len(counts) - covered,
            "min_per_class": min_per_class,
            "allocation_rule": "guaranteed floor of min_per_class per "
                               "grapheme root, remainder proportional to "
                               "residual availability",
            "rarest_strata": [
                {"root": k, "available": counts[k], "drawn": alloc[k]}
                for k in sorted(counts, key=lambda x: counts[x])[:15]
            ],
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, help="Ekush JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None,
                    help="where to write the stratification report "
                         "(default: <out>.report.json)")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--min-per-class", type=int, default=8)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--label-key", default="text")
    args = ap.parse_args(argv)

    records = [
        json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    result = stratified_sample(
        records, args.n, args.min_per_class, args.seed, args.label_key
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for rec in result["records"]:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report_path = Path(args.report) if args.report else out.with_suffix(
        out.suffix + ".report.json"
    )
    report_path.write_text(
        json.dumps(result["report"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(result["report"], indent=2, ensure_ascii=False))
    print(f"\nwrote {out} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
