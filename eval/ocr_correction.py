#!/usr/bin/env python3
"""
OCR-correction scoring: correction, over-correction and false-positive rates.

Paper Sec. V-B reports that "Llama-3.2-11B and Llama-3.1-8B led with 88%
and 85% correct fixes and the lowest over-correction and false-positive
rates", and that "Mistral-7B showed the worst pattern in the set: a 35%
correction rate paired with a 30% over-correction rate".

``docs/ERRATA.md`` B11 records the problem with those numbers as printed:
**no denominators are given.** 88% of how many available errors, over how
many items? Three different denominators are defensible and they give
different answers, so a rate without its denominator is not a checkable
figure. This script fixes the denominators in code and reports every count
alongside every rate.

Paper Sec. IV-D also states the motivation for tracking the three rates
separately: "so that a model cannot inflate its apparent correction rate
simply by editing text aggressively."

Definitions
-----------
Every record carries three strings: the noisy OCR output (``noisy``), the
ground truth (``reference``), and the model's corrected output
(``hypothesis``). Character-level Levenshtein alignment of ``noisy`` to
``reference`` partitions the reference into two disjoint sets:

``error_sites``
    reference positions where ``noisy`` differs from ``reference``. These
    are the characters a corrector *should* change.

``correct_sites``
    reference positions where ``noisy`` already matches ``reference``.
    These are the characters a corrector *should not* touch.

The three rates then are:

``correction_rate``
    fixed error sites / total error sites.
    Denominator: the number of errors actually available to fix. A model
    cannot score above 100% by editing more.

``over_correction_rate``
    damaged correct sites / total correct sites.
    Denominator: the characters that were already right. This is the rate
    that catches a model "actively introducing new errors".

``false_positive_rate``
    edits landing on correct sites / total edits made.
    Denominator: the model's own edit budget. This answers "of everything
    this model changed, what fraction should it have left alone?" and is
    the rate a reader most often assumes ``over_correction_rate`` means.

``net_error_reduction``
    (edit distance before - edit distance after) / edit distance before.
    The single number worth quoting if only one is quoted, because it
    cannot be gamed in either direction: aggressive editing shows up as a
    negative value.

Input JSONL, one record per item::

    {"item_id": "...", "noisy": "...", "reference": "...",
     "hypothesis": "...", "model": "..."}

``model`` is optional; if present, results are broken down by it.

Usage
-----
    python eval/ocr_correction.py --pred benchmarks/raw/ocr_correction.jsonl \\
        --out eval/ocr_correction.json
"""

import argparse
import json
import os
import sys
import unicodedata
from collections import defaultdict

# eval/ocr_cer.py already implements Levenshtein alignment with
# backtracking. Reuse it rather than writing a second, subtly different
# aligner -- two aligners in one repository is how two incompatible CERs
# get published.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from ocr_cer import align as _align, norm as _norm  # noqa: E402
except ImportError:  # pragma: no cover - fallback for odd import paths
    _align = _norm = None


def norm(s):
    if _norm is not None:
        return _norm(s)
    return unicodedata.normalize("NFC", s or "")


def align(ref, hyp):
    if _align is not None:
        return _align(ref, hyp)
    raise RuntimeError("eval/ocr_cer.py must be importable for alignment")


def edit_distance(a, b):
    return sum(1 for op, _, _ in align(a, b) if op != "match")


def _reference_status(reference, other):
    """Map each reference index to whether ``other`` reproduces it correctly.

    Returns a list of booleans, one per reference character: True where
    ``other`` has the right character at the aligned position.
    """
    status = []
    for op, rc, _ in align(reference, other):
        if op == "ins":
            continue  # insertions consume no reference position
        status.append(op == "match")
    # Alignment always consumes every reference character exactly once.
    assert len(status) == len(reference), (
        f"alignment consumed {len(status)} of {len(reference)} reference chars"
    )
    return status


def score_item(record):
    """Score one correction item. Pure function, unit-testable."""
    ref = norm(record["reference"])
    noisy = norm(record["noisy"])
    hyp = norm(record["hypothesis"])

    before = _reference_status(ref, noisy)   # True where noisy was already right
    after = _reference_status(ref, hyp)      # True where the correction is right

    error_sites = sum(1 for ok in before if not ok)
    correct_sites = sum(1 for ok in before if ok)

    fixed = sum(1 for b, a in zip(before, after) if not b and a)
    still_wrong = sum(1 for b, a in zip(before, after) if not b and not a)
    damaged = sum(1 for b, a in zip(before, after) if b and not a)
    preserved = sum(1 for b, a in zip(before, after) if b and a)

    # Edits the model made, measured against the noisy input rather than the
    # reference: this is the model's own action, independent of ground truth.
    edits_made = edit_distance(noisy, hyp)

    d_before = edit_distance(ref, noisy)
    d_after = edit_distance(ref, hyp)

    return {
        "item_id": record.get("item_id"),
        "model": record.get("model"),
        "reference_chars": len(ref),
        "error_sites": error_sites,
        "correct_sites": correct_sites,
        "fixed": fixed,
        "still_wrong": still_wrong,
        "damaged": damaged,
        "preserved": preserved,
        "edits_made": edits_made,
        "distance_before": d_before,
        "distance_after": d_after,
    }


def aggregate(items):
    """Pool counts across items, then compute rates. Micro-averaged.

    Micro-averaging (pool the counts, divide once) rather than averaging
    per-item rates: an item with one error site would otherwise carry the
    same weight as an item with forty, and short items dominate.
    """
    tot = defaultdict(int)
    for it in items:
        for k in ("error_sites", "correct_sites", "fixed", "still_wrong",
                  "damaged", "preserved", "edits_made", "distance_before",
                  "distance_after", "reference_chars"):
            tot[k] += it[k]

    def rate(num, den):
        return round(num / den, 4) if den else None

    # Edits that landed on a site that was already correct. `damaged` counts
    # those that made it wrong; that is the measurable lower bound on
    # misdirected edits at character granularity.
    misdirected = tot["damaged"]

    return {
        "n_items": len(items),
        "counts": {
            "reference_chars": tot["reference_chars"],
            "error_sites_available": tot["error_sites"],
            "correct_sites_available": tot["correct_sites"],
            "errors_fixed": tot["fixed"],
            "errors_remaining": tot["still_wrong"],
            "correct_chars_damaged": tot["damaged"],
            "correct_chars_preserved": tot["preserved"],
            "edits_made": tot["edits_made"],
            "edit_distance_before": tot["distance_before"],
            "edit_distance_after": tot["distance_after"],
        },
        "rates": {
            "correction_rate": rate(tot["fixed"], tot["error_sites"]),
            "correction_rate_denominator": tot["error_sites"],
            "over_correction_rate": rate(misdirected, tot["correct_sites"]),
            "over_correction_rate_denominator": tot["correct_sites"],
            "false_positive_rate": rate(misdirected, tot["edits_made"]),
            "false_positive_rate_denominator": tot["edits_made"],
            "net_error_reduction": rate(
                tot["distance_before"] - tot["distance_after"],
                tot["distance_before"],
            ),
            "cer_before": rate(tot["distance_before"], tot["reference_chars"]),
            "cer_after": rate(tot["distance_after"], tot["reference_chars"]),
        },
    }


DEFINITIONS = {
    "correction_rate": "errors_fixed / error_sites_available. Denominator is "
                       "the number of reference characters the noisy input "
                       "got wrong.",
    "over_correction_rate": "correct_chars_damaged / correct_sites_available. "
                            "Denominator is the number of reference "
                            "characters the noisy input already had right.",
    "false_positive_rate": "correct_chars_damaged / edits_made. Denominator "
                           "is the number of character edits the model itself "
                           "made, measured noisy -> hypothesis.",
    "net_error_reduction": "(edit_distance_before - edit_distance_after) / "
                           "edit_distance_before. Negative means the model "
                           "made the text worse overall.",
    "alignment": "Character-level Levenshtein with backtracking, shared with "
                 "eval/ocr_cer.py. Unicode NFC normalisation.",
    "averaging": "Micro-averaged: counts pooled across items, then divided.",
    "why_three_rates": "Paper Sec. IV-D: tracked separately 'so that a model "
                       "cannot inflate its apparent correction rate simply by "
                       "editing text aggressively'.",
    "errata": "docs/ERRATA.md B11 records that Sec. V-B reports these rates "
              "without denominators. Every denominator is reported here.",
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="JSONL of correction items")
    ap.add_argument("--out", default="eval/ocr_correction.json")
    ap.add_argument("--per-item", action="store_true")
    args = ap.parse_args()

    with open(args.pred, encoding="utf-8") as fh:
        records = [json.loads(l) for l in fh if l.strip()]
    for i, r in enumerate(records):
        missing = [k for k in ("noisy", "reference", "hypothesis") if k not in r]
        if missing:
            sys.exit(f"record {i} is missing {missing}; "
                     f"every record needs noisy, reference and hypothesis")

    items = [score_item(r) for r in records]

    payload = {
        "source": args.pred,
        "metric_definitions": DEFINITIONS,
        "overall": aggregate(items),
    }

    by_model = defaultdict(list)
    for it in items:
        if it.get("model"):
            by_model[it["model"]].append(it)
    if by_model:
        payload["per_model"] = {m: aggregate(v) for m, v in sorted(by_model.items())}

    if args.per_item:
        payload["per_item"] = items

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    printable = {k: v for k, v in payload.items() if k != "per_item"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
