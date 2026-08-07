#!/usr/bin/env python3
"""
A definition for the "confidence score" in Table VII, and code to compute it.

Table VII reports a confidence score rising from 0.68 to 0.82 after
fine-tuning. ``docs/ERRATA.md`` B11 records the problem: *confidence is not
a defined output of an autoregressive vision-language model.* The entry
gives two options -- supply a definition or withdraw the number. This file
supplies the definition.

Definition adopted
------------------
    confidence = exp( mean over generated tokens of log P(token | prefix) )

which is the sequence's per-token geometric-mean probability. Properties
that make it the right choice here:

* It lies in (0, 1] and equals 1 only for a sequence the model considered
  certain at every step, so it reads on the same scale a reader expects
  from the word "confidence".
* It is **length-normalised**. The raw sequence probability shrinks
  geometrically with length, so an unnormalised score would report that the
  model is less confident about longer transcriptions purely because they
  are longer. Table VII compares a before/after pair whose output lengths
  differ, so this is not a nicety.
* It is computable from any HuggingFace ``generate`` call with
  ``output_scores=True``, requiring no extra forward pass.

What it is not
--------------
This is a *calibration-free* self-reported quantity. A model can be
confidently wrong, and 4-bit quantisation perturbs the logits that produce
it. Reporting it next to CER is fine; reporting it as evidence of accuracy
is not. ``expected_calibration_error`` is therefore computed alongside
whenever references are available, so the number carries its own health
check: a confidence score whose ECE is large is not measuring what its name
suggests.

Two alternatives, for the record
--------------------------------
* Mean token probability (arithmetic rather than geometric). Dominated by
  high-probability tokens; a single near-zero step barely moves it.
* Minimum token probability. Sensitive to exactly the failure that matters
  for OCR (one badly-placed diacritic) but noisy and not comparable across
  lengths.
Both are reported as secondary fields so a reader can see the spread.

Input JSONL, one record per generation::

    {"item_id": "...", "token_logprobs": [-0.01, -0.5, ...],
     "hypothesis": "...", "reference": "...", "model": "..."}

``reference`` is optional and enables the calibration section.
``token_logprobs`` must be natural-log probabilities of the *generated*
tokens only, excluding the prompt.

Collecting token_logprobs from HuggingFace generate
---------------------------------------------------
    out = model.generate(**inputs, max_new_tokens=128,
                         output_scores=True, return_dict_in_generate=True)
    tr = model.compute_transition_scores(
        out.sequences, out.scores, normalize_logits=True)
    token_logprobs = tr[0].tolist()   # already natural log

Usage
-----
    python eval/confidence.py --pred eval/ocr_predictions_logprobs.jsonl \\
        --out eval/confidence.json
"""

import argparse
import json
import math
import os
import sys
import unicodedata
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from ocr_cer import align as _align
except ImportError:  # pragma: no cover
    _align = None

DEFAULT_N_BINS = 10


def sequence_confidence(token_logprobs):
    """exp(mean log p) over the generated tokens.

    Returns ``None`` for an empty sequence rather than 0.0 or 1.0: a model
    that generated nothing has no confidence to report, and folding that
    into either extreme distorts the corpus mean.
    """
    lps = [float(x) for x in (token_logprobs or [])]
    if not lps:
        return None
    return math.exp(sum(lps) / len(lps))


def token_probabilities(token_logprobs):
    return [math.exp(float(x)) for x in (token_logprobs or [])]


def summarise_item(record):
    lps = record.get("token_logprobs") or []
    probs = token_probabilities(lps)
    return {
        "item_id": record.get("item_id"),
        "model": record.get("model"),
        "n_tokens": len(lps),
        # Primary definition.
        "confidence": (
            round(sequence_confidence(lps), 6)
            if sequence_confidence(lps) is not None else None
        ),
        # Secondary, for the spread discussion in the module docstring.
        "mean_token_probability": (
            round(sum(probs) / len(probs), 6) if probs else None
        ),
        "min_token_probability": round(min(probs), 6) if probs else None,
        "total_logprob": round(sum(float(x) for x in lps), 6) if lps else None,
    }


def _char_accuracy(reference, hypothesis):
    """Fraction of reference characters correctly reproduced."""
    ref = unicodedata.normalize("NFC", reference or "")
    hyp = unicodedata.normalize("NFC", hypothesis or "")
    if not ref:
        return None
    if _align is None:
        return None
    ops = _align(ref, hyp)
    matches = sum(1 for op, _, _ in ops if op == "match")
    return matches / len(ref)


def calibration(pairs, n_bins=DEFAULT_N_BINS):
    """Expected calibration error over (confidence, accuracy) pairs.

    Bins items by reported confidence and compares the mean confidence in
    each bin with the mean measured character accuracy. ECE is the
    sample-weighted mean absolute gap. A well-calibrated 0.82 means items
    scored 0.82 are right about 82% of the time; ECE says how far from that
    the number actually is.
    """
    pairs = [(c, a) for c, a in pairs if c is not None and a is not None]
    if not pairs:
        return {"n": 0, "expected_calibration_error": None,
                "note": "no items carried both a confidence and a reference"}

    bins = defaultdict(list)
    for conf, acc in pairs:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, acc))

    total = len(pairs)
    ece = 0.0
    rows = []
    for idx in sorted(bins):
        items = bins[idx]
        mean_conf = sum(c for c, _ in items) / len(items)
        mean_acc = sum(a for _, a in items) / len(items)
        ece += (len(items) / total) * abs(mean_conf - mean_acc)
        rows.append({
            "bin": f"[{idx / n_bins:.1f}, {(idx + 1) / n_bins:.1f})",
            "n": len(items),
            "mean_confidence": round(mean_conf, 4),
            "mean_char_accuracy": round(mean_acc, 4),
            "gap": round(mean_conf - mean_acc, 4),
        })

    return {
        "n": total,
        "n_bins": n_bins,
        "expected_calibration_error": round(ece, 4),
        "bins": rows,
        "interpretation":
            "ECE is the sample-weighted mean |confidence - accuracy| across "
            "bins. A positive gap means the model is over-confident. Report "
            "this next to any Table VII confidence figure: an uncalibrated "
            "confidence score is a decoder statistic, not a quality measure.",
    }


def aggregate(items):
    vals = [i["confidence"] for i in items if i["confidence"] is not None]
    if not vals:
        # Every generation was empty. The count still has to be reported:
        # a silently absent n_empty_generations reads as "no empty
        # generations" rather than "all of them".
        return {
            "n_items": len(items),
            "n_with_confidence": 0,
            "n_empty_generations": len(items),
            "mean_confidence": None,
        }
    vals_sorted = sorted(vals)
    mid = len(vals_sorted) // 2
    median = (
        vals_sorted[mid] if len(vals_sorted) % 2
        else (vals_sorted[mid - 1] + vals_sorted[mid]) / 2
    )
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {
        "n_items": len(items),
        "n_with_confidence": len(vals),
        "n_empty_generations": len(items) - len(vals),
        "mean_confidence": round(mean, 4),
        "median_confidence": round(median, 4),
        "stdev_confidence": round(math.sqrt(var), 4),
        "min_confidence": round(min(vals), 4),
        "max_confidence": round(max(vals), 4),
        "mean_tokens": round(sum(i["n_tokens"] for i in items) / len(items), 2),
    }


DEFINITION = {
    "confidence": "exp(mean_t log P(token_t | prefix)) over generated tokens "
                  "only; the per-token geometric-mean probability.",
    "range": "(0, 1]. Length-normalised, so it is comparable across "
             "transcriptions of different lengths.",
    "excluded": "Prompt tokens are excluded. Empty generations yield null "
                "rather than 0.0 or 1.0.",
    "caveat": "Self-reported and calibration-free. 4-bit NF4 quantisation "
              "perturbs the logits this is computed from. Read it with the "
              "expected_calibration_error in the same file.",
    "errata": "docs/ERRATA.md B11 flags Table VII's 0.68 -> 0.82 as an "
              "undefined quantity. This is the definition proposed to "
              "replace it; figures regenerated with this script are the ones "
              "to cite.",
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True,
                    help="JSONL with token_logprobs per record")
    ap.add_argument("--out", default="eval/confidence.json")
    ap.add_argument("--bins", type=int, default=DEFAULT_N_BINS)
    ap.add_argument("--per-item", action="store_true")
    args = ap.parse_args()

    with open(args.pred, encoding="utf-8") as fh:
        records = [json.loads(l) for l in fh if l.strip()]
    if not records:
        sys.exit("no records in --pred")
    if not any("token_logprobs" in r for r in records):
        sys.exit("no record carries 'token_logprobs'; see the module "
                 "docstring for how to collect them from generate()")

    items = [summarise_item(r) for r in records]

    payload = {
        "source": args.pred,
        "definition": DEFINITION,
        "overall": aggregate(items),
    }

    cal_pairs = [
        (summarise_item(r)["confidence"],
         _char_accuracy(r.get("reference", ""), r.get("hypothesis", "")))
        for r in records if r.get("reference")
    ]
    payload["calibration"] = calibration(cal_pairs, args.bins)

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

    print(json.dumps({k: v for k, v in payload.items() if k != "per_item"},
                     indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
