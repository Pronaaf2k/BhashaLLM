#!/usr/bin/env python3
"""
Aggregate human ratings and compute inter-rater reliability.

Sec. III-E specifies weighted Cohen's Kappa (2 raters) or Krippendorff's
alpha (3+), interpreted against Landis and Koch with kappa >= 0.60 as the
target. Sec. VI-D states the value had not been computed. This script
computes it, so Table V can carry a reliability figure.

Krippendorff's alpha is implemented here from the coincidence-matrix
definition rather than imported, so the repository has no dependency that
a reader has to install to check the headline number. It handles missing
ratings natively, which Cohen's kappa does not, and supports ordinal
weighting, which a 1-5 rubric requires -- a 1-vs-5 disagreement is worse
than a 3-vs-4 disagreement, and nominal alpha cannot express that.

Input CSV columns: rater_id,item_id,model_id,dimension,score

Reporting rule, worth stating up front: report whatever alpha comes out.
Sec. VI-D pre-commits to doing so, and following through on a
pre-registered reporting commitment is a credibility asset. Re-running
until it looks better is p-hacking under another name.

Usage:
    python eval/aggregate_human_eval.py --ratings human_eval/ratings.csv \
        --out human_eval/alpha_by_dimension.json
"""

import argparse
import json
import os
from collections import defaultdict
from itertools import product


# ------------------------------------------------- Krippendorff's alpha

def _delta2(levels, marginals, level_of_measurement):
    """Squared difference function over the observed value levels."""
    idx = {v: i for i, v in enumerate(levels)}
    d = {}
    for c, k in product(levels, repeat=2):
        if level_of_measurement == "nominal":
            d[(c, k)] = 0.0 if c == k else 1.0
        elif level_of_measurement == "interval":
            d[(c, k)] = float((c - k) ** 2)
        elif level_of_measurement == "ordinal":
            lo, hi = (idx[c], idx[k]) if idx[c] <= idx[k] else (idx[k], idx[c])
            s = sum(marginals[levels[g]] for g in range(lo, hi + 1))
            s -= (marginals[c] + marginals[k]) / 2.0
            d[(c, k)] = float(s ** 2)
        else:
            raise ValueError(level_of_measurement)
    return d


def krippendorff_alpha(units, level_of_measurement="ordinal"):
    """units: iterable of lists of observed values (one list per unit).

    Units rated fewer than twice carry no pairable information and are
    dropped, per the standard definition.
    """
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None, 0, 0

    levels = sorted({v for u in units for v in u})
    if len(levels) == 1:
        # every rater agreed on every unit; D_e is 0 and alpha is undefined
        return 1.0, len(units), sum(len(u) for u in units)

    # coincidence matrix
    o = defaultdict(float)
    for u in units:
        m = len(u)
        counts = defaultdict(int)
        for v in u:
            counts[v] += 1
        for c, k in product(levels, repeat=2):
            pairs = counts[c] * (counts[k] - (1 if c == k else 0))
            if pairs:
                o[(c, k)] += pairs / (m - 1)

    marginals = {c: sum(o[(c, k)] for k in levels) for c in levels}
    n = sum(marginals.values())
    if n <= 1:
        return None, len(units), int(n)

    d2 = _delta2(levels, marginals, level_of_measurement)

    do = sum(o[(c, k)] * d2[(c, k)] for c, k in product(levels, repeat=2))
    de = sum(marginals[c] * (marginals[k] - (1 if c == k else 0))
             * d2[(c, k)] for c, k in product(levels, repeat=2)) / (n - 1)

    if de == 0:
        return None, len(units), int(n)
    return 1.0 - do / de, len(units), int(n)


def landis_koch(a):
    if a is None:
        return "undefined"
    for bound, label in ((0.0, "poor"), (0.20, "slight"), (0.40, "fair"),
                         (0.60, "moderate"), (0.80, "substantial")):
        if a < bound:
            return label
    return "almost perfect" if a >= 0.80 else "substantial"


# ------------------------------------------------- aggregation

def mean_sd(xs):
    n = len(xs)
    if n == 0:
        return None, None
    mu = sum(xs) / n
    if n == 1:
        return round(mu, 3), None
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return round(mu, 3), round(var ** 0.5, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--level", default="ordinal",
                    choices=["nominal", "ordinal", "interval"])
    ap.add_argument("--out", default="human_eval/alpha_by_dimension.json")
    args = ap.parse_args()

    import csv
    with open(args.ratings, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["score"] = float(r["score"])

    raters = sorted({r["rater_id"] for r in rows})
    dims = sorted({r["dimension"] for r in rows})
    models = sorted({r["model_id"] for r in rows})

    # reliability per dimension: a "unit" is one (item, model) pair
    alpha = {}
    for dim in dims:
        units = defaultdict(list)
        for r in rows:
            if r["dimension"] == dim:
                units[(r["item_id"], r["model_id"])].append(r["score"])
        a, n_units, n_obs = krippendorff_alpha(list(units.values()), args.level)
        alpha[dim] = {
            "alpha": round(a, 4) if a is not None else None,
            "level_of_measurement": args.level,
            "n_units": n_units,
            "n_observations": n_obs,
            "landis_koch": landis_koch(a),
            "meets_0.60_target": (a is not None and a >= 0.60),
        }

    # scores per model per dimension, as means with SD and N
    scores = {}
    for model in models:
        scores[model] = {}
        totals = defaultdict(list)
        for dim in dims:
            xs = [r["score"] for r in rows
                  if r["model_id"] == model and r["dimension"] == dim]
            mu, sd = mean_sd(xs)
            scores[model][dim] = {"mean": mu, "sd": sd, "n_ratings": len(xs)}
            for r in rows:
                if r["model_id"] == model and r["dimension"] == dim:
                    totals[(r["item_id"], r["rater_id"])].append(r["score"])
        per_pass = [sum(v) for v in totals.values() if len(v) == len(dims)]
        mu, sd = mean_sd(per_pass)
        scores[model]["total"] = {"mean": mu, "sd": sd, "n": len(per_pass),
                                  "max_possible": 5 * len(dims)}

    payload = {
        "source": args.ratings,
        "n_raters": len(raters),
        "n_items": len({r["item_id"] for r in rows}),
        "n_models": len(models),
        "dimensions": dims,
        "reliability": alpha,
        "scores_by_model": scores,
        "note": "Table V should report the 'total' means with SD and n, not "
                "bare integers out of 15. Integers imply a single rating pass; "
                "means with dispersion imply a study.",
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print(f"raters={len(raters)}  items={payload['n_items']}  models={len(models)}")
    for dim, a in alpha.items():
        print(f"  {dim:22s} alpha={a['alpha']}  ({a['landis_koch']}, "
              f"n_units={a['n_units']})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
