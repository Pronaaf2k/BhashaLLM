#!/usr/bin/env python3
"""
Energy and CO2 per 1,000 inferences, recomputed from measured latency.

Paper Sec. VI-F: "At an estimated 250W sustained draw, 1,000 Llama-3.2-11B
inferences at roughly 600ms each consume approximately 42Wh, on the order
of 20g of CO2 under an average grid mix."

``docs/ERRATA.md`` B4 records two problems with that:

1. **The latency is one the paper contradicts.** The arithmetic is correct
   for 600 ms (250 W x 600 s = 41.7 Wh), but Sec. V-C reports 1450 ms for
   the same model -- a factor of 2.4. Recomputed at 1450 ms:
   250 W x 1450 s ~= **101 Wh** per 1,000 inferences.

2. **The grid carbon intensity factor is unstated.** 20 g CO2 for 42 Wh
   implies roughly 0.48 kg CO2/kWh, which is below published figures for
   the Bangladesh grid. Any restatement should cite the factor used.

This script recomputes both from a measured latency and *requires* the grid
factor to be named. It will not emit a CO2 figure with an anonymous
constant behind it, which is what produced the discrepancy in the first
place.

Grid intensity references (cite whichever you use)
--------------------------------------------------
The values below are provided as named starting points, not as
authoritative figures. Grid intensity varies by year, season and time of
day, and the correct move is to cite a source with an access date rather
than to inherit a number from a script. Pass ``--grid-factor`` with your
own value and ``--grid-source`` with the citation.

Usage
-----
    # from a measured latency
    python eval/energy.py --latency-ms 1450 --power-w 250 \\
        --grid-factor 0.68 --grid-source "Ember, Bangladesh 2024, accessed 2026-08" \\
        --out eval/energy.json

    # straight from benchmarks/latency.json
    python eval/energy.py --from-latency-json benchmarks/latency.json \\
        --grid-factor 0.68 --grid-source "..." --out eval/energy.json
"""

import argparse
import json
import os
import sys

# Sec. VI-F's stated sustained draw. Note this is an *estimate* in the paper,
# not a measurement; --power-w accepts a measured figure from nvidia-smi.
DEFAULT_POWER_W = 250.0

# The paper's own figures, kept so every output can show the delta.
PAPER = {
    "latency_ms": 600.0,
    "power_w": 250.0,
    "wh_per_1000": 42.0,
    "g_co2_per_1000": 20.0,
    "implied_grid_factor_kg_per_kwh": round(0.020 / 0.042, 3),  # ~0.476
    "sec_v_c_latency_ms": 1450.0,
}


def compute(latency_ms, power_w, grid_factor_kg_per_kwh, n_inferences=1000):
    """Energy and CO2 for ``n_inferences`` at the given latency and draw.

    Energy is ``power x time``; time is ``latency x n``. The only subtlety
    is unit bookkeeping, which is exactly where the paper's figure went
    wrong, so every intermediate is returned rather than folded away.
    """
    seconds = (latency_ms / 1000.0) * n_inferences
    joules = power_w * seconds
    wh = joules / 3600.0
    kwh = wh / 1000.0
    g_co2 = kwh * grid_factor_kg_per_kwh * 1000.0
    return {
        "n_inferences": n_inferences,
        "latency_ms": latency_ms,
        "power_w": power_w,
        "grid_factor_kg_co2_per_kwh": grid_factor_kg_per_kwh,
        "gpu_seconds": round(seconds, 1),
        "energy_wh": round(wh, 2),
        "energy_kwh": round(kwh, 5),
        "co2_g": round(g_co2, 1),
        "co2_g_per_inference": round(g_co2 / n_inferences, 4),
        "wh_per_inference": round(wh / n_inferences, 4),
    }


def latency_from_json(path, model_substr=None):
    """Pull a median latency out of a benchmarks/latency.json file."""
    payload = json.loads(open(path, encoding="utf-8").read())
    results = payload.get("results", [])
    if model_substr:
        results = [r for r in results
                   if model_substr.lower() in str(r.get("model", "")).lower()]
    if not results:
        sys.exit(f"no matching result in {path}")
    r = results[0]
    return r["median_ms"], {
        "source_file": path,
        "source_model": r.get("model"),
        "source_token_budget": r.get("max_new_tokens"),
        "source_mode": r.get("mode"),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--latency-ms", type=float,
                     help="measured per-inference latency")
    src.add_argument("--from-latency-json", metavar="PATH",
                     help="read the median latency from benchmarks/latency.json")
    ap.add_argument("--model", default=None,
                    help="substring selecting a result from the latency JSON")
    ap.add_argument("--power-w", type=float, default=DEFAULT_POWER_W,
                    help="sustained draw in watts. Sec. VI-F estimates 250; "
                         "measure with `nvidia-smi --query-gpu=power.draw` "
                         "under load for a real figure.")
    ap.add_argument("--grid-factor", type=float, required=True,
                    help="REQUIRED. kg CO2 per kWh. The paper's unstated "
                         "factor is what docs/ERRATA.md B4 objects to; this "
                         "script will not invent one.")
    ap.add_argument("--grid-source", required=True,
                    help="REQUIRED. Citation and access date for the grid "
                         "factor.")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="eval/energy.json")
    args = ap.parse_args()

    provenance = {}
    if args.from_latency_json:
        latency_ms, provenance = latency_from_json(
            args.from_latency_json, args.model)
    else:
        latency_ms = args.latency_ms

    measured = compute(latency_ms, args.power_w, args.grid_factor, args.n)

    # The paper's own numbers, recomputed under the same grid factor, so the
    # comparison is like-for-like rather than a mix of two constants.
    at_600 = compute(PAPER["latency_ms"], args.power_w, args.grid_factor, args.n)
    at_1450 = compute(PAPER["sec_v_c_latency_ms"], args.power_w,
                      args.grid_factor, args.n)

    payload = {
        "grid_factor_kg_co2_per_kwh": args.grid_factor,
        "grid_factor_source": args.grid_source,
        "power_source": (
            f"{args.power_w} W sustained (Sec. VI-F estimates 250 W)"
        ),
        "latency_provenance": provenance or {"latency_ms": latency_ms,
                                             "origin": "--latency-ms"},
        "measured": measured,
        "paper_as_printed": {
            **PAPER,
            "note":
                "Sec. VI-F asserts 42 Wh and 20 g CO2 per 1,000 inferences at "
                "~600 ms. The arithmetic is right for 600 ms, but Sec. V-C "
                "reports 1450 ms for the same model. The implied grid factor "
                f"of {PAPER['implied_grid_factor_kg_per_kwh']} kg/kWh is below "
                "published Bangladesh figures and is not cited in the paper. "
                "See docs/ERRATA.md B4.",
        },
        "recomputed_at_paper_latencies": {
            "600ms_as_used_in_sec_VI_F": at_600,
            "1450ms_as_reported_in_sec_V_C": at_1450,
            "ratio": round(
                at_1450["energy_wh"] / at_600["energy_wh"], 2
            ) if at_600["energy_wh"] else None,
        },
        "mitigations_claimed_sec_VI_F": [
            "route simpler tasks to Qwen-1.5B rather than defaulting to the "
            "largest available model",
            "batch requests to improve throughput per watt",
            "4-bit quantisation to reduce compute per inference",
        ],
        "caveat":
            "This accounts for GPU draw only. It excludes CPU, RAM, PSU "
            "conversion loss, cooling and any datacentre PUE multiplier, so "
            "it is a lower bound on system energy, not a full footprint.",
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    m = payload["measured"]
    print(f"  latency        {m['latency_ms']} ms @ {m['power_w']} W")
    print(f"  per {args.n:,} inf  {m['energy_wh']} Wh   {m['co2_g']} g CO2")
    print(f"  grid factor    {args.grid_factor} kg/kWh  ({args.grid_source})")
    print(f"\n  paper as printed: {PAPER['wh_per_1000']} Wh / "
          f"{PAPER['g_co2_per_1000']} g at {PAPER['latency_ms']} ms")
    print(f"  at Sec. V-C's 1450 ms: {at_1450['energy_wh']} Wh / "
          f"{at_1450['co2_g']} g")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
