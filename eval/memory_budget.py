#!/usr/bin/env python3
"""
Table II, term by term: what each fine-tuning method costs on one 16 GB card.

Paper Sec. III-C argues that most parameter-efficient methods are ruled out
before output quality enters the discussion, "and the term that rules them
out is the same in each case: the memory held by the frozen base weights."
Table II records the conclusion. The arithmetic behind it was never
committed, and ``docs/ERRATA.md`` B5 found an error in the sentence that
states it:

    Sec. III-C gives ~21 GB weights, ~21 GB gradients, and "roughly four
    times the weight size again for the AdamW moments and master copy",
    totalling "near 170 GB". But 21 + 21 + (4 x 21) = 128 GB, not 170.
    Reaching ~170 GB requires *six* times the fp16 weight size for
    optimiser state, which is what fp32 first moment + fp32 second moment +
    fp32 master weights actually costs. The stated total is right and the
    stated multiplier is wrong.

This script computes every term so the table stops being hand arithmetic.
Run it and the multiplier is whatever the terms say.

The accounting
--------------
For a model with ``P`` parameters:

================  ==========================================================
frozen weights    ``P x bytes_per_weight``. fp16 = 2, NF4 = 0.5 plus the
                  quantisation constants (double quantisation costs about
                  0.127 extra bits per parameter, which is included).
gradients         ``P_trainable x 2`` (fp16). Frozen parameters have none,
                  which is the whole point of LoRA -- and equally the whole
                  reason LoRA alone does not save you, because the frozen
                  weights are still resident.
AdamW state       ``P_trainable x 12``: fp32 exp_avg (4) + fp32
                  exp_avg_sq (4) + fp32 master copy (4).
activations       Estimated from sequence length, batch, layers and hidden
                  size. Gradient checkpointing replaces the per-layer term
                  with roughly ``sqrt(layers)`` stored checkpoints, which is
                  the fourth technique Sec. VI-E lists.
================  ==========================================================

What this does *not* do
-----------------------
It does not claim to predict peak VRAM to the megabyte. Allocator
fragmentation, kernel workspaces and the CUDA context add a gigabyte or so
that no formula captures. It answers the question Table II actually asks --
*does this method fit at all* -- and it makes the 170 GB figure checkable.
For a measured number, run a phase and read ``peak_vram_gb`` out of
``logs/phase*_summary.json``.

Usage
-----
    python eval/memory_budget.py --table-ii
    python eval/memory_budget.py --params 10.7e9 --method qlora --rank 16
    python eval/memory_budget.py --table-ii --out eval/memory_budget.json
"""

import argparse
import json
import os

GB = 1e9

# Bytes per stored parameter.
BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    # NF4 is 4 bits, plus quantisation constants. With double quantisation
    # (Table III: bnb_4bit_use_double_quant) the overhead is about 0.127
    # bits/param, per the QLoRA paper Sec. 3.
    "nf4": (4 + 0.127) / 8,
}

# AdamW: fp32 first moment + fp32 second moment + fp32 master weights.
ADAMW_BYTES_PER_TRAINABLE = 12.0
GRAD_BYTES_PER_TRAINABLE = 2.0  # fp16 gradients

# Llama-3.2-11B-Vision, the model Table II is written about.
LLAMA_11B = {
    "params": 10.7e9,
    "layers": 40,
    "hidden": 4096,
    "name": "Llama-3.2-11B (Table II subject)",
}


def lora_trainable(params, layers, hidden, rank, n_target_modules=2):
    """Trainable parameters for LoRA at a given rank.

    Each targeted projection contributes ``2 x rank x hidden``: an A matrix
    of shape (rank, hidden) and a B matrix of shape (hidden, rank).
    Table III targets q_proj and v_proj, hence the default of 2.
    """
    return layers * n_target_modules * 2 * rank * hidden


def activation_bytes(layers, hidden, seq_len, batch, checkpointing,
                     bytes_per_activation=2.0):
    """Rough activation memory.

    The per-layer constant of 16 covers the tensors a transformer block
    keeps for backward (attention projections, the MLP intermediate, the
    residual stream). It is an approximation and is labelled as one.

    Gradient checkpointing stores roughly ``sqrt(layers)`` checkpoints and
    recomputes the rest, which is why Sec. VI-E lists it as one of the four
    techniques that made the run fit.
    """
    per_layer = batch * seq_len * hidden * 16 * bytes_per_activation
    effective_layers = layers ** 0.5 if checkpointing else layers
    return per_layer * effective_layers


def budget(params, method, layers, hidden, rank=16, n_target_modules=2,
           seq_len=512, batch=1, checkpointing=True,
           adapter_bottleneck=64, prefix_vectors=10):
    """Memory budget for one method. Returns every term, not just the total."""
    method = method.lower()

    if method in ("full", "full_fine_tuning"):
        weight_dtype, trainable = "fp16", params
    elif method in ("lora", "lora_fp16"):
        weight_dtype = "fp16"
        trainable = lora_trainable(params, layers, hidden, rank, n_target_modules)
    elif method == "qlora":
        weight_dtype = "nf4"
        trainable = lora_trainable(params, layers, hidden, rank, n_target_modules)
    elif method == "adapters":
        # Houlsby-style bottleneck adapters: two per layer, each a
        # down-projection and an up-projection through the bottleneck.
        weight_dtype = "fp16"
        trainable = layers * 2 * (2 * hidden * adapter_bottleneck)
    elif method in ("prefix", "prefix_tuning"):
        # Trainable key and value prefixes at every layer.
        weight_dtype = "fp16"
        trainable = layers * 2 * prefix_vectors * hidden
    else:
        raise ValueError(f"unknown method {method!r}")

    frozen = params - (trainable if method in ("full", "full_fine_tuning") else 0)

    weights_b = params * BYTES[weight_dtype]
    grads_b = trainable * GRAD_BYTES_PER_TRAINABLE
    optim_b = trainable * ADAMW_BYTES_PER_TRAINABLE
    acts_b = activation_bytes(layers, hidden, seq_len, batch, checkpointing)
    total_b = weights_b + grads_b + optim_b + acts_b

    return {
        "method": method,
        "weight_storage_dtype": weight_dtype,
        "total_params": int(params),
        "trainable_params": int(trainable),
        "trainable_pct": round(100 * trainable / params, 4),
        "frozen_params": int(frozen),
        "terms_gb": {
            "frozen_weights": round(weights_b / GB, 2),
            "gradients": round(grads_b / GB, 3),
            "adamw_state": round(optim_b / GB, 3),
            "activations_est": round(acts_b / GB, 3),
        },
        "total_gb": round(total_b / GB, 2),
        "fits_in_16gb": total_b / GB <= 16.0,
        "gradient_checkpointing": checkpointing,
        "seq_len": seq_len,
        "batch": batch,
    }


def table_ii(seq_len=512, batch=1, checkpointing=True):
    """Reproduce Table II for Llama-3.2-11B on one 16 GB card."""
    p, layers, hidden = LLAMA_11B["params"], LLAMA_11B["layers"], LLAMA_11B["hidden"]
    rows = [
        ("Full fine-tuning", budget(p, "full", layers, hidden,
                                    seq_len=seq_len, batch=batch,
                                    checkpointing=checkpointing)),
        ("Adapters (bottleneck 64)", budget(p, "adapters", layers, hidden,
                                            seq_len=seq_len, batch=batch,
                                            checkpointing=checkpointing)),
        ("Prefix tuning (10 vec.)", budget(p, "prefix", layers, hidden,
                                           seq_len=seq_len, batch=batch,
                                           checkpointing=checkpointing)),
        ("LoRA, r=16 (fp16)", budget(p, "lora", layers, hidden, rank=16,
                                     seq_len=seq_len, batch=batch,
                                     checkpointing=checkpointing)),
        ("QLoRA, r=16 (NF4)", budget(p, "qlora", layers, hidden, rank=16,
                                     seq_len=seq_len, batch=batch,
                                     checkpointing=checkpointing)),
    ]

    full = rows[0][1]
    w = full["terms_gb"]["frozen_weights"]
    optim_multiplier = round(
        (full["terms_gb"]["adamw_state"] + full["terms_gb"]["gradients"]) / w, 2)

    return {
        "model": LLAMA_11B["name"],
        "card": "16 GB (RTX 5070 Ti, paper Sec. IV-A)",
        "rows": [{"label": label, **b} for label, b in rows],
        "errata_b5_check": {
            "paper_sentence":
                "~21 GB weights, ~21 GB gradients, and 'roughly four times "
                "the weight size again for the AdamW moments and master "
                "copy', totalling 'near 170 GB'.",
            "computed_weights_gb": w,
            "computed_gradients_gb": full["terms_gb"]["gradients"],
            "computed_adamw_gb": full["terms_gb"]["adamw_state"],
            "computed_total_gb": full["total_gb"],
            "stated_multiplier": 4,
            "computed_multiplier_over_fp16_weights": optim_multiplier,
            "verdict":
                "21 + 21 + (4 x 21) = 128 GB, not 170. The optimiser state "
                "alone is 6x the fp16 weight size (fp32 exp_avg + fp32 "
                "exp_avg_sq + fp32 master copy = 12 bytes per parameter "
                "against 2 bytes stored), which is what reaches ~170 GB. "
                "The paper's total is defensible; its multiplier is not. "
                "See docs/ERRATA.md B5.",
        },
        "table_ii_reference_values": {
            "qlora_trainable_params": 9_400_000,
            "qlora_peak_vram_gb": 9.4,
            "note":
                "Table II's 9.4 GB is an estimate from a feasibility "
                "analysis, not a measurement -- docs/ERRATA.md B1 records "
                "that no 11B training run exists. For a measured figure, "
                "read peak_vram_gb from logs/phase*_summary.json.",
        },
        "assumptions": {
            "adamw_bytes_per_trainable": ADAMW_BYTES_PER_TRAINABLE,
            "grad_bytes_per_trainable": GRAD_BYTES_PER_TRAINABLE,
            "nf4_bytes_per_param": round(BYTES["nf4"], 4),
            "nf4_note": "4 bits plus ~0.127 bits/param of quantisation "
                        "constants under double quantisation (Table III).",
            "activation_model": "batch x seq_len x hidden x 16 x 2 bytes per "
                                "layer; sqrt(layers) layers retained under "
                                "gradient checkpointing. Approximate.",
            "excluded": "allocator fragmentation, kernel workspaces, CUDA "
                        "context (~1 GB combined). This answers 'does it "
                        "fit', not 'what is peak VRAM to the megabyte'.",
        },
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table-ii", action="store_true",
                    help="reproduce Table II for Llama-3.2-11B")
    ap.add_argument("--params", type=float, default=None)
    ap.add_argument("--method", default="qlora",
                    choices=["full", "lora", "qlora", "adapters", "prefix"])
    ap.add_argument("--layers", type=int, default=LLAMA_11B["layers"])
    ap.add_argument("--hidden", type=int, default=LLAMA_11B["hidden"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--target-modules", type=int, default=2,
                    help="Table III targets q_proj and v_proj, so 2")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--no-checkpointing", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.table_ii or args.params is None:
        payload = table_ii(args.seq_len, args.batch, not args.no_checkpointing)
        print(f"{payload['model']} on {payload['card']}\n")
        header = (f"  {'Method':28s} {'Base':6s} {'Trainable':>12s} "
                  f"{'Weights':>9s} {'Grad':>7s} {'Adam':>8s} {'Act':>7s} "
                  f"{'Total':>9s}  Fits")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in payload["rows"]:
            t = r["terms_gb"]
            print(f"  {r['label']:28s} {r['weight_storage_dtype']:6s} "
                  f"{r['trainable_params']:>12,d} "
                  f"{t['frozen_weights']:>8.1f}G {t['gradients']:>6.2f}G "
                  f"{t['adamw_state']:>7.2f}G {t['activations_est']:>6.2f}G "
                  f"{r['total_gb']:>8.1f}G  "
                  f"{'yes' if r['fits_in_16gb'] else 'NO'}")
        e = payload["errata_b5_check"]
        print(f"\n  ERRATA B5: paper says the optimiser term is "
              f"{e['stated_multiplier']}x the weight size; computed "
              f"{e['computed_multiplier_over_fp16_weights']}x "
              f"(grad + AdamW over fp16 weights).")
        print(f"  Full fine-tuning total: {e['computed_total_gb']} GB "
              f"(paper: 'near 170 GB').")

        print("\n  Computed vs Table II as printed:")
        for label, stated in (("Adapters (bottle. 64)", 42_000_000),
                              ("Prefix tuning (10 vec.)", 3_300_000),
                              ("LoRA / QLoRA r=16", 9_400_000)):
            key = ("adapters" if "Adapter" in label
                   else "prefix" if "Prefix" in label else "qlora")
            got = next(r["trainable_params"] for r in payload["rows"]
                       if r["method"] == key)
            delta = 100 * (got - stated) / stated
            print(f"    {label:24s} stated {stated:>12,d}   "
                  f"computed {got:>12,d}   ({delta:+.1f}%)")
        print("  The two adapter rows reproduce Table II almost exactly. The "
              "LoRA row is ~11% above the printed 9.4M, which is what a "
              "36-layer rather than 40-layer count would give -- worth "
              "checking which layer count Table II assumed.")
    else:
        payload = budget(args.params, args.method, args.layers, args.hidden,
                         args.rank, args.target_modules, args.seq_len,
                         args.batch, not args.no_checkpointing)
        print(json.dumps(payload, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
