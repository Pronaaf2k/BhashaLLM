#!/usr/bin/env python3
"""
Capture the out-of-memory traceback behind Table II's "attempted" row.

Table II records ``LoRA, r=16 | fp16 | 9.4M | >16GB (OOM) | attempted``.
``docs/TRACEABILITY.md`` names ``logs/oom_attempt.txt`` -- "the traceback
itself" -- as the evidence file for that row and marks it `MISSING`. An OOM
claim is one of the few claims whose evidence is a *failure*, and a failure
is only evidence if it was recorded.

This script reruns the configuration Table II says failed and writes
whatever happens to ``logs/oom_attempt.txt``: the traceback if it OOMs, or
a successful-load report if it does not. Both outcomes are useful. A load
that succeeds would mean Table II's row needs revising, and finding that
out is worth more than assuming the row is right.

``eval/memory_budget.py --table-ii`` computes that fp16 LoRA on this model
needs about 22 GB against the card's 16 GB, so the row is very likely
correct. This produces the artifact rather than the estimate.

Usage
-----
    python scripts/capture_oom_attempt.py
    python scripts/capture_oom_attempt.py --model meta-llama/Llama-3.2-11B-Vision-Instruct
    python scripts/capture_oom_attempt.py --dry-run     # no weights loaded
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DEFAULT_OUT = BASE_DIR / "logs" / "oom_attempt.txt"


def gpu_report():
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not importable"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "cuda not available"}
    p = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": p.name,
        "capability": f"sm_{p.major}{p.minor}",
        "total_memory_gb": round(p.total_memory / 1e9, 3),
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def attempt(model_id, rank, alpha, dropout, targets):
    """Load ``model_id`` in fp16 and attach a LoRA adapter. Report the outcome."""
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # fp16, unquantised -- deliberately the configuration Table II says
    # fails. Do NOT add quantization_config here; that would be QLoRA and
    # would not test this row.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map={"": 0},
        trust_remote_code=True,
    )
    peft_config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=list(targets),
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    peak = torch.cuda.max_memory_allocated() / 1e9

    return {
        "outcome": "LOADED",
        "trainable_params": trainable,
        "total_params": total,
        "peak_vram_gb": round(peak, 3),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rank", type=int, default=16)       # Table II / Table III
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", nargs="+",
                    default=["q_proj", "v_proj"])          # Table III
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--dry-run", action="store_true",
                    help="record the environment and the predicted budget "
                         "without loading any weights")
    args = ap.parse_args()

    header = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Evidence for Table II row: 'LoRA, r=16 | fp16 | >16GB "
                   "(OOM) | attempted'. See docs/TRACEABILITY.md.",
        "model": args.model,
        "configuration": {
            "quantisation": "none (fp16) -- deliberately NOT QLoRA",
            "lora_r": args.rank,
            "lora_alpha": args.alpha,
            "lora_dropout": args.dropout,
            "target_modules": args.target_modules,
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": gpu_report(),
    }

    # Predicted budget, so the artifact carries the expectation next to the
    # outcome regardless of which way the run goes.
    try:
        sys.path.insert(0, str(BASE_DIR / "eval"))
        from memory_budget import LLAMA_11B, budget
        header["predicted"] = budget(
            LLAMA_11B["params"], "lora", LLAMA_11B["layers"],
            LLAMA_11B["hidden"], rank=args.rank,
            n_target_modules=len(args.target_modules))
    except Exception as exc:  # noqa: BLE001
        header["predicted"] = {"error": str(exc)}

    body = ""
    if args.dry_run:
        header["outcome"] = "DRY_RUN"
        header["note"] = "No weights were loaded. Rerun without --dry-run on "\
                         "the 16 GB card to capture the real outcome."
    else:
        try:
            header.update(attempt(args.model, args.rank, args.alpha,
                                  args.dropout, args.target_modules))
            header["interpretation"] = (
                "The model LOADED in fp16. Table II's 'OOM / attempted' row "
                "does not reproduce under this configuration and should be "
                "revised, or the conditions that produced the OOM (different "
                "target modules, longer sequence, other resident processes) "
                "should be stated."
            )
        except Exception as exc:  # noqa: BLE001
            body = traceback.format_exc()
            is_oom = "out of memory" in str(exc).lower() or \
                     "CUDA out of memory" in body
            header["outcome"] = "OOM" if is_oom else "FAILED"
            header["exception_type"] = type(exc).__name__
            header["exception"] = str(exc)
            header["interpretation"] = (
                "Out of memory, as Table II records. The traceback below is "
                "the evidence for that row."
                if is_oom else
                "The attempt failed, but NOT with an out-of-memory error. "
                "This does not support Table II's OOM claim; read the "
                "traceback and rerun once the underlying problem is fixed."
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(header, indent=2, ensure_ascii=False, default=str)
    if body:
        text += "\n\n" + "=" * 70 + "\nTRACEBACK\n" + "=" * 70 + "\n" + body
    out.write_text(text + "\n", encoding="utf-8")

    print(text[:2000])
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
