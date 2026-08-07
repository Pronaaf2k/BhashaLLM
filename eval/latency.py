#!/usr/bin/env python3
"""
End-to-end latency and throughput at a *declared* token budget.

Paper Sec. V-C reports Llama-3.2-11B at 1450 ms / 69 tokens/s and
Qwen-1.5B at 680 ms / 147 tokens/s. ``docs/ERRATA.md`` B3 records why those
need a qualifier: both products give **100 tokens**, not the 256 that
Sec. IV-C fixes for decoding.

    69 tok/s  x 1.45 s  ~= 100
    147 tok/s x 0.68 s  ~= 100

At 69 tokens/s, 256 tokens would take about 3.7 seconds. The latency
figures are correct for a 100-token budget; the 256-token setting applies
to the quality evaluations, not the latency measurements. Reporting a
latency without its token budget is reporting nothing, so this script
refuses to write a result that does not carry one.

What is measured
----------------
``max_new_tokens`` is forced with ``min_new_tokens`` set equal to it, so
every timed generation emits exactly the budget. Without that, an early EOS
produces a fast generation and a latency number that describes the prompt
rather than the model. Warm-up iterations are discarded (first-call CUDA
graph capture and kernel autotuning are not representative), and the
reported figure is the median rather than the mean, which is robust to a
single scheduler hiccup.

Where it is measured
--------------------
``--mode local`` times ``model.generate`` directly. ``--mode api`` times a
POST to a running BhashaLLM endpoint, which is what paper Sec. IV-D
describes ("System latency was measured end-to-end at the API layer"). The
two differ by serialisation and HTTP overhead; the mode is recorded in the
output so they are never mixed.

Usage
-----
    # local, the two models compared in Sec. V-C
    python eval/latency.py --mode local \\
        --models Qwen/Qwen2.5-1.5B-Instruct meta-llama/Llama-3.2-11B-Vision \\
        --max-new-tokens 100 --load-4bit --out benchmarks/latency.json

    # at the API layer, against a running server
    python eval/latency.py --mode api --url http://localhost:5000/api/v1/generate \\
        --max-new-tokens 100 --out benchmarks/latency.json
"""

import argparse
import json
import os
import statistics
import time

# Paper Sec. IV-C fixed decoding, used so latency is measured under the same
# settings as the quality evaluations.
DECODING = {
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": True,
}

DEFAULT_PROMPT = (
    "বাংলা ভাষার ইতিহাস এবং সাহিত্যিক ঐতিহ্য সম্পর্কে একটি সংক্ষিপ্ত অনুচ্ছেদ লিখুন।"
)


def _percentiles(values):
    vals = sorted(values)
    def pct(p):
        if not vals:
            return None
        k = (len(vals) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)
    return {
        "median_ms": round(statistics.median(vals), 1) if vals else None,
        "mean_ms": round(statistics.fmean(vals), 1) if vals else None,
        "stdev_ms": round(statistics.stdev(vals), 1) if len(vals) > 1 else 0.0,
        "min_ms": round(min(vals), 1) if vals else None,
        "p90_ms": round(pct(0.90), 1) if vals else None,
        "max_ms": round(max(vals), 1) if vals else None,
    }


def _result(model_id, durations_ms, n_tokens, mode, extra=None):
    stats = _percentiles(durations_ms)
    median_s = (stats["median_ms"] or 0) / 1000.0
    return {
        "model": model_id,
        "mode": mode,
        # The field that ERRATA B3 exists because the paper omitted.
        "max_new_tokens": n_tokens,
        "tokens_per_second": (
            round(n_tokens / median_s, 1) if median_s else None
        ),
        "n_measured": len(durations_ms),
        **stats,
        **(extra or {}),
    }


def bench_local(model_id, prompt, n_tokens, n_iter, warmup, load_4bit, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"dtype": torch.float16, "device_map": device}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        # Table III quantisation, so latency reflects the deployed config.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, **kwargs).eval()

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    peak_gb = None

    def one():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                # Forces exactly the budget: an early EOS would otherwise
                # produce a latency that describes the prompt, not the model.
                min_new_tokens=n_tokens,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                **DECODING,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    for _ in range(warmup):
        one()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    durations = [one() for _ in range(n_iter)]
    if torch.cuda.is_available():
        peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return _result(model_id, durations, n_tokens, "local", {
        "quantisation": "nf4-4bit" if load_4bit else "fp16",
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "peak_vram_gb": peak_gb,
    })


def bench_api(url, prompt, n_tokens, n_iter, warmup, field):
    import urllib.request

    payload = json.dumps({
        "prompt": prompt, "max_new_tokens": n_tokens, **DECODING,
    }).encode("utf-8")

    def one():
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as resp:
            resp.read()
        return (time.perf_counter() - t0) * 1000.0

    for _ in range(warmup):
        one()
    durations = [one() for _ in range(n_iter)]
    return _result(url, durations, n_tokens, "api", {"response_field": field})


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["local", "api"], default="local")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--url", default="http://localhost:5000/api/v1/generate")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, required=True,
                    help="REQUIRED. Sec. V-C's figures correspond to 100; "
                         "Sec. IV-C's quality evaluations use 256. A latency "
                         "without its token budget is not interpretable "
                         "(docs/ERRATA.md B3).")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="benchmarks/latency.json")
    args = ap.parse_args()

    if args.mode == "local" and not args.models:
        ap.error("--mode local requires --models")

    results = []
    if args.mode == "local":
        for m in args.models:
            print(f"benchmarking {m} at {args.max_new_tokens} new tokens ...")
            results.append(bench_local(
                m, args.prompt, args.max_new_tokens, args.iterations,
                args.warmup, args.load_4bit, args.device))
    else:
        print(f"benchmarking {args.url} at {args.max_new_tokens} new tokens ...")
        results.append(bench_api(
            args.url, args.prompt, args.max_new_tokens, args.iterations,
            args.warmup, "text"))

    payload = {
        "token_budget": args.max_new_tokens,
        "decoding": DECODING,
        "iterations": args.iterations,
        "warmup_discarded": args.warmup,
        "prompt": args.prompt,
        "statistic": "median of n_measured timed generations",
        "note":
            "Paper Sec. V-C reports 1450 ms / 69 tok/s (Llama-3.2-11B) and "
            "680 ms / 147 tok/s (Qwen-1.5B). Both products give 100 tokens, "
            "not the 256 fixed in Sec. IV-C. See docs/ERRATA.md B3. Compare "
            "these results only against figures at the same token budget.",
        "results": results,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    for r in results:
        print(f"  {str(r['model'])[:45]:45s} "
              f"{r['median_ms']:>8.1f} ms  "
              f"{r['tokens_per_second']:>6.1f} tok/s  "
              f"@ {r['max_new_tokens']} tokens")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
