#!/usr/bin/env python3
"""
Bits-per-character for base-model selection.

Sec. III-B selects Qwen-2.5-1.5B over XGLM-1.7B on per-token perplexity
(3.8 vs 73.15). That comparison is not valid as stated: per-token
perplexity is normalised by token count, and the two models segment
Bangla into different numbers of tokens for the same string, so the
denominators differ and the ratio does not mean what it appears to.
A reviewer in this area raises this immediately.

Bits-per-character normalises the same total negative log-likelihood by
*character* count instead, which is identical across tokenizers and
therefore comparable. If Qwen still wins on BPC, the selection argument
in Sec. III-B survives and is now defensible.

Three things this script does that a naive loop gets wrong:

  1. Padding is masked. Averaging loss over pad tokens deflates it.
  2. Long documents are scored with a sliding window and only the
     newly-predicted positions are counted, so no character is scored
     twice and none is scored with truncated context.
  3. The first token of each window has no prediction target and is
     excluded, rather than silently counted as free.

Usage:
    python eval/compute_bpc.py \
        --models Qwen/Qwen2.5-1.5B-Instruct facebook/xglm-1.7b \
        --corpus data/splits/test.txt \
        --out eval/bpc_comparison.json
"""

import argparse
import json
import math
import os
import unicodedata


def read_corpus(path):
    with open(path, encoding="utf-8") as fh:
        docs = [unicodedata.normalize("NFC", d.strip())
                for d in fh.read().split("\n\n")]
    return [d for d in docs if d]


def bpc_for_model(model_id, docs, max_len, stride, device, load_4bit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"dtype": torch.float16, "device_map": device}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()

    total_nll = 0.0
    total_tokens = 0
    total_chars = 0

    for doc in docs:
        total_chars += len(doc)
        ids = tok(doc, return_tensors="pt").input_ids.to(model.device)
        n = ids.size(1)
        if n < 2:
            continue
        prev_end = 0
        for start in range(0, n, stride):
            end = min(start + max_len, n)
            window = ids[:, start:end]
            # only positions after prev_end are newly predicted
            n_new = end - prev_end
            target = window.clone()
            target[:, :-n_new] = -100
            with torch.no_grad():
                out = model(window, labels=target)
            # HF returns mean over non-ignored, already-shifted positions
            n_scored = int((target[:, 1:] != -100).sum())
            if n_scored:
                total_nll += out.loss.item() * n_scored
                total_tokens += n_scored
            prev_end = end
            if end == n:
                break

    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    return {
        "model": model_id,
        "bits_per_character": round(total_nll / total_chars / math.log(2), 4),
        "nats_per_token": round(total_nll / total_tokens, 4),
        "token_perplexity": round(math.exp(total_nll / total_tokens), 4),
        "chars_evaluated": total_chars,
        "tokens_evaluated": total_tokens,
        "chars_per_token": round(total_chars / total_tokens, 3),
        "note": "token_perplexity is reported for continuity with Sec. III-B "
                "only; it is NOT comparable across models with different "
                "tokenizers. Compare bits_per_character.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--corpus", required=True,
                    help="held-out text, documents separated by blank lines")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--out", default="eval/bpc_comparison.json")
    args = ap.parse_args()

    docs = read_corpus(args.corpus)
    print(f"{len(docs)} documents, "
          f"{sum(len(d) for d in docs)} characters")

    results = [bpc_for_model(m, docs, args.max_len, args.stride,
                             args.device, args.load_4bit)
               for m in args.models]
    for r in results:
        print(f"  {r['model']:45s} BPC {r['bits_per_character']:.4f}  "
              f"(ppl {r['token_perplexity']:.2f}, "
              f"{r['chars_per_token']:.2f} chars/token)")

    payload = {
        "corpus": args.corpus,
        "n_documents": len(docs),
        "max_len": args.max_len,
        "stride": args.stride,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
