#!/usr/bin/env python3
"""
ROUGE-1/2/L, BLEU and chrF++ for Bangla, with the tokenisation stated.

Why this file implements ROUGE rather than importing it:

  The reference implementation (google-research `rouge_score`) tokenises
  with `re.sub(r"[^a-z0-9]+", " ", text.lower())`. That expression
  removes every non-ASCII character, so it reduces any Bangla string to
  the empty string and returns 0.0 for every pair. Any ROUGE number
  computed on Bangla with default settings is therefore either zero or
  came from a different tokenizer -- and if it came from a different
  tokenizer, the paper has to say which one. This implementation uses an
  explicit Unicode-aware word tokenizer and prints it in the output, so
  the Table VI numbers carry their own definition.

  Note also that `rouge_score` is not in the repository's dependency
  file, so Table VI cannot currently be regenerated from a clean install
  of this project. This file closes that gap.

BLEU and chrF++ use sacrebleu when it is installed, and report the
sacrebleu signature verbatim, which is what makes a BLEU number
comparable to anyone else's. chrF++ is the more informative of the two
for a morphologically rich language, because it scores character
n-grams and is not punished by legitimate morphological variation the
way word-level BLEU is.

Input JSONL: {"item_id":..., "hypothesis": str, "reference": str}

Usage:
    python eval/text_metrics.py --pred benchmarks/raw/llama-3.2-11b.jsonl \
        --out eval/rouge_results.json
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter

TOKENIZER_NAME = "alnum_plus_combining_marks (NFC, casefold)"


def _is_word_char(ch):
    """A character continues a word if it is alphanumeric OR a combining mark.

    The mark clause is not optional for Bangla. Python's `\\w` matches only
    what `str.isalnum()` accepts, and `isalnum()` is False for Unicode
    categories Mn and Mc -- which is what Bangla vowel signs (matra) and the
    hasant are. A `[^\\W_]+` tokenizer therefore splits every word at its
    first matra: 'বাংলাদেশের' comes apart into 'ব', 'ল', 'দ', 'শ', 'র'.
    That inflates unigram overlap and destroys bigram overlap, so ROUGE
    computed that way is wrong in both directions and not comparable to
    anything.
    """
    return ch.isalnum() or unicodedata.category(ch) in ("Mn", "Mc")


def tokenize(text):
    text = unicodedata.normalize("NFC", text or "").casefold()
    tokens, cur = [], []
    for ch in text:
        if _is_word_char(ch):
            cur.append(ch)
        elif cur:
            tokens.append("".join(cur))
            cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _prf(match, n_hyp, n_ref):
    p = match / n_hyp if n_hyp else 0.0
    r = match / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "fmeasure": round(f, 4)}


def rouge_n(hyp, ref, n):
    h, r = _ngrams(hyp, n), _ngrams(ref, n)
    match = sum((h & r).values())
    return _prf(match, max(sum(h.values()), 0), max(sum(r.values()), 0))


def lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(hyp, ref):
    return _prf(lcs_len(hyp, ref), len(hyp), len(ref))


def corpus_rouge(pairs):
    """Average of per-item F-measures (ROUGE is conventionally averaged,
    not micro-pooled)."""
    acc = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
    for hyp, ref in pairs:
        h, r = tokenize(hyp), tokenize(ref)
        acc["rouge1"].append(rouge_n(h, r, 1)["fmeasure"])
        acc["rouge2"].append(rouge_n(h, r, 2)["fmeasure"])
        acc["rougeL"].append(rouge_l(h, r)["fmeasure"])
    return {k: round(sum(v) / len(v), 4) if v else None for k, v in acc.items()}


def corpus_bleu_chrf(pairs):
    try:
        import sacrebleu
    except ImportError:
        return {"error": "sacrebleu not installed -- pip install sacrebleu"}
    hyps = [h for h, _ in pairs]
    refs = [[r for _, r in pairs]]
    # tokenize='char' is the defensible choice for Bangla: the default '13a'
    # tokenizer is built for Latin script and its punctuation rules do not
    # apply. State whichever you use; the signature records it.
    bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize="char")
    chrf = sacrebleu.corpus_chrf(hyps, refs, word_order=2)
    return {
        "bleu": round(bleu.score, 2),
        "bleu_signature": str(bleu.get_signature()),
        "chrf++": round(chrf.score, 2),
        "chrf++_signature": str(chrf.get_signature()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="eval/text_metrics.json")
    ap.add_argument("--skip-bleu", action="store_true")
    args = ap.parse_args()

    with open(args.pred, encoding="utf-8") as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    pairs = [(r["hypothesis"], r["reference"]) for r in recs]

    payload = {
        "source": args.pred,
        "n_items": len(pairs),
        "tokenizer": TOKENIZER_NAME,
        "rouge": corpus_rouge(pairs),
        "rouge_note": "ROUGE computed with the Unicode word tokenizer named "
                      "above. The google-research rouge_score default "
                      "tokenizer strips non-ASCII and returns 0.0 on Bangla.",
    }
    if not args.skip_bleu:
        payload.update(corpus_bleu_chrf(pairs))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
