#!/usr/bin/env python3
"""
Tokenisation and morphology sanity checks for Bangla (paper Sec. IV-B).

    "Tokenisation and morphology behaviour were sanity-checked against a
    Bangla spelling-checker resource [24], an 80k-entry word-frequency list
    [25], a morphological-analysis gold dataset [26], and a Bengali
    sentence collection drawn from OSCAR [27]."

No such check existed in the repository. This file is it, and it also
supplies the evidence for a claim the paper makes but never measures: that
subword tokenisation of Bangla "is inefficient enough that context is the
scarce resource" (Sec. III-C), and the tokeniser-geometry hypothesis for
script confusion in Sec. VI-A.

The five checks
---------------

**1. Fertility.** Subword tokens per whitespace word, and characters per
token. Sec. III-C's argument for rejecting prefix tuning rests entirely on
Bangla tokenisation being inefficient; this measures how inefficient. A
fertility above ~3 means a 512-token budget holds fewer than 170 words,
which is the concrete form of "context is the scarce resource".

**2. Conjunct integrity.** Whether the tokeniser splits a conjunct at the
hasant (U+09CD). This is the Roy et al. problem the paper cites [5]:
"Bangla conjuncts are not always single Unicode codepoints, and naively
splitting them during tokenisation trains a model on fragments that no
longer match the rendered glyph." A tokeniser that ends a token on a
hasant has produced exactly such a fragment.

**3. Matra integrity.** Whether a token boundary falls between a consonant
and its vowel sign. Same failure, different mark class, and the one that
also breaks naive `\\w`-based metric tokenizers -- see
``eval/text_metrics.py``.

**4. Round-trip fidelity.** Whether decode(encode(x)) == NFC(x). A
tokeniser that loses or reorders combining marks corrupts training targets
silently, and Bangla is unusually good at exposing this.

**5. Cross-script adjacency (Sec. VI-A).** The paper hypothesises that
"shared multilingual tokenisers often place phonetically similar Bangla and
Devanagari sequences at adjacent token boundaries; if the two scripts'
tokens sit close together in embedding space, drifting from one to the
other during generation may be an easier error than a clean single-script
mistake." Sec. VI-A is explicit that no tokenizer-level ablation was run.
This check does the cheap half: it counts Devanagari tokens in the
vocabulary and reports how many Bangla tokens have a Devanagari
counterpart that is adjacent *in token-id space*. Adjacent ids are not
adjacent embeddings, and this script says so rather than overclaiming --
but a vocabulary where the two scripts interleave by id is at least
consistent with the hypothesis, and one where they occupy disjoint ranges
is evidence against it.

Reference resources (paper refs [24]-[27])
------------------------------------------
Pass any of them and the corresponding check runs over real vocabulary
instead of the built-in probe list:

``--wordlist``     80k word-frequency list [25], or the spelling checker [24]
``--morphology``   morphological-analysis gold dataset [26]
``--corpus``       OSCAR Bengali sentences [27], or data/splits/test.txt

Usage
-----
    python eval/tokenizer_sanity.py --models Qwen/Qwen2.5-1.5B-Instruct \\
        facebook/xglm-1.7b --out eval/tokenizer_sanity.json

    python eval/tokenizer_sanity.py --models Qwen/Qwen2.5-1.5B-Instruct \\
        --wordlist data/raw/bangla_word_frequency_80k.txt \\
        --corpus data/splits/test.txt
"""

import argparse
import json
import os
import sys
import unicodedata
from collections import Counter

HASANT = "্"
BENGALI = (0x0980, 0x09FF)
DEVANAGARI = (0x0900, 0x097F)
SHARED_INDIC_PUNCTUATION = {0x0964, 0x0965}

# Built-in probe set: conjuncts, matra-bearing words and common vocabulary.
# Used when no external wordlist is supplied so the script is runnable
# without any of refs [24]-[27] on disk.
PROBE_WORDS = [
    "বাংলাদেশ", "বাংলাদেশের", "রবীন্দ্রনাথ", "ঠাকুর", "নজরুল",
    "বিশ্ববিদ্যালয়", "কৃষ্ণচূড়া", "সংস্কৃতি", "স্বাধীনতা", "উচ্চারণ",
    "বিদ্যুৎ", "সংখ্যা", "ক্ষুধা", "জ্ঞান", "উজ্জ্বল", "দ্বন্দ্ব",
    "প্রত্যেক", "শুভেচ্ছা", "অন্তর্ভুক্ত", "পরিপ্রেক্ষিত",
]

PROBE_SENTENCES = [
    "বাংলাদেশের রাজধানী ঢাকা।",
    "রবীন্দ্রনাথ ঠাকুর একজন বিশ্ববিখ্যাত কবি ছিলেন।",
    "আমি প্রতিদিন সকালে বই পড়তে ভালোবাসি।",
]


def is_bengali(ch):
    return BENGALI[0] <= ord(ch) <= BENGALI[1]


def is_devanagari(ch):
    cp = ord(ch)
    return DEVANAGARI[0] <= cp <= DEVANAGARI[1] and cp not in SHARED_INDIC_PUNCTUATION


def _script_of(text):
    """Dominant script of a token string."""
    b = sum(1 for c in text if is_bengali(c))
    d = sum(1 for c in text if is_devanagari(c))
    if b and not d:
        return "bengali"
    if d and not b:
        return "devanagari"
    if b and d:
        return "mixed"
    return "other"


# ------------------------------------------------------------------ checks


def check_fertility(tok, words, sentences):
    """Tokens per word and characters per token (paper Sec. III-C)."""
    n_tokens = sum(len(tok.encode(w, add_special_tokens=False)) for w in words)
    n_chars = sum(len(w) for w in words)

    sent_tokens = sum(len(tok.encode(s, add_special_tokens=False))
                      for s in sentences)
    sent_words = sum(len(s.split()) for s in sentences)

    fertility = n_tokens / len(words) if words else None
    return {
        "n_probe_words": len(words),
        "tokens_per_word": round(fertility, 3) if fertility else None,
        "chars_per_token": round(n_chars / n_tokens, 3) if n_tokens else None,
        "tokens_per_word_in_sentences": (
            round(sent_tokens / sent_words, 3) if sent_words else None),
        "bangla_words_in_512_tokens": (
            int(512 / fertility) if fertility else None),
        "interpretation":
            "Sec. III-C rejects prefix tuning because 'the subword tokeniser "
            "is inefficient enough that context is the scarce resource'. "
            "bangla_words_in_512_tokens is that claim as a number: it is how "
            "much Bangla fits in the Table III sequence budget.",
    }


def check_conjunct_integrity(tok, words):
    """Do token boundaries fall on the hasant, fragmenting conjuncts? [5]"""
    broken, examined = [], 0
    for w in words:
        if HASANT not in w:
            continue
        examined += 1
        ids = tok.encode(w, add_special_tokens=False)
        pieces = [tok.decode([i]) for i in ids]
        # A conjunct is fragmented if a piece ends with the hasant: the
        # dependent consonant then begins a different token, so the model
        # never sees the conjunct as a unit.
        if any(p.endswith(HASANT) for p in pieces):
            broken.append({"word": w, "pieces": pieces})
    return {
        "words_with_conjuncts": examined,
        "conjuncts_split_at_hasant": len(broken),
        "rate": round(len(broken) / examined, 4) if examined else None,
        "examples": broken[:8],
        "interpretation":
            "Paper ref [5] (Roy et al.): naively splitting conjuncts during "
            "tokenisation 'trains a model on fragments that no longer match "
            "the rendered glyph'. A token ending in U+09CD is such a "
            "fragment.",
    }


def check_matra_integrity(tok, words):
    """Do token boundaries separate a consonant from its vowel sign?"""
    broken, examined = [], 0
    for w in words:
        marks = [c for c in w if unicodedata.category(c) in ("Mn", "Mc")]
        if not marks:
            continue
        examined += 1
        ids = tok.encode(w, add_special_tokens=False)
        pieces = [tok.decode([i]) for i in ids]
        # A piece that *begins* with a combining mark means the boundary
        # fell between the base consonant and its matra.
        if any(p and unicodedata.category(p[0]) in ("Mn", "Mc") for p in pieces):
            broken.append({"word": w, "pieces": pieces})
    return {
        "words_with_marks": examined,
        "words_split_before_a_mark": len(broken),
        "rate": round(len(broken) / examined, 4) if examined else None,
        "examples": broken[:8],
        "interpretation":
            "A token starting with a combining mark means the boundary fell "
            "between a consonant and its matra. The same class of error "
            "breaks naive \\\\w-based metric tokenizers -- see "
            "eval/text_metrics.py.",
    }


def check_roundtrip(tok, words, sentences):
    """decode(encode(x)) == NFC(x)?  Silent corruption is the failure mode."""
    failures = []
    for s in list(words) + list(sentences):
        target = unicodedata.normalize("NFC", s)
        got = unicodedata.normalize(
            "NFC", tok.decode(tok.encode(s, add_special_tokens=False)))
        if got != target:
            failures.append({"input": s, "output": got})
    total = len(words) + len(sentences)
    return {
        "n_tested": total,
        "n_failed": len(failures),
        "rate": round(len(failures) / total, 4) if total else None,
        "examples": failures[:8],
        "interpretation":
            "A tokeniser that loses or reorders combining marks corrupts "
            "training targets silently. Any non-zero rate here undermines "
            "every downstream metric.",
    }


def check_script_adjacency(tok):
    """Vocabulary-level evidence for the Sec. VI-A hypothesis."""
    try:
        vocab = tok.get_vocab()
    except Exception:  # noqa: BLE001
        return {"error": "tokenizer exposes no vocabulary"}

    by_id = {}
    for token, idx in vocab.items():
        try:
            surface = tok.convert_tokens_to_string([token])
        except Exception:  # noqa: BLE001
            surface = token
        script = _script_of(surface)
        if script in ("bengali", "devanagari", "mixed"):
            by_id[idx] = script

    counts = Counter(by_id.values())
    bengali_ids = sorted(i for i, s in by_id.items() if s == "bengali")
    devanagari_ids = sorted(i for i, s in by_id.items() if s == "devanagari")
    dev_set = set(devanagari_ids)

    # How many Bangla tokens sit immediately beside a Devanagari token in id
    # space? Interleaving is consistent with the hypothesis; disjoint ranges
    # are evidence against it.
    adjacent = sum(1 for i in bengali_ids
                   if (i - 1) in dev_set or (i + 1) in dev_set)

    def span(ids):
        return [min(ids), max(ids)] if ids else None

    return {
        "vocab_size": len(vocab),
        "bengali_tokens": counts.get("bengali", 0),
        "devanagari_tokens": counts.get("devanagari", 0),
        "mixed_script_tokens": counts.get("mixed", 0),
        "bengali_id_range": span(bengali_ids),
        "devanagari_id_range": span(devanagari_ids),
        "bengali_tokens_adjacent_to_devanagari_id": adjacent,
        "adjacency_rate": (
            round(adjacent / len(bengali_ids), 4) if bengali_ids else None),
        "ranges_overlap": bool(
            bengali_ids and devanagari_ids
            and max(min(bengali_ids), min(devanagari_ids))
            <= min(max(bengali_ids), max(devanagari_ids))),
        "interpretation":
            "Sec. VI-A hypothesises that Bangla and Devanagari sit at "
            "adjacent token boundaries and close together in embedding "
            "space. This measures ADJACENCY IN TOKEN-ID SPACE ONLY. Token-id "
            "adjacency is not embedding proximity, and this is not the "
            "tokenizer ablation Sec. VI-A says was not run. Interleaved "
            "ranges are consistent with the hypothesis; disjoint ranges are "
            "evidence against it. Testing it properly needs embedding "
            "distances, or the byte-level comparison of Sec. VI-C.",
    }


# ------------------------------------------------------------------ driver


def load_words(path, limit):
    """Read a wordlist. Tolerates plain, CSV and 'word<TAB>count' formats."""
    words = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            token = line.replace("\t", ",").split(",")[0].strip().strip('"')
            if token and any(is_bengali(c) for c in token):
                words.append(token)
            if len(words) >= limit:
                break
    return words


def load_sentences(path, limit):
    text = open(path, encoding="utf-8").read()
    sents = [s.strip() + "।" for s in text.split("।") if s.strip()]
    return sents[:limit]


def analyse(model_id, words, sentences):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return {
        "model": model_id,
        "tokenizer_class": type(tok).__name__,
        "fertility": check_fertility(tok, words, sentences),
        "conjunct_integrity": check_conjunct_integrity(tok, words),
        "matra_integrity": check_matra_integrity(tok, words),
        "roundtrip": check_roundtrip(tok, words, sentences),
        "script_adjacency": check_script_adjacency(tok),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+",
                    default=["Qwen/Qwen2.5-1.5B-Instruct"])
    ap.add_argument("--wordlist", default=None,
                    help="paper refs [24]/[25]: spelling checker or the "
                         "80k word-frequency list")
    ap.add_argument("--morphology", default=None,
                    help="paper ref [26]: morphological-analysis gold data")
    ap.add_argument("--corpus", default=None,
                    help="paper ref [27]: OSCAR Bengali sentences, or "
                         "data/splits/test.txt")
    ap.add_argument("--limit-words", type=int, default=5000)
    ap.add_argument("--limit-sentences", type=int, default=200)
    ap.add_argument("--out", default="eval/tokenizer_sanity.json")
    args = ap.parse_args()

    words, sources = list(PROBE_WORDS), {"probe_words": len(PROBE_WORDS)}
    for label, path in (("wordlist_ref_24_25", args.wordlist),
                        ("morphology_ref_26", args.morphology)):
        if path:
            loaded = load_words(path, args.limit_words)
            words += loaded
            sources[label] = {"path": path, "words": len(loaded)}
        else:
            sources[label] = None

    sentences = list(PROBE_SENTENCES)
    if args.corpus:
        loaded = load_sentences(args.corpus, args.limit_sentences)
        sentences += loaded
        sources["corpus_ref_27"] = {"path": args.corpus, "sentences": len(loaded)}
    else:
        sources["corpus_ref_27"] = None

    words = list(dict.fromkeys(words))  # de-duplicate, preserve order

    if all(v is None for k, v in sources.items() if k != "probe_words"):
        print("NOTE: no external resources supplied, so these checks run over "
              f"the {len(PROBE_WORDS)} built-in probe words only. Paper "
              "Sec. IV-B claims checks against refs [24]-[27]; pass "
              "--wordlist / --morphology / --corpus to reproduce that.\n",
              file=sys.stderr)

    results = []
    for m in args.models:
        print(f"analysing {m} ...")
        try:
            results.append(analyse(m, words, sentences))
        except Exception as exc:  # noqa: BLE001
            results.append({"model": m, "error": f"{type(exc).__name__}: {exc}"})

    payload = {
        "paper_section": "IV-B (sanity checks), III-C (fertility), VI-A (script adjacency)",
        "resources": sources,
        "n_words": len(words),
        "n_sentences": len(sentences),
        "results": results,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    for r in results:
        if "error" in r:
            print(f"  {r['model']:45s} ERROR {r['error']}")
            continue
        f = r["fertility"]
        c = r["conjunct_integrity"]
        a = r["script_adjacency"]
        print(f"\n  {r['model']}")
        print(f"    fertility          {f['tokens_per_word']} tokens/word, "
              f"{f['chars_per_token']} chars/token")
        print(f"    512-token budget   ~{f['bangla_words_in_512_tokens']} "
              f"Bangla words")
        print(f"    conjuncts split    {c['conjuncts_split_at_hasant']}/"
              f"{c['words_with_conjuncts']} (rate {c['rate']})")
        print(f"    matra split        "
              f"{r['matra_integrity']['words_split_before_a_mark']}/"
              f"{r['matra_integrity']['words_with_marks']}")
        print(f"    roundtrip failures {r['roundtrip']['n_failed']}/"
              f"{r['roundtrip']['n_tested']}")
        if "error" not in a:
            print(f"    vocab scripts      {a['bengali_tokens']} bengali, "
                  f"{a['devanagari_tokens']} devanagari, ranges overlap="
                  f"{a['ranges_overlap']}, adjacency={a['adjacency_rate']}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
