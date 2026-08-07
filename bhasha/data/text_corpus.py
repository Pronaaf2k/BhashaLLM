"""Bangla text-corpus preprocessing and the 80/10/10 by-document split.

Implements paper Sec. IV-B, which the repository described but did not
carry any code for:

    "Bangla text was normalised with NFKC, stray Latin characters were
    stripped, and the resulting corpus of roughly 6.6M tokens was split
    80/10/10 for training, validation, and testing."

and Sec. IV-C:

    "The 6.6M-token Bangla corpus was split 80/10/10 by document rather
    than by sentence, so no work by the same author appears on both
    sides."

Two things here are load-bearing and easy to get wrong.

**By document, not by sentence.** A sentence-level shuffle puts adjacent
sentences of the same Tagore poem in train and test. The resulting
validation loss measures memorisation of a specific text, not Bangla
competence. ``split_by_document`` shuffles whole documents and, when an
``author`` key is present, keeps every document by one author on a single
side of the split — which is the property Sec. IV-C actually claims.

**Token counts are tokenizer-dependent.** The paper's 6.6M figure has no
tokenizer attached. ``count_tokens`` therefore reports the tokenizer name
in its output and also reports a character count, which is
tokenizer-independent and is the figure to compare across models. See
``docs/ERRATA.md`` B6 for the same argument applied to perplexity.

Outputs are written as blank-line-separated documents, which is the format
``eval/compute_bpc.py`` already expects (it splits on ``"\\n\\n"``).

Usage
-----
    python -m bhasha.data.text_corpus \\
        --input data/raw/nazrul data/raw/tagore \\
        --out-dir data/splits \\
        --tokenizer Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Paper Sec. IV-C: "All runs used seed 42".
DEFAULT_SEED = 42

# Paper Sec. IV-B / Table IV: 80/10/10.
DEFAULT_RATIOS = (0.8, 0.1, 0.1)

BENGALI_BLOCK = (0x0980, 0x09FF)

# U+0964 DANDA and U+0965 DOUBLE DANDA sit in the Devanagari block but are
# shared Indic punctuation and the correct sentence terminators in Bangla.
# eval/script_integrity.py exempts them for the same reason; stripping them
# here would destroy sentence boundaries in the training corpus.
SHARED_INDIC_PUNCTUATION = {"।", "॥"}

# Latin letters only. Digits, Bangla digits, whitespace and punctuation are
# kept: paper Sec. IV-B says "stray Latin characters", not "everything
# non-Bangla". Removing punctuation as well would strip the danda and the
# quotation marks that carry dialogue structure in the literary corpus.
_LATIN_RE = re.compile(r"[A-Za-z]+")


def strip_latin(text: str) -> str:
    """Remove Latin alphabetic runs, collapsing the whitespace they leave.

    Applied after normalisation. Latin *digits* are retained deliberately:
    dates and verse numbers appear throughout the literary corpus and are
    not script confusion.
    """
    text = _LATIN_RE.sub(" ", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def normalise(text: str, form: str = "NFKC", remove_latin: bool = True) -> str:
    """NFKC-normalise and optionally strip Latin, per paper Sec. IV-B.

    NFKC is what the paper specifies for the corpus. Note that it is *not*
    what ``bhasha/data/dataset.py`` uses for OCR transcriptions, which need
    NFC — compatibility folding would rewrite presentation forms away from
    what was written on the page. The divergence is intentional; see that
    module's ``normalise_bangla``.
    """
    text = unicodedata.normalize(form, text or "")
    if remove_latin:
        text = strip_latin(text)
    # Normalise line endings, then collapse *every* run of blank lines to a
    # single newline. This is not cosmetic: write_split joins documents with
    # "\n\n" and eval/compute_bpc.py splits the file on "\n\n", so a blank
    # line inside a document would be read downstream as a document
    # boundary. A poem with stanza breaks would silently become five
    # documents, inflating the document count and breaking the by-document
    # split guarantee this module exists to provide.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def bangla_ratio(text: str) -> float:
    """Fraction of non-whitespace characters inside the Bengali block.

    Used to drop documents that survived normalisation but are mostly
    metadata, transliteration or markup. A corpus quality gate the paper
    does not describe but that any reader would expect to exist.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    n = sum(1 for c in chars if BENGALI_BLOCK[0] <= ord(c) <= BENGALI_BLOCK[1])
    return n / len(chars)


# --------------------------------------------------------------- documents


def load_documents(
    paths: Sequence[str | Path],
    min_chars: int = 200,
    min_bangla_ratio: float = 0.5,
    normalise_form: str = "NFKC",
    remove_latin: bool = True,
) -> List[Dict[str, object]]:
    """Read ``.txt`` files into normalised document records.

    Each input path may be a file or a directory (searched recursively for
    ``*.txt``). The immediate parent directory name becomes the ``author``
    key, which is what makes the author-disjoint split of Sec. IV-C
    possible without a separate metadata file.
    """
    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.txt")))
        elif p.exists():
            files.append(p)
        else:
            raise FileNotFoundError(p)

    docs: List[Dict[str, object]] = []
    for path in files:
        text = normalise(
            path.read_text(encoding="utf-8", errors="replace"),
            form=normalise_form,
            remove_latin=remove_latin,
        )
        if len(text) < min_chars:
            continue
        ratio = bangla_ratio(text)
        if ratio < min_bangla_ratio:
            continue
        # doc_id must be unique across the whole corpus, not just within one
        # author's directory. Two authors each having a `d0.txt` would
        # otherwise collide into a single group and be split as one unit,
        # silently halving the number of groups.
        docs.append({
            "doc_id": f"{path.parent.name}/{path.stem}",
            "author": path.parent.name,
            "path": str(path),
            "text": text,
            "chars": len(text),
            "bangla_ratio": round(ratio, 4),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    return docs


def split_by_document(
    docs: List[Dict[str, object]],
    ratios: Tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
    group_key: Optional[str] = "author",
) -> Dict[str, List[Dict[str, object]]]:
    """80/10/10 split at document (or author) granularity.

    ``group_key='author'`` implements the Sec. IV-C guarantee that "no work
    by the same author appears on both sides". With only two literary
    corpora (Nazrul and Tagore, refs [21]/[22]) an author-disjoint split is
    degenerate — two authors cannot be split three ways — so the function
    falls back to document-level grouping and *says so* in the returned
    metadata rather than silently producing an author-overlapping split
    while the caller believes otherwise.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    groups: Dict[object, List[Dict[str, object]]] = defaultdict(list)
    key = group_key or "doc_id"
    for d in docs:
        groups[d.get(key, d["doc_id"])].append(d)

    fallback_reason = None
    if group_key and len(groups) < 3:
        fallback_reason = (
            f"only {len(groups)} distinct {group_key!r} values; a "
            f"{group_key}-disjoint 3-way split is not possible. Fell back to "
            "document-level grouping. The Sec. IV-C claim that no work by "
            "the same author appears on both sides does NOT hold for this "
            "input."
        )
        groups = defaultdict(list)
        for d in docs:
            groups[d["doc_id"]].append(d)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    n = len(keys)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    bounds = {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }
    out = {name: [d for k in ks for d in groups[k]] for name, ks in bounds.items()}

    # An empty split is almost always a mistake at this scale, and it fails
    # silently downstream: eval/compute_bpc.py on an empty test.txt reports
    # a BPC over zero characters rather than erroring.
    empty = [name for name in ("train", "val", "test") if not out[name]]
    empty_warning = None
    if empty:
        empty_warning = (
            f"splits {empty} are empty: {n} groups cannot be divided "
            f"{ratios} without one side rounding to zero. Supply more "
            f"documents, or adjust --ratios."
        )

    out["_meta"] = [{  # type: ignore[assignment]
        "seed": seed,
        "ratios": list(ratios),
        "group_key": group_key,
        "n_groups": n,
        "grouping_fallback": fallback_reason,
        "empty_splits": empty_warning,
    }]
    return out


# ------------------------------------------------------------------ tokens


def count_tokens(texts: Iterable[str], tokenizer_id: Optional[str]) -> Dict[str, object]:
    """Token and character counts, with the tokenizer named.

    The paper's "roughly 6.6M tokens" (Sec. IV-B, Table IV) has no
    tokenizer attached, so it is not a checkable figure. This returns both
    the token count under a named tokenizer and the character count, which
    is tokenizer-independent. Report both.
    """
    texts = list(texts)
    chars = sum(len(t) for t in texts)
    payload: Dict[str, object] = {
        "characters": chars,
        "tokenizer": tokenizer_id,
        "note": "Token counts are tokenizer-dependent. The character count "
                "is not, and is the comparable figure. See docs/ERRATA.md B6.",
    }
    if not tokenizer_id:
        payload["tokens"] = None
        return payload
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_id)
        payload["tokens"] = sum(len(tok(t).input_ids) for t in texts)
        payload["chars_per_token"] = round(chars / max(payload["tokens"], 1), 3)
    except Exception as exc:  # noqa: BLE001
        payload["tokens"] = None
        payload["tokenizer_error"] = str(exc)
    return payload


def write_split(docs: List[Dict[str, object]], path: Path) -> None:
    """Write documents separated by blank lines.

    This is the format ``eval/compute_bpc.py`` reads (it splits the file on
    ``"\\n\\n"``), so the split files feed the base-model selection of
    Sec. III-B without a conversion step.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n\n".join(str(d["text"]) for d in docs) + "\n", encoding="utf-8"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", nargs="+", required=True,
                    help="text files or directories of .txt files")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--ratios", type=float, nargs=3, default=list(DEFAULT_RATIOS))
    ap.add_argument("--group-by", default="author",
                    help="'author' for the Sec. IV-C guarantee, 'doc_id' for "
                         "document-level only, or '' to disable grouping")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer id used to report the token count")
    ap.add_argument("--normalise-form", default="NFKC",
                    choices=["NFC", "NFD", "NFKC", "NFKD"])
    ap.add_argument("--keep-latin", action="store_true",
                    help="skip the Latin-stripping step of Sec. IV-B")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--min-bangla-ratio", type=float, default=0.5)
    args = ap.parse_args(argv)

    docs = load_documents(
        args.input,
        min_chars=args.min_chars,
        min_bangla_ratio=args.min_bangla_ratio,
        normalise_form=args.normalise_form,
        remove_latin=not args.keep_latin,
    )
    if not docs:
        raise SystemExit("no documents survived filtering; check --input")

    splits = split_by_document(
        docs, tuple(args.ratios), args.seed, args.group_by or None
    )
    meta = splits.pop("_meta")[0]

    out_dir = Path(args.out_dir)
    summary: Dict[str, object] = {
        "seed": args.seed,
        "ratios": args.ratios,
        "normalisation": args.normalise_form,
        "latin_stripped": not args.keep_latin,
        "n_documents": len(docs),
        "grouping": meta,
        "splits": {},
    }
    for name in ("train", "val", "test"):
        write_split(splits[name], out_dir / f"{name}.txt")
        summary["splits"][name] = {  # type: ignore[index]
            "n_documents": len(splits[name]),
            "document_ids": [d["doc_id"] for d in splits[name]],
            **count_tokens((str(d["text"]) for d in splits[name]), args.tokenizer),
        }

    manifest = out_dir / "split_manifest.json"
    manifest.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Checksums, so a reader can verify they hold the same corpus.
    # doc_id already carries the author prefix, so it is the full identifier.
    lines = [f"{d['sha256']}  {d['doc_id']}" for d in docs]
    (out_dir / "text_raw.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(json.dumps(
        {k: v for k, v in summary.items() if k != "splits"}, indent=2,
        ensure_ascii=False,
    ))
    for name in ("train", "val", "test"):
        s = summary["splits"][name]  # type: ignore[index]
        print(f"  {name:5s} {s['n_documents']:5d} docs  "
              f"{s['characters']:>10,d} chars  tokens={s['tokens']}")
    for key in ("grouping_fallback", "empty_splits"):
        if meta.get(key):
            print(f"\nWARNING: {meta[key]}")
    print(f"\nwrote {out_dir}/{{train,val,test}}.txt, {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
