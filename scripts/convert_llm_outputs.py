#!/usr/bin/env python3
"""
Convert `llm outputs/*.md` into the JSONL every script in `eval/` reads.

`APPLY_THIS_PATCH.md` step 0 calls this "the cheapest work in the whole
plan": before regenerating anything, find out what evidence you already
hold. The generations behind Tables V, VI and part of Sec. V-B **are** in
this repository — they are in `llm outputs/`, one Markdown file per model.
But every evaluation script reads JSONL from `benchmarks/raw/`, so nothing
could consume them and `docs/TRACEABILITY.md` marked the rows `CHECK`.

This script closes that gap. It parses the Markdown the Ollama benchmark
runner emitted and writes `benchmarks/raw/<model>.jsonl` in the schema
documented in `benchmarks/README.md`, so the existing evidence flows
straight into `eval/script_integrity.py`, `eval/text_metrics.py` and
`eval/ocr_correction.py` without anyone re-running a model.

What it also records
--------------------
Four things about that benchmark are not what the paper describes, and the
converter writes them into every record rather than leaving them to be
rediscovered:

* **The backend was Ollama, serving GGUF at Q4_K_M** — not the NF4
  BitsAndBytes stack of Sec. III-C/IV-A (`docs/ERRATA.md` C11).
* **Temperature was 0.3**, not the 0.7 fixed in Sec. IV-C, and no explicit
  repetition penalty was set (C13).
* **N is 1 per task.** The runner defines four prompts, one per category.
  Table VI's ROUGE figures are computed on a single summary per model (C12).
* **`llama3.2:latest` is the 3B model**, so the file named
  `Llama_3.2_11B.md` was very probably not produced by an 11B model
  (`docs/ERRATA.md` A00).

References
----------
The OCR-Fix prompt embeds its own ground truth as
``(Expected: ...)``, so those records get a `reference` and a `noisy`
string and can be scored directly by `eval/ocr_correction.py`. The
translation and summarisation prompts carry no reference, so those records
are emitted without one and `eval/text_metrics.py` will need an item set
with references before it can score them — which is the `benchmarks/items/`
work `docs/TRACEABILITY.md` still lists as `MISSING`.

Usage
-----
    python scripts/convert_llm_outputs.py
    python scripts/convert_llm_outputs.py --out-dir benchmarks/raw
    python scripts/convert_llm_outputs.py --print-summary

    # then, with no model re-run needed:
    python eval/script_integrity.py "benchmarks/raw/*.jsonl" \\
        --out eval/script_integrity.json
    python eval/ocr_correction.py --pred benchmarks/raw/ocr_correction.jsonl \\
        --out eval/ocr_correction.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SRC = BASE_DIR / "llm outputs"


def _rel(path: Path) -> str:
    """Repo-relative path where possible, absolute otherwise.

    `--out-dir` may point outside the repository (a scratch directory
    during testing), and `Path.relative_to` raises rather than falling
    back. A reporting helper must never be the thing that fails a run.
    """
    try:
        return str(Path(path).resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path)
DEFAULT_OUT = BASE_DIR / "benchmarks" / "raw"

# The four categories run by bhasha/llm/run_benchmark_suite.py, mapped onto
# the paper's task names.
TASK_OF_CATEGORY = {
    "Translation": "translation",
    "Summarization": "summarisation",
    "OCR Fix": "ocr_correction",
    "Creative": "creative_writing",
}

# Provenance stamped on every record. These are facts about how the
# generations were produced, and they are what makes the numbers readable
# six months from now.
PROVENANCE = {
    "backend": "ollama",
    "quantisation": "Q4_K_M (GGUF, Ollama default)",
    "quantisation_source": "docs/ERRATA.md C11",
    "temperature": 0.3,
    "top_p": 0.9,
    "repetition_penalty": None,
    "max_new_tokens": 256,
    "decoding_note":
        "Paper Sec. IV-C specifies temperature 0.7 and repetition penalty "
        "1.1. The runner used 0.3 and set no repetition penalty. See "
        "docs/ERRATA.md C13.",
    "n_items_per_task": 1,
    "n_note":
        "The runner defines one prompt per category, so N=1 per task per "
        "model. See docs/ERRATA.md C12.",
}

# Ollama tag -> what it actually resolves to. The first entry is the one
# that matters; see docs/ERRATA.md A00.
OLLAMA_TAGS = {
    "Llama_3.2_11B": {
        "tag": "llama3.2:latest",
        "resolves_to": "llama3.2:3b (3.21B parameters, Q4_K_M, ~2.0 GB)",
        "paper_label": "Llama-3.2-11B",
        "mismatch": True,
        "note": "Ollama's llama3.2 tag defaults to the 3B TEXT model. The "
                "11B model is Llama-3.2-11B-Vision, served as "
                "llama3.2-vision:11b. This file was very probably NOT "
                "produced by an 11B model. See docs/ERRATA.md A00.",
    },
    "Llama_3.1_8B": {"tag": "llama3.1:latest", "resolves_to": "llama3.1:8b", "mismatch": False},
    "Mistral_7B": {"tag": "mistral:latest", "resolves_to": "mistral:7b (v0.3)", "mismatch": False},
    "Nemo_12B": {"tag": "mistral-nemo:latest", "resolves_to": "mistral-nemo:12b", "mismatch": False},
    "Gemma_9B": {"tag": "gemma2:9b", "resolves_to": "gemma2:9b", "mismatch": False},
    "Gemma_2B": {"tag": "gemma2:2b", "resolves_to": "gemma2:2b", "mismatch": False},
    "Qwen_1.5B": {"tag": "qwen2.5:1.5b", "resolves_to": "qwen2.5:1.5b", "mismatch": False},
    "Qwen_3B": {"tag": "qwen2.5:3b", "resolves_to": "qwen2.5:3b", "mismatch": False},
    "bn_rag_8B": {
        "tag": "hf.co/BanglaLLM/bangla-llama-13b-base-v0.1-GGUF",
        "resolves_to": "bangla-llama-13b-base-v0.1 (GGUF)",
        "mismatch": True,
        "note": "A retrieval/Bangla-specific model that is NOT in Table I, "
                "and retrieval is the approach Sec. III-D says was not "
                "used. See docs/ERRATA.md C5 and C14.",
    },
}

SECTION_RE = re.compile(r"^###\s+(.+?)\s*$", re.M)
RESPONSE_RE = re.compile(r"^\*\*Response:\*\*\s*(?:\(([\d.]+)s\))?\s*$", re.M)
EXPECTED_RE = re.compile(r"\(Expected:\s*(.+?)\)\s*$", re.S)


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", (s or "").strip())


def _strip_quote(block: str) -> str:
    """Un-indent a Markdown blockquote."""
    lines = [re.sub(r"^>\s?", "", ln) for ln in block.splitlines()]
    return "\n".join(lines).strip()


def parse_markdown(text: str) -> List[Dict[str, object]]:
    """Pull (category, prompt, response, seconds) out of one output file."""
    sections = list(SECTION_RE.finditer(text))
    records = []
    for i, m in enumerate(sections):
        category = m.group(1).strip()
        if category not in TASK_OF_CATEGORY:
            continue
        body = text[m.end():(sections[i + 1].start() if i + 1 < len(sections)
                             else len(text))]

        rm = RESPONSE_RE.search(body)
        if not rm:
            continue
        seconds = float(rm.group(1)) if rm.group(1) else None

        prompt_block = body[:rm.start()]
        prompt = ""
        pm = re.search(r"\*\*Prompt:\*\*\s*\n(.*)", prompt_block, re.S)
        if pm:
            prompt = _strip_quote(pm.group(1))

        response = body[rm.end():]
        # Sections are separated by a horizontal rule.
        response = re.split(r"^---\s*$", response, maxsplit=1, flags=re.M)[0]

        records.append({
            "category": category,
            "prompt": _norm(prompt),
            "response": _norm(response),
            "seconds": seconds,
        })
    return records


def build_records(model: str, parsed: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    tag_info = OLLAMA_TAGS.get(model, {})
    for idx, rec in enumerate(parsed):
        task = TASK_OF_CATEGORY[str(rec["category"])]
        prompt = str(rec["prompt"])
        response = str(rec["response"])

        record: Dict[str, object] = {
            "item_id": f"{task}_001",
            "model": model,
            "task": task,
            "prompt": prompt,
            # `output` is what eval/script_integrity.py reads; `hypothesis`
            # is what eval/text_metrics.py and eval/ocr_correction.py read.
            # Same string under both keys so one file feeds all three.
            "output": response,
            "hypothesis": response,
            "latency_ms": round(rec["seconds"] * 1000, 1) if rec["seconds"] else None,
            "max_new_tokens": PROVENANCE["max_new_tokens"],
            "provenance": PROVENANCE,
        }
        if tag_info:
            record["ollama"] = tag_info

        if task == "ocr_correction":
            # The OCR-Fix prompt embeds its own ground truth, so these
            # records are directly scorable by eval/ocr_correction.py.
            em = EXPECTED_RE.search(prompt)
            if em:
                record["reference"] = _norm(em.group(1))
            # The corrupted string is the first single-quoted span.
            qm = re.search(r"'([^']{10,})'", prompt)
            if qm:
                record["noisy"] = _norm(qm.group(1))
        else:
            record["reference"] = None
            record["reference_note"] = (
                "No reference in the prompt. eval/text_metrics.py needs one; "
                "commit benchmarks/items/*.jsonl (see benchmarks/README.md)."
            )
        out.append(record)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--print-summary", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"source directory not found: {src}")
    files = sorted(src.glob("*.md"))
    if not files:
        sys.exit(f"no .md files in {src}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, all_correction = [], []
    for f in files:
        model = f.stem
        parsed = parse_markdown(f.read_text(encoding="utf-8"))
        records = build_records(model, parsed)
        if not records:
            summary.append({"model": model, "n": 0,
                            "note": "no parsable benchmark sections"})
            continue

        target = out_dir / f"{model}.jsonl"
        with open(target, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        all_correction += [r for r in records if r["task"] == "ocr_correction"
                           and r.get("reference") and r.get("noisy")]

        tag = OLLAMA_TAGS.get(model, {})
        summary.append({
            "model": model,
            "n": len(records),
            "tasks": sorted({str(r["task"]) for r in records}),
            "file": _rel(target),
            "ollama_tag": tag.get("tag"),
            "label_mismatch": tag.get("mismatch", None),
        })

    # A single pooled file for the correction task, which is the one that
    # carries references and is therefore scorable today.
    if all_correction:
        pooled = out_dir / "ocr_correction.jsonl"
        with open(pooled, "w", encoding="utf-8") as fh:
            for r in all_correction:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "source": _rel(src),
        "out_dir": _rel(out_dir),
        "provenance": PROVENANCE,
        "models": summary,
        "pooled_correction_file": (
            _rel(out_dir / "ocr_correction.jsonl")
            if all_correction else None),
        "n_correction_records": len(all_correction),
    }
    (out_dir / "_conversion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for s in summary:
        flag = ""
        if s.get("label_mismatch"):
            flag = "  <-- LABEL MISMATCH, see docs/ERRATA.md A00/C14"
        print(f"  {s['model']:18s} {s['n']} records  "
              f"{s.get('ollama_tag') or '':45s}{flag}")

    print(f"\n  {len(files)} files -> {out_dir}")
    print(f"  {len(all_correction)} scorable OCR-correction records "
          f"(they carry their own reference)")
    print(f"\n  Every record is stamped: backend=ollama, "
          f"quantisation=Q4_K_M, temperature=0.3, N=1 per task.")
    print("\nNext:")
    print(f'  python eval/script_integrity.py "{out_dir}/*.jsonl" '
          f'--out eval/script_integrity.json')
    if all_correction:
        print(f"  python eval/ocr_correction.py --pred "
              f"{out_dir}/ocr_correction.jsonl --out eval/ocr_correction.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
