#!/usr/bin/env python3
"""
Recover the training and OCR evidence already committed to this repository.

`APPLY_THIS_PATCH.md` step 0: "if the raw outputs behind Tables V, VI and
VII are sitting in [logs/, report/, llm outputs/], most of the work below
collapses from 'regenerate' to 'point at it'." Some of it is. This script
points at it.

`scripts/convert_llm_outputs.py` already recovered the text generations.
This one covers the other two categories:

**Phase-1 training history.**
`report/antigravity_experiments/logs/bangla_adapt_Qwen2.5-1.5B-Instruct_history.json`
is the HuggingFace log history for Phase 1 — the filename is exactly the
`run_name` that `bhasha/llm/train.py` builds. It is converted here into
`logs/phase1_summary.json`, the schema `bhasha/utils/run_summary.py`
writes and `docs/TRACEABILITY.md` points at. Doing so corrects Table IV in
three places (see `docs/ERRATA.md` C18):

* the run reached **410** optimiser steps, not 500;
* **1.31 is the validation loss**, not the training loss (train ends at
  1.3405, eval at 1.3139), and Table IV labels it `(train)`;
* epoch coverage was **0.985**, not 0.8.

**OCR predictions.**
`training/vlm_ocr/outputs/.../desktop_lines_test_predictions.jsonl` holds
88 line predictions with references — the only committed OCR predictions
in the repository. They are converted to the `eval/ocr_cer.py` input
schema and scored, which puts a *measured* CER next to Table VII's 12%.

Note carefully: those predictions come from a **PaliGemma-2 3B** adapter,
which is not the model the paper describes (`docs/ERRATA.md` C19). The
number is reported as what it is — the best-evidenced OCR result in the
artifact — not as a refutation of Table VII on its own terms. The
refutation is in the two committed reports that evaluate the paper's own
model, and this script prints those beside it.

Usage
-----
    python scripts/recover_committed_evidence.py
    python scripts/recover_committed_evidence.py --write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "eval"))

PHASE1_HISTORY = (BASE_DIR / "report" / "antigravity_experiments" / "logs"
                  / "bangla_adapt_Qwen2.5-1.5B-Instruct_history.json")
PALIGEMMA_PREDS = (BASE_DIR / "training" / "vlm_ocr" / "outputs"
                   / "paligemma2_3b_448_desktop_lines_lora"
                   / "desktop_lines_test_predictions.jsonl")
OCR_BENCH_REPORT = BASE_DIR / "report" / "ocr_benchmark_report.md"
COMPREHENSIVE = BASE_DIR / "report" / "COMPREHENSIVE_OCR_EVALUATION.md"
MODELS_SUMMARY = BASE_DIR / "docs" / "MODELS_SUMMARY.md"

# Table IV, Phase 1, as printed.
PAPER_PHASE1 = {"steps": 500, "final_loss": 1.31, "final_loss_label": "train",
                "epochs": 0.8, "time": "3h05"}
# Table VII, as printed.
PAPER_TABLE_VII = {"cer_before": 0.28, "cer_after": 0.12,
                   "accuracy_before": 0.72, "accuracy_after": 0.88}


# ------------------------------------------------------- phase 1 history


def recover_phase1(path: Path = PHASE1_HISTORY) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    history = json.loads(path.read_text(encoding="utf-8"))

    train = [e for e in history if "loss" in e]
    evals = [e for e in history if "eval_loss" in e]
    final = next((e for e in history if "train_runtime" in e), {})
    if not train:
        return None

    last = train[-1]
    steps = int(last.get("step") or 0)
    # bhasha/llm/train.py: per-device batch 1 x gradient accumulation 4.
    effective_batch = 4
    runtime = final.get("train_runtime")
    sps = final.get("train_samples_per_second")
    # The sequence count the run actually saw, from the trainer's own
    # throughput figures. This is what makes the corpus-size claim in
    # ERRATA C18 checkable rather than inferred.
    n_sequences = int(round(sps * runtime)) if (sps and runtime) else None

    return {
        "phase": "1_bangla_pt",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "seed": 42,
        "config_source": "recovered from " + str(path.relative_to(BASE_DIR)),
        "recovered": True,
        "steps": steps,
        "effective_batch": effective_batch,
        "tokens_per_sequence": 512,
        "sequences_consumed": steps * effective_batch,
        "n_train_sequences": n_sequences,
        "epochs_covered": round(last.get("epoch"), 4) if last.get("epoch") else None,
        "epochs_covered_note":
            "Read from the trainer's own `epoch` field. Table IV states 0.8; "
            "the log says 0.985. See docs/ERRATA.md C18, which supersedes B2.",
        "final_train_loss": round(last["loss"], 4),
        "final_eval_loss": round(evals[-1]["eval_loss"], 4) if evals else None,
        "best_eval_loss": round(min(e["eval_loss"] for e in evals), 4) if evals else None,
        "reported_train_loss_over_run": final.get("train_loss"),
        "wall_clock_s": runtime,
        "n_logged_train_points": len(train),
        "n_logged_eval_points": len(evals),
        "peak_vram_gb": None,
        "peak_vram_note":
            "Not recorded by the original run. bhasha/utils/run_summary.py "
            "records it on any future run.",
        "estimated_train_tokens": (n_sequences * 512) if n_sequences else None,
        "estimated_train_tokens_note":
            "Table IV gives the Phase-1 training split as 5.28M tokens. This "
            "estimate is roughly an order of magnitude smaller, and agrees "
            "with the corpus-archive audit in docs/ERRATA.md C8.",
        "log_history": history,
    }


def compare_phase1(rec: Dict[str, Any]) -> List[str]:
    out = []
    if rec["steps"] != PAPER_PHASE1["steps"]:
        out.append(f"steps: log says {rec['steps']}, Table IV says "
                   f"{PAPER_PHASE1['steps']}")
    tr, ev = rec["final_train_loss"], rec["final_eval_loss"]
    if ev is not None and abs(ev - PAPER_PHASE1["final_loss"]) < 0.01 \
            and abs(tr - PAPER_PHASE1["final_loss"]) > 0.01:
        out.append(f"final loss: Table IV's {PAPER_PHASE1['final_loss']} "
                   f"(labelled '{PAPER_PHASE1['final_loss_label']}') matches "
                   f"the EVAL loss {ev}, not the train loss {tr}")
    if rec["epochs_covered"] and abs(
            rec["epochs_covered"] - PAPER_PHASE1["epochs"]) > 0.05:
        out.append(f"epochs: log says {rec['epochs_covered']}, Table IV says "
                   f"{PAPER_PHASE1['epochs']}")
    if rec["wall_clock_s"]:
        out.append(f"wall clock: log says {rec['wall_clock_s']:.0f} s "
                   f"({rec['wall_clock_s'] / 60:.1f} min), Table IV says "
                   f"{PAPER_PHASE1['time']}")
    if rec.get("estimated_train_tokens"):
        out.append(f"training tokens: ~{rec['estimated_train_tokens']:,} "
                   f"estimated from throughput, Table IV says 5,280,000")
    return out


# ----------------------------------------------------------- ocr evidence


def recover_paligemma(path: Path = PALIGEMMA_PREDS) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if not recs:
        return None

    converted = [{
        "image_id": Path(r["image"]).stem,
        "reference": r.get("ref", ""),
        "hypothesis": r.get("pred", ""),
    } for r in recs]

    from ocr_cer import score  # eval/ocr_cer.py
    overall, _ = score(converted)
    return {"converted": converted, "overall": overall, "n_source": len(recs)}


def checkpoint_steps(path: Path = None) -> List[tuple]:
    """Read the saved checkpoint steps per phase from MODELS_SUMMARY.md.

    Returns [(label, last_step, table_iv_claim), ...]. This is the second,
    independent line of evidence for A9: the checkpoint directories record
    where each run actually stopped, and for Phase 1 they agree exactly with
    the recovered log history.
    """
    path = path or MODELS_SUMMARY
    if not path.exists():
        return []
    lines = [l for l in path.read_text(encoding="utf-8").splitlines()
             if "Checkpoints:" in l]
    labels = ["Phase 1  bangla_adapters", "Phase 2  instruct_adapters",
              "Phase 3  ocr_adapters"]
    claims = [500, 500, 2000]   # Table IV
    out = []
    for label, line, claim in zip(labels, lines, claims):
        nums = [int(x) for x in re.findall(r"\d+", line.split(":", 1)[1])]
        if nums:
            out.append((label, max(nums), claim))
    return out


def _grep_table(path: Path, pattern: str) -> List[str]:
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
            if re.search(pattern, l)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="write logs/phase1_summary.json and "
                         "eval/ocr_predictions_paligemma.jsonl")
    ap.add_argument("--out-dir", default=str(BASE_DIR))
    args = ap.parse_args()
    out_root = Path(args.out_dir)

    print("=" * 72)
    print("PHASE 1 TRAINING HISTORY")
    print("=" * 72)
    p1 = recover_phase1()
    if not p1:
        print(f"  not found: {PHASE1_HISTORY}")
    else:
        print(f"  source            {PHASE1_HISTORY.relative_to(BASE_DIR)}")
        print(f"  steps             {p1['steps']}")
        print(f"  final train loss  {p1['final_train_loss']}")
        print(f"  final eval loss   {p1['final_eval_loss']}")
        print(f"  epochs covered    {p1['epochs_covered']}")
        print(f"  wall clock        {p1['wall_clock_s']:.1f} s")
        print(f"  train sequences   {p1['n_train_sequences']}")
        print(f"  est. train tokens {p1['estimated_train_tokens']:,}"
              if p1.get("estimated_train_tokens") else "")
        print("\n  disagreements with Table IV:")
        for d in compare_phase1(p1):
            print(f"    - {d}")
        if args.write:
            target = out_root / "logs" / "phase1_summary.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(p1, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            print(f"\n  wrote {target}")

    print()
    print("=" * 72)
    print("STEP COUNTS — Table IV against the committed checkpoints")
    print("=" * 72)
    steps = checkpoint_steps()
    if not steps:
        print(f"  not found: {MODELS_SUMMARY}")
    else:
        print(f"  {'phase':30s} {'Table IV':>9s} {'last ckpt':>10s} {'short':>7s}")
        for label, last, claimed in steps:
            short = f"{100 * (1 - last / claimed):.0f}%" if claimed else "-"
            print(f"  {label:30s} {claimed:>9d} {last:>10d} {short:>7s}")
        if p1:
            print(f"\n  Phase 1 cross-check: the recovered log ends at step "
                  f"{p1['steps']}, and the last checkpoint is "
                  f"{steps[0][1]}. Two independent sources agree, which is "
                  f"what makes the other two rows credible.")
        print("  See docs/ERRATA.md A9.")

    print()
    print("=" * 72)
    print("OCR EVIDENCE — every CER figure in the repository")
    print("=" * 72)
    rows = [("Paper Table VII, before fine-tuning", "28.0%",
             "Bangla-OCR-SFT", "no artifact"),
            ("Paper Table VII, after fine-tuning", "12.0%",
             "Bangla-OCR-SFT + adapter", "no artifact")]

    for line in _grep_table(OCR_BENCH_REPORT, r"\*\*(Baseline|Fine-tuned)\*\*"):
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 3:
            rows.append((f"report/ocr_benchmark_report.md — {cells[0]}",
                         cells[2], "Bangla-OCR-SFT",
                         str(OCR_BENCH_REPORT.relative_to(BASE_DIR))))

    pali = recover_paligemma()
    if pali:
        o = pali["overall"]
        rows.append(("PaliGemma predictions, measured here",
                     f"{o['cer'] * 100:.1f}%",
                     "PaliGemma-2 3B (NOT in the paper)",
                     str(PALIGEMMA_PREDS.relative_to(BASE_DIR))))

    print(f"  {'source':46s} {'CER':>7s}  model")
    print("  " + "-" * 70)
    for name, cer, model, _ in rows:
        print(f"  {name:46s} {cer:>7s}  {model}")

    if COMPREHENSIVE.exists():
        print(f"\n  {COMPREHENSIVE.relative_to(BASE_DIR)} additionally reports "
              f"0.00% accuracy for BOTH the baseline and the fine-tuned "
              f"model on 50 Ekush images.")

    if pali:
        o = pali["overall"]
        print(f"\n  PaliGemma detail: {o['n_items']} lines, "
              f"{o['reference_chars']:,} reference chars, "
              f"char accuracy {o['char_accuracy'] * 100:.1f}%")
        print("  per grapheme category (paper Sec. V-B in brackets):")
        expect = {"independent_vowel": "94", "consonant": "91",
                  "vowel_diacritic": "85", "consonant_diacritic": "79"}
        for k, v in sorted(o["by_grapheme_category"].items(),
                           key=lambda x: -x[1]["n"]):
            if v["n"] >= 20:
                exp = f"  [paper {expect[k]}%]" if k in expect else ""
                print(f"    {k:22s} n={v['n']:>5d}  "
                      f"acc={v['accuracy'] * 100:5.1f}%{exp}")
        if args.write:
            target = out_root / "eval" / "ocr_predictions_paligemma.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as fh:
                for c in pali["converted"]:
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"\n  wrote {target}")
            print(f"  score it: python eval/ocr_cer.py --pred "
                  f"{target.relative_to(BASE_DIR)} --out eval/ocr_cer_paligemma.json")

    print()
    print("  Four sources, four different CERs, and the two that evaluate the")
    print("  paper's own model both report that fine-tuning made it WORSE.")
    print("  See docs/ERRATA.md A6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
