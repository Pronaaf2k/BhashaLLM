"""The hybrid handwritten Bangla OCR pipeline of paper Sec. III-A.

This is the third headline contribution in Section I:

    "A hybrid handwritten Bangla OCR pipeline combining a fine-tuned
    vision-language recognizer with LLM-based error correction, reaching
    88% character-level accuracy on a held-out, self-collected test set."

and the architecture of Sec. III-A:

    "BhashaLLM decouples character segmentation (detection) from character
    classification (recognition), which allows the detector and the
    recognizer to be optimised and swapped independently."

Why this file exists
--------------------
``bhasha/ocr/pipeline.py`` is the pipeline the repository shipped. It
differs from the paper in two ways that matter:

1. Its recognizer is a PaddleOCR + Tesseract ensemble. Neither engine
   appears anywhere in the manuscript, and the paper's recognizer is the
   QLoRA-fine-tuned Qwen-VL model of Sec. IV-C Phase 3.
2. Its correction stage, ``correct_text_with_llm``, is
   ``return text  # Placeholder``. The LLM correction stage is the half of
   contribution #3 that distinguishes it from ordinary OCR, and the
   correction/over-correction/false-positive rates of Sec. V-B are
   measurements of it.

``bhasha/ocr/pipeline.py`` is **not modified**; it is working code and a
useful classical baseline. This module is the paper's pipeline, built
beside it. See ``docs/ERRATA.md`` C4.

Stage 1 — detection
-------------------
Sec. III-A decouples detection from recognition, and Sec. II-A explains the
reasoning: a Bangla grapheme can encode a root, a vowel diacritic and a
consonant diacritic simultaneously, so keeping that complexity inside the
recognition stage is what makes the split worth having.

Two detectors are provided behind one interface, chosen with
``--detector``:

``projection``
    Horizontal-projection-profile line segmentation. No dependencies
    beyond OpenCV, deterministic, and appropriate for the page and line
    images in the self-collected set. This is the default.
``paddle``
    PP-OCRv3's text detector, reusing the engine
    ``bhasha/ocr/pipeline.py`` already depends on. Sec. II-A cites
    PaddleOCR as precedent for the detect-then-recognize split [9].
``none``
    Treat the whole image as one line. Correct for the Ekush and
    BanglaWriting character crops, and for already-segmented line images
    such as those under ``training/vlm_ocr/data_desktop/``.

Stage 2 — recognition
---------------------
The Phase-3 QLoRA adapter over ``swapnillo/Bangla-OCR-SFT``, driven through
``bhasha.app.adapter_manager`` so that the resizing and normalisation match
training exactly and only one adapter is resident (Sec. III-F).

Stage 3 — LLM correction
------------------------
The text model with the Bangla adapter, prompted in Bangla only, asked to
repair recognition errors. Three properties are enforced here rather than
left to the prompt, because Sec. IV-D requires that a model "cannot inflate
its apparent correction rate simply by editing text aggressively":

* **The correction is rejected if it is too far from the input.** A model
  that rewrites the line rather than repairing it produces a large edit
  distance; past ``--max-edit-ratio`` the original is kept and the
  rejection is recorded. This is a guardrail, not a metric — the metric is
  ``eval/ocr_correction.py``.
* **Empty or script-invalid corrections are rejected.** A correction that
  drops the Bangla or introduces Devanagari is worse than no correction;
  the check reuses ``eval/script_integrity.py``'s definition.
* **Every stage output is retained.** The record carries the raw
  recognition, the correction, and whether the correction was accepted, so
  the correction stage can be scored separately from the recognizer, which
  is what Sec. V-B reports.

Usage
-----
    python -m bhasha.ocr.hybrid_pipeline --image page.png
    python -m bhasha.ocr.hybrid_pipeline --image page.png --detector projection
    python -m bhasha.ocr.hybrid_pipeline --image line.png --detector none --no-correct

    # batch, writing the JSONL that eval/ocr_cer.py and
    # eval/ocr_correction.py read
    python -m bhasha.ocr.hybrid_pipeline --manifest data/handwriting/manifest.csv \\
        --split test --out eval/ocr_predictions.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Bangla-only correction prompt. Paper Sec. IV-C: an English instruction
# wrapper "made language reversion markedly more likely in the smaller
# models", and correction runs on the same family of models.
CORRECTION_PROMPT = (
    "নিচের লেখাটি হাতে লেখা বাংলা ছবি থেকে যন্ত্রে পড়া হয়েছে, তাই এতে "
    "বানান ও যুক্তাক্ষরের ভুল থাকতে পারে। শুধুমাত্র ভুলগুলো সংশোধন করুন। "
    "নতুন কোনো শব্দ যোগ করবেন না, কোনো শব্দ বাদ দেবেন না, এবং সঠিক অংশ "
    "অপরিবর্তিত রাখুন। শুধু সংশোধিত লেখাটি লিখুন।\n\n"
    "লেখা: {text}\n\n"
    "সংশোধিত:"
)

# A correction whose character edit distance exceeds this fraction of the
# input length is treated as a rewrite rather than a repair and is
# rejected. 0.4 is deliberately permissive: Sec. V-B reports Mistral-7B
# over-correcting at 30%, and the guardrail should let that behaviour
# through to be *measured*, not silently suppress it.
DEFAULT_MAX_EDIT_RATIO = 0.4

BENGALI = (0x0980, 0x09FF)
DEVANAGARI = (0x0900, 0x097F)
SHARED_INDIC_PUNCTUATION = {0x0964, 0x0965}  # danda, double danda


# --------------------------------------------------------------- utilities


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. Small, self-contained, no dependency."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def script_ok(text: str) -> bool:
    """True if the text is Bangla and free of Devanagari.

    Same definition as ``eval/script_integrity.py``, including the
    exemption for U+0964/U+0965. Those two codepoints live in the
    Devanagari block but are shared Indic punctuation and the correct
    sentence terminators in Bangla; a check without the exemption rejects
    every correctly punctuated Bangla sentence.
    """
    text = _norm(text)
    bengali = devanagari = 0
    for ch in text:
        cp = ord(ch)
        if cp in SHARED_INDIC_PUNCTUATION:
            continue
        if DEVANAGARI[0] <= cp <= DEVANAGARI[1]:
            devanagari += 1
        elif BENGALI[0] <= cp <= BENGALI[1]:
            bengali += 1
    return devanagari == 0 and bengali > 0


# --------------------------------------------------------- stage 1: detect


def detect_lines_projection(
    image_path: str | Path,
    min_line_height: int = 8,
    smoothing: int = 3,
) -> List[Tuple[int, int, int, int]]:
    """Segment a page into line boxes by horizontal projection profile.

    Returns a list of ``(x, y, w, h)`` boxes, top to bottom.

    The profile method is chosen over a learned detector for a reason worth
    stating: it is deterministic and has no trained parameters, so a
    difference in end-to-end CER between two runs is attributable to the
    recognizer rather than to detector variance. Sec. III-A's whole point
    is that the two stages can be optimised independently, which requires
    holding one fixed while varying the other.

    Falls back to a single full-image box when OpenCV is unavailable or no
    line separation is found, so the caller always receives at least one
    region.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    h, w = img.shape[:2]

    # Otsu, inverted so ink is high. Adaptive thresholding is better for
    # uneven lighting but produces a noisier projection profile; Otsu is
    # the right trade-off when the output feeds a profile rather than a
    # recognizer.
    _, binary = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    profile = binary.sum(axis=1).astype(float) / 255.0
    if smoothing > 1:
        kernel = np.ones(smoothing) / smoothing
        profile = np.convolve(profile, kernel, mode="same")

    # A row is "ink" if it carries more than a small fraction of the mean.
    # An absolute threshold fails on faint pencil; a relative one adapts.
    threshold = max(profile.mean() * 0.15, 1.0)
    inked = profile > threshold

    boxes: List[Tuple[int, int, int, int]] = []
    start = None
    for y, on in enumerate(inked):
        if on and start is None:
            start = y
        elif not on and start is not None:
            if y - start >= min_line_height:
                boxes.append((0, start, w, y - start))
            start = None
    if start is not None and h - start >= min_line_height:
        boxes.append((0, start, w, h - start))

    return boxes


def detect_lines_paddle(image_path: str | Path) -> List[Tuple[int, int, int, int]]:
    """Text-region detection via PP-OCRv3's detector.

    Reuses the engine ``bhasha/ocr/pipeline.py`` already depends on. Paper
    Sec. II-A cites PaddleOCR's detect-then-recognize split as precedent
    for the decoupling in Sec. III-A [9].

    Only the *detector* is used. PaddleOCR's Bangla recognizer is not the
    paper's recognizer and its output is discarded here.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return []
    try:
        engine = PaddleOCR(use_angle_cls=False, lang="bn", show_log=False)
        result = engine.ocr(str(image_path), det=True, rec=False, cls=False)
    except Exception:  # noqa: BLE001 - engine variation across versions
        return []
    if not result or not result[0]:
        return []

    boxes = []
    for quad in result[0]:
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        x, y = int(min(xs)), int(min(ys))
        boxes.append((x, y, int(max(xs)) - x, int(max(ys)) - y))
    boxes.sort(key=lambda b: (b[1], b[0]))  # reading order: top-down, left-right
    return boxes


DETECTORS = {
    "projection": detect_lines_projection,
    "paddle": detect_lines_paddle,
    "none": lambda p, **kw: [],
}


# ------------------------------------------------------------- the pipeline


@dataclass
class LineResult:
    """One detected line, carried through every stage.

    Each stage's output is kept rather than overwritten, so the recognizer
    and the corrector can be scored independently -- which is exactly what
    Table VII (recognition) and Sec. V-B (correction) report separately.
    """
    index: int
    box: Optional[Tuple[int, int, int, int]]
    recognised: str = ""
    corrected: str = ""
    correction_applied: bool = False
    rejection_reason: Optional[str] = None
    edit_distance: int = 0
    confidence: Optional[float] = None
    recognise_ms: float = 0.0
    correct_ms: float = 0.0


@dataclass
class PageResult:
    image: str
    detector: str
    n_lines: int
    lines: List[LineResult] = field(default_factory=list)
    total_ms: float = 0.0

    @property
    def raw_text(self) -> str:
        """Recognition output only, before correction. Table VII's subject."""
        return "\n".join(l.recognised for l in self.lines)

    @property
    def text(self) -> str:
        """Final text after the correction stage. Contribution #3's output."""
        return "\n".join(l.corrected or l.recognised for l in self.lines)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["raw_text"] = self.raw_text
        d["text"] = self.text
        d["n_corrections_applied"] = sum(
            1 for l in self.lines if l.correction_applied)
        d["n_corrections_rejected"] = sum(
            1 for l in self.lines if l.rejection_reason)
        return d


class HybridOCRPipeline:
    """Detection -> QLoRA recognition -> LLM correction (paper Sec. III-A).

    Parameters
    ----------
    manager:
        An ``AdapterManager``. Built lazily if not supplied, so importing
        this module costs nothing.
    detector:
        ``projection`` | ``paddle`` | ``none``.
    correct:
        Run stage 3. ``False`` gives the recognizer's raw output, which is
        the configuration Table VII's CER measures.
    correction_adapter:
        Which text adapter to correct with. ``bangla`` is the Phase-1
        adapter; ``None`` uses the bare base model.
    max_edit_ratio:
        Reject a correction whose edit distance exceeds this fraction of
        the recognised length. See the module docstring.
    """

    def __init__(
        self,
        manager: Any = None,
        detector: str = "projection",
        correct: bool = True,
        correction_adapter: Optional[str] = "bangla",
        max_edit_ratio: float = DEFAULT_MAX_EDIT_RATIO,
        image_size: int = 256,
        use_adapter: bool = True,
    ) -> None:
        if detector not in DETECTORS:
            raise ValueError(
                f"unknown detector {detector!r}; choose from {sorted(DETECTORS)}")
        self.detector = detector
        self.correct = correct
        self.correction_adapter = correction_adapter
        self.max_edit_ratio = max_edit_ratio
        self.image_size = image_size
        self.use_adapter = use_adapter
        self._manager = manager

    @property
    def manager(self):
        if self._manager is None:
            from bhasha.app.adapter_manager import get_manager
            self._manager = get_manager()
        return self._manager

    # ------------------------------------------------------ stage 3 helper

    def correct_line(self, text: str) -> Tuple[str, bool, Optional[str], int]:
        """Run the LLM correction stage on one recognised line.

        Returns ``(text, applied, rejection_reason, edit_distance)``.

        This is the function ``bhasha/ocr/pipeline.py::correct_text_with_llm``
        was a placeholder for.
        """
        source = _norm(text).strip()
        if not source:
            return text, False, "empty_input", 0

        prompt = self.manager.format_chatml([
            {"role": "user", "content": CORRECTION_PROMPT.format(text=source)}
        ])
        out = self.manager.generate(
            prompt,
            task=self.correction_adapter,
            # Greedy. Correction is a constrained rewriting task, and
            # sampling at temperature 0.7 introduces exactly the kind of
            # gratuitous edit the over-correction rate is meant to catch.
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=max(32, int(len(source) * 1.5)),
        )
        candidate = _norm(out["text"]).strip().split("\n")[0].strip()

        if not candidate:
            return text, False, "empty_correction", 0
        if not script_ok(candidate):
            # A correction that drops the Bangla or introduces Devanagari is
            # worse than no correction.
            return text, False, "script_invalid", 0

        distance = _edit_distance(source, candidate)
        if distance == 0:
            return candidate, False, None, 0
        if distance > self.max_edit_ratio * max(len(source), 1):
            return text, False, "rewrite_exceeds_max_edit_ratio", distance
        return candidate, True, None, distance

    # ------------------------------------------------------------ full run

    def run(self, image_path: str | Path) -> PageResult:
        from PIL import Image

        t0 = time.perf_counter()
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        # -- stage 1: detection
        detect = DETECTORS[self.detector]
        boxes = detect(image_path) if self.detector != "none" else []
        page = Image.open(image_path).convert("RGB")
        if not boxes:
            # 'none', an unavailable detector, or a page with no separable
            # lines. One region covering the whole image.
            boxes = [(0, 0, page.width, page.height)]

        result = PageResult(image=str(image_path), detector=self.detector,
                            n_lines=len(boxes))

        for i, (x, y, w, h) in enumerate(boxes):
            crop = page.crop((x, y, x + w, y + h))
            line = LineResult(index=i, box=(x, y, w, h))

            # -- stage 2: recognition (QLoRA Qwen-VL, paper Sec. IV-C)
            t1 = time.perf_counter()
            rec = self.manager.ocr(
                crop, image_size=self.image_size, return_logprobs=True,
                use_adapter=self.use_adapter,
            )
            line.recognise_ms = round((time.perf_counter() - t1) * 1000, 1)
            line.recognised = rec["text"].strip()
            line.confidence = rec.get("confidence")
            line.corrected = line.recognised

            # -- stage 3: LLM correction (contribution #3)
            if self.correct:
                t2 = time.perf_counter()
                corrected, applied, reason, dist = self.correct_line(
                    line.recognised)
                line.correct_ms = round((time.perf_counter() - t2) * 1000, 1)
                line.corrected = corrected
                line.correction_applied = applied
                line.rejection_reason = reason
                line.edit_distance = dist

            result.lines.append(line)

        result.total_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result


# ------------------------------------------------------------------- CLI


def _batch(pipeline: HybridOCRPipeline, args) -> int:
    rows = list(csv.DictReader(
        open(args.manifest, encoding="utf-8", newline="")))
    if args.split:
        rows = [r for r in rows if r.get("split") == args.split]
    if not rows:
        sys.exit(f"no rows in {args.manifest}"
                 + (f" with split={args.split}" if args.split else ""))

    root = Path(args.image_root) if args.image_root else Path(args.manifest).parent
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            image_id = row.get("image_id", "")
            candidate = Path(row.get("image_path") or (root / f"{image_id}.png"))
            try:
                res = pipeline.run(candidate)
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                print(f"  [{i}/{len(rows)}] {image_id}: {exc}", file=sys.stderr)
                continue

            record: Dict[str, Any] = {
                "image_id": image_id,
                "writer_id": row.get("writer_id"),
                # `hypothesis` is the post-correction text -- the pipeline's
                # actual output, and what eval/ocr_cer.py should score.
                "hypothesis": res.text,
                # `noisy` is the pre-correction recognition. Naming it this
                # way means the same file feeds eval/ocr_correction.py
                # without a conversion step: noisy -> hypothesis is exactly
                # the correction stage.
                "noisy": res.raw_text,
                "detector": res.detector,
                "n_lines": res.n_lines,
                "latency_ms": res.total_ms,
                "n_corrections_applied": sum(
                    1 for l in res.lines if l.correction_applied),
                "n_corrections_rejected": sum(
                    1 for l in res.lines if l.rejection_reason),
            }
            ref = row.get("reference") or row.get("text")
            if ref:
                record["reference"] = ref
            confs = [l.confidence for l in res.lines if l.confidence is not None]
            if confs:
                record["confidence"] = round(sum(confs) / len(confs), 4)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"  {i}/{len(rows)} ...", file=sys.stderr)

    print(f"wrote {out_path}  ({n_ok} ok, {n_fail} failed)")
    print("\nNext:")
    print(f"  python eval/ocr_cer.py --pred {out_path} "
          f"--manifest {args.manifest} --group-by writer_id --out eval/ocr_cer.json")
    print(f"  python eval/ocr_correction.py --pred {out_path} "
          f"--out eval/ocr_correction.json   # scores the correction stage")
    return 0 if n_fail == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="single image")
    src.add_argument("--manifest", help="batch over a handwriting manifest")
    ap.add_argument("--split", default=None, help="train | val | test")
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--out", default="eval/ocr_predictions.jsonl")
    ap.add_argument("--detector", default="projection",
                    choices=sorted(DETECTORS),
                    help="stage-1 detector (paper Sec. III-A)")
    ap.add_argument("--no-correct", action="store_true",
                    help="skip stage 3. This is the configuration Table "
                         "VII's CER measures -- recognition only.")
    ap.add_argument("--no-adapter", action="store_true",
                    help="recognise with the un-adapted base model. This is "
                         "Table VII's 'Before' column (28%% CER).")
    ap.add_argument("--correction-adapter", default="bangla",
                    help="text adapter used for stage 3; 'none' for base")
    ap.add_argument("--max-edit-ratio", type=float,
                    default=DEFAULT_MAX_EDIT_RATIO)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    pipeline = HybridOCRPipeline(
        detector=args.detector,
        correct=not args.no_correct,
        correction_adapter=(
            None if args.correction_adapter in ("none", "null")
            else args.correction_adapter),
        max_edit_ratio=args.max_edit_ratio,
        image_size=args.image_size,
        use_adapter=not args.no_adapter,
    )

    if args.manifest:
        return _batch(pipeline, args)

    res = pipeline.run(args.image)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"detector={res.detector}  lines={res.n_lines}  "
              f"{res.total_ms} ms")
        print("-" * 60)
        print("recognised (stage 2):")
        print(res.raw_text)
        if not args.no_correct:
            print("-" * 60)
            print("corrected (stage 3):")
            print(res.text)
            applied = sum(1 for l in res.lines if l.correction_applied)
            rejected = [l.rejection_reason for l in res.lines
                        if l.rejection_reason]
            print(f"\n{applied} of {res.n_lines} lines corrected; "
                  f"{len(rejected)} rejected {rejected or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
