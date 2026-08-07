"""Vision-language OCR dataset for the Phase-3 QLoRA fine-tune.

Restores ``bhasha.data.dataset``, which ``bhasha/ocr/train.py``,
``bhasha/scripts/train_ocr_improved.py`` and
``bhasha/scripts/debug_dataset_shapes.py`` all import and which was absent
from the repository. Those three modules could not be imported before this
file existed, so paper Sec. IV-C Phase 3 was not runnable from a clean
checkout.

The record format is the one already used by the committed JSONL files
under ``training/vlm_ocr/data_desktop/``::

    {"image": "<path>", "text": "<bangla transcription>",
     "source_dataset": "<optional>", "image_id": "<optional>",
     "writer_id": "<optional>"}

``image_id`` and ``writer_id`` are optional and are carried through
untouched so that predictions written during evaluation can be joined
against ``data/handwriting/manifest.csv`` for the writer-disjoint CER of
paper Sec. VI-D.

Design notes
------------
* Images are converted to RGB and resized to a square edge. The default is
  256, which is the ``image_size`` recorded in
  ``configs/phase3_ocr_sft.yaml`` and matches paper Sec. V-C, where moving
  from 224 to 256 costs 40 ms per image and gains 0.02 reported confidence.
* Prompt text is Bangla-only. Paper Sec. IV-C: "Prompts were held constant
  per task and written in Bangla only, since a pilot run showed that an
  English instruction wrapper made language reversion markedly more likely
  in the smaller models."
* Only the answer span is supervised. Prompt tokens are masked to -100 so
  the model is not trained to reproduce its own instruction, which would
  otherwise dominate the loss on short character-level targets.
* The collator pads to the longest item in the batch rather than to a fixed
  length. With ``per_device_train_batch_size: 1`` (Table III) this is a
  no-op, but it keeps the loader correct if the batch size is raised.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

# Bangla-only instruction. See module docstring for why this is not English.
DEFAULT_PROMPT = "এই ছবিতে লেখা বাংলা লেখাটি হুবহু লিখুন।"

# configs/phase3_ocr_sft.yaml -> data.image_size
DEFAULT_IMAGE_SIZE = 256

# Table III: max sequence length 512. OCR targets are far shorter, but the
# cap protects against a pathological transcription blowing up memory.
DEFAULT_MAX_LENGTH = 512

IGNORE_INDEX = -100


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read a JSONL file, skipping blank lines.

    Raises on malformed JSON rather than silently dropping records: a
    silently shortened training set is the kind of error that shows up as an
    unexplained loss difference three phases later.
    """
    path = Path(path)
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
    return records


def normalise_bangla(text: str) -> str:
    """NFC-normalise a transcription.

    Paper Sec. IV-B normalises the *text corpus* with NFKC. Transcriptions
    are normalised with NFC instead, deliberately: NFKC applies
    compatibility folding, which rewrites some Bangla presentation forms and
    would make the reference string differ from what was written on the
    page. Every evaluation script in ``eval/`` also normalises with NFC, so
    training targets and scoring references agree.
    """
    return unicodedata.normalize("NFC", text or "")


class OCRDataset(Dataset):
    """Image/transcription pairs for a Qwen-VL-style OCR model.

    Parameters
    ----------
    jsonl_path:
        Path to a JSONL file with ``image`` and ``text`` keys per record.
    processor:
        A HuggingFace ``AutoProcessor`` for the vision-language base model
        (``swapnillo/Bangla-OCR-SFT`` in paper Sec. IV-C).
    prompt:
        Instruction text. Defaults to the Bangla-only prompt above.
    image_size:
        Square edge length in pixels. Defaults to 256 per
        ``configs/phase3_ocr_sft.yaml``.
    max_length:
        Token cap, Table III's 512 by default.
    image_root:
        Optional directory that relative ``image`` paths are resolved
        against. Needed because the committed JSONL files carry absolute
        paths from the collection machine.
    skip_missing:
        If True, records whose image file is absent are dropped at
        construction time and the count is reported on ``self.n_skipped``.
        If False (the default) a missing file raises, so a partially
        available dataset cannot quietly shrink the training set.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        processor: Any,
        prompt: str = DEFAULT_PROMPT,
        image_size: int = DEFAULT_IMAGE_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
        image_root: Optional[str | Path] = None,
        skip_missing: bool = False,
    ) -> None:
        self.processor = processor
        self.prompt = prompt
        self.image_size = image_size
        self.max_length = max_length
        self.image_root = Path(image_root) if image_root else None

        records = read_jsonl(jsonl_path)
        self.n_skipped = 0
        if skip_missing:
            kept = []
            for rec in records:
                if self._resolve(rec.get("image", "")).exists():
                    kept.append(rec)
                else:
                    self.n_skipped += 1
            records = kept
        self.records = records
        self.source = str(jsonl_path)

    # ------------------------------------------------------------------ util

    def _resolve(self, image_path: str) -> Path:
        p = Path(image_path)
        if not p.is_absolute() and self.image_root is not None:
            p = self.image_root / p
        return p

    def _load_image(self, image_path: str):
        from PIL import Image  # imported lazily so the module imports without PIL

        path = self._resolve(image_path)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path} (from {self.source})")
        img = Image.open(path).convert("RGB")
        if self.image_size:
            img = img.resize((self.image_size, self.image_size))
        return img

    # -------------------------------------------------------------- protocol

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        image = self._load_image(rec["image"])
        target = normalise_bangla(rec.get("text", ""))

        # Build the chat-formatted prompt. Vision-language processors expose
        # apply_chat_template; fall back to a bare prompt string for
        # processors that do not.
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": self.prompt},
            ],
        }]
        try:
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001 - processor variation is expected
            prompt_text = self.prompt

        full_text = prompt_text + target

        encoded = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encoded["input_ids"][0]
        labels = input_ids.clone()

        # Mask the prompt span so loss is computed on the transcription only.
        prompt_only = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        n_prompt = int(prompt_only["input_ids"].shape[-1])
        labels[:n_prompt] = IGNORE_INDEX

        item: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": encoded.get(
                "attention_mask", torch.ones_like(input_ids)
            )[0],
            "labels": labels,
            "pixel_values": encoded.get("pixel_values"),
            "image_grid_thw": encoded.get("image_grid_thw"),
        }
        # Squeeze the batch dimension the processor adds, where present.
        for key in ("pixel_values", "image_grid_thw"):
            val = item[key]
            if isinstance(val, torch.Tensor) and val.dim() > 1 and val.shape[0] == 1:
                item[key] = val[0]

        # Provenance keys, carried through for evaluation joins. These are
        # popped by collate_fn so they never reach the model.
        for key in ("image_id", "writer_id", "source_dataset"):
            if key in rec:
                item[key] = rec[key]
        return item


def _pad_stack(
    tensors: Sequence[torch.Tensor], pad_value: int
) -> torch.Tensor:
    longest = max(t.shape[0] for t in tensors)
    out = torch.full((len(tensors), longest), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.shape[0]] = t
    return out


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad a batch to its longest item and stack the vision tensors.

    Text tensors are right-padded; ``labels`` are padded with -100 so the
    padding contributes no loss, and ``attention_mask`` with 0 so it
    contributes no attention. Padding ``labels`` with the pad token instead
    is a common and silent error: it trains the model to emit padding and
    deflates the reported loss.

    Provenance keys (``image_id``, ``writer_id``, ``source_dataset``) are
    collected into plain lists and returned alongside. ``Trainer`` is
    configured with ``remove_unused_columns=False`` in the OCR phase, so
    these are dropped here rather than passed to the model.
    """
    pad_id = 0
    text_keys = {"input_ids", "attention_mask", "labels"}

    out: Dict[str, Any] = {
        "input_ids": _pad_stack([b["input_ids"] for b in batch], pad_id),
        "attention_mask": _pad_stack(
            [b["attention_mask"] for b in batch], 0
        ),
        "labels": _pad_stack([b["labels"] for b in batch], IGNORE_INDEX),
    }

    for key in ("pixel_values", "image_grid_thw"):
        vals = [b.get(key) for b in batch]
        if all(v is not None for v in vals):
            try:
                out[key] = torch.stack(vals)
            except RuntimeError:
                # Variable patch counts (Qwen-VL packs a variable number of
                # visual tokens per image); concatenate instead of stacking.
                out[key] = torch.cat(vals, dim=0)

    for key in ("image_id", "writer_id", "source_dataset"):
        vals = [b.get(key) for b in batch]
        if any(v is not None for v in vals):
            out[key] = vals

    assert not (text_keys - set(out)), "collate_fn dropped a required text key"
    return out
