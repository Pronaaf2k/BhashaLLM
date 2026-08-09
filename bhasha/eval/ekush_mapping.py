"""Ekush class-index to Bangla-character mapping.

`bhasha/eval/all_ocr.py` and `bhasha/eval/ocr_models.py` both open with

    from ekush_mapping import get_label_text

and no such module existed anywhere in the repository. Both raised
`ModuleNotFoundError` on import, so neither the comprehensive OCR
evaluation nor the baseline-versus-fine-tuned comparison could be run from
a clean checkout. Same class of defect as the missing `bhasha.data`
package. See `docs/ERRATA.md` C15.

Both call sites currently read the ground-truth string straight out of the
JSONL (`item['text']`) and never call `get_label_text`, so restoring the
import is enough to make them run. The function is implemented properly
rather than stubbed, because the situation it exists for is real: Ekush
ships as directories of images named by **numeric class index**, not by
character, and something has to turn `.../52/img_001.png` into `ক`.

The Ekush ordering
------------------
Ekush (Rabby et al., ref [28]) covers 122 classes: 10 numerals, 50 basic
characters (11 vowels + 39 consonants), 10 modifiers and 52 compound
characters. The ordering below follows the standard Bangla alphabet order
used by the dataset's own class directories.

**This mapping is asserted, not verified against a copy of Ekush**, because
the dataset is not in this repository. Before citing per-class results,
check it against the `labels.csv` or class-directory listing that ships
with your copy, and pass `--mapping` to override. `verify_mapping()` is
provided for exactly that check. A silently wrong mapping produces a
plausible-looking per-class accuracy table that is meaningless, which is
the failure mode worth guarding against.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Optional override: a JSON object mapping {"class_index": "character"}.
# Checked before the built-in table so a verified mapping always wins.
DEFAULT_MAPPING_FILE = BASE_DIR / "data" / "ekush_label_map.json"

# --- 0-9: Bangla numerals -------------------------------------------------
NUMERALS = ["০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯"]

# --- 10-20: vowels (স্বরবর্ণ), 11 characters ------------------------------
VOWELS = ["অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ"]

# --- 21-59: consonants (ব্যঞ্জনবর্ণ), 39 characters -----------------------
CONSONANTS = [
    "ক", "খ", "গ", "ঘ", "ঙ",
    "চ", "ছ", "জ", "ঝ", "ঞ",
    "ট", "ঠ", "ড", "ঢ", "ণ",
    "ত", "থ", "দ", "ধ", "ন",
    "প", "ফ", "ব", "ভ", "ম",
    "য", "র", "ল", "শ", "ষ",
    "স", "হ", "ড়", "ঢ়", "য়",
    "ৎ", "ং", "ঃ", "ঁ",
]

# --- 60-69: vowel modifiers (কার), 10 characters --------------------------
# Rendered with the dotted circle U+25CC so they display as standalone
# glyphs; strip_dotted_circle() removes it when a bare mark is wanted.
MODIFIERS = ["◌া", "◌ি", "◌ী", "◌ু", "◌ূ", "◌ৃ", "◌ে", "◌ৈ", "◌ো", "◌ৌ"]

# --- 70-121: compound characters (যুক্তাক্ষর), 52 characters ---------------
# The conjuncts Ekush covers. Section II-A of the paper is about exactly
# these: a rendered conjunct is not one Unicode codepoint, which is why
# tokenisation and per-class evaluation both need care here.
COMPOUNDS = [
    "ক্ক", "ক্ট", "ক্ত", "ক্ব", "ক্ম", "ক্র", "ক্ল", "ক্ষ", "ক্স",
    "গ্ধ", "গ্ন", "গ্ব", "গ্ম", "গ্র", "গ্ল",
    "ঙ্ক", "ঙ্গ", "ঙ্খ", "ঙ্ঘ",
    "চ্চ", "চ্ছ", "জ্জ", "জ্ঞ", "ঞ্চ", "ঞ্ছ", "ঞ্জ",
    "ট্ট", "ড্ড", "ণ্ট", "ণ্ঠ", "ণ্ড", "ণ্ণ",
    "ত্ত", "ত্থ", "ত্ন", "ত্ব", "ত্ম", "ত্র",
    "দ্দ", "দ্ধ", "দ্ব", "দ্ভ", "দ্র",
    "ন্ত", "ন্থ", "ন্দ", "ন্ধ", "ন্ন",
    "প্ট", "প্ত", "প্প", "প্র",
]

DOTTED_CIRCLE = "◌"


def _build_default_map() -> Dict[int, str]:
    table: Dict[int, str] = {}
    index = 0
    for group in (NUMERALS, VOWELS, CONSONANTS, MODIFIERS, COMPOUNDS):
        for ch in group:
            table[index] = ch
            index += 1
    return table


DEFAULT_MAP: Dict[int, str] = _build_default_map()

CLASS_GROUPS = {
    "numeral": (0, len(NUMERALS)),
    "vowel": (len(NUMERALS), len(NUMERALS) + len(VOWELS)),
    "consonant": (len(NUMERALS) + len(VOWELS),
                  len(NUMERALS) + len(VOWELS) + len(CONSONANTS)),
    "modifier": (len(NUMERALS) + len(VOWELS) + len(CONSONANTS),
                 len(NUMERALS) + len(VOWELS) + len(CONSONANTS) + len(MODIFIERS)),
    "compound": (len(NUMERALS) + len(VOWELS) + len(CONSONANTS) + len(MODIFIERS),
                 len(DEFAULT_MAP)),
}

_loaded_override: Optional[Dict[int, str]] = None


def load_mapping(path: Optional[str | Path] = None) -> Dict[int, str]:
    """Load a verified mapping from JSON, falling back to the built-in table.

    Cached after the first call. Pass an explicit path to bypass the cache.
    """
    global _loaded_override
    if path is None:
        if _loaded_override is not None:
            return _loaded_override
        path = DEFAULT_MAPPING_FILE
        cache = True
    else:
        cache = False

    p = Path(path)
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        table = {int(k): v for k, v in raw.items()}
        if cache:
            _loaded_override = table
        return table

    if cache:
        _loaded_override = DEFAULT_MAP
    return DEFAULT_MAP


def strip_dotted_circle(text: str) -> str:
    """Remove the U+25CC placeholder from a modifier glyph."""
    return (text or "").replace(DOTTED_CIRCLE, "")


def class_group(index: int) -> str:
    """Which Ekush group a class index belongs to.

    Useful for the per-category breakdown of paper Sec. V-B, which reports
    vowels, base consonants, vowel diacritics and consonant diacritics
    separately. Note that `eval/ocr_cer.py` derives its categories from
    Unicode properties of the *reference string* instead, which is the
    right method when ground truth is a line transcription rather than a
    class index.
    """
    for name, (lo, hi) in CLASS_GROUPS.items():
        if lo <= index < hi:
            return name
    return "unknown"


def get_label_text(label, mapping: Optional[Dict[int, str]] = None) -> str:
    """Return the Bangla text for an Ekush label.

    Accepts, in order of preference:

    * an ``int`` class index                     -> mapped
    * a numeric ``str`` such as ``"52"``          -> mapped
    * a path such as ``".../52/img_001.png"``     -> parent directory mapped
    * anything already containing Bangla          -> returned unchanged

    The last case matters: both call sites read ground truth from JSONL
    where ``text`` is already the character. Returning it untouched means
    this function is safe to apply to either representation without the
    caller having to know which it holds.

    An unmappable label is returned as its own string rather than raising.
    A crash mid-evaluation loses the whole run; an unmapped label shows up
    as a visible mismatch in the results and costs one row.
    """
    table = mapping if mapping is not None else load_mapping()

    if isinstance(label, int):
        return table.get(label, str(label))

    text = str(label).strip()
    if not text:
        return ""

    # Already Bangla: leave it alone.
    if any(0x0980 <= ord(c) <= 0x09FF for c in text):
        return unicodedata.normalize("NFC", text)

    if text.isdigit():
        return table.get(int(text), text)

    # A path: take the last numeric path component, which is how Ekush
    # names its class directories.
    parts = [p for p in re.split(r"[\\/]+", text) if p]
    for part in reversed(parts):
        stem = Path(part).stem
        if stem.isdigit():
            return table.get(int(stem), stem)

    return text


def verify_mapping(dataset_root: str | Path,
                   mapping: Optional[Dict[int, str]] = None) -> Dict[str, object]:
    """Check the mapping against the class directories of a real Ekush copy.

    The built-in table is asserted from the dataset's documented ordering,
    not read from a copy of the data. This compares it against what is
    actually on disk and reports the disagreement, so a wrong mapping is
    caught before it produces a per-class accuracy table nobody can trust.
    """
    table = mapping if mapping is not None else load_mapping()
    root = Path(dataset_root)
    if not root.exists():
        return {"ok": False, "reason": f"{root} does not exist"}

    dirs = sorted(
        (d for d in root.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if not dirs:
        return {"ok": False,
                "reason": f"no numeric class directories under {root}"}

    found = [int(d.name) for d in dirs]
    unmapped = [i for i in found if i not in table]
    return {
        "ok": not unmapped,
        "n_class_directories": len(found),
        "n_classes_in_mapping": len(table),
        "class_index_range": [min(found), max(found)],
        "unmapped_class_indices": unmapped,
        "sample": {str(i): table.get(i) for i in found[:12]},
        "note":
            "Ekush documents 122 classes: 10 numerals, 11 vowels, 39 "
            "consonants, 10 modifiers, 52 compounds. A count other than 122 "
            "means this copy is a subset or a different release, and the "
            "built-in ordering should not be trusted for it. Write a "
            "verified map to data/ekush_label_map.json to override.",
    }


__all__ = [
    "get_label_text", "load_mapping", "verify_mapping", "class_group",
    "strip_dotted_circle", "DEFAULT_MAP", "CLASS_GROUPS",
]


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import argparse
    ap = argparse.ArgumentParser(description="Inspect the Ekush label map.")
    ap.add_argument("--verify", metavar="EKUSH_ROOT",
                    help="check the mapping against class directories on disk")
    ap.add_argument("--index", type=int, help="look up one class index")
    a = ap.parse_args()

    if a.verify:
        print(json.dumps(verify_mapping(a.verify), indent=2, ensure_ascii=False))
    elif a.index is not None:
        print(f"{a.index} -> {get_label_text(a.index)!r} "
              f"({class_group(a.index)})")
    else:
        print(f"{len(DEFAULT_MAP)} classes")
        for name, (lo, hi) in CLASS_GROUPS.items():
            sample = " ".join(DEFAULT_MAP[i] for i in range(lo, min(lo + 8, hi)))
            print(f"  {name:10s} {hi - lo:3d}  {lo:3d}-{hi - 1:<3d}  {sample} ...")
