"""Dataset and preprocessing components for BhashaLLM.

This package was referenced by three modules in the repository
(``bhasha/ocr/train.py``, ``bhasha/scripts/train_ocr_improved.py`` and
``bhasha/scripts/debug_dataset_shapes.py``) but was never committed, so the
Phase-3 OCR training path raised ``ModuleNotFoundError`` on import. The
modules here restore that import and additionally implement the
preprocessing steps the paper describes but the repository did not carry:

``dataset``
    ``OCRDataset`` / ``collate_fn`` for the vision-language OCR phase
    (paper Sec. IV-C, Phase 3).

``text_corpus``
    NFKC normalisation, Latin stripping and the 80/10/10 **by-document**
    split described in paper Sec. IV-B.

``ekush_sampling``
    Stratified sampling by grapheme root, described in paper Sec. IV-C as
    the method used to draw 6,000 Ekush images "so that rare conjuncts
    survived sampling".

``manifest``
    The handwriting manifest schema. ``writer_id`` is what makes the
    writer-disjoint evaluation of Sec. VI-D possible at all.

Nothing in this package replaces existing functionality; every symbol here
is new.
"""

from pathlib import Path

# Repository root, resolved from this file so nothing depends on the caller's
# working directory. bhasha/data/__init__.py -> bhasha/data -> bhasha -> root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

__all__ = ["BASE_DIR"]
